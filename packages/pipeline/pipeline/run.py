from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Sequence

from .aggregate import aggregate_metrics
from .asr_pipe import run_asr
from .audio_pipe import run_audio
from .events import detect_events
from .html_render import render_report_html
from .openface_pipe import run_openface_au
from .probe import extract_audio_wav, probe_video
from .qc import build_qc_segments
from .report_builder import build_and_write
from .utils import ensure_dir
from .vision import run_vision

DEFAULT_PACKAGES = [
    "visual_base",
    "gaze_basic",
    "audio_acoustic",
    "asr_basic",
    "events_basic",
]


def normalize_packages(packages: Optional[Sequence[str]]) -> list[str]:
    if not packages:
        return list(DEFAULT_PACKAGES)
    out = []
    for p in packages:
        for part in str(p).split(","):
            part = part.strip()
            if part and part not in out:
                out.append(part)
    return out or list(DEFAULT_PACKAGES)


def run_pipeline(
    video_path: Path,
    out_dir: Path,
    *,
    asr_model: str = "tiny",
    max_frames: Optional[int] = None,
    frame_stride: int = 1,
    enable_asr: bool = True,
    packages: Optional[Sequence[str]] = None,
    template_html: Optional[Path] = None,
) -> dict[str, Any]:
    video_path = Path(video_path)
    out_dir = ensure_dir(Path(out_dir))
    t0 = time.time()

    pkgs = normalize_packages(packages)
    if not enable_asr and "asr_basic" in pkgs:
        pkgs = [p for p in pkgs if p != "asr_basic"]
    want_vision = any(p in pkgs for p in ("visual_base", "gaze_basic", "au_advanced"))
    want_audio = "audio_acoustic" in pkgs or "asr_basic" in pkgs or "events_basic" in pkgs
    want_asr = "asr_basic" in pkgs and enable_asr
    want_events = "events_basic" in pkgs
    want_au = "au_advanced" in pkgs

    media = probe_video(video_path)
    wav_path = out_dir / "process" / "_audio_16k.wav"
    ensure_dir(wav_path.parent)
    audio_ok = False
    if want_audio:
        audio_ok = extract_audio_wav(video_path, wav_path)
    if not audio_ok:
        wav_path = None  # type: ignore

    if want_vision:
        vision = run_vision(str(video_path), max_frames=max_frames, stride=frame_stride)
    else:
        vision = {
            "fps": media["fps"],
            "frame_count_scanned": 0,
            "frame_count_video": media["frame_count"],
            "width": media["width"],
            "height": media["height"],
            "rows": [],
            "track": {
                "primary_track_id": "track_0",
                "candidate_face_count_max": 0,
                "track_switches": 0,
                "face_gap_segments": [],
                "bbox_point_count": 0,
            },
            "coverage": {
                "face_detected_frames": 0,
                "face_detected_rate": 0.0,
                "pose_detected_frames": 0,
                "pose_detected_rate": 0.0,
            },
        }

    if vision.get("frame_count_video"):
        media["frame_count"] = vision["frame_count_video"]
        media["fps"] = vision["fps"]
        media["width"] = vision["width"]
        media["height"] = vision["height"]
        media["duration_ms"] = int(round(media["frame_count"] / max(media["fps"], 1e-6) * 1000))

    if want_audio:
        audio = run_audio(str(wav_path) if wav_path else None, media["duration_ms"])
    else:
        audio = {
            "ok": False,
            "reason": "disabled",
            "hop_ms": 25,
            "rows": [],
            "vad_segments": [],
            "speech_duration_ms": 0,
            "vad_segment_count": 0,
        }

    if want_asr:
        asr = run_asr(str(wav_path) if wav_path else None, model_size=asr_model)
    else:
        asr = {
            "ok": False,
            "reason": "disabled",
            "words": [],
            "utterances": [],
            "text": "",
            "asr_confidence": None,
            "char_count": 0,
            "filler_count": 0,
            "asr_word_count": 0,
        }

    qc_segments = build_qc_segments(vision["rows"]) if want_vision else []
    face_ms = int(vision["coverage"]["face_detected_frames"] / max(vision["fps"], 1e-6) * 1000)
    if want_events:
        events, event_summary = detect_events(
            vision["rows"], audio.get("vad_segments") or [], face_detected_ms=face_ms
        )
    else:
        events, event_summary = [], []

    au: dict[str, Any]
    if want_au:
        au_work = ensure_dir(out_dir / "process" / "openface")
        au = run_openface_au(video_path, au_work)
    else:
        au = {
            "ok": False,
            "reason": "package_disabled",
            "frames": [],
            "extraction_mode": None,
            "model_version": None,
        }

    metrics = aggregate_metrics(
        vision["rows"],
        audio.get("rows") or [],
        asr,
        speech_duration_ms=int(audio.get("speech_duration_ms") or 0),
        duration_ms=int(media["duration_ms"]),
        enable_au=want_au,
        au_frames=au.get("frames") or [],
    )
    # When packages omit modalities, mark related metric groups as not_extracted
    if not want_vision:
        for m in metrics:
            if m["group"] in ("face_head", "gaze", "au"):
                m["status"] = "not_extracted"
                for k in ("n_valid", "n_total", "missing_rate", "mean", "std", "min", "p25", "median", "p75", "max"):
                    m[k] = None
    if not want_audio:
        for m in metrics:
            if m["group"] == "acoustic":
                m["status"] = "not_extracted"
                for k in ("n_valid", "n_total", "missing_rate", "mean", "std", "min", "p25", "median", "p75", "max"):
                    m[k] = None
    if not want_asr:
        for m in metrics:
            if m["group"] == "asr":
                m["status"] = "not_extracted"
                for k in ("n_valid", "n_total", "missing_rate", "mean", "std", "min", "p25", "median", "p75", "max"):
                    m[k] = None

    bundle = build_and_write(
        out_dir=out_dir,
        media=media,
        vision=vision,
        audio=audio,
        asr=asr,
        qc_segments=qc_segments,
        events=events,
        event_summary=event_summary,
        metrics=metrics,
        packages=pkgs,
        au=au,
    )

    html = render_report_html(bundle, template_path=template_html)
    (out_dir / "report.html").write_text(html, encoding="utf-8")

    elapsed = time.time() - t0
    run_log = out_dir / "process" / "run_log.json"
    if run_log.exists():
        import json

        data = json.loads(run_log.read_text(encoding="utf-8"))
        data["elapsed_sec"] = round(elapsed, 2)
        data["realtime_factor"] = round(elapsed / max(media["duration_ms"] / 1000, 1e-6), 3)
        data["packages"] = pkgs
        run_log.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return bundle
