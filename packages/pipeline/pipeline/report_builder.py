from __future__ import annotations

import csv
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from . import METRIC_REGISTRY_VERSION, SCHEMA_VERSION, __version__
from .utils import ensure_dir, ms_to_mmss, sha256_file, write_json, write_jsonl


def build_and_write(
    out_dir: Path,
    media: dict[str, Any],
    vision: dict[str, Any],
    audio: dict[str, Any],
    asr: dict[str, Any],
    qc_segments: list[dict[str, Any]],
    events: list[dict[str, Any]],
    event_summary: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    packages: list[str],
    au: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    process_dir = ensure_dir(out_dir / "process")
    features_dir = ensure_dir(out_dir / "features")

    job_id = f"job_{uuid4().hex[:12]}"
    report_id = f"rpt_{uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # timeseries merge (second resolution for demo size)
    import pandas as pd

    vdf = pd.DataFrame(vision["rows"])
    if not vdf.empty:
        vdf["second"] = (vdf["timestamp_ms"] // 1000).astype(int)
        vsec = vdf.groupby("second", as_index=False).mean(numeric_only=True)
    else:
        vsec = pd.DataFrame()

    adf = pd.DataFrame(audio.get("rows") or [])
    if not adf.empty:
        adf["second"] = (adf["timestamp_ms"] // 1000).astype(int)
        asec = adf.groupby("second", as_index=False).mean(numeric_only=True)
    else:
        asec = pd.DataFrame()

    if not vsec.empty and not asec.empty:
        ts = pd.merge(vsec, asec, on="second", how="outer", suffixes=("", "_aud")).sort_values("second")
    elif not vsec.empty:
        ts = vsec
    else:
        ts = asec
    ts_path = features_dir / "timeseries.parquet"
    if not ts.empty:
        ts.to_parquet(ts_path, index=False)
    else:
        pd.DataFrame({"second": []}).to_parquet(ts_path, index=False)

    write_jsonl(features_dir / "events.jsonl", events)
    write_json(process_dir / "media_probe.json", media)
    write_jsonl(
        process_dir / "face_track.jsonl",
        [
            {
                "timestamp_ms": r["timestamp_ms"],
                "face_detected": r["face_detected"],
                "bbox": r.get("face_bbox"),
                "face_confidence": r.get("face_confidence"),
            }
            for r in vision["rows"]
            if r.get("face_detected") == 1
        ],
    )
    write_jsonl(process_dir / "vad_segments.jsonl", audio.get("vad_segments") or [])
    write_jsonl(process_dir / "asr_words.jsonl", asr.get("words") or [])
    write_jsonl(process_dir / "utterances.jsonl", asr.get("utterances") or [])
    write_json(process_dir / "qc_segments.json", qc_segments)
    au = au or {}
    au_frames = au.get("frames") or []
    if au_frames:
        write_jsonl(features_dir / "au_frames.jsonl", au_frames)
    # Frame-level series for the portrait layer (100ms alignment). The second-level
    # parquet stays as the base summary.
    _vision_keys = (
        "timestamp_ms", "frame_idx", "face_detected", "face_confidence",
        "head_yaw", "head_pitch", "head_roll", "gaze_yaw", "gaze_pitch",
        "pupil_left_x", "pupil_left_y", "iris_left_ratio", "iris_right_ratio",
        "ear_left", "ear_right", "pose_detected", "body_visibility", "profile_like",
    )
    write_jsonl(
        features_dir / "vision_frames.jsonl",
        [{k: r.get(k) for k in _vision_keys} for r in vision["rows"]],
    )
    write_jsonl(features_dir / "audio_frames.jsonl", audio.get("rows") or [])
    write_json(
        process_dir / "openface_summary.json",
        {
            "ok": bool(au.get("ok")),
            "extraction_mode": au.get("extraction_mode"),
            "model_version": au.get("model_version"),
            "frame_count": au.get("frame_count"),
            "face_success_frames": au.get("face_success_frames"),
            "csv_path": au.get("csv_path"),
            "docker_image": au.get("docker_image"),
            "reason": au.get("reason"),
            "detail": au.get("detail"),
        },
    )
    align_manifest = {
        "visual_clock": "round(frame_idx/fps*1000)",
        "audio_hop_ms": audio.get("hop_ms", 25),
        "output_resolution": "second",
        "av_offset_ms": 0,
        "resize_max_side": None,
        "frame_stride": 1,
        "interpolation": "mean_within_second",
    }
    write_json(process_dir / "align_manifest.json", align_manifest)
    repro = {
        "packages": packages,
        "confidence_threshold": 0.5,
        "models": {
            "face": "mediapipe_face_mesh",
            "pose": "mediapipe_pose",
            "vad": "energy_vad",
            "asr": f"faster-whisper-{asr.get('model_size')}" if asr.get("ok") else None,
            "au": au.get("extraction_mode") if au.get("ok") else None,
        },
        "metric_registry_version": METRIC_REGISTRY_VERSION,
        "pipeline_version": __version__,
    }
    write_json(process_dir / "repro_manifest.json", repro)
    write_json(
        process_dir / "run_log.json",
        {
            "finished_at": now,
            "vision_frames": len(vision["rows"]),
            "audio_ok": bool(audio.get("ok")),
            "asr_ok": bool(asr.get("ok")),
            "au_ok": bool(au.get("ok")),
            "au_mode": au.get("extraction_mode"),
        },
    )

    # metrics csv
    csv_path = out_dir / "metrics_summary.csv"
    if metrics:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
            w.writeheader()
            w.writerows(metrics)

    coverage = {
        **vision["coverage"],
        "speech_duration_ms": int(audio.get("speech_duration_ms") or 0),
        "speech_time_rate": (audio.get("speech_duration_ms") or 0) / max(media["duration_ms"], 1),
        "asr_char_count": int(asr.get("char_count") or 0),
        "low_confidence_frames": sum(1 for r in vision["rows"] if r.get("profile_like") == 1),
    }

    process = {
        "media": {
            "container": media.get("container"),
            "video_codec": media.get("video_codec"),
            "width": media.get("width"),
            "height": media.get("height"),
            "fps": media.get("fps"),
            "frame_count": media.get("frame_count"),
            "duration_ms": media.get("duration_ms"),
            "audio_present": media.get("audio_present"),
            "audio_codec": media.get("audio_codec"),
            "audio_src_sr": media.get("audio_src_sr"),
            "audio_out_sr": media.get("audio_out_sr"),
            "audio_channels_src": media.get("audio_channels_src"),
        },
        "track": vision["track"],
        "align": align_manifest,
        "vad_segments_preview": (audio.get("vad_segments") or [])[:8],
        "vad_segment_count": int(audio.get("vad_segment_count") or 0),
        "asr_word_count": int(asr.get("asr_word_count") or 0),
        "qc_segments": qc_segments,
        "repro": repro,
    }

    def art(rel: str) -> dict[str, str]:
        p = out_dir / rel
        return {"url": f"./{rel}", "checksum": sha256_file(p) if p.exists() else ""}

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "report_id": report_id,
        "job_id": job_id,
        "generated_at": now,
        "pipeline_version": __version__,
        "metric_registry_version": METRIC_REGISTRY_VERSION,
        "source": {
            "video_id": f"vid_{uuid4().hex[:8]}",
            "filename": media.get("filename"),
            "duration_ms": media.get("duration_ms"),
            "fps": media.get("fps"),
            "frame_count": media.get("frame_count"),
            "resolution": {"w": media.get("width"), "h": media.get("height")},
            "audio_sample_rate": 16000,
            "video_codec": media.get("video_codec"),
        },
        "config_echo": {
            "temporal_resolution": "second",
            "packages": packages,
            "include_confidence": True,
            "confidence_threshold": 0.5,
        },
        "process": process,
        "coverage": coverage,
        "metrics": metrics,
        "event_summary": event_summary,
        "events_preview": events[:50],
        "asr_text": asr.get("text") if asr.get("ok") else None,
        "artifacts": {
            "process": {
                "media_probe_url": art("process/media_probe.json")["url"],
                "face_track_url": art("process/face_track.jsonl")["url"],
                "vad_segments_url": art("process/vad_segments.jsonl")["url"],
                "asr_words_url": art("process/asr_words.jsonl")["url"],
                "utterances_url": art("process/utterances.jsonl")["url"],
                "qc_segments_url": art("process/qc_segments.json")["url"],
                "align_manifest_url": art("process/align_manifest.json")["url"],
                "repro_manifest_url": art("process/repro_manifest.json")["url"],
                "run_log_url": art("process/run_log.json")["url"],
            },
            "features": {
                "timeseries_url": art("features/timeseries.parquet")["url"],
                "events_url": art("features/events.jsonl")["url"],
            },
            "summary": {
                "metrics_csv_url": art("metrics_summary.csv")["url"],
                "report_json_url": "./report.json",
            },
        },
        "disclaimer": "本报告提供客观可测量的行为特征与处理过程信息，供研究、产品验证与算法开发使用，不构成情绪、人格或医学诊断结论。",
    }
    write_json(out_dir / "report.json", bundle)

    # full zip
    zip_path = out_dir / "full.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out_dir.rglob("*"):
            if p.is_file() and p.name != "full.zip":
                zf.write(p, arcname=str(p.relative_to(out_dir)))

    readme = out_dir / "README.txt"
    readme.write_text(
        f"Behavior feature demo package\njob={job_id}\nduration={ms_to_mmss(media.get('duration_ms'))}\n",
        encoding="utf-8",
    )
    return bundle
