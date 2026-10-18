from __future__ import annotations

from typing import Any, Optional

from .openface_pipe import AU_SPEC
from .utils import safe_stats

FACE_METRICS = [
    ("face_confidence", "面部检测置信度", "face_head", "0-1"),
    ("head_yaw", "头部左右转角", "face_head", "deg"),
    ("head_pitch", "头部上下点头角", "face_head", "deg"),
    ("head_roll", "头部左右倾斜角", "face_head", "deg"),
]
GAZE_METRICS = [
    ("gaze_yaw", "视线水平偏角", "gaze", "deg"),
    ("gaze_pitch", "视线垂直偏角", "gaze", "deg"),
    ("iris_left_ratio", "左眼瞳孔占比", "gaze", "ratio"),
    ("iris_right_ratio", "右眼瞳孔占比", "gaze", "ratio"),
    ("pupil_left_x", "左瞳孔水平位置", "gaze", "0-1"),
    ("pupil_left_y", "左瞳孔垂直位置", "gaze", "0-1"),
]
ACOUSTIC_METRICS = [
    ("f0_hz", "基频", "acoustic", "Hz"),
    ("energy_db", "短时能量", "acoustic", "dB"),
    ("spectral_centroid", "频谱质心", "acoustic", "Hz"),
    ("zcr", "过零率", "acoustic", "ratio"),
] + [(f"mfcc_{i}", f"MFCC 第 {i} 维", "acoustic", "—") for i in range(1, 14)]


def _metric_row(code: str, name: str, group: str, unit: str, values: list, n_total: int, status: str = "ok") -> dict[str, Any]:
    st = safe_stats(values)
    n_valid = int(st["n_valid"] or 0)
    if status == "ok" and n_valid == 0:
        status = "insufficient_data"
    return {
        "metric_code": code,
        "metric_name": name,
        "group": group,
        "unit": unit,
        "status": status,
        "n_valid": n_valid if status != "not_extracted" else None,
        "n_total": n_total if status != "not_extracted" else None,
        "missing_rate": (1.0 - n_valid / n_total) if status == "ok" and n_total else None,
        "mean": st["mean"],
        "std": st["std"],
        "min": st["min"],
        "p25": st["p25"],
        "median": st["median"],
        "p75": st["p75"],
        "max": st["max"],
        "mean_confidence": None,
    }


def aggregate_metrics(
    vision_rows: list[dict[str, Any]],
    audio_rows: list[dict[str, Any]],
    asr: dict[str, Any],
    speech_duration_ms: int,
    duration_ms: int,
    enable_au: bool = False,
    au_frames: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    n_vis = len(vision_rows)
    for code, name, group, unit in FACE_METRICS + GAZE_METRICS:
        vals = [r[code] for r in vision_rows if r.get(code) is not None]
        metrics.append(_metric_row(code, name, group, unit, vals, n_vis))

    au_frames = au_frames or []
    if not enable_au:
        for _col, code, name in AU_SPEC:
            metrics.append(_metric_row(code, name, "au", "0-1", [], 0, status="not_extracted"))
    else:
        n_au = len(au_frames)
        for _col, code, name in AU_SPEC:
            vals = [r[code] for r in au_frames if r.get(code) is not None]
            metrics.append(_metric_row(code, name, "au", "0-1", vals, n_au if n_au else 0))

    n_aud = len(audio_rows)
    for code, name, group, unit in ACOUSTIC_METRICS:
        vals = [r[code] for r in audio_rows if r.get(code) is not None]
        status = "ok" if audio_rows else "not_extracted"
        metrics.append(_metric_row(code, name, group, unit, vals, n_aud if n_aud else 0, status=status))

    # speech scalars
    from .asr_pipe import speech_rate_cpm

    if asr.get("ok"):
        chars = int(asr.get("char_count") or 0)
        rate = speech_rate_cpm(chars, speech_duration_ms) if chars > 0 and speech_duration_ms > 0 else None
        non_speech = max(duration_ms - speech_duration_ms, 0)
        # ASR ran successfully: always emit fields (null allowed when no speech content)
        scalars = [
            ("speech_rate_cpm", "语速", "asr", "chars_per_min", rate, "ok" if rate is not None else "insufficient_data"),
            ("non_speech_duration_ms", "非语音时长", "asr", "ms", float(non_speech), "ok"),
            ("utterance_count", "语段数", "asr", "count", float(len(asr.get("utterances") or [])), "ok"),
            ("filler_count", "填充词次数", "asr", "count", float(asr.get("filler_count") or 0), "ok"),
            ("asr_confidence", "识别平均置信度", "asr", "0-1", asr.get("asr_confidence"), "ok" if asr.get("asr_confidence") is not None else "insufficient_data"),
        ]
        for code, name, group, unit, val, status in scalars:
            metrics.append(
                {
                    "metric_code": code,
                    "metric_name": name,
                    "group": group,
                    "unit": unit,
                    "status": status,
                    "n_valid": 1 if val is not None else 0,
                    "n_total": 1,
                    "missing_rate": 0.0 if val is not None else 1.0,
                    "mean": val,
                    "std": None,
                    "min": val,
                    "p25": None,
                    "median": val,
                    "p75": None,
                    "max": val,
                    "mean_confidence": None,
                }
            )
    else:
        for code, name in [
            ("speech_rate_cpm", "语速"),
            ("non_speech_duration_ms", "非语音时长"),
            ("utterance_count", "语段数"),
            ("filler_count", "填充词次数"),
            ("asr_confidence", "识别平均置信度"),
        ]:
            metrics.append(_metric_row(code, name, "asr", "—", [], 0, status="not_extracted"))

    return metrics


def metric_value(metrics: list[dict[str, Any]], code: str) -> Optional[float]:
    for m in metrics:
        if m["metric_code"] == code:
            return m.get("mean")
    return None
