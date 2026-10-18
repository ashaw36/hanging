from __future__ import annotations

from typing import Any


def build_qc_segments(vision_rows: list[dict[str, Any]], min_run_ms: int = 1500) -> list[dict[str, Any]]:
    """Merge consecutive profile_like / no-face runs into QC segments (single source)."""
    segs: list[dict[str, Any]] = []
    if not vision_rows:
        return segs

    def flush(kind: str, start: int, end: int, frames: int):
        if end - start < min_run_ms and frames < 10:
            return
        if kind == "profile":
            segs.append(
                {
                    "start_ms": start,
                    "end_ms": end,
                    "reason_code": "QC_PROFILE",
                    "reason": "侧脸角度较大，注视相关指标可靠度较低",
                    "affected_metrics": ["gaze_yaw", "gaze_pitch", "iris_left_ratio", "iris_right_ratio"],
                    "frame_count": frames,
                }
            )
        elif kind == "noface":
            segs.append(
                {
                    "start_ms": start,
                    "end_ms": end,
                    "reason_code": "QC_NO_FACE",
                    "reason": "未检出面部",
                    "affected_metrics": ["head_yaw", "head_pitch", "head_roll", "gaze_yaw", "gaze_pitch"],
                    "frame_count": frames,
                }
            )

    cur_kind = None
    start_ms = 0
    frames = 0
    for r in vision_rows:
        kind = None
        if r.get("face_detected") != 1:
            kind = "noface"
        elif r.get("profile_like") == 1:
            kind = "profile"
        if kind != cur_kind:
            if cur_kind is not None:
                flush(cur_kind, start_ms, int(r["timestamp_ms"]), frames)
            cur_kind = kind
            start_ms = int(r["timestamp_ms"])
            frames = 1 if kind else 0
        else:
            if kind:
                frames += 1
    if cur_kind is not None and vision_rows:
        flush(cur_kind, start_ms, int(vision_rows[-1]["timestamp_ms"]), frames)
    return segs
