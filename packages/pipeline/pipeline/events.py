from __future__ import annotations

from typing import Any, Optional


def _adaptive_ear_threshold(ears: list[float], default: float = 0.18) -> Optional[float]:
    """Return blink threshold or None if eyes too stable (still-face / no blink signal)."""
    if len(ears) < 10:
        return default
    import statistics

    med = statistics.median(ears)
    std = statistics.pstdev(ears) if len(ears) > 1 else 0.0
    # still face: almost no variance → do not invent blinks
    if std < 0.015:
        return None
    thr = med - max(0.04, 1.5 * std)
    # keep sane bounds
    return float(min(max(thr, 0.10), 0.28))


def detect_events(
    vision_rows: list[dict[str, Any]],
    vad_segments: list[dict[str, Any]],
    ear_thr: float = 0.18,
    face_detected_ms: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    eid = 1

    ears = []
    for r in vision_rows:
        if r.get("face_detected") == 1 and r.get("ear_left") is not None and r.get("ear_right") is not None:
            ears.append((float(r["ear_left"]) + float(r["ear_right"])) / 2.0)
    thr = _adaptive_ear_threshold(ears, default=ear_thr)

    # blink via EAR (skip when thr is None = still face)
    if thr is not None:
        closed = False
        blink_start: Optional[int] = None
        for r in vision_rows:
            ear_l = r.get("ear_left")
            ear_r = r.get("ear_right")
            if ear_l is None or ear_r is None or r.get("face_detected") != 1:
                continue
            ear = (ear_l + ear_r) / 2.0
            ts = int(r["timestamp_ms"])
            if ear < thr and not closed:
                closed = True
                blink_start = ts
            elif ear >= thr and closed:
                closed = False
                start = blink_start if blink_start is not None else ts
                dur = max(ts - start, 1)
                if 40 <= dur <= 350:
                    events.append(
                        {
                            "event_id": f"evt_{eid:04d}",
                            "type": "blink",
                            "start_ms": start,
                            "end_ms": ts,
                            "duration_ms": dur,
                            "attributes": {"left_eye": True, "right_eye": True, "ear_thr": thr},
                        }
                    )
                    eid += 1
                blink_start = None

    # pause >= 2s on merged VAD non_speech
    for seg in vad_segments:
        if seg.get("label") == "non_speech" and int(seg.get("duration_ms") or 0) >= 2000:
            events.append(
                {
                    "event_id": f"evt_{eid:04d}",
                    "type": "pause_2s",
                    "start_ms": int(seg["start_ms"]),
                    "end_ms": int(seg["end_ms"]),
                    "duration_ms": int(seg["duration_ms"]),
                    "attributes": {},
                }
            )
            eid += 1

    # speech_onset: only speech segments with meaningful duration
    for seg in vad_segments:
        if seg.get("label") == "speech" and int(seg.get("duration_ms") or 0) >= 250:
            events.append(
                {
                    "event_id": f"evt_{eid:04d}",
                    "type": "speech_onset",
                    "start_ms": int(seg["start_ms"]),
                    "end_ms": None,
                    "duration_ms": None,
                    "attributes": {"segment_duration_ms": int(seg["duration_ms"])},
                }
            )
            eid += 1

    # head_nod best-effort
    pitches = [(int(r["timestamp_ms"]), r.get("head_pitch")) for r in vision_rows if r.get("head_pitch") is not None]
    i = 0
    while i < len(pitches) - 5:
        t0, _p0 = pitches[i]
        window = [x for x in pitches[i:] if x[0] - t0 <= 800]
        if len(window) >= 5:
            vals = [p for _, p in window if p is not None]
            if vals:
                delta = max(vals) - min(vals)
                if delta >= 10:
                    events.append(
                        {
                            "event_id": f"evt_{eid:04d}",
                            "type": "head_nod",
                            "start_ms": t0,
                            "end_ms": window[-1][0],
                            "duration_ms": window[-1][0] - t0,
                            "attributes": {"mean_pitch_change_deg": float(delta)},
                        }
                    )
                    eid += 1
                    i += max(len(window), 1)
                    continue
        i += 1

    def summary_for(typ: str) -> dict[str, Any]:
        subset = [e for e in events if e["type"] == typ]
        out: dict[str, Any] = {"type": typ, "count": len(subset)}
        durs = [e["duration_ms"] for e in subset if e.get("duration_ms")]
        if durs:
            out["mean_duration_ms"] = float(sum(durs) / len(durs))
            out["total_duration_ms"] = int(sum(durs))
        if typ == "blink" and face_detected_ms > 0:
            mins = face_detected_ms / 60000.0
            out["rate_per_min"] = float(len(subset) / mins) if mins > 0 else 0.0
            out["rate_per_min_denominator"] = "face_detected_minutes"
            if thr is None:
                out["note"] = "still_face_no_blink_signal"
        if typ == "head_nod":
            deltas = [e["attributes"].get("mean_pitch_change_deg") for e in subset if e.get("attributes")]
            deltas = [d for d in deltas if d is not None]
            if deltas:
                out["mean_pitch_change_deg"] = float(sum(deltas) / len(deltas))
        return out

    summary = [summary_for(t) for t in ["blink", "pause_2s", "speech_onset", "head_nod", "saccade"]]
    for s in summary:
        if s["type"] == "saccade" and s["count"] == 0:
            s["note"] = "best_effort_not_detected"
    return events, summary
