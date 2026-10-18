from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _energy_vad(y: np.ndarray, sr: int, frame_ms: int = 25, hop_ms: int = 25, thr_db: float = -40.0):
    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if frame < 1:
        frame = 1
    if hop < 1:
        hop = 1
    energies = []
    flags = []
    for start in range(0, max(len(y) - frame, 1), hop):
        seg = y[start : start + frame]
        rms = float(np.sqrt(np.mean(seg**2) + 1e-12))
        db = 20 * np.log10(rms + 1e-12)
        energies.append(db)
        flags.append(1 if db > thr_db else 0)
    return np.array(energies), np.array(flags), hop


def _flags_to_segments(flags: np.ndarray, hop_ms: int) -> list[dict[str, Any]]:
    segs = []
    if len(flags) == 0:
        return segs
    cur = int(flags[0])
    start_i = 0
    for i in range(1, len(flags)):
        if int(flags[i]) != cur:
            segs.append(
                {
                    "start_ms": start_i * hop_ms,
                    "end_ms": i * hop_ms,
                    "label": "speech" if cur == 1 else "non_speech",
                    "duration_ms": (i - start_i) * hop_ms,
                }
            )
            cur = int(flags[i])
            start_i = i
    segs.append(
        {
            "start_ms": start_i * hop_ms,
            "end_ms": len(flags) * hop_ms,
            "label": "speech" if cur == 1 else "non_speech",
            "duration_ms": (len(flags) - start_i) * hop_ms,
        }
    )
    return segs


def _merge_vad_segments(
    segs: list[dict[str, Any]],
    *,
    min_speech_ms: int = 250,
    min_non_speech_ms: int = 200,
) -> list[dict[str, Any]]:
    """Absorb short flips into neighbors to reduce VAD chatter."""
    if not segs:
        return segs
    merged: list[dict[str, Any]] = [dict(segs[0])]
    for seg in segs[1:]:
        prev = merged[-1]
        if seg["label"] == prev["label"]:
            prev["end_ms"] = seg["end_ms"]
            prev["duration_ms"] = prev["end_ms"] - prev["start_ms"]
            continue
        # short opposite segment: absorb into previous
        thr = min_speech_ms if seg["label"] == "speech" else min_non_speech_ms
        if int(seg["duration_ms"]) < thr:
            prev["end_ms"] = seg["end_ms"]
            prev["duration_ms"] = prev["end_ms"] - prev["start_ms"]
            continue
        merged.append(dict(seg))
    # second pass: drop leading/trailing micro segments by merging into neighbor
    out: list[dict[str, Any]] = []
    for seg in merged:
        thr = min_speech_ms if seg["label"] == "speech" else min_non_speech_ms
        if out and int(seg["duration_ms"]) < thr:
            out[-1]["end_ms"] = seg["end_ms"]
            out[-1]["duration_ms"] = out[-1]["end_ms"] - out[-1]["start_ms"]
            continue
        if out and seg["label"] == out[-1]["label"]:
            out[-1]["end_ms"] = seg["end_ms"]
            out[-1]["duration_ms"] = out[-1]["end_ms"] - out[-1]["start_ms"]
        else:
            out.append(seg)
    return out


def run_audio(wav_path: Optional[str], duration_ms: int) -> dict[str, Any]:
    """Acoustic features + energy VAD. If no wav, return empty/disabled."""
    if not wav_path:
        return {
            "ok": False,
            "reason": "no_audio",
            "hop_ms": 25,
            "rows": [],
            "vad_segments": [],
            "speech_duration_ms": 0,
            "vad_segment_count": 0,
        }

    import librosa

    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    hop_ms = 25
    energies, flags, hop = _energy_vad(y, sr, hop_ms=hop_ms)
    # adaptive threshold: high percentile of low-energy floor
    if len(energies):
        floor = float(np.percentile(energies, 20))
        thr = max(floor + 6.0, float(np.median(energies)) - 1.0)
        flags = (energies > thr).astype(int)
        # if still all silent but signal exists, mark upper half as speech for continuity
        if flags.sum() == 0 and float(np.max(energies)) > -60:
            thr2 = float(np.percentile(energies, 50))
            flags = (energies >= thr2).astype(int)

    segs = _merge_vad_segments(_flags_to_segments(flags, hop_ms))
    speech_ms = int(sum(s["duration_ms"] for s in segs if s["label"] == "speech"))

    # frame features
    frame = int(sr * hop_ms / 1000)
    hop_len = max(frame, 1)
    f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr, frame_length=max(frame * 4, 512), hop_length=hop_len)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_len)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_len)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_len)
    n = min(len(energies), len(f0), len(cent), len(zcr), mfcc.shape[1], len(flags))

    rows = []
    for i in range(n):
        rows.append(
            {
                "timestamp_ms": i * hop_ms,
                "speech_detected": int(flags[i]) if i < len(flags) else 0,
                "energy_db": float(energies[i]) if i < len(energies) else None,
                "f0_hz": float(f0[i]) if np.isfinite(f0[i]) and flags[i] == 1 else None,
                "spectral_centroid": float(cent[i]),
                "zcr": float(zcr[i]),
                **{f"mfcc_{k+1}": float(mfcc[k, i]) for k in range(13)},
            }
        )

    return {
        "ok": True,
        "hop_ms": hop_ms,
        "sample_rate": sr,
        "rows": rows,
        "vad_segments": segs,
        "speech_duration_ms": speech_ms,
        "vad_segment_count": len(segs),
        "duration_ms_audio": int(len(y) / sr * 1000),
        "duration_ms_video": duration_ms,
    }
