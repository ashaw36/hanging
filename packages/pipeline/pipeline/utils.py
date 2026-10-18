from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ms_to_clock(ms: Optional[float]) -> str:
    if ms is None:
        return "—"
    ms = int(round(ms))
    s, milli = divmod(ms, 1000)
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{sec:02d}.{milli:03d}"
    return f"{m:02d}:{sec:02d}.{milli:03d}"


def ms_to_mmss(ms: Optional[float]) -> str:
    if ms is None:
        return "—"
    s = int(round(ms / 1000.0))
    m, sec = divmod(s, 60)
    return f"{m:02d}:{sec:02d}"


def safe_stats(values: Sequence[float]) -> dict[str, Optional[float]]:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {
            "n_valid": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    vals_sorted = sorted(vals)
    n = len(vals_sorted)

    def pct(p: float) -> float:
        if n == 1:
            return vals_sorted[0]
        idx = (n - 1) * p
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return vals_sorted[lo]
        w = idx - lo
        return vals_sorted[lo] * (1 - w) + vals_sorted[hi] * w

    return {
        "n_valid": n,
        "mean": float(statistics.fmean(vals_sorted)),
        "std": float(statistics.pstdev(vals_sorted)) if n > 1 else 0.0,
        "min": float(vals_sorted[0]),
        "p25": float(pct(0.25)),
        "median": float(pct(0.5)),
        "p75": float(pct(0.75)),
        "max": float(vals_sorted[-1]),
    }


def find_ffmpeg() -> Optional[str]:
    import os
    import shutil

    for key in ("FFMPEG_BINARY", "IMAGEIO_FFMPEG_EXE"):
        v = os.environ.get(key)
        if v and Path(v).exists():
            return v
    which = shutil.which("ffmpeg")
    if which:
        return which
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
