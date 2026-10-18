"""OpenFace AU extraction via native binary or Docker sidecar."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

# OpenFace intensity columns (0–5) → report metric codes (0–1)
AU_SPEC: list[tuple[str, str, str]] = [
    ("AU01_r", "au_01", "AU01 内眉上扬强度"),
    ("AU02_r", "au_02", "AU02 外眉上扬强度"),
    ("AU04_r", "au_04", "AU04 皱眉强度"),
    ("AU05_r", "au_05", "AU05 上眼睑上扬强度"),
    ("AU06_r", "au_06", "AU06 提颊强度"),
    ("AU07_r", "au_07", "AU07 眼睑收紧强度"),
    ("AU09_r", "au_09", "AU09 皱鼻强度"),
    ("AU10_r", "au_10", "AU10 上唇上扬强度"),
    ("AU12_r", "au_12", "AU12 嘴角上扬强度"),
    ("AU14_r", "au_14", "AU14 嘴角收紧强度"),
    ("AU15_r", "au_15", "AU15 嘴角下拉强度"),
    ("AU17_r", "au_17", "AU17 抬颏强度"),
    ("AU20_r", "au_20", "AU20 嘴角水平拉伸强度"),
    ("AU23_r", "au_23", "AU23 唇收紧强度"),
    ("AU25_r", "au_25", "AU25 唇分开强度"),
    ("AU26_r", "au_26", "AU26 下颌下落强度"),
    ("AU45_r", "au_45", "AU45 眨眼强度"),
]

DEFAULT_DOCKER_IMAGE = "benbuleong/openface-cambridge-2.2.0"
OPENFACE_VERSION = "2.2.0"


def _normalize_intensity(raw: float) -> float:
    return round(max(0.0, min(1.0, raw / 5.0)), 4)


def resolve_openface_bin(openface_bin: str = "") -> Optional[str]:
    candidate = (openface_bin or os.environ.get("OPENFACE_BIN", "")).strip()
    if candidate:
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        return None
    return shutil.which("FeatureExtraction")


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def parse_openface_csv(csv_path: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        # OpenFace headers often have leading spaces
        if reader.fieldnames:
            reader.fieldnames = [h.strip() for h in reader.fieldnames]
        for row in reader:
            cleaned = {(k or "").strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            try:
                frame_idx = int(float(cleaned.get("frame", len(frames) + 1)))
            except (TypeError, ValueError):
                frame_idx = len(frames) + 1
            try:
                ts = float(cleaned.get("timestamp", 0) or 0)
            except (TypeError, ValueError):
                ts = 0.0
            timestamp_ms = int(round(ts * 1000))
            success = str(cleaned.get("success", "0")).lower() in {"1", "true", "yes"}
            conf_raw = cleaned.get("confidence")
            try:
                confidence = float(conf_raw) if conf_raw not in (None, "", "NaN") else None
            except ValueError:
                confidence = None

            item: dict[str, Any] = {
                "frame": frame_idx,
                "timestamp_ms": timestamp_ms,
                "face_detected": 1 if success else 0,
                "face_confidence": confidence,
            }
            for col, code, _name in AU_SPEC:
                raw = cleaned.get(col)
                if not success or raw in (None, "", "NaN"):
                    item[code] = None
                    continue
                try:
                    item[code] = _normalize_intensity(float(raw))
                except ValueError:
                    item[code] = None
            frames.append(item)
    return frames


def _run_native(bin_path: str, video_path: Path, out_dir: Path) -> Path:
    cmd = [bin_path, "-f", str(video_path), "-out_dir", str(out_dir), "-aus", "-q"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "unknown error").strip()
        raise RuntimeError(f"OpenFace native failed: {detail[:800]}")
    csv_path = out_dir / f"{video_path.stem}.csv"
    if not csv_path.exists():
        csvs = sorted(out_dir.glob("*.csv"))
        if not csvs:
            raise RuntimeError("OpenFace produced no CSV")
        csv_path = csvs[0]
    return csv_path


def _run_docker(video_path: Path, out_dir: Path, image: str) -> Path:
    video_path = video_path.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Mount parent dir so Chinese / spaced paths stay stable inside container
    host_dir = video_path.parent
    cmd = [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "-v",
        f"{host_dir}:/data:ro",
        "-v",
        f"{out_dir}:/out",
        "-w",
        "/opt/OpenFace/build/bin",
        image,
        "./FeatureExtraction",
        "-f",
        f"/data/{video_path.name}",
        "-out_dir",
        "/out",
        "-aus",
        "-q",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "unknown error").strip()
        raise RuntimeError(f"OpenFace docker failed: {detail[:800]}")
    csv_path = out_dir / f"{video_path.stem}.csv"
    if not csv_path.exists():
        csvs = sorted(out_dir.glob("*.csv"))
        if not csvs:
            raise RuntimeError("OpenFace docker produced no CSV")
        csv_path = csvs[0]
    return csv_path


def run_openface_au(
    video_path: str | Path,
    work_dir: Path,
    *,
    openface_bin: str = "",
    docker_image: Optional[str] = None,
) -> dict[str, Any]:
    """Extract AU intensities. Prefer native binary, else Docker image."""
    video_path = Path(video_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    image = (docker_image or os.environ.get("OPENFACE_DOCKER_IMAGE") or DEFAULT_DOCKER_IMAGE).strip()

    bin_path = resolve_openface_bin(openface_bin)
    mode = "native" if bin_path else None
    csv_path: Optional[Path] = None
    error: Optional[str] = None

    try:
        if bin_path:
            csv_path = _run_native(bin_path, video_path, work_dir)
            mode = "native"
        elif docker_available():
            csv_path = _run_docker(video_path, work_dir, image)
            mode = "docker"
        else:
            return {
                "ok": False,
                "reason": "openface_unavailable",
                "detail": "Neither OPENFACE_BIN nor usable Docker found",
                "frames": [],
                "csv_path": None,
                "extraction_mode": None,
                "model_version": OPENFACE_VERSION,
            }
    except Exception as exc:  # noqa: BLE001 — surface to report empty-state path
        error = str(exc)
        return {
            "ok": False,
            "reason": "openface_failed",
            "detail": error,
            "frames": [],
            "csv_path": None,
            "extraction_mode": mode,
            "model_version": OPENFACE_VERSION,
        }

    frames = parse_openface_csv(csv_path)
    success_n = sum(1 for f in frames if f.get("face_detected") == 1)
    return {
        "ok": True,
        "reason": None,
        "detail": None,
        "frames": frames,
        "csv_path": str(csv_path),
        "extraction_mode": f"openface_{OPENFACE_VERSION}_{mode}",
        "model_version": OPENFACE_VERSION,
        "frame_count": len(frames),
        "face_success_frames": success_n,
        "docker_image": image if mode == "docker" else None,
    }
