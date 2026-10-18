from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .utils import find_ffmpeg


def probe_video(video_path: Path) -> dict[str, Any]:
    """Probe media with OpenCV (+ optional ffprobe via bundled ffmpeg)."""
    import cv2

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 1e-6:
        fps = 25.0
    duration_ms = int(round(frame_count / fps * 1000)) if frame_count else 0

    media: dict[str, Any] = {
        "container": video_path.suffix.lstrip(".").lower() or "unknown",
        "video_codec": "unknown",
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_ms": duration_ms,
        "audio_present": False,
        "audio_codec": None,
        "audio_src_sr": None,
        "audio_out_sr": 16000,
        "audio_channels_src": None,
        "filename": video_path.name,
        "path": str(video_path.resolve()),
    }

    ff = find_ffmpeg()
    if ff:
        try:
            cmd = [
                ff.replace("ffmpeg", "ffprobe") if "ffmpeg" in Path(ff).name else "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(video_path),
            ]
            # imageio-ffmpeg only ships ffmpeg; use ffmpeg -i parse fallback
            probe_bin = ff
            cmd = [
                probe_bin,
                "-hide_banner",
                "-i",
                str(video_path),
                "-f",
                "null",
                "-",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            err = (proc.stderr or "") + (proc.stdout or "")
            if "Audio:" in err:
                media["audio_present"] = True
                media["audio_codec"] = "embedded"
            if "Video: h264" in err or "Video: avc" in err:
                media["video_codec"] = "h264"
            elif "Video:" in err:
                for line in err.splitlines():
                    if "Video:" in line:
                        parts = line.split("Video:")[-1].strip().split()
                        if parts:
                            media["video_codec"] = parts[0].strip(",")
                        break
        except Exception:
            pass

    # OpenCV cannot reliably detect audio; if ffmpeg missing, assume present for common containers
    if not media["audio_present"] and video_path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}:
        media["audio_present"] = True
        media["audio_codec"] = media["audio_codec"] or "assumed"

    return media


def extract_audio_wav(video_path: Path, wav_path: Path, sample_rate: int = 16000) -> bool:
    """Extract mono wav. Returns False if extraction failed."""
    ff = find_ffmpeg()
    if not ff:
        return False
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ff,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 44
