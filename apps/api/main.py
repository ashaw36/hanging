"""Local web API for 鸿心影鉴."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "packages" / "pipeline"))
sys.path.insert(0, str(ROOT / "packages" / "portrait"))

DATA = ROOT / "data"
JOBS = DATA / "jobs"
DB_PATH = DATA / "app.db"
WEB = ROOT / "apps" / "web"
CONFIG = ROOT / "config"
MAX_DURATION_MS = 30 * 60 * 1000

PACKAGES = os.environ.get(
    "PACKAGES",
    "visual_base,gaze_basic,audio_acoustic,asr_basic,events_basic,au_advanced",
)
ASR_MODEL = os.environ.get("ASR_MODEL", "small")
STRIDE = int(os.environ.get("STRIDE", "2"))

app = FastAPI(title="鸿心影鉴")
_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _db() -> sqlite3.Connection:
    DATA.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            filename TEXT,
            status TEXT,
            stage TEXT,
            error TEXT,
            narrate INTEGER,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    conn.commit()
    return conn


def _update(job_id: str, **fields) -> None:
    fields["updated_at"] = _now()
    keys = list(fields)
    sql = "UPDATE jobs SET " + ", ".join(f"{key}=?" for key in keys) + " WHERE id=?"
    with _lock:
        conn = _db()
        conn.execute(sql, [fields[key] for key in keys] + [job_id])
        conn.commit()
        conn.close()


def _row(job_id: str) -> dict | None:
    conn = _db()
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def _job_view(row: dict) -> dict:
    portrait = None
    path = JOBS / row["id"] / "portrait.json"
    if path.exists() and row["status"] == "done":
        portrait = json.loads(path.read_text(encoding="utf-8"))
    return {
        "id": row["id"],
        "filename": row["filename"],
        "status": row["status"],
        "stage": row["stage"],
        "error": row["error"],
        "narrate": bool(row["narrate"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "video_url": f"/api/jobs/{row['id']}/video",
        "portrait": portrait,
    }


def _work(job_id: str) -> None:
    job_dir = JOBS / job_id
    video = job_dir / "input.mp4"
    feature_dir = job_dir / "features"
    try:
        _update(job_id, status="running", stage="extracting", error=None)
        from pipeline.probe import probe_video
        from pipeline.run import run_pipeline

        media = probe_video(video)
        if int(media.get("duration_ms") or 0) > MAX_DURATION_MS:
            raise RuntimeError("视频长于 30 分钟，未进入画像")
        packages = [part.strip() for part in PACKAGES.split(",") if part.strip()]
        run_pipeline(
            video,
            feature_dir,
            asr_model=ASR_MODEL,
            frame_stride=STRIDE,
            packages=packages,
            template_html=None,
        )
        _update(job_id, stage="scoring")
        from portrait.engine import build_portrait

        row = _row(job_id)
        portrait = build_portrait(feature_dir, CONFIG, use_deepseek=bool(row and row["narrate"]))
        portrait["filename"] = row["filename"] if row else video.name
        portrait["job_id"] = job_id
        (job_dir / "portrait.json").write_text(json.dumps(portrait, ensure_ascii=False, indent=2), encoding="utf-8")
        _update(job_id, status="done", stage="done")
    except Exception as exc:  # noqa: BLE001 — surface the failure on the job
        trace = traceback.format_exc(limit=8)
        (job_dir / "error.log").write_text(trace, encoding="utf-8")
        _update(job_id, status="failed", stage="failed", error=str(exc)[:500])


def _openface_health() -> dict:
    from pipeline.openface_pipe import (
        DEFAULT_DOCKER_IMAGE,
        OPENFACE_VERSION,
        docker_available,
        resolve_openface_bin,
    )

    image = (os.environ.get("OPENFACE_DOCKER_IMAGE") or DEFAULT_DOCKER_IMAGE).strip()
    native = resolve_openface_bin()
    docker_ok = docker_available()
    image_present = False
    if docker_ok:
        try:
            proc = subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            image_present = proc.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            image_present = False

    if native:
        return {
            "available": True,
            "mode": "native",
            "image": image,
            "version": OPENFACE_VERSION,
            "binary": native,
        }
    if docker_ok and image_present:
        return {
            "available": True,
            "mode": "docker",
            "image": image,
            "version": OPENFACE_VERSION,
            "binary": None,
        }
    return {
        "available": False,
        "mode": "none",
        "image": image,
        "version": OPENFACE_VERSION,
        "binary": None,
        "docker_running": docker_ok,
        "image_present": image_present,
    }


def _models_health() -> dict:
    mediapipe_ok = False
    try:
        import mediapipe as mp  # noqa: F401

        mediapipe_ok = True
    except Exception:  # noqa: BLE001
        mediapipe_ok = False
    return {
        "mediapipe": mediapipe_ok,
        "mediapipe_face": mediapipe_ok,
        "mediapipe_pose": mediapipe_ok,
    }


def _versions_health() -> dict:
    versions = {
        "pipeline": "0.1.0",
        "indicators": "indicators-0.1.0",
        "rules": "rules-0.1.0",
        "topics": "topics-0.1.0",
    }
    for name, key in (
        ("indicators-0.1.0.yaml", "indicators"),
        ("rules-0.1.0.yaml", "rules"),
        ("topics-0.1.0.yaml", "topics"),
    ):
        path = CONFIG / name
        if path.exists():
            versions[key] = path.stem
    try:
        from importlib.metadata import version as pkg_version

        versions["pipeline"] = pkg_version("pipeline")
    except Exception:  # noqa: BLE001
        init_path = ROOT / "packages" / "pipeline" / "pipeline" / "__init__.py"
        if init_path.exists():
            text = init_path.read_text(encoding="utf-8")
            for line in text.splitlines():
                if "__version__" in line and "=" in line:
                    versions["pipeline"] = line.split("=", 1)[1].strip().strip("'\"")
                    break
    return versions


def _system_health() -> dict:
    info: dict = {"cpu_percent": None, "mem_percent": None, "disk_free_gb": None, "device": "CPU"}
    try:
        import psutil

        info["cpu_percent"] = round(psutil.cpu_percent(interval=0.1), 1)
        info["mem_percent"] = round(psutil.virtual_memory().percent, 1)
        info["disk_free_gb"] = round(psutil.disk_usage(str(ROOT)).free / (1024**3), 1)
    except Exception:  # noqa: BLE001
        try:
            usage = shutil.disk_usage(str(ROOT))
            info["disk_free_gb"] = round(usage.free / (1024**3), 1)
        except Exception:  # noqa: BLE001
            pass
    return info


@app.get("/api/health")
def health():
    key = bool(os.environ.get("DEEPSEEK_API_KEY", "").strip())
    return {
        "ok": True,
        "deepseek_configured": key,
        "asr_model": ASR_MODEL,
        "stride": STRIDE,
        "openface": _openface_health(),
        "models": _models_health(),
        "versions": _versions_health(),
        "system": _system_health(),
    }


@app.get("/api/jobs")
def list_jobs():
    conn = _db()
    rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    conn.close()
    return [_job_view(dict(row)) for row in rows]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    row = _row(job_id)
    if not row:
        raise HTTPException(404, "任务不存在")
    return _job_view(row)


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str):
    path = JOBS / job_id / "input.mp4"
    if not path.exists():
        raise HTTPException(404, "视频不存在")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/metrics.csv")
def job_csv(job_id: str):
    path = JOBS / job_id / "features" / "metrics_summary.csv"
    if not path.exists():
        raise HTTPException(404, "指标表还不存在")
    return FileResponse(path, media_type="text/csv", filename=f"{job_id}-metrics.csv")


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), narrate: str = Form("true")):
    name = Path(file.filename or "video.mp4").name
    suffix = Path(name).suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".webm"}:
        raise HTTPException(400, "请上传 mp4、mov、mkv 或 webm")
    job_id = uuid4().hex[:12]
    job_dir = JOBS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    target = job_dir / "input.mp4"
    content = await file.read()
    if not content:
        raise HTTPException(400, "文件是空的")
    target.write_bytes(content)
    flag = narrate.strip().lower() in {"1", "true", "yes", "on"}
    conn = _db()
    conn.execute(
        "INSERT INTO jobs (id, filename, status, stage, error, narrate, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
        (job_id, name, "queued", "queued", None, 1 if flag else 0, _now(), _now()),
    )
    conn.commit()
    conn.close()
    threading.Thread(target=_work, args=(job_id,), daemon=True).start()
    return JSONResponse(_job_view(_row(job_id)), status_code=201)


# --- Static 18-page prototype UI ---
HTML_PAGES = {
    "index",
    "analysis",
    "batch",
    "capture",
    "tasks",
    "data",
    "semantic",
    "markers",
    "alignment",
    "indicators",
    "special",
    "alerts",
    "reports",
    "logs",
    "models",
    "settings",
    "profile",
    "report",
}


@app.get("/")
def index():
    path = WEB / "index.html"
    if not path.exists():
        raise HTTPException(404, "前端未构建：请运行 apps/web/build.py")
    return FileResponse(path)


@app.get("/{page}.html")
def html_page(page: str):
    if page not in HTML_PAGES:
        raise HTTPException(404, "页面不存在")
    path = WEB / f"{page}.html"
    if not path.exists():
        raise HTTPException(404, "页面文件缺失")
    return FileResponse(path)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8765")))


if __name__ == "__main__":
    main()
