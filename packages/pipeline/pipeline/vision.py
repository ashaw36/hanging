from __future__ import annotations

import math
import urllib.request
from pathlib import Path
from typing import Any, Optional

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[3] / "models"
FACE_MODEL = MODEL_DIR / "face_landmarker.task"
POSE_MODEL = MODEL_DIR / "pose_landmarker_lite.task"

FACE_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
POSE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"


def _ensure_models() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in ((FACE_MODEL, FACE_URL), (POSE_MODEL, POSE_URL)):
        if path.exists() and path.stat().st_size > 1000:
            continue
        urllib.request.urlretrieve(url, path)


def _wrap_deg(a: float) -> float:
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return float(a)


def _euler_from_matrix(rmat: np.ndarray) -> tuple[float, float, float]:
    """Yaw/pitch/roll in degrees from 3x3 rotation (OpenCV camera convention)."""
    sy = math.sqrt(float(rmat[0, 0]) ** 2 + float(rmat[1, 0]) ** 2)
    singular = sy < 1e-6
    if not singular:
        yaw = math.degrees(math.atan2(float(rmat[1, 0]), float(rmat[0, 0])))
        pitch = math.degrees(math.atan2(-float(rmat[2, 0]), sy))
        roll = math.degrees(math.atan2(float(rmat[2, 1]), float(rmat[2, 2])))
    else:
        yaw = math.degrees(math.atan2(-float(rmat[0, 1]), float(rmat[1, 1])))
        pitch = math.degrees(math.atan2(-float(rmat[2, 0]), sy))
        roll = 0.0
    return _wrap_deg(yaw), _wrap_deg(pitch), _wrap_deg(roll)


def _head_euler_from_matrix(matrix_4x4) -> Optional[tuple[float, float, float]]:
    try:
        m = np.array(matrix_4x4, dtype=np.float64).reshape(4, 4)
        rmat = m[:3, :3]
        return _euler_from_matrix(rmat)
    except Exception:
        return None


def _head_euler_geometric(landmarks, w: int, h: int) -> tuple[float, float, float]:
    """Stable fallback: eye-line roll + solvePnP yaw/pitch with roll replaced by eye tilt."""
    idx = {"nose": 1, "chin": 152, "left_eye": 33, "right_eye": 263, "left_mouth": 61, "right_mouth": 291}
    pts = {}
    for k, i in idx.items():
        lm = landmarks[i]
        pts[k] = np.array([lm.x * w, lm.y * h, lm.z * w], dtype=np.float64)

    # Roll from eye corners — robust for customer-facing "head tilt"
    roll = math.degrees(
        math.atan2(pts["right_eye"][1] - pts["left_eye"][1], pts["right_eye"][0] - pts["left_eye"][0] + 1e-6)
    )

    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1),
        ],
        dtype=np.float64,
    )
    image_points = np.array(
        [pts["nose"][:2], pts["chin"][:2], pts["left_eye"][:2], pts["right_eye"][:2], pts["left_mouth"][:2], pts["right_mouth"][:2]],
        dtype=np.float64,
    )
    import cv2

    focal = float(w)
    camera = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        model_points, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        # yaw/pitch crude fallback
        mid_eye_x = (pts["left_eye"][0] + pts["right_eye"][0]) / 2.0
        yaw = ((pts["nose"][0] - mid_eye_x) / max(w, 1)) * 90.0
        pitch = ((pts["nose"][1] - (pts["left_eye"][1] + pts["right_eye"][1]) / 2.0) / max(h, 1)) * 90.0
        return _wrap_deg(yaw), _wrap_deg(pitch), _wrap_deg(roll)

    rmat, _ = cv2.Rodrigues(rvec)
    yaw, pitch, _roll_pnp = _euler_from_matrix(rmat)
    # Prefer geometric roll; PnP roll often jumps near ±180 on frontal faces
    return yaw, pitch, _wrap_deg(roll)


def _ear(landmarks, idxs: list[int]) -> float:
    pts = [(landmarks[i].x, landmarks[i].y) for i in idxs]

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    v = (dist(pts[1], pts[5]) + dist(pts[2], pts[4])) / 2.0
    h = dist(pts[0], pts[3]) + 1e-6
    return float(v / h)


def _gaze(landmarks) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    try:
        left_iris = landmarks[468]
        right_iris = landmarks[473]
        left_outer, left_inner = landmarks[33], landmarks[133]
        right_outer, right_inner = landmarks[362], landmarks[263]
    except Exception:
        return None, None, None, None, None, None

    def ratio(iris, outer, inner):
        span = (inner.x - outer.x) + 1e-6
        return float(np.clip((iris.x - outer.x) / span, 0, 1))

    lx = ratio(left_iris, left_outer, left_inner)
    rx = ratio(right_iris, right_outer, right_inner)
    ly, ry = float(left_iris.y), float(right_iris.y)
    gaze_yaw = ((lx + rx) / 2.0 - 0.5) * 40.0
    gaze_pitch = ((ly + ry) / 2.0 - 0.45) * 30.0
    left_ratio = float(min(1.0, max(0.0, 0.35 + (0.5 - abs(lx - 0.5)))))
    right_ratio = float(min(1.0, max(0.0, 0.35 + (0.5 - abs(rx - 0.5)))))
    return gaze_yaw, gaze_pitch, lx, ly, left_ratio, right_ratio


def run_vision(video_path: str, max_frames: Optional[int] = None, stride: int = 1) -> dict[str, Any]:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision as mp_vision

    _ensure_models()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    base = python.BaseOptions
    face_opts = mp_vision.FaceLandmarkerOptions(
        base_options=base(model_asset_path=str(FACE_MODEL)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=base(model_asset_path=str(POSE_MODEL)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )
    face_lm = mp_vision.FaceLandmarker.create_from_options(face_opts)
    pose_lm = mp_vision.PoseLandmarker.create_from_options(pose_opts)
    MpImage = mp.Image
    ImageFormat = mp.ImageFormat

    rows: list[dict[str, Any]] = []
    track_points = 0
    gaps: list[dict[str, int]] = []
    gap_start: Optional[int] = None
    idx = 0
    processed = 0
    LEFT_EAR_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EAR_IDX = [362, 385, 387, 263, 373, 380]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride != 0:
            idx += 1
            continue
        if max_frames is not None and processed >= max_frames:
            break

        ts_ms = int(round(idx / fps * 1000))
        ts_s = idx / fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)
        face_res = face_lm.detect_for_video(mp_image, int(ts_s * 1000))
        pose_res = pose_lm.detect_for_video(mp_image, int(ts_s * 1000))

        row: dict[str, Any] = {
            "timestamp_ms": ts_ms,
            "frame_idx": idx,
            "face_detected": 0,
            "face_confidence": None,
            "head_yaw": None,
            "head_pitch": None,
            "head_roll": None,
            "gaze_yaw": None,
            "gaze_pitch": None,
            "pupil_left_x": None,
            "pupil_left_y": None,
            "iris_left_ratio": None,
            "iris_right_ratio": None,
            "ear_left": None,
            "ear_right": None,
            "pose_detected": 0,
            "body_visibility": None,
            "face_bbox": None,
            "profile_like": 0,
        }

        if face_res.face_landmarks:
            lms = face_res.face_landmarks[0]
            row["face_detected"] = 1
            row["face_confidence"] = 0.9
            pose = None
            if getattr(face_res, "facial_transformation_matrixes", None):
                pose = _head_euler_from_matrix(face_res.facial_transformation_matrixes[0])
            if pose is None:
                yaw, pitch, roll = _head_euler_geometric(lms, w, h)
            else:
                yaw, pitch, roll = pose
                # stabilize roll with eye-line (matrix roll can still be noisy)
                _, _, roll_geo = _head_euler_geometric(lms, w, h)
                if abs(roll) > 45:
                    roll = roll_geo
            row["head_yaw"], row["head_pitch"], row["head_roll"] = yaw, pitch, roll
            gy, gp, plx, ply, ilr, irr = _gaze(lms)
            row["gaze_yaw"], row["gaze_pitch"] = gy, gp
            row["pupil_left_x"], row["pupil_left_y"] = plx, ply
            row["iris_left_ratio"], row["iris_right_ratio"] = ilr, irr
            row["ear_left"] = _ear(lms, LEFT_EAR_IDX)
            row["ear_right"] = _ear(lms, RIGHT_EAR_IDX)
            xs = [lm.x for lm in lms]
            ys = [lm.y for lm in lms]
            row["face_bbox"] = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            row["profile_like"] = 1 if abs(yaw) >= 35 else 0
            track_points += 1
            if gap_start is not None:
                gaps.append({"start_ms": int(round(gap_start / fps * 1000)), "end_ms": ts_ms, "gap_frames": idx - gap_start})
                gap_start = None
        else:
            if gap_start is None:
                gap_start = idx

        if pose_res.pose_landmarks:
            row["pose_detected"] = 1
            vis = [getattr(lm, "visibility", 1.0) or 1.0 for lm in pose_res.pose_landmarks[0]]
            row["body_visibility"] = float(sum(vis) / max(len(vis), 1))

        rows.append(row)
        idx += 1
        processed += 1

    if gap_start is not None and total:
        gaps.append(
            {
                "start_ms": int(round(gap_start / fps * 1000)),
                "end_ms": int(round((total - 1) / fps * 1000)),
                "gap_frames": max(0, total - gap_start),
            }
        )

    cap.release()
    face_lm.close()
    pose_lm.close()

    face_frames = sum(1 for r in rows if r["face_detected"] == 1)
    pose_frames = sum(1 for r in rows if r["pose_detected"] == 1)
    return {
        "fps": fps,
        "frame_count_scanned": len(rows),
        "frame_count_video": total,
        "width": w,
        "height": h,
        "rows": rows,
        "track": {
            "primary_track_id": "track_0",
            "candidate_face_count_max": 1,
            "track_switches": 0,
            "face_gap_segments": gaps,
            "bbox_point_count": track_points,
        },
        "coverage": {
            "face_detected_frames": face_frames,
            "face_detected_rate": face_frames / max(len(rows), 1),
            "pose_detected_frames": pose_frames,
            "pose_detected_rate": pose_frames / max(len(rows), 1),
        },
    }
