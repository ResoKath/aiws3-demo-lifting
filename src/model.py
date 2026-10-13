"""
Core Ergonomics Models, MediaPipe Pose Detection, 3D Geometry & LiftCounter
"""

import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .dataset import (
    COL_NAMES,
    REACH_LIMITS_CM,
    ROW_NAMES,
    WEIGHT_LIMITS_KG,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "runs" / "checkpoints" / "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# ค่าคงที่สรีระและเรขาคณิต
NOSE_ABOVE_FOOT = 0.905
CALIB_SECONDS = 1.5
CAMERA_FOV_DEG = 70.0
WAIST_FRACTION = 0.30
TWIST_WARN_DEG = 45.0
TABLE_LENGTH_CM = 55.0
CELL_ALPHA = 0.45

# Landmark IDs
FEET = [29, 30, 31, 32]
LOWER = [27, 28, 29, 30, 31, 32]
PNP_POINTS = [0, 11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
LEFT_HAND, RIGHT_HAND = [15, 17, 19], [16, 18, 20]


@dataclass
class AnalysisConfig:
    """พารามิเตอร์การวิเคราะห์ที่ปรับแต่งได้"""
    subject_height_cm: float = 170.0
    calib_seconds: float = CALIB_SECONDS
    nose_above_foot: float = NOSE_ABOVE_FOOT
    camera_fov_deg: float = CAMERA_FOV_DEG
    waist_fraction: float = WAIST_FRACTION
    twist_warn_deg: float = TWIST_WARN_DEG
    table_length_cm: float = TABLE_LENGTH_CM
    cell_alpha: float = CELL_ALPHA
    min_confidence: float = 0.5
    world_window: int = 5
    min_rep_duration_s: float = 0.8
    min_vertical_travel_cm: float = 18.0


def ensure_model(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    """ตรวจสอบไฟล์โมเดล MediaPipe หากยังไม่มีให้ดาวน์โหลดอัตโนมัติ"""
    model_path = Path(model_path)
    if not model_path.is_file():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(POSE_MODEL_URL, model_path)
    return model_path


def create_pose_detector(model_path: Path = DEFAULT_MODEL_PATH):
    """สร้าง MediaPipe Pose Landmarker detector"""
    ensure_model(model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def extract_keypoints(detector, frame: np.ndarray, frame_index: int, fps: float):
    """สกัด 33 keypoints 2D (พิกเซล), 3D World (เมตร) และค่าความมั่นใจ (Confidence)"""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    timestamp_ms = int(frame_index * 1000 / fps)
    result = detector.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
    )

    xy = np.full((33, 2), np.nan)
    world = np.full((33, 3), np.nan)
    conf = np.full(33, np.nan)
    if result.pose_landmarks:
        for i, lm in enumerate(result.pose_landmarks[0]):
            xy[i] = [lm.x * w, lm.y * h]
            conf[i] = min(lm.visibility, lm.presence)
        for i, lm in enumerate(result.pose_world_landmarks[0]):
            world[i] = [lm.x, lm.y, lm.z]
    return xy, world, conf


def unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v


def mean_point(points: np.ndarray, ids: List[int]) -> np.ndarray:
    return np.nanmean(points[ids], axis=0)


def calc_cm_per_px(
    xy_frames: List[np.ndarray],
    height_cm: float,
    nose_above_foot: float = NOSE_ABOVE_FOOT,
) -> float:
    distances = []
    for xy in xy_frames:
        floor_y = np.nanmax(xy[FEET, 1])
        distances.append(floor_y - xy[0, 1])
    med = np.nanmedian(distances)
    if not np.isfinite(med) or med <= 1e-3:
        return height_cm / 500.0
    return height_cm * nose_above_foot / med


def camera_matrix(
    width: int, height: int, fov_deg: float = CAMERA_FOV_DEG
) -> np.ndarray:
    f = (max(width, height) / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])


def calc_world_scale(
    world_frames: List[np.ndarray],
    height_cm: float,
    nose_above_foot: float = NOSE_ABOVE_FOOT,
) -> float:
    heights = []
    for world in world_frames:
        axis = unit(mean_point(world, [23, 24]) - mean_point(world, LOWER))
        heights.append(np.nanmax((world[0] - world[FEET]) @ axis) * 100.0)
    med = np.nanmedian(heights)
    if not np.isfinite(med) or med <= 1e-3:
        return 1.0
    return height_cm * nose_above_foot / med


def solve_camera_pose(
    P: np.ndarray,
    xy: np.ndarray,
    observed: np.ndarray,
    K: np.ndarray,
    previous: Optional[Dict] = None,
) -> Optional[Dict]:
    ids = [
        i
        for i in PNP_POINTS
        if observed[i] and np.isfinite(P[i]).all() and np.isfinite(xy[i]).all()
    ]
    if len(ids) < 6:
        return None
    if previous is None:
        ok, rvec, tvec = cv2.solvePnP(P[ids], xy[ids], K, None)
    else:
        ok, rvec, tvec = cv2.solvePnP(
            P[ids],
            xy[ids],
            K,
            None,
            previous["rvec"].copy(),
            previous["tvec"].copy(),
            True,
        )
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return {"R": R, "t": tvec.ravel(), "rvec": rvec, "tvec": tvec}


def to_camera_3d(
    xy: np.ndarray, P: np.ndarray, pose: Dict, K: np.ndarray
) -> np.ndarray:
    P_cam = P @ pose["R"].T + pose["t"]
    Z = P_cam[:, 2]
    X = (xy[:, 0] - K[0, 2]) * Z / K[0, 0]
    Y = (xy[:, 1] - K[1, 2]) * Z / K[1, 1]
    C = np.column_stack([X, Y, Z])
    missing = np.isnan(xy[:, 0])
    C[missing] = P_cam[missing]
    return C


def project(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    u = K[0, 0] * points[..., 0] / points[..., 2] + K[0, 2]
    v = K[1, 1] * points[..., 1] / points[..., 2] + K[1, 2]
    return np.stack([u, v], axis=-1)


def body_frame(
    C: np.ndarray, up: np.ndarray, floor: Optional[np.ndarray] = None
) -> Optional[Dict]:
    ankle = mean_point(C, [27, 28])
    side = C[24] - C[23]
    side = unit(side - (side @ up) * up)
    front = unit(np.cross(side, up))
    toe, heel = mean_point(C, [31, 32]), mean_point(C, [29, 30])
    if (toe - heel) @ front < 0:
        front = -front
    if floor is None:
        drop = np.nanmin((C[FEET] - ankle) @ up)
    else:
        drop = (floor - ankle) @ up
    O = ankle + up * drop
    if not (np.isfinite(O).all() and np.isfinite(front).all()):
        return None
    return {"O": O, "up": up, "side": side, "front": front}


def body_coords(p: np.ndarray, body: Dict) -> Tuple[float, float, float]:
    d = p - body["O"]
    return float(d @ body["front"]), float(d @ body["up"]), float(d @ body["side"])


def orientation_angle(C: np.ndarray) -> float:
    """มุมระหว่างเส้นสะโพกกับระนาบกล้อง (0 = หันเข้ากล้อง, 90 = หันข้าง)"""
    d = C[24] - C[23]
    yaw = abs(math.degrees(math.atan2(d[2], d[0])))
    return min(yaw, 180.0 - yaw)


def twist_angle(world: np.ndarray) -> float:
    axis = unit(mean_point(world, [23, 24]) - mean_point(world, LOWER))
    hip = world[24] - world[23]
    shoulder = world[12] - world[11]
    hip = unit(hip - (hip @ axis) * axis)
    shoulder = unit(shoulder - (shoulder @ axis) * axis)
    return math.degrees(math.atan2(np.cross(hip, shoulder) @ axis, hip @ shoulder))


def twist_from_baseline(world: np.ndarray, calib: Dict) -> float:
    diff = twist_angle(world) - calib["twist_baseline"]
    diff = (diff + 180.0) % 360.0 - 180.0
    return abs(diff)


def calibrate(
    samples: List[Dict],
    height_cm: float,
    width: int,
    height: int,
    cfg: Optional[AnalysisConfig] = None,
) -> Dict:
    cfg = cfg or AnalysisConfig(subject_height_cm=height_cm)
    calib = {
        "height_cm": height_cm,
        "table_length_cm": cfg.table_length_cm,
        "cell_alpha": cfg.cell_alpha,
    }
    calib["cm_per_px"] = calc_cm_per_px(
        [s["xy"] for s in samples], height_cm, cfg.nose_above_foot
    )
    calib["K"] = camera_matrix(width, height, cfg.camera_fov_deg)
    calib["camera_dist_cm"] = calib["K"][0, 0] * calib["cm_per_px"]
    calib["world_scale"] = calc_world_scale(
        [s["world"] for s in samples], height_cm, cfg.nose_above_foot
    )

    C_list, axes, pose = [], [], None
    for s in samples:
        P = s["world"] * 100.0 * calib["world_scale"]
        pose = solve_camera_pose(P, s["xy"], s["observed"], calib["K"], pose)
        if pose is None:
            continue
        C = to_camera_3d(s["xy"], P, pose, calib["K"])
        C_list.append(C)
        axes.append(unit(mean_point(C, [23, 24]) - mean_point(C, LOWER)))

    calib["up"] = (
        unit(np.nanmean(axes, axis=0)) if axes else np.array([0.0, -1.0, 0.0])
    )
    floors = []
    for C in C_list:
        body = body_frame(C, calib["up"])
        if body is not None:
            floors.append(body["O"])
    calib["floor"] = (
        np.nanmedian(floors, axis=0)
        if floors
        else np.array([0.0, 0.0, calib["camera_dist_cm"]])
    )

    knee, hip, shoulder, hand_v, hand_h = [], [], [], [], []
    for C in C_list:
        body = body_frame(C, calib["up"], calib["floor"])
        if body is None:
            continue
        knee.append(body_coords(mean_point(C, [25, 26]), body)[1])
        hip.append(body_coords(mean_point(C, [23, 24]), body)[1])
        shoulder.append(body_coords(mean_point(C, [11, 12]), body)[1])
        hands_3d = np.nanmean(
            [mean_point(C, LEFT_HAND), mean_point(C, RIGHT_HAND)], axis=0
        )
        if np.isfinite(hands_3d).all():
            f_c, v_c, s_c = body_coords(hands_3d, body)
            hand_v.append(v_c)
            hand_h.append(math.hypot(f_c, s_c))

    calib["knee_cm"] = float(np.nanmedian(knee)) if knee else height_cm * 0.285
    calib["shoulder_cm"] = (
        float(np.nanmedian(shoulder)) if shoulder else height_cm * 0.818
    )
    hip_med = float(np.nanmedian(hip)) if hip else height_cm * 0.530
    calib["hip_cm"] = hip_med
    calib["waist_cm"] = hip_med + cfg.waist_fraction * (
        calib["shoulder_cm"] - hip_med
    )
    calib["rest_v_cm"] = float(np.nanmedian(hand_v)) if hand_v else hip_med * 0.92
    calib["rest_h_cm"] = float(np.nanmedian(hand_h)) if hand_h else 18.0
    calib["twist_baseline"] = float(
        np.nanmedian([twist_angle(s["world"]) for s in samples])
    )
    return calib


def table_row(v_cm: float, calib: Dict) -> int:
    if v_cm >= calib["shoulder_cm"]:
        return 0
    if v_cm >= calib["waist_cm"]:
        return 1
    if v_cm >= calib["knee_cm"]:
        return 2
    return 3


def table_col(h_cm: float) -> int:
    if h_cm < REACH_LIMITS_CM[0]:
        return 0
    if h_cm < REACH_LIMITS_CM[1]:
        return 1
    return 2


def measure_hands(C: np.ndarray, xy: np.ndarray, calib: Dict) -> Optional[Dict]:
    body = body_frame(C, calib["up"], calib["floor"])
    if body is None:
        return None
    hands_3d = np.nanmean(
        [mean_point(C, LEFT_HAND), mean_point(C, RIGHT_HAND)], axis=0
    )
    hands_px = np.nanmean(
        [mean_point(xy, LEFT_HAND), mean_point(xy, RIGHT_HAND)], axis=0
    )
    if np.isnan(hands_3d).any():
        return None
    if np.isnan(hands_px).any():
        hands_px = project(hands_3d, calib["K"])

    front, v_cm, side = body_coords(hands_3d, body)
    h_cm = math.hypot(front, side)
    row, col = table_row(v_cm, calib), table_col(h_cm)
    return {
        "H": h_cm,
        "V": v_cm,
        "front": front,
        "side": side,
        "row": row,
        "col": col,
        "limit_kg": WEIGHT_LIMITS_KG[row][col],
        "hands_px": hands_px,
        "body": body,
        "direction": -1 if front < -REACH_LIMITS_CM[0] else 1,
    }


class LiftCounter:
    """ระบบตรวจจับและนับรอบการยกอัตโนมัติ (3-Phase State Machine)"""

    def __init__(self, calib: Dict, fps: float, cfg: Optional[AnalysisConfig] = None):
        self.calib = calib
        self.fps = fps
        self.cfg = cfg or AnalysisConfig()
        self.count = 0
        self.state = "IDLE"
        self.reps: List[Dict] = []

        self.knee_cm = calib["knee_cm"]
        self.waist_cm = calib["waist_cm"]
        self.shoulder_cm = calib["shoulder_cm"]
        self.rest_v = calib.get("rest_v_cm", calib["hip_cm"] * 0.92)
        self.rest_h = min(calib.get("rest_h_cm", 18.0), 22.0)

        self._reset_current_rep()
        self.idle_frames = 0
        self.active_frames = 0

    def _reset_current_rep(self):
        self.start_frame = None
        self.start_time = None
        self.v_history: List[float] = []
        self.h_history: List[float] = []
        self.twist_history: List[float] = []
        self.limit_history: List[int] = []
        self.zone_history: List[str] = []
        self.cycle_stage = 0
        self.peak_v = None
        self.trough_v = None

    def _is_in_rest(self, v_cm: float, h_cm: float) -> bool:
        v_near_rest = abs(v_cm - self.rest_v) <= 12.0
        h_near_rest = h_cm <= max(self.rest_h + 6.0, 25.0)
        return v_near_rest and h_near_rest

    def update(
        self,
        frame_number: int,
        time_s: float,
        m: Optional[Dict],
        twist_deg: float,
    ) -> Dict:
        if m is None:
            return {
                "lift_count": self.count,
                "lift_state": self.state,
                "rep_just_completed": False,
            }

        v_cm, h_cm = m["V"], m["H"]
        zone = ROW_NAMES[m["row"]]
        limit_kg = m["limit_kg"]
        in_rest = self._is_in_rest(v_cm, h_cm)
        rep_completed = False

        if self.state == "IDLE":
            if not in_rest:
                self.active_frames += 1
                if self.active_frames >= max(5, int(0.18 * self.fps)):
                    self.state = "LIFTING"
                    self.idle_frames = 0
                    self.start_frame = frame_number
                    self.start_time = time_s
                    self.v_history = [v_cm]
                    self.h_history = [h_cm]
                    self.twist_history = [twist_deg]
                    self.limit_history = [limit_kg]
                    self.zone_history = [zone]
                    self.cycle_stage = 0
                    self.peak_v = v_cm
                    self.trough_v = v_cm
            else:
                self.active_frames = 0

        elif self.state == "LIFTING":
            self.v_history.append(v_cm)
            self.h_history.append(h_cm)
            self.twist_history.append(twist_deg)
            self.limit_history.append(limit_kg)
            self.zone_history.append(zone)

            v_min, v_max = min(self.v_history), max(self.v_history)
            v_span = v_max - v_min
            duration = time_s - self.start_time

            if self.cycle_stage == 0:
                if (
                    v_min < self.knee_cm + 8.0
                    and v_max >= self.waist_cm - 3.0
                    and v_span >= 40.0
                ):
                    idx_min = int(np.argmin(self.v_history))
                    idx_max = int(np.argmax(self.v_history))
                    self.cycle_stage = 1 if idx_min < idx_max else -1
                    self.peak_v = v_max
                    self.trough_v = v_min
            elif self.cycle_stage == 1:
                if v_cm > self.peak_v:
                    self.peak_v = v_cm
                elif (
                    v_cm < self.knee_cm + 5.0
                    and (self.peak_v - v_cm) >= 45.0
                    and duration >= 2.2
                ):
                    rep_completed = self._finalize_rep(frame_number, time_s)
                    self.state = "LIFTING"
                    self.start_frame = frame_number
                    self.start_time = time_s
                    self.v_history = [v_cm]
                    self.h_history = [h_cm]
                    self.twist_history = [twist_deg]
                    self.limit_history = [limit_kg]
                    self.zone_history = [zone]
                    self.cycle_stage = 0
                    self.peak_v = v_cm
                    self.trough_v = v_cm
                    return {
                        "lift_count": self.count,
                        "lift_state": self.state,
                        "rep_just_completed": rep_completed,
                    }

            if in_rest:
                self.idle_frames += 1
                if self.idle_frames >= max(12, int(0.45 * self.fps)):
                    if (
                        v_span >= self.cfg.min_vertical_travel_cm
                        and duration >= self.cfg.min_rep_duration_s
                    ):
                        rep_completed = self._finalize_rep(frame_number, time_s)
                    else:
                        self._reset_current_rep()
                    self.state = "IDLE"
                    self.active_frames = 0
            else:
                self.idle_frames = 0

        return {
            "lift_count": self.count,
            "lift_state": self.state,
            "rep_just_completed": rep_completed,
        }

    def flush_at_end(self, last_frame: int, last_time: float):
        if self.state == "LIFTING" and self.v_history:
            v_span = max(self.v_history) - min(self.v_history)
            duration = last_time - (self.start_time or 0.0)
            if (
                v_span >= self.cfg.min_vertical_travel_cm
                and duration >= self.cfg.min_rep_duration_s
            ):
                self._finalize_rep(last_frame, last_time)
                self.state = "IDLE"

    def _finalize_rep(self, end_frame: int, end_time: float) -> bool:
        self.count += 1
        idx_min = int(np.argmin(self.v_history))
        idx_max = int(np.argmax(self.v_history))
        min_zone = ROW_NAMES[table_row(self.v_history[idx_min], self.calib)]
        max_zone = ROW_NAMES[table_row(self.v_history[idx_max], self.calib)]

        if idx_min < idx_max:
            origin_zone, dest_zone = min_zone, max_zone
        else:
            origin_zone, dest_zone = max_zone, min_zone

        rep_info = {
            "rep_number": self.count,
            "start_frame": self.start_frame,
            "end_frame": end_frame,
            "start_time_s": round(self.start_time, 2),
            "end_time_s": round(end_time, 2),
            "duration_s": round(end_time - self.start_time, 2),
            "origin_zone": origin_zone,
            "dest_zone": dest_zone,
            "zone_path": f"{origin_zone} -> {dest_zone}",
            "min_V_cm": round(float(np.min(self.v_history)), 1),
            "max_V_cm": round(float(np.max(self.v_history)), 1),
            "vertical_travel_cm": round(
                float(np.max(self.v_history) - np.min(self.v_history)), 1
            ),
            "max_H_cm": round(float(np.max(self.h_history)), 1),
            "max_twist_deg": round(float(np.max(self.twist_history)), 1),
            "min_limit_kg": int(np.min(self.limit_history)),
        }
        self.reps.append(rep_info)
        self._reset_current_rep()
        return True

