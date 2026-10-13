"""
Lifting Weight Limit & Ergonomics Analysis Engine
รายวิชา CP413705 AI Workshop III - วิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น (กลุ่ม G10)

รวมการคำนวณทั้งหมด:
1. สกัด Keypoints 33 จุดด้วย MediaPipe Pose Landmarker
2. ลดการสั่นไหวด้วย Kalman Filter + Moving Average (PoseSmoother)
3. คำนวณมาตราส่วน cm/pixel, พิกัด 3 มิติ (solvePnP) และกรอบอ้างอิงร่างกาย (Body Frame)
4. คำนวณมุมการหันของร่างกายเทียบกับระนาบกล้อง (Orientation Angle)
5. คำนวณมุมการบิดตัวของลำตัวขณะยก (Twist Angle)
6. คำนวณระยะเอื้อมแนวราบ (H) และความสูงของมือ (V) พร้อมตาราง NIOSH 3x4 (Perspective Transform)
7. ระบบนับรอบการยกอัตโนมัติ (LiftCounter) พร้อมสรุปสถิติรายรอบ
"""

import argparse
import math
import re
import subprocess
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# ค่าตั้งต้น (Default Constants & Hyperparameters)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# ค่าสำหรับการกรองสัญญาณ (Tracking & Smoothing)
MIN_CONFIDENCE = 0.5        # ค่าวัดที่ confidence ต่ำกว่านี้ไม่ใช้
MAX_PREDICT_FRAMES = 5      # ใช้ค่าทายแทนได้ติดกันกี่เฟรม
HISTORY_FRAMES = 3          # moving average = เฟรมนี้ + 3 เฟรมก่อนหน้า
WORLD_WINDOW = 5            # จำนวนเฟรมที่เฉลี่ยโครงร่าง 3 มิติ

# ค่าสำหรับการเทียบมาตราส่วนและกล้อง (Calibration & 3D Geometry)
NOSE_ABOVE_FOOT = 0.905     # อัตราส่วน พื้น->จมูก / ส่วนสูงจริง
CALIB_SECONDS = 1.5         # ช่วงเวลายืนนิ่งต้นคลิป (วินาที)
CAMERA_FOV_DEG = 70.0       # มุมมองกล้องตามด้านยาวของภาพ (องศา)
WAIST_FRACTION = 0.30       # ตำแหน่งเอวระหว่างสะโพกกับไหล่

# เกณฑ์แจ้งเตือนและขนาดตาราง NIOSH 3x4
TWIST_WARN_DEG = 45.0       # เกณฑ์แจ้งเตือนมุมบิดตัว (องศา)
REACH_LIMITS_CM = [7 * 2.54, 12 * 2.54]  # [17.78, 30.48] ซม. (7 นิ้ว, 12 นิ้ว)
ROW_NAMES = ["Above Shoulder", "Waist-Shoulder", "Knee-Waist", "Below Knee"]
COL_NAMES = ["Near", "Mild", "Extended"]
WEIGHT_LIMITS_KG = [
    [29, 18, 14],  # Above Shoulder
    [32, 23, 18],  # Waist to Shoulder
    [41, 25, 18],  # Knee to Waist (สูงสุด 41 kg)
    [31, 23, 16],  # Below Knee
]

TABLE_LENGTH_CM = 55.0      # ความยาวตารางด้านหน้า (ซม.)
ABOVE_MARGIN_CM = 30.0      # ความสูงของแถว Above Shoulder เหนือไหล่
HEADER_CM = 10.0            # แถบหัวคอลัมน์
PX_PER_CM = 8               # ความละเอียดของภาพตาราง
CELL_ALPHA = 0.45           # ความทึบของช่องตาราง

# จุดอ้างอิงโครงกระดูก MediaPipe
SKELETON = [
    (11, 12), (11, 23), (12, 24), (23, 24),              # ลำตัว
    (11, 13), (13, 15), (15, 17), (15, 19),              # แขนซ้าย
    (12, 14), (14, 16), (16, 18), (16, 20),              # แขนขวา
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),    # ขาซ้าย
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),    # ขาขวา
    (0, 11), (0, 12),                                    # ศีรษะ
]
FEET = [29, 30, 31, 32]
LOWER = [27, 28, 29, 30, 31, 32]
PNP_POINTS = [0, 11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
LEFT_HAND, RIGHT_HAND = [15, 17, 19], [16, 18, 20]

# ข้อมูลส่วนสูงของผู้ยกในกลุ่ม G10 (สำหรับตั้งค่าอัตโนมัติจากชื่อไฟล์)
G10_SUBJECT_HEIGHTS = {
    "S01": 170.0,
    "S02": 168.0,
    "S03": 174.0,
    "S04": 170.0,
    "S05": 170.0,
}

ZONE_CODE = {
    "AS": "Above Shoulder",
    "WS": "Waist-Shoulder",
    "KW": "Knee-Waist",
    "BK": "Below Knee",
}


@dataclass
class AnalysisConfig:
    """พารามิเตอร์สำหรับการประมวลผลที่ปรับจูนได้"""
    subject_height_cm: float = 170.0
    calib_seconds: float = CALIB_SECONDS
    nose_above_foot: float = NOSE_ABOVE_FOOT
    camera_fov_deg: float = CAMERA_FOV_DEG
    waist_fraction: float = WAIST_FRACTION
    twist_warn_deg: float = TWIST_WARN_DEG
    table_length_cm: float = TABLE_LENGTH_CM
    cell_alpha: float = CELL_ALPHA
    min_confidence: float = MIN_CONFIDENCE
    world_window: int = WORLD_WINDOW
    # เกณฑ์การนับรอบยก
    min_rep_duration_s: float = 0.8       # ระยะเวลาขั้นต่ำต่อ 1 รอบการยก (วินาที)
    min_vertical_travel_cm: float = 18.0  # ระยะการเคลื่อนที่แนวดิ่งขั้นต่ำของมือใน 1 รอบ (ซม.)
    active_reach_cm: float = 18.0         # ระยะเอื้อม H ที่ถือว่ากำลังยื่นมือจับ/วางวัตถุ (ซม.)


def infer_subject_height(video_name: str, default_height: float = 170.0) -> float:
    """ดึงส่วนสูงของผู้ยกอัตโนมัติจากรหัส S01..S05 ในชื่อไฟล์"""
    m = re.search(r"_(S0[1-9])_", video_name)
    if m and m.group(1) in G10_SUBJECT_HEIGHTS:
        return G10_SUBJECT_HEIGHTS[m.group(1)]
    return default_height


# ==============================================================================
# ขั้นที่ ① & ② · อ่านวิดีโอและสกัดจุดพิกัดร่างกาย (MediaPipe Pose)
# ==============================================================================
def ensure_model(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    if not model_path.is_file():
        import urllib.request
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(POSE_MODEL_URL, model_path)
    return model_path


def open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"ไม่สามารถเปิดไฟล์วิดีโอได้: {path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    return cap, info


def create_pose_detector(model_path: Path = DEFAULT_MODEL_PATH):
    ensure_model(model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def extract_keypoints(detector, frame: np.ndarray, frame_index: int, fps: float):
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


# ==============================================================================
# ขั้นที่ ③ · การติดตามและลดการสั่นไหวของพิกัด (Tracking & Smoothing)
# ==============================================================================
def create_kalman(x: float, y: float):
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
    )
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    acc = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]], np.float32)
    kf.processNoiseCov = 16.0 * acc @ acc.T
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 25.0
    kf.errorCovPost = np.diag([25, 25, 400, 400]).astype(np.float32)
    kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
    return kf


class PointTracker:
    def __init__(self, width: int, height: int, min_conf: float = MIN_CONFIDENCE):
        self.width, self.height = width, height
        self.min_conf = min_conf
        self.gate = max(25.0, 0.04 * math.hypot(width, height))
        self.kalman = None
        self.missing = 0
        self.history = []

    def update(self, x: float, y: float, conf: float):
        good = (
            conf >= self.min_conf
            and 0 <= x < self.width
            and 0 <= y < self.height
        )
        point, observed = None, False

        if self.kalman is None:
            if good:
                self.kalman = create_kalman(x, y)
                point, observed = np.array([x, y]), True
        else:
            predicted = self.kalman.predict()[:2, 0]
            jump = math.hypot(x - predicted[0], y - predicted[1]) if good else 0
            if good and jump <= self.gate:
                self.kalman.correct(np.array([[x], [y]], np.float32))
                point, observed = np.array([x, y]), True
                self.missing = 0
            else:
                self.missing += 1
                inside = (
                    0 <= predicted[0] < self.width
                    and 0 <= predicted[1] < self.height
                )
                if self.missing <= MAX_PREDICT_FRAMES and inside:
                    point = predicted
                else:
                    self.kalman, self.missing, self.history = None, 0, []

        if point is None:
            return np.array([np.nan, np.nan]), False
        self.history = (self.history + [point])[-(HISTORY_FRAMES + 1) :]
        return np.mean(self.history, axis=0), observed


class PoseSmoother:
    def __init__(
        self,
        width: int,
        height: int,
        min_conf: float = MIN_CONFIDENCE,
        world_window: int = WORLD_WINDOW,
    ):
        self.trackers = [
            PointTracker(width, height, min_conf=min_conf) for _ in range(33)
        ]
        self.world_history = deque(maxlen=world_window)

    def update(self, xy: np.ndarray, world: np.ndarray, conf: np.ndarray):
        xy_s = np.full((33, 2), np.nan)
        observed = np.zeros(33, dtype=bool)
        for i in range(33):
            xy_s[i], observed[i] = self.trackers[i].update(
                xy[i, 0], xy[i, 1], conf[i]
            )
        self.world_history.append(world)
        world_s = np.nanmean(np.array(self.world_history), axis=0)
        return xy_s, world_s, observed


# ==============================================================================
# ขั้นที่ ④ · การวาดโครงกระดูกและ UI พื้นฐาน
# ==============================================================================
def pt(p) -> Tuple[int, int]:
    return (int(round(p[0])), int(round(p[1])))


def ui_scale(img: np.ndarray) -> float:
    return img.shape[1] / 720.0


def draw_skeleton(img: np.ndarray, xy_s: np.ndarray, observed: np.ndarray):
    s = ui_scale(img)
    for a, b in SKELETON:
        if np.isfinite(xy_s[a]).all() and np.isfinite(xy_s[b]).all():
            cv2.line(
                img,
                pt(xy_s[a]),
                pt(xy_s[b]),
                (255, 255, 255),
                max(1, int(2 * s)),
                cv2.LINE_AA,
            )
    for a, b in SKELETON:
        for p in (a, b):
            if np.isfinite(xy_s[p]).all():
                color = (0, 200, 0) if observed[p] else (0, 140, 255)
                cv2.circle(
                    img, pt(xy_s[p]), max(2, int(4 * s)), color, -1, cv2.LINE_AA
                )
    return img


# ==============================================================================
# ขั้นที่ ⑤, ⑥, ⑦ · มาตราส่วน พิกัด 3 มิติ มุมหัน และมุมบิดตัว
# ==============================================================================
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


def twist_angle(world: np.ndarray) -> float:
    axis = unit(mean_point(world, [23, 24]) - mean_point(world, LOWER))
    hip = world[24] - world[23]
    shoulder = world[12] - world[11]
    hip = unit(hip - (hip @ axis) * axis)
    shoulder = unit(shoulder - (shoulder @ axis) * axis)
    return math.degrees(math.atan2(np.cross(hip, shoulder) @ axis, hip @ shoulder))


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

    if axes:
        calib["up"] = unit(np.nanmean(axes, axis=0))
    else:
        calib["up"] = np.array([0.0, -1.0, 0.0])

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
    calib["rest_h_cm"] = float(np.nanmedian(hand_h)) if hand_h else 14.0
    calib["twist_baseline"] = float(
        np.nanmedian([twist_angle(s["world"]) for s in samples])
    )
    return calib


def orientation_angle(C: np.ndarray) -> float:
    """⑥ มุมระหว่างเส้นสะโพกกับระนาบกล้อง (0 = หันเข้ากล้อง, 90 = หันข้าง)"""
    d = C[24] - C[23]
    yaw = abs(math.degrees(math.atan2(d[2], d[0])))
    return min(yaw, 180.0 - yaw)


def twist_from_baseline(world: np.ndarray, calib: Dict) -> float:
    """⑦ มุมบิดตัว (องศา, ค่าบวก) เทียบกับท่ายืนตรง"""
    diff = twist_angle(world) - calib["twist_baseline"]
    diff = (diff + 180.0) % 360.0 - 180.0
    return abs(diff)


# ==============================================================================
# ขั้นที่ ⑧ · ตำแหน่งมือและตารางเกณฑ์น้ำหนักปลอดภัย 3×4 (Perspective Transform)
# ==============================================================================
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


def cell_color(kg: int) -> Tuple[int, int, int]:
    t = (41 - kg) / (41 - 14)
    light, dark = np.array([160, 225, 255]), np.array([20, 20, 170])
    return tuple(int(c) for c in light + (dark - light) * t)


def make_table_image(
    calib: Dict, highlight: Tuple[int, int], axis_on_left: bool
) -> Tuple[np.ndarray, np.ndarray]:
    table_len = calib.get("table_length_cm", TABLE_LENGTH_CM)
    c_alpha = calib.get("cell_alpha", CELL_ALPHA)
    top_cm = calib["shoulder_cm"] + ABOVE_MARGIN_CM + HEADER_CM
    w, h = int(table_len * PX_PER_CM), int(top_cm * PX_PER_CM)
    color = np.zeros((h, w, 3), np.uint8)
    alpha_u8 = np.zeros((h, w), np.uint8)

    col_edges = [0, REACH_LIMITS_CM[0], REACH_LIMITS_CM[1], table_len]
    row_edges = [
        calib["shoulder_cm"] + ABOVE_MARGIN_CM,
        calib["shoulder_cm"],
        calib["waist_cm"],
        calib["knee_cm"],
        0,
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX

    def to_x(cm: float) -> int:
        x = int(cm * PX_PER_CM)
        return x if axis_on_left else w - x

    def to_y(cm: float) -> int:
        return int((top_cm - cm) * PX_PER_CM)

    def write(text: str, pos: Tuple[int, int], size: float, thick: int, rgb=(255, 255, 255)):
        cv2.putText(color, text, pos, font, size, rgb, thick, cv2.LINE_AA)
        cv2.putText(alpha_u8, text, pos, font, size, 255, thick, cv2.LINE_AA)

    for r in range(4):
        for c in range(3):
            x0, x1 = sorted([to_x(col_edges[c]), to_x(col_edges[c + 1])])
            y0, y1 = to_y(row_edges[r]), to_y(row_edges[r + 1])
            kg = WEIGHT_LIMITS_KG[r][c]
            is_hand_cell = (r, c) == highlight
            cv2.rectangle(
                color,
                (x0, y0),
                (x1, y1),
                (0, 230, 255) if is_hand_cell else cell_color(kg),
                -1,
            )
            alpha_u8[y0:y1, x0:x1] = int(round((0.85 if is_hand_cell else c_alpha) * 255))
            cv2.rectangle(color, (x0, y0), (x1, y1), (255, 255, 255), 2)
            cv2.rectangle(alpha_u8, (x0, y0), (x1, y1), 255, 2)
            write(
                str(kg),
                ((x0 + x1) // 2 - 22, (y0 + y1) // 2 + 15),
                1.6,
                4,
                (0, 0, 0) if is_hand_cell else (255, 255, 255),
            )
        x_label = 8 if not axis_on_left else w - 190
        write(ROW_NAMES[r], (x_label, to_y(row_edges[r]) + 26), 0.7, 2)
    for c in range(3):
        x0, x1 = sorted([to_x(col_edges[c]), to_x(col_edges[c + 1])])
        write(COL_NAMES[c], (x0 + 8, to_y(row_edges[0]) - 14), 0.7, 2)
    write("MAX WEIGHT (kg)", (w // 2 - 120, 30), 0.8, 2)
    return color, alpha_u8.astype(np.float32) / 255.0


def draw_weight_table(img: np.ndarray, calib: Dict, m: Dict) -> np.ndarray:
    body, K = m["body"], calib["K"]
    table_len = calib.get("table_length_cm", TABLE_LENGTH_CM)
    O, up, front = body["O"], body["up"], body["front"] * m["direction"]
    top_cm = calib["shoulder_cm"] + ABOVE_MARGIN_CM + HEADER_CM
    corners_3d = np.array(
        [
            O,
            O + front * table_len,
            O + front * table_len + up * top_cm,
            O + up * top_cm,
        ]
    )
    if (corners_3d[:, 2] < 10).any():
        return img
    corners_px = project(corners_3d, K).astype(np.float32)
    if abs(cv2.contourArea(corners_px)) < 0.002 * img.shape[0] * img.shape[1]:
        return img

    axis_on_left = corners_px[1, 0] >= corners_px[0, 0]
    table, alpha = make_table_image(calib, (m["row"], m["col"]), axis_on_left)
    th, tw = alpha.shape
    x_body, x_end = (0, tw) if axis_on_left else (tw, 0)
    table_corners = np.float32(
        [[x_body, th], [x_end, th], [x_end, 0], [x_body, 0]]
    )

    M = cv2.getPerspectiveTransform(table_corners, corners_px)
    size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(table, M, size)
    warped_alpha = cv2.warpPerspective(alpha, M, size)[:, :, None]
    img[:] = (img * (1 - warped_alpha) + warped * warped_alpha).astype(np.uint8)
    return img


def draw_hand_label(img: np.ndarray, calib: Dict, m: Dict) -> np.ndarray:
    body, K, s = m["body"], calib["K"], ui_scale(img)
    table_len = calib.get("table_length_cm", TABLE_LENGTH_CM)
    O, up, front = body["O"], body["up"], body["front"] * m["direction"]
    on_table = project(O + front * m["H"] + up * m["V"], K)
    hands = m["hands_px"]
    cv2.line(
        img,
        pt(hands),
        pt(on_table),
        (255, 255, 255),
        max(1, int(2 * s)),
        cv2.LINE_AA,
    )
    cv2.circle(img, pt(on_table), int(7 * s), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(
        img, pt(hands), int(16 * s), (0, 230, 255), max(2, int(3 * s)), cv2.LINE_AA
    )

    anchor = project(O + front * (table_len + 3) + up * m["V"], K)
    box_w, box_h = int(230 * s), int(80 * s)
    x0 = anchor[0] if anchor[0] >= project(O, K)[0] else anchor[0] - box_w
    x0 = int(np.clip(x0, 5, img.shape[1] - box_w - 5))
    y0 = int(np.clip(anchor[1] - box_h / 2, 5, img.shape[0] - box_h - 5))
    cv2.line(
        img,
        pt(on_table),
        (x0 + box_w // 2, y0 + box_h // 2),
        (0, 230, 255),
        max(2, int(3 * s)),
    )
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (0, 230, 255), -1)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img,
        f"MAX {m['limit_kg']} kg",
        (x0 + int(10 * s), y0 + int(35 * s)),
        font,
        1.0 * s,
        (0, 0, 0),
        max(2, int(3 * s)),
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"{ROW_NAMES[m['row']]} | {COL_NAMES[m['col']]}",
        (x0 + int(10 * s), y0 + int(55 * s)),
        font,
        0.45 * s,
        (0, 0, 0),
        max(1, int(1.5 * s)),
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"H {m['H']:.0f} cm   V {m['V']:.0f} cm",
        (x0 + int(10 * s), y0 + int(72 * s)),
        font,
        0.45 * s,
        (0, 0, 0),
        max(1, int(1.5 * s)),
        cv2.LINE_AA,
    )
    return img


# ==============================================================================
# ขั้นที่ ⑧.5 · ระบบนับรอบการยกอัตโนมัติ (Lift Repetition Counter)
# ==============================================================================
class LiftCounter:
    """
    ระบบตรวจจับและนับรอบการยกอัตโนมัติแบบ 3 จังหวะ (3-Phase State Machine)
    รองรับทั้ง:
    1. การก้มหยิบจากที่ต่ำขึ้นที่สูง (เช่น BK -> WS, KW -> WS)
    2. การหยิบจากที่สูงลงที่ต่ำ (เช่น AS -> WS, WS -> KW)
    3. การยกต่อเนื่องหลายรอบ (Multi-lift repetitions)

    โครงสร้าง 1 รอบการยก (1 Lifting Cycle):
    - IDLE (ท่ายืนพัก): มืออยู่ใกล้ตำแหน่งพักข้างลำตัว (V ใกล้ rest_v_cm และระยะเอื้อม H ต่ำ)
    - LIFTING (กำลังปฏิบัติงานยก):
        * จังหวะเริ่ม: มือเคลื่อนออกจากท่าพัก (ก้มลง / ยกขึ้น / เอื้อมมือออกไปหยิบของ)
        * จังหวะเคลื่อนย้าย: มือเคลื่อนที่ผ่านระยะแนวดิ่ง (Vertical Travel) หรือเอื้อมไปวางของ
        * จังหวะจบรอบ:
          (ก) ผู้ยกวางของเสร็จและดึงมือกลับเข้าสู่ท่าพัก (IDLE) นานเกิน 0.25 วินาที หรือ
          (ข) กรณียกต่อเนื่องไม่หยุดพัก: ผู้ยกเคลื่อนมือจากจุดต่ำสุดไปจุดสูงสุดแล้วก้มกลับลงมาจุดต่ำสุดอีกครั้ง
    """

    def __init__(self, calib: Dict, fps: float, cfg: Optional[AnalysisConfig] = None):
        self.calib = calib
        self.fps = fps
        self.cfg = cfg or AnalysisConfig()

        self.count = 0
        self.state = "IDLE"  # IDLE, LIFTING
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
        self.cycle_stage = 0  # 0=เริ่มออกตัว, 1=ถึงจุดหยิบ(ต่ำ/สูง), 2=ถึงจุดวาง(ตรงข้าม)
        self.peak_v = None
        self.trough_v = None

    def _is_in_rest(self, v_cm: float, h_cm: float) -> bool:
        """ตรวจว่ามืออยู่ในท่ายืนพักข้างลำตัว (ใกล้ระดับมือขณะยืนตรง และไม่ได้ยื่นแขนออกไป)"""
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

            # กรณีที่ 1: ยกต่อเนื่องโดยไม่หยุดพักในท่ายืนตรง (Continuous Multi-Lift)
            # เมื่อมือเคลื่อนจากโซนต่ำ (<= เข่า) ขึ้นไปโซนสูง (>= เอว) แล้วก้มกลับลงไปโซนต่ำอีกครั้ง
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
                # ยกขึ้นถึงระดับเอว/ไหล่แล้ว ก้มกลับลงไปต่ำกว่าระดับเข่าเพื่อหยิบชิ้นต่อไปทันที
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

            # กรณีที่ 2 (มาตรฐาน): ผู้ยกวางวัตถุเสร็จและดึงมือกลับเข้าสู่ท่ายืนพัก (IDLE)
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
        """ปิดรอบสุดท้ายหากคลิปตัดจบขณะที่ผู้ยกเพิ่งยกถึงจุดหมายแต่ยังไม่ทันลดมือลง"""
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

        # เรียงลำดับโซนเริ่มต้น -> โซนปลายทางตามเวลาที่เกิดจุดต่ำสุดและจุดสูงสุด
        if idx_min < idx_max:
            origin_zone, dest_zone = min_zone, max_zone
        else:
            origin_zone, dest_zone = max_zone, min_zone

        distinct_zones = []
        for z in self.zone_history:
            if not distinct_zones or distinct_zones[-1] != z:
                distinct_zones.append(z)

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


# ==============================================================================
# ขั้นที่ ⑨ · แผงข้อมูลบนหน้าจอ (HUD Overlay) และ Pipeline หลัก
# ==============================================================================
def draw_info_panel(img: np.ndarray, lines: List[Tuple[str, Tuple[int, int, int]]]):
    s = ui_scale(img)
    line_h = int(26 * s)
    panel_w = int(390 * s)
    panel_h = int(16 * s) + line_h * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.60, img, 0.40, 0, dst=img)
    for i, (text, color) in enumerate(lines):
        cv2.putText(
            img,
            text,
            (int(12 * s), int(28 * s) + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * s,
            color,
            max(1, int(1.6 * s)),
            cv2.LINE_AA,
        )
    return img


def draw_rep_counter_badge(
    img: np.ndarray, lift_count: int, lift_state: str, min_rep_limit: Optional[int] = None
):
    """วาดป้ายแสดงจำนวนรอบการยก (Lift Counter Badge) ที่มุมขวาบนให้เห็นชัดเจน"""
    s = ui_scale(img)
    w = img.shape[1]
    box_w, box_h = int(230 * s), int(92 * s)
    x0 = max(10, w - box_w - int(15 * s))
    y0 = int(15 * s)

    overlay = img.copy()
    bg_color = (20, 90, 20) if lift_state == "LIFTING" else (30, 30, 30)
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), bg_color, -1)
    cv2.addWeighted(overlay, 0.70, img, 0.30, 0, dst=img)
    border_color = (0, 255, 120) if lift_state == "LIFTING" else (200, 200, 200)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), border_color, max(1, int(2 * s)))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img,
        f"LIFT COUNT: {lift_count}",
        (x0 + int(12 * s), y0 + int(36 * s)),
        font,
        0.75 * s,
        (0, 230, 255),
        max(2, int(2.2 * s)),
        cv2.LINE_AA,
    )
    state_color = (0, 255, 120) if lift_state == "LIFTING" else (200, 200, 200)
    cv2.putText(
        img,
        f"STATE: {lift_state}",
        (x0 + int(12 * s), y0 + int(62 * s)),
        font,
        0.52 * s,
        state_color,
        max(1, int(1.5 * s)),
        cv2.LINE_AA,
    )
    if min_rep_limit is not None:
        cv2.putText(
            img,
            f"LAST REP MIN: {min_rep_limit} kg",
            (x0 + int(12 * s), y0 + int(82 * s)),
            font,
            0.46 * s,
            (255, 255, 255),
            max(1, int(1.3 * s)),
            cv2.LINE_AA,
        )
    return img


def convert_to_h264(raw_mp4: Path, final_mp4: Path) -> Path:
    """แปลงวิดีโอ mp4v เป็น H.264 (yuv420p) เพื่อให้เล่นใน Browser / VS Code / Streamlit ได้"""
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(raw_mp4),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "fast",
                "-crf",
                "23",
                str(final_mp4),
            ],
            check=True,
        )
        if raw_mp4.exists() and raw_mp4 != final_mp4:
            raw_mp4.unlink()
        return final_mp4
    except Exception:
        # หากไม่มี ffmpeg ให้ใช้ไฟล์ raw แทน
        if raw_mp4 != final_mp4 and raw_mp4.exists():
            raw_mp4.replace(final_mp4)
        return final_mp4


def run_pipeline(
    video_path: Path,
    out_dir: Path,
    cfg: Optional[AnalysisConfig] = None,
    max_frames: Optional[int] = None,
    progress_callback=None,
    show_window: bool = False,
) -> Dict:
    """
    รันการวิเคราะห์ทั้งหมดตั้งแต่ต้นจนจบ:
    คืนค่า dict ประกอบด้วย:
      - results_df: ตารางข้อมูลรายเฟรม
      - reps_df: ตารางสรุปแต่ละรอบการยก
      - calib: ค่าที่ได้จากการ Calibrate
      - video_out: Path ของไฟล์วิดีโอ H.264
      - csv_out: Path ของไฟล์ CSV รายเฟรม
      - reps_csv_out: Path ของไฟล์ CSV สรุปรายรอบ
      - keyframes: ภาพสำคัญ 3 จังหวะ (Calibrate, Max Reach H, Worst Weight Limit)
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg is None:
        inferred_h = infer_subject_height(video_path.name, 170.0)
        cfg = AnalysisConfig(subject_height_cm=inferred_h)

    cap, info = open_video(video_path)
    fps = info["fps"]
    total_frames = (
        min(info["frames"], max_frames) if max_frames else info["frames"]
    )

    stem = video_path.stem
    raw_out = out_dir / f"{stem}_raw_mp4v.mp4"
    final_out = out_dir / f"{stem}_analyzed.mp4"
    csv_out = out_dir / f"{stem}_frames.csv"
    reps_csv_out = out_dir / f"{stem}_reps_summary.csv"

    writer = cv2.VideoWriter(
        str(raw_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (info["width"], info["height"]),
    )
    smoother = PoseSmoother(
        info["width"],
        info["height"],
        min_conf=cfg.min_confidence,
        world_window=cfg.world_window,
    )
    n_calib = max(5, int(round(cfg.calib_seconds * fps)))
    calib, samples, pose = None, [], None
    counter: Optional[LiftCounter] = None
    records = []

    WHITE = (255, 255, 255)
    YELLOW = (0, 230, 255)
    RED = (80, 80, 255)
    GREEN = (80, 255, 120)

    # เก็บภาพนิ่งจังหวะสำคัญ
    calib_frame_img = None
    max_h_val, max_h_img = -1.0, None
    min_limit_val, min_limit_img = 999, None

    with create_pose_detector() as detector:
        i = 0
        while max_frames is None or i < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            xy, world, conf = extract_keypoints(detector, frame, i, fps)
            xy_s, world_s, observed = smoother.update(xy, world, conf)
            img = frame.copy()
            time_s = (i + 1) / fps
            row = {
                "frame_number": i + 1,
                "time_s": round(time_s, 3),
                "orientation_deg": np.nan,
                "twist_deg": np.nan,
                "H_cm": np.nan,
                "V_cm": np.nan,
                "zone": None,
                "reach_class": None,
                "limit_kg": np.nan,
                "lift_count": 0,
                "lift_state": "CALIBRATING",
                "rep_completed": False,
            }

            if calib is None:
                samples.append(
                    {"xy": xy_s, "world": world_s, "observed": observed}
                )
                if len(samples) >= n_calib:
                    calib = calibrate(
                        samples,
                        cfg.subject_height_cm,
                        info["width"],
                        info["height"],
                        cfg,
                    )
                    counter = LiftCounter(calib, fps, cfg)
                img = draw_skeleton(img, xy_s, observed)
                img = draw_info_panel(
                    img,
                    [
                        (
                            f"CALIBRATING... stand still {len(samples)}/{n_calib}",
                            YELLOW,
                        ),
                        (f"Subject Height: {cfg.subject_height_cm:.0f} cm", WHITE),
                    ],
                )
                if len(samples) == n_calib // 2:
                    calib_frame_img = img.copy()
            else:
                P = world_s * 100.0 * calib["world_scale"]
                pose = solve_camera_pose(P, xy_s, observed, calib["K"], pose)
                m = None
                if pose is not None:
                    C = to_camera_3d(xy_s, P, pose, calib["K"])
                    row["orientation_deg"] = orientation_angle(C)
                    m = measure_hands(C, xy_s, calib)
                row["twist_deg"] = twist_from_baseline(world_s, calib)

                # อัปเดตตัวนับรอบการยก
                rep_status = counter.update(i + 1, time_s, m, row["twist_deg"])
                row["lift_count"] = rep_status["lift_count"]
                row["lift_state"] = rep_status["lift_state"]
                row["rep_completed"] = rep_status["rep_just_completed"]

                if m is not None:
                    img = draw_weight_table(img, calib, m)
                img = draw_skeleton(img, xy_s, observed)

                twist_is_high = row["twist_deg"] > cfg.twist_warn_deg
                lines = [
                    (f"scale   {calib['cm_per_px']:.3f} cm/px", WHITE),
                    (
                        f"orient  {row.get('orientation_deg', np.nan):.0f} deg to camera",
                        WHITE,
                    ),
                    (
                        f"twist   {row['twist_deg']:.0f} deg"
                        + (" [WARN!]" if twist_is_high else ""),
                        RED if twist_is_high else WHITE,
                    ),
                ]
                if m is not None:
                    img = draw_hand_label(img, calib, m)
                    lines += [
                        (f"hands   H {m['H']:.0f} cm | V {m['V']:.0f} cm", WHITE),
                        (
                            f"zone    {ROW_NAMES[m['row']]} | {COL_NAMES[m['col']]}",
                            WHITE,
                        ),
                        (f"LIMIT   {m['limit_kg']} kg", YELLOW),
                    ]
                    row.update(
                        {
                            "H_cm": m["H"],
                            "V_cm": m["V"],
                            "zone": ROW_NAMES[m["row"]],
                            "reach_class": COL_NAMES[m["col"]],
                            "limit_kg": m["limit_kg"],
                        }
                    )
                lines.append(
                    (
                        f"LIFTS   {row['lift_count']} reps ({row['lift_state']})",
                        GREEN if row["lift_state"] == "LIFTING" else YELLOW,
                    )
                )
                img = draw_info_panel(img, lines)
                last_rep_min = (
                    counter.reps[-1]["min_limit_kg"] if counter.reps else None
                )
                img = draw_rep_counter_badge(
                    img, row["lift_count"], row["lift_state"], last_rep_min
                )

                # เก็บภาพนิ่งจังหวะสำคัญ
                if m is not None:
                    if m["H"] > max_h_val:
                        max_h_val = m["H"]
                        max_h_img = img.copy()
                    if m["limit_kg"] < min_limit_val:
                        min_limit_val = m["limit_kg"]
                        min_limit_img = img.copy()

            writer.write(img)
            records.append(row)
            i += 1

            if progress_callback is not None:
                progress_callback(i, total_frames, img, row)

            if show_window:
                cv2.imshow("Lifting Weight Limit & Repetition Analyzer", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    writer.release()
    if show_window:
        cv2.destroyAllWindows()

    # ปิดรอบการยกสุดท้ายหากคลิปจบก่อนลดมือ
    if counter is not None and records:
        prev_count = counter.count
        counter.flush_at_end(records[-1]["frame_number"], records[-1]["time_s"])
        if counter.count > prev_count:
            records[-1]["lift_count"] = counter.count
            records[-1]["rep_completed"] = True

    results_df = pd.DataFrame(records)
    reps_df = pd.DataFrame(counter.reps if counter else [])

    results_df.to_csv(csv_out, index=False)
    reps_df.to_csv(reps_csv_out, index=False)
    final_video = convert_to_h264(raw_out, final_out)

    return {
        "video_info": info,
        "calib": calib,
        "results_df": results_df,
        "reps_df": reps_df,
        "video_out": final_video,
        "csv_out": csv_out,
        "reps_csv_out": reps_csv_out,
        "keyframes": {
            "calibrating": calib_frame_img,
            "max_reach": max_h_img,
            "min_weight_limit": min_limit_img,
        },
    }


# ==============================================================================
# ส่วนรันผ่าน Command Line (CLI)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="วิเคราะห์การยกน้ำหนักปลอดภัย NIOSH + นับรอบการยกอัตโนมัติ"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="videos/G10_S02_MultiLift_3Reps_Test.mp4",
        help="Path ไปยังไฟล์วิดีโอ .mp4",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help="ส่วนสูงจริงของผู้ยก (ซม.) หากไม่ระบุจะดึงจากรหัส S01..S05 อัตโนมัติ",
    )
    parser.add_argument(
        "--calib-sec",
        type=float,
        default=CALIB_SECONDS,
        help="ช่วงเวลายืนนิ่งต้นคลิปเพื่อ Calibrate (วินาที)",
    )
    parser.add_argument(
        "--twist-warn",
        type=float,
        default=TWIST_WARN_DEG,
        help="เกณฑ์แจ้งเตือนมุมบิดตัว (องศา)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="weight_limit_output",
        help="โฟลเดอร์สำหรับบันทึกผลลัพธ์",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="จำกัดจำนวนเฟรมสูงสุด (เช่น 150 สำหรับทดสอบเร็วๆ)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="เปิดหน้าต่างแสดงวิดีโอขณะประมวลผลแบบ Real-time",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    height_cm = (
        args.height
        if args.height is not None
        else infer_subject_height(video_path.name, 170.0)
    )
    cfg = AnalysisConfig(
        subject_height_cm=height_cm,
        calib_seconds=args.calib_sec,
        twist_warn_deg=args.twist_warn,
    )

    print("=" * 65)
    print(f"🎬 วิดีโอ        : {video_path}")
    print(f"📏 ส่วนสูงผู้ยก   : {cfg.subject_height_cm:.1f} ซม.")
    print(f"⏱️ ช่วง Calibrate : {cfg.calib_seconds:.1f} วินาที")
    print("=" * 65)

    t0 = time.time()

    def _cli_progress(frame_idx, total, _, row):
        if frame_idx % 50 == 0 or frame_idx == total:
            print(
                f"  ประมวลผลเฟรม {frame_idx:4d}/{total} | "
                f"H={row['H_cm'] if pd.notna(row['H_cm']) else 0:4.1f} cm | "
                f"V={row['V_cm'] if pd.notna(row['V_cm']) else 0:5.1f} cm | "
                f"Twist={row['twist_deg'] if pd.notna(row['twist_deg']) else 0:4.1f}° | "
                f"Lifts={row['lift_count']}"
            )

    res = run_pipeline(
        video_path=video_path,
        out_dir=Path(args.out_dir),
        cfg=cfg,
        max_frames=args.max_frames,
        progress_callback=_cli_progress,
        show_window=args.show,
    )

    elapsed = time.time() - t0
    df = res["results_df"]
    reps_df = res["reps_df"]
    valid = df.dropna(subset=["limit_kg"])

    print("-" * 65)
    print(f"✅ ประมวลผลเสร็จสิ้นใน {elapsed:.1f} วินาที")
    print(f"📹 วิดีโอผลลัพธ์   : {res['video_out']}")
    print(f"📊 ข้อมูลรายเฟรม   : {res['csv_out']}")
    print(f"🔄 สรุปรายรอบยก  : {res['reps_csv_out']}")
    if not valid.empty:
        print(
            f"🏋️ น้ำหนักปลอดภัยต่ำสุดตลอดคลิป : {valid['limit_kg'].min():.0f} kg "
            f"| ระยะเอื้อมสูงสุด H : {valid['H_cm'].max():.1f} cm "
            f"| มุมบิดตัวสูงสุด : {df['twist_deg'].max():.1f}°"
        )
    print(f"🔢 จำนวนรอบการยกที่ตรวจพบทั้งหมด: {len(reps_df)} รอบ")
    if not reps_df.empty:
        print("\nตารางสรุปรายรอบการยก (Per-Repetition Summary):")
        print(reps_df.to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    main()
