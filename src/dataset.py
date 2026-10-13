"""
Dataset utilities, video readers, and protocol parsers
"""

import re
from pathlib import Path
from typing import Dict, Optional, Tuple
import cv2

# ตารางเกณฑ์น้ำหนักปลอดภัยสูงสุดมาตรฐาน NIOSH 3x4 (กิโลกรัม)
ROW_NAMES = ["Above Shoulder", "Waist-Shoulder", "Knee-Waist", "Below Knee"]
COL_NAMES = ["Near", "Mild", "Extended"]
WEIGHT_LIMITS_KG = [
    [29, 18, 14],  # Above Shoulder
    [32, 23, 18],  # Waist to Shoulder
    [41, 25, 18],  # Knee to Waist (สูงสุด 41 kg)
    [31, 23, 16],  # Below Knee
]

# เกณฑ์ระยะเอื้อม H ตามหน่วยนิ้วแปลงเป็นเซนติเมตร
REACH_LIMITS_CM = [7 * 2.54, 12 * 2.54]  # 17.78, 30.48 ซม.

# ส่วนสูงจริงของผู้ยกในกลุ่ม G10
G10_SUBJECT_HEIGHTS = {
    "S01": 170.0,
    "S02": 168.0,
    "S03": 174.0,
    "S04": 170.0,
    "S05": 170.0,
}

# รหัสโซนตามโพรโทคอล Workshop 7
ZONE_CODE = {
    "AS": "Above Shoulder",
    "WS": "Waist-Shoulder",
    "KW": "Knee-Waist",
    "BK": "Below Knee",
}


def open_video(path: Path) -> Tuple[cv2.VideoCapture, Dict]:
    """เปิดไฟล์วิดีโอและอ่านข้อมูล metadata (fps, width, height, total frames)"""
    path = Path(path)
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


def read_frame(path: Path, frame_number: int):
    """อ่านภาพเฟรมที่ต้องการ (นับเริ่มจาก 1)"""
    cap, _ = open_video(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def infer_subject_height(video_name: str, default_height: float = 170.0) -> float:
    """ระบุส่วนสูงของผู้ยกอัตโนมัติจากรหัส S01..S05 ในชื่อไฟล์"""
    m = re.search(r"_(S0[1-9])_", video_name)
    if m and m.group(1) in G10_SUBJECT_HEIGHTS:
        return G10_SUBJECT_HEIGHTS[m.group(1)]
    return default_height


def parse_protocol_filename(stem: str) -> Optional[Dict]:
    """
    แยกข้อมูลการทดลองจากชื่อไฟล์ตามโพรโทคอล
    ตัวอย่าง: G10_S02_BK-WS_RM_T0_PN_01
    """
    found = re.search(r"_([A-Z]{2})-([A-Z]{2})_(R[NMF])_(T[01])_", stem)
    if not found:
        return None
    origin, dest, reach_code, twist_code = found.groups()
    reach_map = {"RN": "Near", "RM": "Mild", "RF": "Extended"}
    twist_map = {"T0": "No Twist (< 45°)", "T1": "With Twist (≥ 45°)"}
    return {
        "origin_code": origin,
        "origin_zone": ZONE_CODE.get(origin, origin),
        "dest_code": dest,
        "dest_zone": ZONE_CODE.get(dest, dest),
        "reach_code": reach_code,
        "reach_expected": reach_map.get(reach_code, reach_code),
        "twist_code": twist_code,
        "twist_expected": twist_map.get(twist_code, twist_code),
    }

