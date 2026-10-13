"""
Helper functions: Kalman tracking, smoothing, HUD drawing, video encoding, and time-series plotting
"""

import math
import subprocess
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dataset import (
    COL_NAMES,
    REACH_LIMITS_CM,
    ROW_NAMES,
    WEIGHT_LIMITS_KG,
)

SKELETON = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19),
    (12, 14), (14, 16), (16, 18), (16, 20),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    (0, 11), (0, 12),
]

PX_PER_CM = 8
ABOVE_MARGIN_CM = 30.0
HEADER_CM = 10.0


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
    def __init__(self, width: int, height: int, min_conf: float = 0.5):
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
                if self.missing <= 5 and inside:
                    point = predicted
                else:
                    self.kalman, self.missing, self.history = None, 0, []

        if point is None:
            return np.array([np.nan, np.nan]), False
        self.history = (self.history + [point])[-(3 + 1) :]
        return np.mean(self.history, axis=0), observed


class PoseSmoother:
    def __init__(
        self,
        width: int,
        height: int,
        min_conf: float = 0.5,
        world_window: int = 5,
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


def cell_color(kg: int) -> Tuple[int, int, int]:
    t = (41 - kg) / (41 - 14)
    light, dark = np.array([160, 225, 255]), np.array([20, 20, 170])
    return tuple(int(c) for c in light + (dark - light) * t)


def make_table_image(
    calib: Dict, highlight: Tuple[int, int], axis_on_left: bool
) -> Tuple[np.ndarray, np.ndarray]:
    table_len = calib.get("table_length_cm", 55.0)
    c_alpha = calib.get("cell_alpha", 0.45)
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
    from .model import project
    body, K = m["body"], calib["K"]
    table_len = calib.get("table_length_cm", 55.0)
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
    from .model import project
    body, K, s = m["body"], calib["K"], ui_scale(img)
    table_len = calib.get("table_length_cm", 55.0)
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
        if raw_mp4 != final_mp4 and raw_mp4.exists():
            raw_mp4.replace(final_mp4)
        return final_mp4


def build_timeseries_figure(
    df: pd.DataFrame,
    reps_df: pd.DataFrame,
    calib: dict,
    twist_warn_deg: float = 45.0,
    calib_seconds: float = 1.5,
):
    fig, axes = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True)
    t = df["time_s"]
    blue, orange, green, red, grey = (
        "#1a73e8",
        "#e8710a",
        "#1e8e3e",
        "#d93025",
        "#9aa0a6",
    )

    axes[0].plot(t, df["V_cm"], color=blue, lw=2, label="Hand Height V (cm)")
    if calib:
        for name, col_line in [
            ("knee", "#fbbc04"),
            ("waist", "#f29900"),
            ("shoulder", "#ea4335"),
        ]:
            lvl = calib.get(f"{name}_cm")
            if lvl is not None:
                axes[0].axhline(lvl, color=col_line, ls="--", lw=1.2, alpha=0.8)
                axes[0].text(
                    t.iloc[-1] if len(t) else 0,
                    lvl,
                    f" {name} ({lvl:.0f}cm)",
                    va="center",
                    fontsize=8.5,
                    color=col_line,
                )
    axes[0].set_ylabel("V (cm)")
    axes[0].set_title("1. Hand Vertical Height (V) & Body Levels", fontsize=10, fontweight="bold")

    axes[1].plot(t, df["H_cm"], color=blue, lw=2, label="Reach H (cm)")
    for cm, label in zip(REACH_LIMITS_CM, ['Near/Mild (7"=17.8cm)', 'Mild/Ext (12"=30.5cm)']):
        axes[1].axhline(cm, color=grey, ls="--", lw=1.2)
        axes[1].text(
            t.iloc[-1] if len(t) else 0,
            cm,
            f" {label}",
            va="center",
            fontsize=8.5,
            color="#5f6368",
        )
    axes[1].set_ylabel("H (cm)")
    axes[1].set_title("2. Horizontal Reach Distance (H)", fontsize=10, fontweight="bold")

    axes[2].plot(t, df["twist_deg"], color=red, lw=2, label="Twist Angle (deg)")
    axes[2].plot(
        t,
        df["orientation_deg"],
        color=orange,
        lw=1.5,
        alpha=0.85,
        label="Camera Orientation (deg)",
    )
    axes[2].axhline(twist_warn_deg, color=red, ls="--", lw=1.2, alpha=0.7)
    axes[2].text(
        t.iloc[-1] if len(t) else 0,
        twist_warn_deg,
        f" Twist Limit ({twist_warn_deg:.0f}°)",
        va="center",
        fontsize=8.5,
        color=red,
    )
    axes[2].set_ylabel("Angle (deg)")
    axes[2].legend(frameon=False, loc="upper left", ncol=2, fontsize=8.5)
    axes[2].set_title("3. Trunk Twist & Orientation Angles", fontsize=10, fontweight="bold")

    axes[3].step(t, df["limit_kg"], where="post", color=green, lw=2, label="Acceptable Weight Limit (kg)")
    axes[3].set_ylim(0, 46)
    axes[3].set_ylabel("Max Weight (kg)")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].set_title("4. NIOSH Max Acceptable Weight Limit (kg)", fontsize=10, fontweight="bold")

    for ax in axes:
        ax.axvspan(0, calib_seconds, color="#e8eaed", alpha=0.7, zorder=0)
        ax.grid(alpha=0.25)
        if not reps_df.empty:
            for _, rep in reps_df.iterrows():
                ax.axvspan(
                    rep["start_time_s"],
                    rep["end_time_s"],
                    color="#e6f4ea",
                    alpha=0.45,
                    zorder=0,
                )
                ax.axvline(rep["end_time_s"], color=green, ls=":", lw=1.3, alpha=0.85)

    if not reps_df.empty:
        for _, rep in reps_df.iterrows():
            mid_t = 0.5 * (rep["start_time_s"] + rep["end_time_s"])
            axes[3].text(
                mid_t,
                42,
                f"Rep #{int(rep['rep_number'])}\n(min {int(rep['min_limit_kg'])}kg)",
                ha="center",
                va="top",
                fontsize=8.5,
                fontweight="bold",
                color=green,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=green, alpha=0.85),
            )

    plt.tight_layout()
    return fig

