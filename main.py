"""
Main Entry-Point Script: NIOSH Lifting Weight Limit & Repetition Analysis
CP413705 AI Workshop III - KKU (Group G10)

Usage:
  python main.py --video data/raw/G10_S02_MultiLift_3Reps_Test.mp4
  python main.py --video data/raw/G10_S02_BK-WS_RM_T0_PN_01.mp4 --height 168
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd

from src.dataset import (
    COL_NAMES,
    ROW_NAMES,
    infer_subject_height,
    open_video,
)
from src.model import (
    AnalysisConfig,
    create_pose_detector,
    extract_keypoints,
    calibrate,
    measure_hands,
    orientation_angle,
    solve_camera_pose,
    to_camera_3d,
    twist_from_baseline,
    LiftCounter,
)
from src.utils import (
    PoseSmoother,
    convert_to_h264,
    draw_hand_label,
    draw_info_panel,
    draw_rep_counter_badge,
    draw_skeleton,
    draw_weight_table,
)


def run_pipeline(
    video_path: Path,
    out_dir: Path,
    cfg: Optional[AnalysisConfig] = None,
    max_frames: Optional[int] = None,
    progress_callback=None,
    show_window: bool = False,
) -> Dict:
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
                cv2.imshow("NIOSH Lifting Analyzer", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    writer.release()
    if show_window:
        cv2.destroyAllWindows()

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


def main():
    parser = argparse.ArgumentParser(
        description="Main Entry-Point: NIOSH Lifting Weight Limit & Repetition Analyzer"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/raw/G10_S02_MultiLift_3Reps_Test.mp4",
        help="Path to input video (.mp4)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help="Subject height in cm (auto-detected from S01..S05 if omitted)",
    )
    parser.add_argument(
        "--calib-sec",
        type=float,
        default=1.5,
        help="Calibration still duration in seconds",
    )
    parser.add_argument(
        "--twist-warn",
        type=float,
        default=45.0,
        help="Twist angle warning threshold in degrees",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/logs",
        help="Output directory for generated logs and videos",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display live preview window with cv2.imshow",
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
    print(f"🎬 Video input   : {video_path}")
    print(f"📏 Subject height : {cfg.subject_height_cm:.1f} cm")
    print(f"⏱️ Calibration    : {cfg.calib_seconds:.1f} seconds")
    print("=" * 65)

    t0 = time.time()

    def _cli_progress(frame_idx, total, _, row):
        if frame_idx % 50 == 0 or frame_idx == total:
            print(
                f"  Frame {frame_idx:4d}/{total} | "
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
    print(f"✅ Processing completed in {elapsed:.1f} seconds")
    print(f"📹 Output video   : {res['video_out']}")
    print(f"📊 Per-frame logs : {res['csv_out']}")
    print(f"🔄 Reps summary   : {res['reps_csv_out']}")
    if not valid.empty:
        print(
            f"🏋️ Min Safe Limit : {valid['limit_kg'].min():.0f} kg | "
            f"Max Reach H: {valid['H_cm'].max():.1f} cm | "
            f"Max Twist: {df['twist_deg'].max():.1f}°"
        )
    print(f"🔢 Total Lifts Detected: {len(reps_df)} reps")
    if not reps_df.empty:
        print("\nPer-Repetition Summary:")
        print(reps_df.to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    main()

