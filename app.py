"""
NIOSH Lifting Weight Limit & Repetition Analyzer - Web Application (Task 1-4)
รายวิชา CP413705 AI Workshop III - วิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น (กลุ่ม G10)

วิธีรัน:
  1. กดปุ่ม Run Python File ใน VS Code ได้เลย (สคริปต์จะเปิด Streamlit ให้อัตโนมัติ)
  2. หรือรันผ่าน Terminal: ./.venv/bin/streamlit run app.py
"""

import os
import re
import sys
import subprocess
from pathlib import Path

# ถ้ารันด้วย `python app.py` โดยตรง ให้เรียก `streamlit run app.py` อัตโนมัติ
if __name__ == "__main__" and os.environ.get("STREAMLIT_RUN_ACTIVE") != "1":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            env = os.environ.copy()
            env["STREAMLIT_RUN_ACTIVE"] = "1"
            cmd = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())] + sys.argv[1:]
            sys.exit(subprocess.call(cmd, env=env))
    except ImportError:
        pass

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from lifting_engine import (
    AnalysisConfig,
    COL_NAMES,
    REACH_LIMITS_CM,
    ROW_NAMES,
    WEIGHT_LIMITS_KG,
    ZONE_CODE,
    infer_subject_height,
    run_pipeline,
)

BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "weight_limit_output"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_timeseries_figure(
    df: pd.DataFrame,
    reps_df: pd.DataFrame,
    calib: dict,
    cfg: AnalysisConfig,
):
    """สร้างกราฟเส้นเวลา 4 แถว: V, H, Angles (Twist/Orient), และ Max Weight Limit พร้อมมาร์คจุดครบรอบยก"""
    fig, axes = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True)
    t = df["time_s"]
    blue, orange, green, red, grey = (
        "#1a73e8",
        "#e8710a",
        "#1e8e3e",
        "#d93025",
        "#9aa0a6",
    )

    # 1. ความสูงของมือ V (cm)
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
    axes[0].set_title("1. Hand Vertical Height (V) & Body Zone Levels", fontsize=10, fontweight="bold")

    # 2. ระยะเอื้อมแนวราบ H (cm)
    axes[1].plot(t, df["H_cm"], color=blue, lw=2, label="Horizontal Reach H (cm)")
    for cm, label in zip(REACH_LIMITS_CM, ['Near/Mild (7" = 17.8cm)', 'Mild/Ext (12" = 30.5cm)']):
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
    axes[1].set_title("2. Horizontal Reach Distance (H) from Ankle Midpoint", fontsize=10, fontweight="bold")

    # 3. มุมบิดตัว (Twist) และ มุมหันเทียบกล้อง (Orientation)
    axes[2].plot(t, df["twist_deg"], color=red, lw=2, label="Twist Angle (deg)")
    axes[2].plot(
        t,
        df["orientation_deg"],
        color=orange,
        lw=1.5,
         alpha=0.85,
        label="Camera Orientation (deg)",
    )
    axes[2].axhline(cfg.twist_warn_deg, color=red, ls="--", lw=1.2, alpha=0.7)
    axes[2].text(
        t.iloc[-1] if len(t) else 0,
        cfg.twist_warn_deg,
        f" Twist Limit ({cfg.twist_warn_deg:.0f}°)",
        va="center",
        fontsize=8.5,
        color=red,
    )
    axes[2].set_ylabel("Angle (deg)")
    axes[2].legend(frameon=False, loc="upper left", ncol=2, fontsize=8.5)
    axes[2].set_title("3. Trunk Twist Angle & Pelvis Orientation to Camera", fontsize=10, fontweight="bold")

    # 4. น้ำหนักสูงสุดที่ยอมรับได้ (Max Acceptable Weight Limit - kg)
    axes[3].step(t, df["limit_kg"], where="post", color=green, lw=2, label="Acceptable Weight Limit (kg)")
    axes[3].set_ylim(0, 46)
    axes[3].set_ylabel("Max Weight (kg)")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].set_title("4. NIOSH Max Acceptable Weight Limit (kg) & Repetition Markers", fontsize=10, fontweight="bold")

    # ไฮไลต์ช่วง Calibrate และช่วงรอบการยกแต่ละรอบ
    for ax in axes:
        ax.axvspan(0, cfg.calib_seconds, color="#e8eaed", alpha=0.7, zorder=0, label="Calibration")
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


def parse_protocol_filename(stem: str):
    """แยกข้อมูลเงื่อนไขจากชื่อไฟล์ตามโพรโทคอล เช่น G10_S02_BK-WS_RM_T0_PN_01"""
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


def main():
    st.set_page_config(
        page_title="NIOSH Lifting Weight Limit & Repetition Analyzer - G10",
        page_icon="🏋️",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        div[data-testid="stMetric"] {
            background-color: #f8f9fa;
            border: 1px solid #e0e3e7;
            padding: 12px 16px;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🏋️ ระบบวิเคราะห์น้ำหนักปลอดภัยและนับรอบการยกอัตโนมัติ (NIOSH 3D Lifting Analyzer)")
    st.caption(
        "รายวิชา **CP413705 การฝึกปฏิบัติงานปัญญาประดิษฐ์ 3 (AI Workshop III)** · "
        "วิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น · **กลุ่ม G10**"
    )

    # ==========================================================================
    # Sidebar: เลือกไฟล์วิดีโอและปรับจูนค่าพารามิเตอร์ (Task 1)
    # ==========================================================================
    with st.sidebar:
        st.header("🎬 1. เลือกวิดีโอทดสอบ")
        source_mode = st.radio(
            "แหล่งที่มาของวิดีโอ",
            ["เลือกจากคลังคลิปกลุ่ม G10 (videos/)", "อัปโหลดคลิปวิดีโอใหม่ (.mp4/.mov)"],
        )

        selected_video_path = None
        if source_mode == "เลือกจากคลังคลิปกลุ่ม G10 (videos/)":
            video_files = sorted(
                [p.name for p in VIDEOS_DIR.glob("*.mp4")],
                key=lambda x: (0 if "MultiLift" in x else 1, x),
            )
            if not video_files:
                st.warning("ไม่พบไฟล์ .mp4 ในโฟลเดอร์ videos/")
            else:
                default_idx = 0
                selected_name = st.selectbox(
                    "เลือกไฟล์วิดีโอ",
                    video_files,
                    index=default_idx,
                    help="คลิป G10_S02_MultiLift_3Reps_Test.mp4 คือคลิปทดสอบการยกต่อเนื่อง 3 รอบ",
                )
                selected_video_path = VIDEOS_DIR / selected_name
        else:
            uploaded = st.file_uploader("อัปโหลดวิดีโอ (.mp4, .mov)", type=["mp4", "mov"])
            if uploaded is not None:
                save_path = VIDEOS_DIR / uploaded.name
                save_path.write_bytes(uploaded.read())
                selected_video_path = save_path
                st.success(f"บันทึกไฟล์แล้ว: {uploaded.name}")

        auto_height = (
            infer_subject_height(selected_video_path.name, 170.0)
            if selected_video_path
            else 170.0
        )

        st.divider()
        st.header("⚙️ 2. ปรับค่าพารามิเตอร์ (Task 1)")
        subject_height_cm = st.number_input(
            "ส่วนสูงจริงของผู้ยก (ซม.)",
            min_value=140.0,
            max_value=210.0,
            value=float(auto_height),
            step=1.0,
            help="ดึงค่าอัตโนมัติจากรหัส S01=170, S02=168, S03=174 ซม. หรือปรับแก้ได้",
        )
        calib_seconds = st.slider(
            "ช่วงเวลายืนนิ่งต้นคลิป (CALIB_SECONDS)",
            min_value=0.5,
            max_value=3.5,
            value=1.5,
            step=0.1,
            help="ใช้ช่วงยืนตรงต้นคลิปเพื่อคำนวณ cm/px และระดับเข่า/เอว/ไหล่",
        )
        twist_warn_deg = st.slider(
            "เกณฑ์แจ้งเตือนมุมบิดตัว (TWIST_WARN_DEG)",
            min_value=15.0,
            max_value=60.0,
            value=45.0,
            step=5.0,
        )

        with st.expander("🔧 ตั้งค่าขั้นสูง (Advanced Calibration & Counter)"):
            nose_above_foot = st.slider(
                "อัตราส่วน พื้น→จมูก / ส่วนสูง (NOSE_ABOVE_FOOT)",
                min_value=0.850,
                max_value=0.950,
                value=0.905,
                step=0.005,
            )
            camera_fov_deg = st.slider(
                "มุมมองกล้อง (CAMERA_FOV_DEG)",
                min_value=55.0,
                max_value=85.0,
                value=70.0,
                step=1.0,
            )
            table_length_cm = st.slider(
                "ความยาวตารางด้านหน้า (TABLE_LENGTH_CM)",
                min_value=40.0,
                max_value=80.0,
                value=55.0,
                step=5.0,
            )
            min_vertical_travel_cm = st.slider(
                "ระยะยกแนวดิ่งขั้นต่ำต่อ 1 รอบ (ซม.)",
                min_value=10.0,
                max_value=40.0,
                value=18.0,
                step=2.0,
            )
            min_rep_duration_s = st.slider(
                "เวลาขั้นต่ำต่อ 1 รอบการยก (วินาที)",
                min_value=0.5,
                max_value=3.0,
                value=0.8,
                step=0.1,
            )

        run_btn = st.button(
            "🚀 เริ่มวิเคราะห์วิดีโอ (Start Analysis)",
            type="primary",
            use_container_width=True,
            disabled=(selected_video_path is None),
        )

    # ==========================================================================
    # ส่วนประมวลผลและแสดงผลลัพธ์
    # ==========================================================================
    if selected_video_path is None:
        st.info("👈 กรุณาเลือกไฟล์วิดีโอจากแถบด้านซ้ายเพื่อเริ่มต้น")
        return

    cfg = AnalysisConfig(
        subject_height_cm=subject_height_cm,
        calib_seconds=calib_seconds,
        nose_above_foot=nose_above_foot,
        camera_fov_deg=camera_fov_deg,
        twist_warn_deg=twist_warn_deg,
        table_length_cm=table_length_cm,
        min_vertical_travel_cm=min_vertical_travel_cm,
        min_rep_duration_s=min_rep_duration_s,
    )

    # ตรวจว่ามีผลลัพธ์ใน session_state หรือไฟล์ที่เคยรันไว้แล้วหรือไม่
    state_key = f"{selected_video_path.name}_{subject_height_cm}_{calib_seconds}_{twist_warn_deg}"
    if run_btn:
        st.subheader(f"⏳ กำลังประมวลผล: `{selected_video_path.name}`")
        prog_bar = st.progress(0, text="กำลังเริ่มต้น MediaPipe Pose Landmarker...")
        live_cols = st.columns([2, 1])
        with live_cols[0]:
            frame_placeholder = st.empty()
        with live_cols[1]:
            live_metrics_placeholder = st.empty()

        def _st_progress(frame_idx, total, img_bgr, row):
            pct = min(1.0, frame_idx / max(1, total))
            prog_bar.progress(
                pct,
                text=f"กำลังวิเคราะห์เฟรมที่ {frame_idx}/{total} ({pct * 100:.0f}%) · รอบการยกที่พบ: {row['lift_count']} รอบ",
            )
            if frame_idx % 8 == 0 or frame_idx == total:
                rgb_preview = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(
                    rgb_preview,
                    caption=f"Live Frame #{frame_idx} ({row['time_s']:.1f}s)",
                    use_container_width=True,
                )
                live_metrics_placeholder.markdown(
                    f"""
                    ### 📊 สถานะแบบ Real-time
                    - **สถานะการยก**: `{row['lift_state']}`
                    - **จำนวนรอบยกสะสม**: **`{row['lift_count']} รอบ`**
                    - **ระยะเอื้อม (H)**: `{row['H_cm']:.1f} ซม.` ({row['reach_class']})
                    - **ความสูงมือ (V)**: `{row['V_cm']:.1f} ซม.` ({row['zone']})
                    - **มุมบิดตัว (Twist)**: `{row['twist_deg']:.1f}°`
                    - **มุมหันเทียบกล้อง**: `{row['orientation_deg']:.1f}°`
                    - **น้ำหนักปลอดภัยสูงสุด**: **`{row['limit_kg'] if pd.notna(row['limit_kg']) else '-'} kg`**
                    """
                )

        res = run_pipeline(
            video_path=selected_video_path,
            out_dir=OUTPUT_DIR,
            cfg=cfg,
            progress_callback=_st_progress,
        )
        prog_bar.empty()
        frame_placeholder.empty()
        live_metrics_placeholder.empty()
        st.session_state["last_result"] = res
        st.session_state["last_video_name"] = selected_video_path.name
        st.session_state["last_cfg"] = cfg

    # หากยังไม่ได้กดรันในรอบนี้ แต่มีไฟล์ที่วิเคราะห์ไว้แล้วใน weight_limit_output ให้โหลดมาแสดงได้ทันที
    res = st.session_state.get("last_result")
    if res is None or st.session_state.get("last_video_name") != selected_video_path.name:
        stem = selected_video_path.stem
        existing_video = OUTPUT_DIR / f"{stem}_analyzed.mp4"
        existing_csv = OUTPUT_DIR / f"{stem}_frames.csv"
        existing_reps = OUTPUT_DIR / f"{stem}_reps_summary.csv"
        if existing_video.is_file() and existing_csv.is_file():
            df_loaded = pd.read_csv(existing_csv)
            reps_loaded = (
                pd.read_csv(existing_reps)
                if existing_reps.is_file() and existing_reps.stat().st_size > 2
                else pd.DataFrame()
            )
            res = {
                "video_info": {"fps": 30.0},
                "calib": {
                    "cm_per_px": 0.25,
                    "knee_cm": subject_height_cm * 0.285,
                    "waist_cm": subject_height_cm * 0.61,
                    "shoulder_cm": subject_height_cm * 0.818,
                },
                "results_df": df_loaded,
                "reps_df": reps_loaded,
                "video_out": existing_video,
                "csv_out": existing_csv,
                "reps_csv_out": existing_reps,
                "keyframes": {},
            }
        else:
            st.info(
                f"🎬 เลือกไฟล์ **`{selected_video_path.name}`** แล้ว — "
                "กดปุ่ม **🚀 เริ่มวิเคราะห์วิดีโอ (Start Analysis)** ที่แถบซ้ายเพื่อเริ่มคำนวณ"
            )
            st.video(str(selected_video_path))
            return

    df = res["results_df"]
    reps_df = res["reps_df"]
    calib = res.get("calib") or {}
    valid = df.dropna(subset=["limit_kg"])

    total_reps = len(reps_df) if not reps_df.empty else int(df["lift_count"].max())
    min_limit_kg = int(valid["limit_kg"].min()) if not valid.empty else 0
    max_h_cm = float(valid["H_cm"].max()) if not valid.empty else 0.0
    max_twist_deg = float(df["twist_deg"].max()) if df["twist_deg"].notna().any() else 0.0
    mean_orient_deg = (
        float(df["orientation_deg"].median())
        if df["orientation_deg"].notna().any()
        else 0.0
    )

    # ==========================================================================
    # แถบ KPI Cards สรุปผลหลัก
    # ==========================================================================
    st.subheader(f"📊 สรุปผลการวิเคราะห์: `{selected_video_path.name}`")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "🔢 จำนวนรอบการยก (Lift Count)",
        f"{total_reps} รอบ",
        delta="Multi-Lift Pass" if total_reps >= 2 else ("Single Lift" if total_reps == 1 else "No Lift"),
    )
    k2.metric(
        "🏋️ น้ำหนักปลอดภัยต่ำสุด (Min Limit)",
        f"{min_limit_kg} kg" if min_limit_kg else "N/A",
        delta="ความเสี่ยงสูง (≤18kg)" if min_limit_kg <= 18 else "เกณฑ์ปกติ",
        delta_color="inverse" if min_limit_kg <= 18 else "normal",
    )
    k3.metric(
        "📏 ระยะเอื้อมสูงสุด (Max H)",
        f"{max_h_cm:.1f} ซม.",
        delta="Extended (>30.5cm)" if max_h_cm >= 30.48 else ("Mild" if max_h_cm >= 17.78 else "Near"),
        delta_color="off",
    )
    k4.metric(
        "🔄 มุมบิดตัวสูงสุด (Max Twist)",
        f"{max_twist_deg:.1f}°",
        delta="เกินเกณฑ์ 45°!" if max_twist_deg > cfg.twist_warn_deg else "ปลอดภัย (<45°)",
        delta_color="inverse" if max_twist_deg > cfg.twist_warn_deg else "normal",
    )
    k5.metric(
        "📐 มุมหันเทียบกล้อง (Orientation)",
        f"{mean_orient_deg:.1f}°",
        delta=f"Scale {calib.get('cm_per_px', 0.0):.3f} cm/px",
        delta_color="off",
    )

    # ==========================================================================
    # แท็บแสดงผลละเอียด 3 ส่วน
    # ==========================================================================
    tab_video, tab_reps, tab_charts = st.tabs(
        [
            "🎬 1. วิดีโอผลลัพธ์ & ภาพจังหวะสำคัญ",
            "🔄 2. สรุปรายรอบการยก & ตรวจสอบโพรโทคอล",
            "📈 3. กราฟเส้นเวลา & ตารางเวลา NIOSH 3×4",
        ]
    )

    with tab_video:
        v_col1, v_col2 = st.columns([3, 2])
        with v_col1:
            st.markdown("#### 📹 วิดีโอที่ประมวลผลแล้ว (3D Perspective Table + Lift Counter)")
            if Path(res["video_out"]).is_file():
                st.video(str(res["video_out"]))
                with open(res["video_out"], "rb") as vf:
                    st.download_button(
                        "📥 ดาวน์โหลดวิดีโอผลลัพธ์ (.mp4)",
                        data=vf,
                        file_name=Path(res["video_out"]).name,
                        mime="video/mp4",
                        use_container_width=True,
                    )
        with v_col2:
            st.markdown("#### 🧭 ข้อมูลสเกลและจุดอ้างอิงร่างกาย (Calibration)")
            st.json(
                {
                    "subject_height_cm": cfg.subject_height_cm,
                    "cm_per_px": round(float(calib.get("cm_per_px", 0.0)), 4),
                    "camera_dist_cm": round(float(calib.get("camera_dist_cm", 0.0)), 1),
                    "knee_level_cm": round(float(calib.get("knee_cm", 0.0)), 1),
                    "waist_level_cm": round(float(calib.get("waist_cm", 0.0)), 1),
                    "shoulder_level_cm": round(float(calib.get("shoulder_cm", 0.0)), 1),
                }
            )
            keyframes = res.get("keyframes") or {}
            if any(v is not None for v in keyframes.values()):
                st.markdown("#### 📸 ภาพนิ่ง 3 จังหวะสำคัญ")
                for label_th, key in [
                    ("1. ช่วงยืนนิ่ง Calibrate", "calibrating"),
                    ("2. จังหวะเอื้อมมือไกลสุด (Max H)", "max_reach"),
                    ("3. จังหวะน้ำหนักปลอดภัยต่ำสุด (Worst Limit)", "min_weight_limit"),
                ]:
                    img_kf = keyframes.get(key)
                    if img_kf is not None:
                        st.image(
                            cv2.cvtColor(img_kf, cv2.COLOR_BGR2RGB),
                            caption=label_th,
                            use_container_width=True,
                        )

    with tab_reps:
        st.markdown("#### 🔄 ตารางสรุปผลการยกแยกตามรอบ (Per-Repetition Summary)")
        if reps_df.empty:
            st.warning("ไม่พบรอบการยกที่สมบูรณ์ในคลิปนี้ (เช่น อาจเป็นคลิป Negative หรือระยะเคลื่อนที่แนวดิ่งต่ำกว่าเกณฑ์)")
        else:
            display_reps = reps_df.rename(
                columns={
                    "rep_number": "รอบที่ (Rep #)",
                    "start_time_s": "เวลาเริ่ม (s)",
                    "end_time_s": "เวลาจบ (s)",
                    "duration_s": "ระยะเวลา (s)",
                    "origin_zone": "โซนเริ่มต้น (Origin)",
                    "dest_zone": "โซนปลายทาง (Destination)",
                    "zone_path": "ทิศทางการยก (Zone Transfer)",
                    "min_V_cm": "ความสูงต่ำสุด V (cm)",
                    "max_V_cm": "ความสูงสูงสุด V (cm)",
                    "vertical_travel_cm": "ระยะยกแนวดิ่ง ΔV (cm)",
                    "max_H_cm": "ระยะเอื้อมสูงสุด H (cm)",
                    "max_twist_deg": "มุมบิดตัวสูงสุด (°)",
                    "min_limit_kg": "น้ำหนักปลอดภัยต่ำสุด (kg)",
                }
            )
            st.dataframe(display_reps, use_container_width=True, hide_index=True)

            if Path(res["reps_csv_out"]).is_file():
                st.download_button(
                    "📥 ดาวน์โหลดตารางสรุปรายรอบ (.csv)",
                    data=Path(res["reps_csv_out"]).read_bytes(),
                    file_name=Path(res["reps_csv_out"]).name,
                    mime="text/csv",
                )

        st.divider()
        st.markdown("#### ✅ ตรวจสอบเทียบกับรหัสในชื่อไฟล์ตามโพรโทคอล (Protocol Check)")
        proto = parse_protocol_filename(selected_video_path.stem)
        if proto:
            p1, p2, p3 = st.columns(3)
            p1.info(
                f"**โซนตามชื่อไฟล์ ({proto['origin_code']} → {proto['dest_code']})**\n\n"
                f"`{proto['origin_zone']} → {proto['dest_zone']}`"
            )
            p2.info(
                f"**ระยะเอื้อมตามชื่อไฟล์ ({proto['reach_code']})**\n\n"
                f"ตั้งใจถ่าย: `{proto['reach_expected']}` | วัดค่า H สูงสุดได้: `{max_h_cm:.1f} ซม.`"
            )
            p3.info(
                f"**การบิดตัวตามชื่อไฟล์ ({proto['twist_code']})**\n\n"
                f"ตั้งใจถ่าย: `{proto['twist_expected']}` | วัดมุมบิดสูงสุดได้: `{max_twist_deg:.1f}°`"
            )
        else:
            st.caption("ชื่อไฟล์นี้เป็นคลิปทดสอบหลายรอบหรือไม่ได้ตั้งตามรหัสโพรโทคอลเดี่ยว")

    with tab_charts:
        st.markdown("#### 📈 กราฟเส้นเวลา (Time-Series Analysis: V, H, Twist, Weight Limit)")
        fig = build_timeseries_figure(df, reps_df, calib, cfg)
        st.pyplot(fig, use_container_width=True)

        st.divider()
        st.markdown("#### ⏱️ เวลาที่มืออยู่ในแต่ละช่องของตาราง NIOSH 3×4 (วินาที)")
        if not valid.empty:
            fps_val = res.get("video_info", {}).get("fps", 30.0)
            seconds = valid.groupby(["zone", "reach_class"]).size() / fps_val
            zone_table = (
                seconds.unstack("reach_class")
                .reindex(index=ROW_NAMES, columns=COL_NAMES)
                .fillna(0.0)
                .round(2)
            )
            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown("**เวลาที่มืออยู่ในแต่ละช่อง (วินาที)**")
                st.dataframe(zone_table, use_container_width=True)
            with c_right:
                st.markdown("**เกณฑ์น้ำหนักปลอดภัยมาตรฐาน NIOSH (กิโลกรัม)**")
                ref_table = pd.DataFrame(
                    WEIGHT_LIMITS_KG, index=ROW_NAMES, columns=COL_NAMES
                )
                st.dataframe(ref_table, use_container_width=True)

        if Path(res["csv_out"]).is_file():
            st.download_button(
                "📥 ดาวน์โหลดข้อมูลรายเฟรมทั้งหมด (.csv)",
                data=Path(res["csv_out"]).read_bytes(),
                file_name=Path(res["csv_out"]).name,
                mime="text/csv",
            )


if __name__ == "__main__":
    main()

