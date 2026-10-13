#!/usr/bin/env bash
"""
Automated Setup & One-Click Installer
CP413705 AI Workshop III - KKU (Group G10)

สคริปต์สำหรับติดตั้ง Environment และดาวน์โหลดโมเดลอัตโนมัติสำหรับการรันครั้งแรก
วิธีใช้งาน:
    python setup.py
"""

import os
import sys
import venv
import shutil
import platform
import subprocess
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VENV_DIR = ROOT_DIR / ".venv"
REQUIREMENTS_FILE = ROOT_DIR / "requirements.txt"
CHECKPOINT_DIR = ROOT_DIR / "runs" / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# รายชื่อโฟลเดอร์ตาม Schema ที่ต้องสร้าง
REQUIRED_DIRS = [
    ROOT_DIR / "data" / "raw",
    ROOT_DIR / "data" / "processed",
    ROOT_DIR / "input",
    ROOT_DIR / "runs" / "checkpoints",
    ROOT_DIR / "runs" / "logs",
    ROOT_DIR / "docs" / "references",
    ROOT_DIR / "notebooks",
    ROOT_DIR / "scripts",
    ROOT_DIR / ".vscode",
]


def print_banner():
    print("=" * 68)
    print(" 🏋️ NIOSH Lifting Weight Limit & Repetition Analyzer ")
    print("    CP413705 AI Workshop III - KKU (Group G10)")
    print("    One-Click Setup & Environment Installer")
    print("=" * 68)


def get_venv_python():
    """หา path ของ python binary ภายใน .venv ตามระบบปฏิบัติการ"""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def get_venv_pip():
    """หา path ของ pip ภายใน .venv"""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


def step_create_dirs():
    print("\n[1/6] 📁 ตรวจสอบและสร้างโครงสร้างโฟลเดอร์...")
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    print("  ✅ สร้างโฟลเดอร์ตาม Schema ครบถ้วน")


def step_create_venv():
    print(f"\n[2/6] 🐍 ตรวจสอบ Virtual Environment (.venv)...")
    venv_python = get_venv_python()
    if not venv_python.exists():
        print(f"  กำลังสร้าง virtual environment ใน {VENV_DIR.name} ...")
        venv.create(VENV_DIR, with_pip=True)
        print("  ✅ สร้าง .venv สำเร็จ")
    else:
        print(f"  ✅ พบ .venv อยู่แล้วที่ {VENV_DIR}")


def step_install_dependencies():
    print("\n[3/6] 📦 ติดตั้งไลบรารีจาก requirements.txt...")
    venv_python = str(get_venv_python())

    if not REQUIREMENTS_FILE.exists():
        print(f"  ❌ ไม่พบไฟล์ {REQUIREMENTS_FILE}")
        return False

    # อัปเกรด pip ก่อน
    try:
        subprocess.run(
            [venv_python, "-m", "pip", "install", "--upgrade", "pip", "-q"],
            check=False,
        )
    except Exception:
        pass

    # ใช้ uv ถ้ามี หรือใช้ pip ปกติ
    uv_path = shutil.which("uv")
    if uv_path:
        print("  ⚡ ตรวจพบ 'uv' กำลังติดตั้งด้วยความเร็วสูง...")
        cmd = [uv_path, "pip", "install", "-r", str(REQUIREMENTS_FILE), "--python", venv_python]
    else:
        print("  กำลังติดตั้งผ่าน pip (อาจใช้เวลาสักครู่)...")
        cmd = [venv_python, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)]

    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("  ✅ ติดตั้งไลบรารีทั้งหมดสำเร็จเรียบร้อย")
        return True
    else:
        print("  ❌ ติดตั้งล้มเหลว กรุณาตรวจสอบการเชื่อมต่ออินเทอร์เน็ต")
        return False


def step_download_model():
    print("\n[4/6] 🤖 ตรวจสอบโมเดล MediaPipe Pose Landmarker...")
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 1000:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"  กำลังดาวน์โหลด {MODEL_PATH.name} จาก Google Storage ...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print(f"  ✅ ดาวน์โหลดโมเดลสำเร็จ ({MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB)")
        except Exception as e:
            print(f"  ❌ ไม่สามารถดาวน์โหลดโมเดลได้: {e}")
            return False
    else:
        print(f"  ✅ พบโมเดลแล้ว ({MODEL_PATH.name} · {MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB)")
    return True


def step_configure_vscode():
    print("\n[5/6] ⚙️ ตั้งค่า VS Code Workspace (.vscode/settings.json)...")
    vscode_settings = ROOT_DIR / ".vscode" / "settings.json"
    content = """{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.analysis.extraPaths": [
        "${workspaceFolder}/src"
    ],
    "python.terminal.activateEnvironment": true
}
"""
    vscode_settings.write_text(content, encoding="utf-8")
    print("  ✅ กำหนดค่า Python Interpreter ใน VS Code ชี้ไปที่ .venv สำเร็จ")


def step_check_ffmpeg():
    print("\n[6/6] 🎬 ตรวจสอบโปรแกรม FFmpeg (สำหรับการแปลงวิดีโอ H.264)...")
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"  ✅ พบ FFmpeg ในระบบ: {ffmpeg_path}")
    else:
        print("  ⚠️ ไม่พบคำสั่ง 'ffmpeg' ในระบบ")
        if platform.system() == "Linux":
            print("     คำแนะนำ: ติดตั้งผ่าน 'sudo apt install ffmpeg' หรือ 'sudo pacman -S ffmpeg'")
        elif platform.system() == "Darwin":
            print("     คำแนะนำ: ติดตั้งผ่าน 'brew install ffmpeg'")
        else:
            print("     คำแนะนำ: ดาวน์โหลดได้ที่ https://ffmpeg.org/download.html")



def main():
    print_banner()
    step_create_dirs()
    step_create_venv()
    ok_deps = step_install_dependencies()
    ok_model = step_download_model()
    step_configure_vscode()
    step_check_ffmpeg()

    print("\n" + "=" * 68)
    if ok_deps and ok_model:
        print("🎉 การติดตั้ง Environment และโมเดลเสร็จสมบูรณ์เรียบร้อยแล้ว!")
        print("=" * 68)
        print("\nวิธีรันโปรแกรม:")
        if platform.system() == "Windows":
            print("  1. เปิดใช้งาน Environment:")
            print("     .\\.venv\\Scripts\\activate")
            print("  2. เริ่มใช้งาน Web App:")
            print("     streamlit run app.py")
        else:
            print("  1. เปิดใช้งาน Environment:")
            print("     source .venv/bin/activate")
            print("  2. เริ่มใช้งาน Web App:")
            print("     streamlit run app.py")
    else:
        print("⚠️ มีบางขั้นตอนไม่สมบูรณ์ กรุณาตรวจสอบข้อความเตือนด้านบน")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()

