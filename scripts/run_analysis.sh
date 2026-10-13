#!/usr/bin/env bash
# สคริปต์สำหรับรันการวิเคราะห์วิดีโอผ่าน CLI
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VIDEO="${1:-data/raw/G10_S02_MultiLift_3Reps_Test.mp4}"
HEIGHT="${2:-}"

if [ -n "$HEIGHT" ]; then
    ./.venv/bin/python main.py --video "$VIDEO" --height "$HEIGHT"
else
    ./.venv/bin/python main.py --video "$VIDEO"
fi

