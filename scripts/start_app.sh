#!/usr/bin/env bash
# สคริปต์สำหรับเริ่มรัน Streamlit Web Application
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "🚀 Starting Streamlit Web App on http://localhost:8501 ..."
./.venv/bin/streamlit run app.py --server.port 8501

