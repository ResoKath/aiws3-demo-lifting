# 🏋️ NIOSH Lifting Weight Limit & Repetition Analyzer

**รายวิชา CP413705 การฝึกปฏิบัติงานปัญญาประดิษฐ์ 3 (AI Workshop III)**
วิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น · **กลุ่ม G10**

ระบบประเมินความเสี่ยงและคำนวณน้ำหนักสูงสุดที่ยอมรับได้ตามมาตรฐาน **NIOSH Lifting Equation** ด้วยโมเดลเรขาคณิต 3 มิติจาก MediaPipe Pose Keypoints พร้อมระบบนับรอบการยกอัตโนมัติ (Lift Repetition Counter)

---

## 📁 โครงสร้างโปรเจกต์ (Project Schema)

```text
project_lifting-weight-DEMO/
│
├── data/                    # ข้อมูลวิดีโอทั้งหมด
│   ├── raw/                 # คลิปวิดีโอต้นฉบับของกลุ่ม G10 (.mp4)
│   └── processed/           # ข้อมูลที่ตัดต่อหรือปรับแต่งแล้ว
│
├── notebooks/               # ไฟล์ทดลองและวิเคราะห์เบื้องต้น
│   └── 01_eda_lifting_demo.ipynb
│
├── src/                     # โมดูลโค้ดหลัก (Clean Python Modules)
│   ├── __init__.py          # รวมการ Export แพ็กเกจ
│   ├── dataset.py           # ตัวอ่านวิดีโอ, Parser รหัสชื่อไฟล์, ค่าส่วนสูงสมาชิก
│   ├── model.py             # MediaPipe Pose, พิกัด 3D, ตาราง NIOSH, ตัวนับรอบ LiftCounter
│   └── utils.py             # Kalman Smoothing, วาด HUD Overlay, แปลง MP4 H.264, กราฟ
│
├── runs/                    # โฟลเดอร์เก็บผลลัพธ์ที่สร้างอัตโนมัติ
│   ├── checkpoints/         # โมเดล pose_landmarker_lite.task
│   └── logs/                # วิดีโอผลลัพธ์ (.mp4), ตารางสถิติรายเฟรมและสรุปรายรอบ (.csv)
│
├── input/                   # วิดีโอใหม่ที่ผู้ใช้อัปโหลดเข้ามาทดสอบ
├── docs/                    # เอกสารประกอบและคู่มือ
│   ├── instruction.md       # รายการ Checklist สิ่งที่ต้องทำและสคริปต์นำเสนอ 5 นาที
│   └── references/          # เอกสารอ้างอิง
│
├── scripts/                 # สคริปต์ช่วยรันคำสั่ง
│   ├── run_analysis.sh      # รันวิเคราะห์ผ่าน Terminal
│   └── start_app.sh         # เปิด Streamlit Web App
│
├── .vscode/                 # การตั้งค่า Environment สำหรับ VS Code
│   └── settings.json
├── main.py                  # สคริปต์หลักสำหรับรันประมวลผลผ่าน CLI
├── app.py                   # ซอฟต์แวร์ Web Application (Streamlit) สำหรับส่ง Task 4
├── requirements.txt         # รายการไลบรารีที่จำเป็น
├── .gitignore               # กรองไฟล์ขนาดใหญ่และแคชออกจาก Git
└── README.md                # คู่มือการใช้งานโปรเจกต์
```

---

## 🚀 วิธีการใช้งาน (Quick Start)

### 1. ติดตั้งไลบรารี (หากยังไม่ได้ติดตั้ง)

```bash
pip install -r requirements.txt
```

### 2. รันวิเคราะห์ผ่าน CLI (main.py)

สามารถรันวิเคราะห์วิดีโอใดๆ ได้ทันที ระบบจะตรวจจับส่วนสูงตามรหัสผู้ยก (`S01=170`, `S02=168`, `S03=174`, `S05=170` ซม.) อัตโนมัติ:

- **ทดสอบคลิปยกต่อเนื่อง 3 รอบ (Multi-Lift):**

  ```bash
  python main.py --video data/raw/G10_S02_MultiLift_3Reps_Test.mp4
  ```
- **ทดสอบคลิปยกปกติของสมาชิก:**

  ```bash
  python main.py --video data/raw/G10_S02_BK-WS_RM_T0_PN_01.mp4
  ```
- **เปิดหน้าต่าง Preview สดขณะประมวลผล:**

  ```bash
  python main.py --video data/raw/G10_S02_MultiLift_3Reps_Test.mp4 --show
  ```

ผลลัพธ์จะถูกบันทึกไว้ใน `runs/logs/`:

- วิดีโอผลลัพธ์: `runs/logs/<ชื่อคลิป>_analyzed.mp4`
- สถิติรายเฟรม: `runs/logs/<ชื่อคลิป>_frames.csv`
- ตารางสรุปรายรอบการยก: `runs/logs/<ชื่อคลิป>_reps_summary.csv`

---

### 3. รันซอฟต์แวร์ Web Application (app.py)

คุณสามารถเปิดแอปผ่านคำสั่ง:

```bash
streamlit run app.py
```

*(หรือกดปุ่ม **Run Python File** ใน VS Code บนไฟล์ `app.py` ระบบจะเปิดเบราว์เซอร์ให้อัตโนมัติ)*

**ความสามารถของ Web App:**

1. **Interactive Dashboard**: เลือกคลิปจากคลัง `data/raw/` หรืออัปโหลดคลิปใหม่จากคอมพิวเตอร์
2. **Real-time Live Progress**: แสดงภาพโครงกระดูกและตาราง Perspective ขณะกำลังวิเคราะห์
3. **KPI Summary Cards**: แสดงจำนวนรอบยก (`Total Lifts`), น้ำหนักปลอดภัยต่ำสุด (`Worst-case Limit`), ระยะเอื้อมสูงสุด (`Max H`), มุมบิดตัว (`Twist`), และมุมหันเทียบกล้อง
4. **Per-Repetition Table**: ตารางแจกแจงสถิติแยกตามรอบการยก พร้อมตรวจสอบเทียบกับรหัสในชื่อไฟล์
5. **Interactive Charts**: กราฟเส้นเวลา 4 แถว ($V$, $H$, มุมบิดตัว, น้ำหนักปลอดภัยพร้อมเส้นแบ่งรอบ) และ Heatmap ตารางเวลา NIOSH 3×4
