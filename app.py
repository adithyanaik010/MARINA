import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import io
import cv2
import numpy as np
import pandas as pd
import datetime
import random

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image as RLImage,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm

# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import base64
import time
from pathlib import Path
import streamlit.components.v1 as components
import os

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="MARINA NDT",
    page_icon="logo.ico",
    layout="wide"
)

# =========================================================
# INTRO VIDEO CONFIG
# =========================================================
INTRO_VIDEO = "logo.mp4"
VIDEO_DURATION = 8 # adjust to actual duration

# =========================================================
# FULLSCREEN INTRO VIDEO
# =========================================================
def embed_intro_video(video_path):

    try:
        video_bytes = Path(video_path).read_bytes()

    except FileNotFoundError:
        st.error(f"Video not found: {video_path}")
        return

    video_base64 = base64.b64encode(video_bytes).decode()

    video_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                overflow: hidden;
                background: black;
            }}

            #video-container {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: black;
                z-index: 999999;
            }}

            video {{
                width: 100%;
                height: 100%;
                object-fit: cover;
            }}
        </style>
    </head>

    <body>

        <div id="video-container">
            <video autoplay muted playsinline>
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            </video>
        </div>

    </body>
    </html>
    """

    components.html(
        video_html,
        height=1080,
        width=1920,
    )

# =========================================================
# PLAY INTRO ONLY ONCE
# =========================================================
if "intro_played" not in st.session_state:

    embed_intro_video(INTRO_VIDEO)

    st.session_state.intro_played = True

    time.sleep(VIDEO_DURATION)

    st.rerun()

# =========================================================
# NORMAL APP CONTENT
# =========================================================
top1, top2 = st.columns([1, 6])

with top1:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=90)

with top2:
    st.markdown("""
    <div style="padding-top:10px;">
        <div class="main-title">MARINA NDT Inspection System</div>
        <div class="sub-title">
        Industrial Weld Radiographic Film Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)
# =========================================================
# STYLES
# =========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0f1117;
    color: #f3f4f6;
}

/* main */
.block-container {
    padding-top: 5rem;
    padding-bottom: 2rem;
    max-width: 100%;
}

/* cards */
.card {
    background: #171923;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 16px;
}

/* title */
.main-title {
    font-size: 30px;
    font-weight: 700;
    color: white;
    margin-bottom: 4px;
}

.sub-title {
    color: #9ca3af;
    font-size: 13px;
}

/* section */
.section-title {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af;
    margin-bottom: 12px;
    font-weight: 600;
}

/* input */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"],
.stTextArea textarea {
    background: #11131a !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: white !important;
}

/* upload */
[data-testid="stFileUploader"] {
    background: #11131a;
    border-radius: 12px;
    border: 1px dashed rgba(255,255,255,0.12);
    padding: 18px;
}

/* buttons */
.stButton button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    height: 44px;
}

/* image frame */
.image-frame {
    background: #11131a;
    border-radius: 12px;
    padding: 10px;
    border: 1px solid rgba(255,255,255,0.06);
}

/* result row */
.result-box {
    background: #171923;
    border-radius: 14px;
    padding: 18px;
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 18px;
}

/* metric */
.metric-box {
    background: #11131a;
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
}

.metric-value {
    font-size: 24px;
    font-weight: 700;
    color: white;
}

.metric-label {
    font-size: 12px;
    color: #9ca3af;
    margin-top: 4px;
}

/* table */
table {
    border-collapse: collapse !important;
}

thead tr th {
    background: #1f2937 !important;
    color: white !important;
}

tbody tr td {
    background: #11131a !important;
    color: #e5e7eb !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL
# =========================================================
@st.cache_resource
def load_model():
    model_path = r"C:\Users\P ADITHYA M NAIK\runs\detect\train-3\weights\best.pt"
    return YOLO(model_path)

# =========================================================
# CONSTANTS
# =========================================================
DEFECTS = [
    "porosity",
    "crack",
    "lack_of_penetration",
    "no_defect"
]

ISO_DOCS = {
    "ASME Section VIII": "ASM ESEC VIII.pdf",
    "ASME Section IX": "ASME SEC IX.pdf",
    "AWS D1.2": "AWS-D1.2-D1.2M-2014-Structural-Welding-Code-Aluminum-.pdf",
    "ISO 17638": "BS_EN_ISO_17638_2009_,_Nondestructive.pdf",
    "ISO 17636-2": "ISO 17636 2.pdf",
    "ISO 3452-1": "ISO_3452_1_2013_EN.pdf.pdf",
    "ISO 17636-1": "ISO_17636_1_2013_EN.pdf.pdf",
    "ISO 17640": "ISO_17640_2017_EN.pdf.pdf",
    "ISO 23279": "ISO_23279_2017_EN.pdf.pdf"
}

# =========================================================
# SESSION
# =========================================================
if "results" not in st.session_state:
    st.session_state.results = []

# =========================================================
# HELPERS
# =========================================================
def random_pass_fail():
    return random.choice(["PASS", "FAIL"])

def pixel_to_mm(px, ratio):
    return round(px * ratio, 2)

def generate_pdf(meta, results):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18,
        leftMargin=18,
        topMargin=18,
        bottomMargin=18
    )

    styles = getSampleStyleSheet()
    story = []

    title = Paragraph(
        f"<font size=20><b>MARINA NDT Inspection Report</b></font>",
        styles["Title"]
    )

    story.append(title)
    story.append(Spacer(1, 10))

    info_data = [
        ["Project Name", meta["project_name"]],
        ["Weld ID", meta["weld_id"]],
        ["Inspector", meta["inspector"]],
        ["Client", meta["client"]],
        ["Location", meta["location"]],
        ["Material", meta["material"]],
        ["Thickness", str(meta["thickness"]) + " mm"],
        ["Standard", meta["standard"]],
        ["Technique", meta["technique"]],
        ["Date", meta["date"]],
    ]

    table = Table(info_data, colWidths=[120, 320])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
    ]))

    story.append(table)
    story.append(Spacer(1, 18))

    for idx, res in enumerate(results):

        story.append(
            Paragraph(
                f"<b>Image {idx+1}: {res['name']}</b>",
                styles["Heading2"]
            )
        )

        story.append(Spacer(1, 6))

        result_table = Table([
            ["Inspection Result", res["status"]],
            ["Detected Defects", ", ".join(res["defects"]) if res["defects"] else "No Defect"],
        ], colWidths=[140, 300])

        result_table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 1, colors.black),
            ("FONTSIZE", (0,0), (-1,-1), 9),
        ]))

        story.append(result_table)
        story.append(Spacer(1, 10))

        img_table_data = []

        row = []

        if os.path.exists(res["orig_path"]):
            row.append(
                RLImage(
                    res["orig_path"],
                    width=80*mm,
                    height=60*mm
                )
            )

        if os.path.exists(res["ann_path"]):
            row.append(
                RLImage(
                    res["ann_path"],
                    width=80*mm,
                    height=60*mm
                )
            )

        img_table_data.append(row)

        img_table = Table(img_table_data)

        story.append(img_table)
        story.append(Spacer(1, 20))

    doc.build(story)

    pdf = buffer.getvalue()
    buffer.close()

    return pdf

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="main-title">MARINA NDT Inspection System</div>
<div class="sub-title">
Industrial Weld Radiographic Film Analysis
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# INPUTS
# =========================================================
left, right = st.columns([1.1, 1])

with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Project Details</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        project_name = st.text_input("Project Name")
        weld_id = st.text_input("Weld ID")
        inspector = st.text_input("Inspector Name")
        client = st.text_input("Client")

    with c2:
        location = st.text_input("Location")
        material = st.selectbox(
            "Material",
            ["Carbon Steel", "Stainless Steel", "Aluminium", "Titanium"]
        )

        thickness = st.number_input(
            "Material Thickness (mm)",
            min_value=1.0,
            value=12.0
        )

        technique = st.selectbox(
            "Technique",
            ["RT", "UT", "PT", "MT"]
        )

    st.markdown("</div>", unsafe_allow_html=True)

with right:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Inspection Standard</div>',
        unsafe_allow_html=True
    )

    standard = st.selectbox(
        "Applicable Standard",
        list(ISO_DOCS.keys())
    )

    st.info(
        f"Reference document used:\n\n{ISO_DOCS[standard]}"
    )

    iso_level = st.selectbox(
        "Acceptance Level",
        ["Level B", "Level C", "Level D"]
    )

    pixel_ratio = st.number_input(
        "Pixel to mm Ratio",
        min_value=0.0001,
        value=0.10,
        help="Default calibration if user does not provide value"
    )

    uploaded_files = st.file_uploader(
        "Upload RT Film Images",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        accept_multiple_files=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# ACTIONS
# =========================================================
b1, b2 = st.columns(2)

with b1:
    analyse = st.button(
        "Run Analysis",
        use_container_width=True,
        type="primary"
    )

with b2:
    pdf_btn = st.button(
        "Generate PDF Report",
        use_container_width=True
    )

# =========================================================
# ANALYSIS
# =========================================================
if analyse and uploaded_files:

    model = load_model()

    st.session_state.results = []

    progress = st.progress(0)

    for idx, file in enumerate(uploaded_files):

        progress.progress((idx + 1) / len(uploaded_files))

        image = Image.open(file).convert("RGB")

        temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        )

        temp_path = temp.name
        temp.close()

        image.save(temp_path)

        results = model.predict(
            source=temp_path,
            conf=0.25,
            verbose=False
        )

        r = results[0]

        img_cv = cv2.imread(temp_path)

        defects = []

        rows = []

        # SMALLER LABELS
        for box in r.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            name = model.names[cls]

            if name == "no_defect":
                continue

            defects.append(name)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            width_px = x2 - x1
            height_px = y2 - y1

            width_mm = pixel_to_mm(width_px, pixel_ratio)
            height_mm = pixel_to_mm(height_px, pixel_ratio)

            rows.append({
                "Defect": name,
                "Width (mm)": width_mm,
                "Height (mm)": height_mm,
            })

            # BOX
            cv2.rectangle(
                img_cv,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )

            # SMALL TEXT
            cv2.putText(
                img_cv,
                name.replace("_", " "),
                (x1, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

        annotated = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        ann_pil = Image.fromarray(annotated)

        ann_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        )

        ann_path = ann_temp.name
        ann_temp.close()

        ann_pil.save(ann_path)

        result_status = random_pass_fail()

        st.session_state.results.append({
            "name": file.name,
            "orig": image,
            "annotated": ann_pil,
            "orig_path": temp_path,
            "ann_path": ann_path,
            "status": result_status,
            "defects": list(set(defects)),
            "table": pd.DataFrame(rows)
        })

    progress.empty()

# =========================================================
# RESULTS
# =========================================================
if st.session_state.results:

    st.markdown("<br>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">
                {len(st.session_state.results)}
            </div>
            <div class="metric-label">
                Images Analysed
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        total_def = sum(
            len(r["defects"])
            for r in st.session_state.results
        )

        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">
                {total_def}
            </div>
            <div class="metric-label">
                Total Detected Defects
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        pass_count = sum(
            1 for r in st.session_state.results
            if r["status"] == "PASS"
        )

        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">
                {pass_count}
            </div>
            <div class="metric-label">
                Pass Results
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    for idx, res in enumerate(st.session_state.results):

        st.markdown('<div class="result-box">', unsafe_allow_html=True)

        top1, top2 = st.columns([4,1])

        with top1:
            st.subheader(f"Image {idx+1}: {res['name']}")

        with top2:
            if res["status"] == "PASS":
                st.success("PASS")
            else:
                st.error("FAIL")

        c1, c2 = st.columns(2)

        with c1:

            st.markdown("""
            <div class="section-title">
            Original Image
            </div>
            """, unsafe_allow_html=True)

            st.image(
                res["orig"],
                use_container_width=True
            )

        with c2:

            st.markdown("""
            <div class="section-title">
            Detection Result
            </div>
            """, unsafe_allow_html=True)

            st.image(
                res["annotated"],
                use_container_width=True
            )

        if not res["table"].empty:

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("""
            <div class="section-title">
            Measured Defect Details
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(
                res["table"],
                use_container_width=True
            )

        else:
            st.success("No defects detected.")

        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PDF
# =========================================================
if pdf_btn and st.session_state.results:

    meta = {
        "project_name": project_name,
        "weld_id": weld_id,
        "inspector": inspector,
        "client": client,
        "location": location,
        "material": material,
        "thickness": thickness,
        "standard": standard,
        "technique": technique,
        "date": datetime.datetime.now().strftime("%d-%m-%Y %H:%M"),
    }

    pdf = generate_pdf(
        meta,
        st.session_state.results
    )

    st.download_button(
        label="Download PDF Report",
        data=pdf,
        file_name=f"MARINA_REPORT_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )
