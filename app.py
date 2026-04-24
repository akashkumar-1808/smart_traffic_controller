import streamlit as st
import cv2
from src.perception.yolo_detector import load_model, TrafficProcessor
from src.utils.config import PERCEPTION_CONFIG

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Traffic Control", layout="wide")

# =========================
# CUSTOM CSS (FUTURISTIC UI)
# =========================
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.big-title {
    font-size: 40px;
    font-weight: bold;
    color: #00FFC6;
}
.metric-box {
    background-color: #161B22;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 15px;
}
.metric-value {
    font-size: 28px;
    color: #00FFC6;
}
.metric-label {
    font-size: 16px;
    color: #A0A0A0;
}
.decision-box {
    background-color: #0B3D2E;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🚦 AI Traffic Control System</div>', unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def get_model():
    return load_model()

model = get_model()
processor = TrafficProcessor(model)

cap = cv2.VideoCapture(PERCEPTION_CONFIG["camera_url"])

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns([3, 1])

frame_placeholder = col1.empty()

lane1_box = col2.empty()
lane2_box = col2.empty()
decision_box = col2.empty()

# =========================
# LOOP
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result = processor.process_frame(frame)

    # Convert frame
    frame_rgb = cv2.cvtColor(result["frame"], cv2.COLOR_BGR2RGB)

    # Display video
    frame_placeholder.image(frame_rgb, channels="RGB")

    # Lane metrics
    lane1_box.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Lane 1 Vehicles</div>
        <div class="metric-value">{result["lane1_count"]}</div>
    </div>
    """, unsafe_allow_html=True)

    lane2_box.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Lane 2 Vehicles</div>
        <div class="metric-value">{result["lane2_count"]}</div>
    </div>
    """, unsafe_allow_html=True)

    # Decision box
    decision_box.markdown(f"""
    <div class="decision-box">
        <h2>🚦 {result["current_green"]} GREEN</h2>
        <h3>⏱ {result["green_time"]} sec</h3>
    </div>
    """, unsafe_allow_html=True)