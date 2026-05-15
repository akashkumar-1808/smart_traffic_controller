import streamlit as st
import cv2
import time
from src.pipeline.main_pipeline import IntegratedController
from src.utils.config import PERCEPTION_CONFIG

# --- SECTION 1: INITIAL SETUP (Runs once) ---
st.set_page_config(page_title="AI Traffic Control", layout="wide")
st.title("🚦 Smart Urban Traffic Control (AI-Driven)")

if 'controller' not in st.session_state:
    with st.spinner("Loading AI Models..."):
        st.session_state.controller = IntegratedController()

# --- SECTION 2: UI PLACEHOLDERS (Runs once - "The Wall Mounts") ---
# We create these objects OUTSIDE the loop so they stay in one place
col1, col2 = st.columns([3, 1])

with col1:
    frame_placeholder = st.empty() # Video will stay here

with col2:
    st.subheader("System Metrics")
    # These "slots" will be updated inside the loop
    status_slot = st.empty()
    confidence_metric_slot = st.empty()
    confidence_bar_slot = st.empty()
    lane_info_slot = st.empty()
    
    st.divider()
    lane_1_stats = st.empty()
    lane_2_stats = st.empty()

# --- SECTION 3: THE LOOP (Runs for every frame - "The TV Show") ---
video_source = PERCEPTION_CONFIG['camera_url'] 
cap = cv2.VideoCapture(video_source)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Get the decision from your pipeline
    action, annotated_frame, is_override, confidence = st.session_state.controller.get_decision(frame)

    # 2. UPDATE THE SLOTS (This is the fix!)
    # We use .image(), .metric(), etc. on the SLOT variables we created above
    
    frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

    if is_override:
        status_slot.warning("🚨 MODE: FAIRNESS OVERRIDE")
    else:
        status_slot.success("🤖 MODE: RL AGENT ACTIVE")

    confidence_metric_slot.metric("AI Confidence", f"{int(confidence * 100)}%")
    confidence_bar_slot.progress(confidence)
    
    current_lane = "Lane 1" if action == 0 else "Lane 2"
    lane_info_slot.info(f"🟢 Signal State: {current_lane} is GREEN")

    # Update counts (Optional - helps to see the data)
    res = st.session_state.controller.processor.process_frame(frame)
    lane_1_stats.metric("Lane 1 Vehicles", res['lane1_count'])
    lane_2_stats.metric("Lane 2 Vehicles", res['lane2_count'])

    # Control playback speed so it doesn't run too fast
    time.sleep(0.01)

cap.release()