"""Perception + Decision module (dashboard-ready version)"""

import os
import cv2
from ultralytics import YOLO
from src.utils.config import PERCEPTION_CONFIG


# =========================
# MODEL
# =========================
def load_model(model_path: str = PERCEPTION_CONFIG['yolo_model_path']):
    return YOLO(model_path)


# =========================
# DETECTION
# =========================
def infer_frame(model, frame):
    results = model(frame)
    detections = []

    for result in results:
        for box in result.boxes:
            detections.append({
                'xmin': int(box.xyxy[0][0]),
                'ymin': int(box.xyxy[0][1]),
                'xmax': int(box.xyxy[0][2]),
                'ymax': int(box.xyxy[0][3]),
                'confidence': float(box.conf[0]),
                'class_id': int(box.cls[0]),
            })

    return detections


# =========================
# FRAME PROCESSOR (CORE)
# =========================
class TrafficProcessor:
    def __init__(self, model):
        self.model = model

        self.initialized = False
        self.lane1_roi = None
        self.lane2_roi = None

        self.current_green = "Lane 1"
        self.green_time = 10
        self.frame_counter = 0

    def process_frame(self, frame):
        # Initialize ROIs once
        if not self.initialized:
            height, width, _ = frame.shape
            self.lane1_roi = (0, 0, width // 2, height)
            self.lane2_roi = (width // 2, 0, width, height)
            self.initialized = True

        detections = infer_frame(self.model, frame)

        lane1_count = 0
        lane2_count = 0

        for det in detections:
            x1, y1 = det['xmin'], det['ymin']
            x2, y2 = det['xmax'], det['ymax']

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Lane assignment
            if self.lane1_roi[0] <= cx <= self.lane1_roi[2]:
                lane1_count += 1
                color = (255, 0, 0)
            else:
                lane2_count += 1
                color = (0, 0, 255)

            cv2.circle(frame, (cx, cy), 4, color, -1)

        # =========================
        # DECISION LOGIC
        # =========================
        if self.frame_counter % 30 == 0:
            if lane1_count > lane2_count:
                self.current_green = "Lane 1"
                self.green_time = max(10, lane1_count * 2)
            else:
                self.current_green = "Lane 2"
                self.green_time = max(10, lane2_count * 2)

        self.frame_counter += 1

        # =========================
        # VISUAL OVERLAY
        # =========================
        # Draw ROIs
        cv2.rectangle(frame, (self.lane1_roi[0], self.lane1_roi[1]),
                      (self.lane1_roi[2], self.lane1_roi[3]), (255, 0, 0), 2)

        cv2.rectangle(frame, (self.lane2_roi[0], self.lane2_roi[1]),
                      (self.lane2_roi[2], self.lane2_roi[3]), (0, 0, 255), 2)

        height, width, _ = frame.shape

        # Counts
        cv2.putText(frame, f"Lane 1: {lane1_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.putText(frame, f"Lane 2: {lane2_count}", (width - 250, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Decision
        cv2.putText(frame, f"GREEN: {self.current_green}",
                    (width // 2 - 150, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.putText(frame, f"Time: {self.green_time}s",
                    (width // 2 - 100, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Highlight active lane
        if self.current_green == "Lane 1":
            cv2.rectangle(frame,
                          (self.lane1_roi[0], self.lane1_roi[1]),
                          (self.lane1_roi[2], self.lane1_roi[3]),
                          (0, 255, 0), 4)
        else:
            cv2.rectangle(frame,
                          (self.lane2_roi[0], self.lane2_roi[1]),
                          (self.lane2_roi[2], self.lane2_roi[3]),
                          (0, 255, 0), 4)

        # =========================
        # RETURN DATA
        # =========================
        return {
            "frame": frame,
            "lane1_count": lane1_count,
            "lane2_count": lane2_count,
            "current_green": self.current_green,
            "green_time": self.green_time
        }


# =========================
# RUN (TEST MODE)
# =========================
def run_camera(camera_url: str = PERCEPTION_CONFIG['camera_url']):
    print("Trying to open:", camera_url)
    print("Absolute path:", os.path.abspath(camera_url))

    model = load_model()
    processor = TrafficProcessor(model)

    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        raise RuntimeError(f'Unable to open camera stream: {camera_url}')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = processor.process_frame(frame)

        # TEMP DISPLAY (for testing only)
        cv2.imshow("AI Traffic Controller", result["frame"])

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_camera()