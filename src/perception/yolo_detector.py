"""Perception module for vehicle detection using YOLOv8."""

import os
import cv2
from ultralytics import YOLO
from src.utils.config import PERCEPTION_CONFIG


def load_model(model_path: str = PERCEPTION_CONFIG['yolo_model_path']):
    return YOLO(model_path)


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


def run_camera(camera_url: str = PERCEPTION_CONFIG['camera_url']):
    print("Trying to open:", camera_url)
    print("Absolute path:", os.path.abspath(camera_url))

    model = load_model()
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        raise RuntimeError(f'Unable to open camera stream: {camera_url}')

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Unable to read first frame")

    height, width, _ = frame.shape

    # Define ROIs
    lane1_roi = (0, 0, width // 2, height)
    lane2_roi = (width // 2, 0, width, height)

    #  Initialize decision variables 
    current_green = "Lane 1"
    green_time = 10
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = infer_frame(model, frame)

        lane1_count = 0
        lane2_count = 0

        for det in detections:
            x1 = det['xmin']
            y1 = det['ymin']
            x2 = det['xmax']
            y2 = det['ymax']

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Lane classification
            if lane1_roi[0] <= cx <= lane1_roi[2]:
                lane1_count += 1
                color = (255, 0, 0)  # Blue
            else:
                lane2_count += 1
                color = (0, 0, 255)  # Red

            # Draw center
            cv2.circle(frame, (cx, cy), 4, color, -1)

        #  Decision update (every 30 frames)
        if frame_counter % 30 == 0:
            if lane1_count > lane2_count:
                current_green = "Lane 1"
                green_time = max(10, lane1_count * 2)
            else:
                current_green = "Lane 2"
                green_time = max(10, lane2_count * 2)

        frame_counter += 1

        #  Draw ROIs
        cv2.rectangle(frame, (lane1_roi[0], lane1_roi[1]),
                      (lane1_roi[2], lane1_roi[3]), (255, 0, 0), 2)

        cv2.rectangle(frame, (lane2_roi[0], lane2_roi[1]),
                      (lane2_roi[2], lane2_roi[3]), (0, 0, 255), 2)

        #  Display counts
        cv2.putText(frame, f"Lane 1: {lane1_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.putText(frame, f"Lane 2: {lane2_count}", (width - 250, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #  Display decision
        cv2.putText(frame, f"GREEN: {current_green}",
                    (width // 2 - 150, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.putText(frame, f"Time: {green_time}s",
                    (width // 2 - 100, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        #  Highlight active lane
        if current_green == "Lane 1":
            cv2.rectangle(frame,
                          (lane1_roi[0], lane1_roi[1]),
                          (lane1_roi[2], lane1_roi[3]),
                          (0, 255, 0), 4)
        else:
            cv2.rectangle(frame,
                          (lane2_roi[0], lane2_roi[1]),
                          (lane2_roi[2], lane2_roi[3]),
                          (0, 255, 0), 4)

        # Show frame
        cv2.imshow("AI Traffic Controller", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_camera()