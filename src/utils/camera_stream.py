"""Camera streaming utilities for CCTV integration."""

import cv2
from src.utils.config import PERCEPTION_CONFIG


def stream_camera(camera_url: str = PERCEPTION_CONFIG['camera_url']):
    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open stream: {camera_url}')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame

    cap.release()
