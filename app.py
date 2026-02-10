import streamlit as st
import cv2
import numpy as np
import math
import time
from ultralytics import YOLO
from collections import deque
import tempfile

st.set_page_config(page_title="Vehicle Speed Detection", layout="wide")
st.title(" Vehicle Speed Detection")

PIXEL_TO_METER = 0.05
SPEED_LIMIT = st.slider("Speed Limit (km/h)", 20, 140, 60)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # bicycle, car, bike, bus, truck


video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    prev_pos = {}
    prev_time = {}
    speed_hist = {}

    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            imgsz=640,
            conf=0.3,
            verbose=False
        )[0]

        if results.boxes is None:
            continue

        for box in results.boxes:
            if box.id is None:
                continue

            cls = int(box.cls[0])
            if cls not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0])

            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            now = time.time()

            speed = 0
            if track_id in prev_pos:
                px, py = prev_pos[track_id]
                dt = now - prev_time[track_id]
                if dt > 0.03:
                    dist_px = math.hypot(cx - px, cy - py)
                    speed = (dist_px * PIXEL_TO_METER / dt) * 3.6

            prev_pos[track_id] = (cx, cy)
            prev_time[track_id] = now

            speed_hist.setdefault(track_id, deque(maxlen=5)).append(speed)
            smooth_speed = int(np.mean(speed_hist[track_id]))

            color = (0, 255, 0)
            label = f"{smooth_speed} km/h"

            if smooth_speed > SPEED_LIMIT:
                color = (0, 0, 255)
                label = f"OVERSPEED {smooth_speed} km/h"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        frame_placeholder.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True
        )

    cap.release()
    st.success("Video processed successfully")
