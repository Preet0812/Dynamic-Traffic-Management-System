import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

# -------------------
# CONFIGURATION
# -------------------
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
direction_labels = ["north", "south", "east", "west"]

# Camera sources — replace with IP/RTSP URLs for Streamlit Cloud
# For local test, use webcam indices [0, 1, 2, 3]
camera_sources = [0, 1, 2, 3]

# Load YOLO model
model = YOLO("yolov8n.pt")

# -------------------
# STREAMLIT UI
# -------------------
st.set_page_config(page_title="YOLO Traffic Monitoring", layout="wide")
st.title("🚦 YOLO Traffic Monitoring")

# Cycle time control
cycle_time = st.number_input("Cycle Time (seconds):", min_value=5, max_value=300, value=60, step=5)

# Vehicle counts table
vehicle_counts = {}

cols = st.columns(4)  # For displaying camera feeds

# Process each camera
for i, src in enumerate(camera_sources):
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        count = 0
    else:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            count = 0
        else:
            frame = cv2.resize(frame, (320, 240))
            results = model(frame, verbose=False)
            detections = results[0].boxes.cls.cpu().numpy()
            count = sum(1 for cls_id in detections if int(cls_id) in vehicle_classes)
            frame = results[0].plot()

    vehicle_counts[direction_labels[i]] = count
    cols[i].image(frame, channels="BGR", caption=f"Cam {i+1} | {count} Vehicles")

    cap.release()

# Show vehicle counts in a table
st.subheader("📊 Vehicle Counts")
st.table(vehicle_counts)

st.write(f"Current cycle time: *{cycle_time} seconds*")

st.caption("Powered by YOLOv8 + Streamlit")
