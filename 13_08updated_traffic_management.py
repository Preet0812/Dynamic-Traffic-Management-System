from flask import Flask, render_template_string, Response, request, jsonify
import cv2
import threading
import time
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Shared data
lock = threading.Lock()
shared_data = {
    "vehicle_counts": {},
    "frames": [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(4)],
    "cycle_time": 60  # default
}

# YOLO model
model = YOLO("yolov8n.pt")
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
direction_labels = ["north", "south", "east", "west"]

# Camera indices or RTSP URLs
camera_sources = [0, 1, 2, 3]  # Change to your webcams
caps = [cv2.VideoCapture(src) for src in camera_sources]

def process_cameras():
    while True:
        frames = []
        vehicle_counts = []
        for i, cap in enumerate(caps):
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

            cv2.putText(frame, f"Cam {i+1} | {count} Vehicles", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            frames.append(frame)
            vehicle_counts.append(count)

        with lock:
            shared_data["vehicle_counts"] = dict(zip(direction_labels, vehicle_counts))
            shared_data["frames"] = frames
        time.sleep(0.05)

@app.route('/')
def index():
    return render_template_string('''
    <html>
    <head>
        <title>YOLO Traffic Monitoring</title>
        <style>
            body { font-family: Arial; text-align: center; }
            table, th, td { border: 1px solid black; border-collapse: collapse; padding: 5px; }
            input { padding: 5px; }
            button { padding: 5px 10px; }
        </style>
        <script>
            async function fetchData() {
                let res = await fetch("/get_counts");
                let data = await res.json();
                let tableBody = document.getElementById("vehicle_table");
                tableBody.innerHTML = "";
                for (let dir in data) {
                    tableBody.innerHTML += <tr><td>${dir}</td><td>${data[dir]}</td></tr>;
                }
            }
            setInterval(fetchData, 2000); // Update every 2 seconds
        </script>
    </head>
    <body onload="fetchData()">
        <h1>YOLO Traffic Monitoring</h1>
        
        <form action="/update_cycle" method="POST" style="margin-bottom: 20px;">
            <label>Cycle Time (seconds):</label>
            <input type="number" name="cycle_time" value="{{ cycle_time }}">
            <button type="submit">Update</button>
        </form>

        <table align="center">
            <tr><th>Direction</th><th>Vehicle Count</th></tr>
            <tbody id="vehicle_table"></tbody>
        </table>

        <br>
        <div>
            {% for i in range(4) %}
            <img src="/video_feed/{{ i }}" width="320" height="240">
            {% endfor %}
        </div>
    </body>
    </html>
    ''', cycle_time=shared_data["cycle_time"])

@app.route('/get_counts')
def get_counts():
    with lock:
        return jsonify(shared_data["vehicle_counts"])

@app.route('/update_cycle', methods=['POST'])
def update_cycle():
    new_time = request.form.get("cycle_time")
    if new_time:
        with lock:
            shared_data["cycle_time"] = int(new_time)
    return ("<script>window.location.href='/'</script>")

def generate_feed(cam_index):
    while True:
        with lock:
            frame = shared_data["frames"][cam_index]
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed/<int:cam_index>')
def video_feed(cam_index):
    return Response(generate_feed(cam_index),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    t = threading.Thread(target=process_cameras, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False)