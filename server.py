from flask import Flask, Response, jsonify, render_template_string, request
import cv2
import numpy as np
import requests
import threading
import time
import json
import csv
import io
from queue import Queue, Empty
from collections import deque
from ultralytics import YOLO  # YOLOv8 object detection library by Ultralytics
from datetime import datetime

# Flask web framework powers the HTTP server and all API endpoints
app = Flask(__name__)

# =========================
# MODEL SETUP
# Uses YOLOv8n (nano) — the lightest YOLOv8 variant for real-time inference
# =========================
# Load the YOLOv8 nano model; downloads weights automatically on first run
model = YOLO("yolov8n.pt")
# Resize input to 320x320 for faster YOLOv8 inference with minimal accuracy loss
model.overrides["imgsz"] = 320

# Warm up the YOLOv8 model with a dummy frame so the first real inference isn't slow
dummy = np.zeros((320, 320, 3), dtype=np.uint8)
model(dummy, verbose=False)
print("[EYECORE] Neural model initialized.")

# MJPEG stream URL — YOLOv8 will run detection on frames pulled from this camera feed
STREAM_URL = "http://10.181.135.37/1280x1024.mjpeg"

# =========================
# EXTENDED CATEGORY MAP
# Maps raw YOLOv8 COCO class names to broader display categories
# YOLOv8 is trained on the COCO dataset (80 classes); we group them here
# =========================
CATEGORY_MAP = {
    "person":       "person",
    "car":          "vehicle",
    "motorcycle":   "vehicle",
    "bus":          "vehicle",
    "truck":        "vehicle",
    "bicycle":      "bicycle",
    "dog":          "animal",
    "cat":          "animal",
    "bird":         "animal",
    "horse":        "animal",
    "backpack":     "bag",
    "handbag":      "bag",
    "suitcase":     "bag",
    "cell phone":   "phone",
    "knife":        "weapon",
    "scissors":     "weapon",
    "umbrella":     "umbrella",
    "fire hydrant": "infra",
    "stop sign":    "infra",
    "chair":        "furniture",
    "bench":        "furniture",
    "bottle":       "object",
    "cup":          "object",
    "laptop":       "object",
    "tv":           "object",
}

# All display categories shown in the Flask UI dashboard
ALL_CATEGORIES = [
    "person", "vehicle", "animal", "bicycle",
    "bag", "phone", "weapon", "umbrella",
    "infra", "furniture", "object", "others"
]

# Alert thresholds: Flask will send an alert event when YOLOv8 detects
# this many objects of a given category in a single frame
ALERT_THRESHOLDS = {
    "person":  5,
    "weapon":  1,  # Any weapon detection triggers an immediate alert
    "vehicle": 4,
}

# =========================
# SHARED STATE
# Thread-safe data shared between the YOLOv8 worker thread and Flask routes
# =========================
data_lock = threading.Lock()   # Protects shared_data dict (detection counts)
frame_lock = threading.Lock()  # Protects the latest JPEG frame bytes

# Live detection counts updated by the YOLOv8 worker and read by Flask /counts
shared_data = {cat: 0 for cat in ALL_CATEGORIES}
shared_data["fps"] = 0
shared_data["total_detections"] = 0

latest_frame = None      # Raw MJPEG frame (no annotations)
latest_annotated = None  # YOLOv8-annotated frame with bounding boxes drawn

# Queue passes raw camera frames from the stream thread to the YOLOv8 worker
frame_queue = Queue(maxsize=2)

# Rolling history of detection snapshots served by the Flask /history endpoint
MAX_HISTORY = 500
detection_history = deque(maxlen=MAX_HISTORY)
history_lock = threading.Lock()

# Per-frame object list (label, category, confidence, bbox) from YOLOv8
confidence_list = []
conf_lock = threading.Lock()

# Cooldown tracker so the same alert category doesn't fire repeatedly
alert_state = {}
ALERT_COOLDOWN = 10  # seconds between repeat alerts for the same category

frame_count = 0
fps_start = time.time()

last_inference_time = 0
INFERENCE_INTERVAL = 0.2  # Run YOLOv8 inference at most 5 times per second


# =========================
# YOLO WORKER
# Background thread that runs YOLOv8 inference on frames from the queue.
# Results are written to shared state and read by Flask API routes.
# =========================
def yolo_worker():
    global latest_annotated, last_inference_time

    while True:
        # Block until a frame is available from the camera stream thread
        try:
            frame = frame_queue.get(timeout=1)
        except Empty:
            continue

        # Throttle YOLOv8 inference to INFERENCE_INTERVAL seconds per frame
        now = time.time()
        if now - last_inference_time < INFERENCE_INTERVAL:
            continue
        last_inference_time = now

        # Run YOLOv8 detection — conf=0.4 filters out low-confidence predictions
        results = model(frame, conf=0.4, verbose=False)

        # Tally detections by category using the COCO-to-category map
        counts = {cat: 0 for cat in ALL_CATEGORIES}
        obj_list = []
        total = 0

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])          # YOLOv8 confidence score (0–1)
                cls = int(box.cls[0])               # YOLOv8 COCO class index
                name = model.names[cls]             # Human-readable COCO class name
                category = CATEGORY_MAP.get(name, "others")  # Map to display category
                counts[category] += 1
                total += 1
                xyxy = box.xyxy[0].tolist()  # Bounding box in [x1,y1,x2,y2] format
                obj_list.append({
                    "label": name,
                    "category": category,
                    "confidence": round(conf, 3),
                    "bbox": [round(x, 1) for x in xyxy]
                })

        # Check alert thresholds and fire events if exceeded (with cooldown)
        ts = datetime.now().strftime("%H:%M:%S")
        alerts_fired = []
        for cat, threshold in ALERT_THRESHOLDS.items():
            if counts.get(cat, 0) >= threshold:
                last_alert = alert_state.get(cat, 0)
                if now - last_alert >= ALERT_COOLDOWN:
                    alert_state[cat] = now
                    alerts_fired.append({
                        "time": ts,
                        "type": "ALERT",
                        "timestamp": now,
                        "category": cat,
                        "count": counts[cat],
                        "message": f"{cat.upper()} threshold exceeded: {counts[cat]}"
                    })

        # Update shared counts read by the Flask /counts endpoint
        with data_lock:
            for cat in ALL_CATEGORIES:
                shared_data[cat] = counts[cat]
            shared_data["total_detections"] = total

        # Update per-object confidence list read by Flask /confidences
        with conf_lock:
            confidence_list.clear()
            confidence_list.extend(obj_list)

        # Append detection snapshot and any alerts to the rolling history
        snapshot = {
            "time": ts,
            "timestamp": now,
            "counts": dict(counts),
            "total": total,
            "objects": obj_list
        }
        with history_lock:
            detection_history.append(snapshot)
            for alert in alerts_fired:
                detection_history.append(alert)

        # Use YOLOv8's built-in plot() to draw bounding boxes on the frame,
        # then JPEG-encode it for streaming via the Flask /video endpoint
        annotated = results[0].plot()
        _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with frame_lock:
            latest_annotated = buf.tobytes()


# =========================
# STREAM THREAD
# Pulls MJPEG frames from the IP camera and feeds them to the YOLOv8 worker.
# Runs as a daemon thread alongside the Flask server.
# =========================
def detection_loop():
    global latest_frame, frame_count, fps_start

    while True:
        try:
            print("[EYECORE] Connecting to retinal feed...")
            # Use requests to consume the MJPEG stream as a chunked HTTP response
            stream = requests.get(STREAM_URL, stream=True, timeout=10)
            buf = b""

            for chunk in stream.iter_content(chunk_size=4096):
                buf += chunk
                # MJPEG frames are delimited by JPEG SOI (0xFFD8) and EOI (0xFFD9) markers
                a = buf.find(b'\xff\xd8')
                b = buf.find(b'\xff\xd9')

                if a != -1 and b != -1:
                    jpg = buf[a:b+2]
                    buf = buf[b+2:]

                    # Decode JPEG bytes into an OpenCV BGR frame
                    frame = cv2.imdecode(
                        np.frombuffer(jpg, dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    if frame is None:
                        continue

                    # Resize to 640x480 before queuing for YOLOv8 inference
                    resized = cv2.resize(frame, (640, 480))
                    if not frame_queue.full():
                        frame_queue.put(resized)

                    # Also store the raw (unannotated) frame as fallback for /video
                    _, raw_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    with frame_lock:
                        latest_frame = raw_buf.tobytes()

                    # Track frames per second and expose via shared_data for Flask /counts
                    frame_count += 1
                    if time.time() - fps_start >= 1:
                        with data_lock:
                            shared_data["fps"] = frame_count
                        frame_count = 0
                        fps_start = time.time()

        except Exception as e:
            print(f"[EYECORE ERROR] {e}")
            time.sleep(3)  # Back off before reconnecting to the camera stream


# =========================
# VIDEO GENERATOR
# Generator function used by the Flask /video route to stream MJPEG over HTTP.
# Yields the latest YOLOv8-annotated frame (or raw frame if inference is pending).
# =========================
def gen_frames():
    last = None
    while True:
        # Prefer the YOLOv8-annotated frame; fall back to raw if not yet available
        with frame_lock:
            frame = latest_annotated or latest_frame

        if frame is None or frame is last:
            time.sleep(0.033)  # ~30 fps cap; avoid busy-waiting
            continue

        last = frame
        # Flask streams this as a multipart/x-mixed-replace MJPEG response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# =========================
# HTML — EYECORE UI
# Single-page dashboard served by Flask at the root route.
# Polls Flask JSON endpoints (/counts, /confidences, /history) every 800ms
# and renders live YOLOv8 stats using Chart.js.
# =========================
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EYECORE — Neural Vision System</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap" rel="stylesheet">
<!-- Chart.js renders the live YOLOv8 detection count graph in the dashboard -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>

<style>
:root {
  --void:      #03050a;
  --deep:      #060c14;
  --panel:     #080f1a;
  --panel2:    #0a1220;
  --iris:      #0af0ff;
  --iris2:     #0066ff;
  --pupil:     #00ffcc;
  --retina:    #0033aa;
  --alert:     #ff2d55;
  --warn:      #ff9f0a;
  --neural:    #7b5ea7;
  --text:      #c8e0f4;
  --dim:       #2a4a6a;
  --dim2:      #1a2a3a;
  --glow:      rgba(10,240,255,0.15);
  --glow2:     rgba(10,240,255,0.05);
  --font-eye:  'Orbitron', monospace;
  --font-body: 'Exo 2', sans-serif;
}

* { margin:0; padding:0; box-sizing:border-box; }

body {
  background: var(--void);
  color: var(--text);
  font-family: var(--font-body);
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  cursor: crosshair;
}

/* ── SCANLINE OVERLAY ── */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(10,240,255,0.012) 2px,
    rgba(10,240,255,0.012) 4px
  );
  pointer-events: none;
  z-index: 9999;
}

/* ── VIGNETTE ── */
body::after {
  content: '';
  position: fixed;
  inset: 0;
  background: radial-gradient(ellipse at center, transparent 55%, rgba(3,5,10,0.7) 100%);
  pointer-events: none;
  z-index: 9998;
}

/* ── HEADER ── */
header {
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  border-bottom: 1px solid rgba(10,240,255,0.12);
  background: rgba(6,12,20,0.98);
  flex-shrink: 0;
  position: relative;
  z-index: 100;
}

header::after {
  content: '';
  position: absolute;
  bottom: -1px; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--iris), transparent);
  opacity: 0.5;
}

/* ── EYE LOGO ── */
.eye-logo {
  display: flex;
  align-items: center;
  gap: 14px;
}

.eye-svg {
  width: 40px;
  height: 40px;
  flex-shrink: 0;
}

.logo-text h1 {
  font-family: var(--font-eye);
  font-size: 18px;
  font-weight: 900;
  color: var(--iris);
  letter-spacing: 6px;
  text-shadow: 0 0 20px rgba(10,240,255,0.6), 0 0 40px rgba(10,240,255,0.2);
  line-height: 1;
}

.logo-text span {
  font-size: 9px;
  color: var(--dim);
  letter-spacing: 3px;
  font-family: var(--font-eye);
  font-weight: 400;
}

/* ── HUD STATS ── */
.hud-stats {
  display: flex;
  gap: 28px;
}

.hud-stat {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  position: relative;
}

.hud-stat::after {
  content: '';
  position: absolute;
  right: -14px;
  top: 50%;
  transform: translateY(-50%);
  width: 1px;
  height: 24px;
  background: var(--dim2);
}

.hud-stat:last-child::after { display: none; }

.hud-stat .lbl {
  font-family: var(--font-eye);
  font-size: 7px;
  color: var(--dim);
  letter-spacing: 2px;
}

.hud-stat .val {
  font-family: var(--font-eye);
  font-size: 13px;
  color: var(--iris);
  text-shadow: 0 0 10px rgba(10,240,255,0.4);
}

.hud-stat .val.online { color: var(--pupil); text-shadow: 0 0 10px rgba(0,255,204,0.4); }
.hud-stat .val.offline { color: var(--alert); text-shadow: 0 0 10px rgba(255,45,85,0.4); }
.hud-stat .val.warn { color: var(--warn); }

/* ── HEADER ACTIONS ── */
.hdr-actions { display: flex; gap: 8px; align-items: center; }

.btn {
  font-family: var(--font-eye);
  font-size: 8px;
  padding: 6px 14px;
  border: 1px solid rgba(10,240,255,0.25);
  background: rgba(10,240,255,0.04);
  color: var(--iris);
  cursor: pointer;
  letter-spacing: 2px;
  transition: all 0.25s;
  border-radius: 2px;
  position: relative;
  overflow: hidden;
}

.btn::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, transparent 40%, rgba(10,240,255,0.08) 100%);
  opacity: 0;
  transition: opacity 0.2s;
}

.btn:hover { border-color: var(--iris); box-shadow: 0 0 12px rgba(10,240,255,0.2); }
.btn:hover::before { opacity: 1; }
.btn.danger { border-color: rgba(255,45,85,0.3); color: var(--alert); background: rgba(255,45,85,0.04); }
.btn.danger:hover { border-color: var(--alert); box-shadow: 0 0 12px rgba(255,45,85,0.2); }

/* ── ALERT BANNER ── */
#alert-banner {
  display: none;
  padding: 5px 20px;
  background: linear-gradient(90deg, rgba(255,45,85,0.2), rgba(255,45,85,0.05), rgba(255,45,85,0.2));
  border-bottom: 1px solid rgba(255,45,85,0.5);
  font-family: var(--font-eye);
  font-size: 10px;
  color: var(--alert);
  letter-spacing: 3px;
  text-align: center;
  animation: alert-pulse 0.8s infinite;
  flex-shrink: 0;
  z-index: 100;
  position: relative;
}
@keyframes alert-pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

/* ── EXPORT MENU ── */
#export-menu {
  position: fixed;
  top: 57px; right: 0;
  background: var(--panel2);
  border: 1px solid rgba(10,240,255,0.2);
  border-top: none;
  border-radius: 0 0 0 6px;
  padding: 8px;
  display: none;
  flex-direction: column;
  gap: 6px;
  z-index: 200;
  box-shadow: -4px 4px 24px rgba(10,240,255,0.08);
}
#export-menu.open { display: flex; }

/* ── LAYOUT ── */
.main {
  flex: 1;
  display: grid;
  grid-template-columns: 1fr 320px;
  gap: 8px;
  padding: 8px;
  overflow: hidden;
}

.left-col {
  display: grid;
  grid-template-rows: 1fr 120px;
  gap: 8px;
  overflow: hidden;
}

/* ── PANEL ── */
.panel {
  background: var(--panel);
  border: 1px solid rgba(10,240,255,0.08);
  border-radius: 6px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  position: relative;
}

.panel::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(10,240,255,0.3), transparent);
}

.panel-hdr {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 12px;
  border-bottom: 1px solid rgba(10,240,255,0.06);
  flex-shrink: 0;
}

.panel-title {
  font-family: var(--font-eye);
  font-size: 8px;
  color: var(--iris);
  letter-spacing: 3px;
  opacity: 0.8;
}

.panel-tag {
  font-family: var(--font-eye);
  font-size: 8px;
  padding: 2px 8px;
  border-radius: 2px;
}

/* ── VIDEO ── */
.video-wrap {
  flex: 1;
  background: #000;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
}

#feed {
  width: 100%; height: 100%;
  object-fit: contain;
}

/* Eye-shaped frame overlay */
.eye-frame {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

/* Animated iris ring */
.iris-ring {
  position: absolute;
  inset: 50%;
  transform: translate(-50%, -50%);
  width: min(70%, 70%);
  aspect-ratio: 2/1;
  border-radius: 50%;
  border: 1px solid rgba(10,240,255,0.2);
  box-shadow: 0 0 0 1px rgba(10,240,255,0.05), inset 0 0 40px rgba(10,240,255,0.03);
  animation: iris-breathe 4s ease-in-out infinite;
}
@keyframes iris-breathe {
  0%,100%{ opacity:0.4; transform:translate(-50%,-50%) scale(1); }
  50%    { opacity:0.8; transform:translate(-50%,-50%) scale(1.01); }
}

/* Corner markers */
.eye-frame::before,
.eye-frame::after {
  content: '';
  position: absolute;
  width: 20px; height: 20px;
  border-color: rgba(10,240,255,0.5);
  border-style: solid;
}
.eye-frame::before { top:6px; left:6px; border-width:1px 0 0 1px; }
.eye-frame::after  { bottom:6px; right:6px; border-width:0 1px 1px 0; }

/* Scan line sweep */
.scan-sweep {
  position: absolute;
  left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, rgba(10,240,255,0.4), transparent);
  animation: sweep 3s linear infinite;
  pointer-events: none;
}
@keyframes sweep {
  0%   { top: 0%; opacity: 0; }
  5%   { opacity: 1; }
  95%  { opacity: 1; }
  100% { top: 100%; opacity: 0; }
}

/* Live badges */
.video-badges {
  position: absolute;
  top: 10px;
  right: 10px;
  display: flex;
  flex-direction: column;
  gap: 3px;
  align-items: flex-end;
}

.vbadge {
  font-family: var(--font-eye);
  font-size: 9px;
  padding: 3px 8px;
  border-radius: 2px;
  border: 1px solid;
  backdrop-filter: blur(8px);
  display: flex;
  align-items: center;
  gap: 8px;
  transition: all 0.4s;
  letter-spacing: 1px;
}
.vbadge.zero { opacity: 0.15; }
.vbadge .vb-count { font-size: 13px; font-weight: 700; line-height: 1; }

.vbadge.person   { color:#00ffcc; border-color:rgba(0,255,204,0.35); background:rgba(0,255,204,0.06); }
.vbadge.vehicle  { color:#0af0ff; border-color:rgba(10,240,255,0.35); background:rgba(10,240,255,0.06); }
.vbadge.animal   { color:#ff9f0a; border-color:rgba(255,159,10,0.35); background:rgba(255,159,10,0.06); }
.vbadge.weapon   { color:#ff2d55; border-color:rgba(255,45,85,0.45); background:rgba(255,45,85,0.1); box-shadow:0 0 8px rgba(255,45,85,0.2); }
.vbadge.bag      { color:#bf5af2; border-color:rgba(191,90,242,0.35); background:rgba(191,90,242,0.06); }
.vbadge.phone    { color:#ffd60a; border-color:rgba(255,214,10,0.35); background:rgba(255,214,10,0.06); }

/* ── COUNT TILES ── */
.tiles-bar {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 6px;
  padding: 8px;
  background: var(--panel2);
  border-top: 1px solid rgba(10,240,255,0.06);
  border-radius: 6px;
}

.tile {
  background: var(--panel);
  border: 1px solid rgba(10,240,255,0.06);
  border-radius: 4px;
  padding: 6px 4px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1px;
  transition: border-color 0.3s, box-shadow 0.3s;
  cursor: default;
  position: relative;
  overflow: hidden;
}

.tile::before {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 2px;
  opacity: 0.4;
  border-radius: 0 0 4px 4px;
}

.tile.alert {
  border-color: rgba(255,45,85,0.4);
  box-shadow: 0 0 12px rgba(255,45,85,0.15), inset 0 0 12px rgba(255,45,85,0.05);
  animation: tile-alert 1s ease-in-out infinite;
}
@keyframes tile-alert { 0%,100%{box-shadow:0 0 12px rgba(255,45,85,0.15)} 50%{box-shadow:0 0 20px rgba(255,45,85,0.3)} }

.tile-name {
  font-family: var(--font-eye);
  font-size: 7px;
  color: var(--dim);
  letter-spacing: 1px;
}

.tile-val {
  font-family: var(--font-eye);
  font-size: 20px;
  font-weight: 700;
  line-height: 1;
}

.tile-delta {
  font-family: var(--font-eye);
  font-size: 7px;
  height: 9px;
  color: var(--dim);
}

/* ── RIGHT COLUMN ── */
.right-col {
  display: flex;
  flex-direction: column;
  gap: 8px;
  overflow: hidden;
}

/* ── CHART ── */
.chart-panel {
  flex: 0 0 210px;
}

.chart-inner {
  flex: 1;
  padding: 8px;
  height: 170px;
  position: relative;
}

/* ── CONFIDENCE ── */
.conf-panel {
  flex: 1;
  min-height: 0;
}

.conf-body {
  flex: 1;
  overflow-y: auto;
  padding: 4px 10px;
}
.conf-body::-webkit-scrollbar { width: 2px; }
.conf-body::-webkit-scrollbar-thumb { background: rgba(10,240,255,0.15); }

.conf-row {
  display: grid;
  grid-template-columns: 1fr 70px 48px;
  gap: 6px;
  padding: 5px 0;
  border-bottom: 1px solid rgba(10,240,255,0.04);
  align-items: center;
}

.conf-label {
  font-family: var(--font-eye);
  font-size: 9px;
  color: var(--text);
  letter-spacing: 0.5px;
}
.conf-cat {
  font-family: var(--font-eye);
  font-size: 7px;
  color: var(--dim);
  letter-spacing: 1px;
  margin-top: 2px;
}

.conf-bar-track {
  height: 10px;
  background: rgba(255,255,255,0.03);
  border-radius: 1px;
  overflow: hidden;
  position: relative;
}
.conf-bar-fill {
  position: absolute;
  left: 0; top: 0; bottom: 0;
  border-radius: 1px;
  transition: width 0.4s ease;
}

.conf-pct {
  font-family: var(--font-eye);
  font-size: 10px;
  text-align: right;
}

/* ── HISTORY ── */
.hist-panel {
  flex: 0 0 170px;
}

.hist-body {
  flex: 1;
  overflow-y: auto;
  padding: 4px 10px;
}
.hist-body::-webkit-scrollbar { width: 2px; }
.hist-body::-webkit-scrollbar-thumb { background: rgba(10,240,255,0.15); }

.h-row {
  display: flex;
  gap: 6px;
  padding: 4px 0;
  border-bottom: 1px solid rgba(10,240,255,0.04);
  align-items: flex-start;
}

.h-time {
  font-family: var(--font-eye);
  font-size: 8px;
  color: var(--dim);
  flex-shrink: 0;
  width: 50px;
  padding-top: 2px;
}

.h-badge {
  font-family: var(--font-eye);
  font-size: 7px;
  padding: 1px 5px;
  border-radius: 2px;
  flex-shrink: 0;
  margin-top: 1px;
  letter-spacing: 1px;
}
.h-badge.DETECTION { background: rgba(0,255,204,0.08); color: var(--pupil); border: 1px solid rgba(0,255,204,0.2); }
.h-badge.ALERT     { background: rgba(255,45,85,0.12); color: var(--alert); border: 1px solid rgba(255,45,85,0.3); animation: alert-pulse 1.5s infinite; }

.h-msg {
  font-size: 9px;
  color: rgba(200,224,244,0.7);
  flex: 1;
  line-height: 1.4;
  font-family: var(--font-body);
}

@media (max-width: 900px) {
  .main { grid-template-columns: 1fr; }
  .right-col { display: none; }
  .tiles-bar { grid-template-columns: repeat(4, 1fr); }
}
</style>
</head>
<body>

<!-- ALERT BANNER -->
<div id="alert-banner">◈ NEURAL ALERT — ANOMALY DETECTED ◈</div>

<!-- HEADER -->
<header>
  <div class="eye-logo">
    <!-- Animated SVG Eye -->
    <svg class="eye-svg" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="irisGrad" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#0af0ff" stop-opacity="0.9"/>
          <stop offset="60%" stop-color="#0066ff" stop-opacity="0.7"/>
          <stop offset="100%" stop-color="#000033" stop-opacity="0.9"/>
        </radialGradient>
        <filter id="glow">
          <feGaussianBlur stdDeviation="1.5" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <!-- Eye outline -->
      <path d="M4 20 Q20 6 36 20 Q20 34 4 20Z" fill="rgba(6,12,20,0.9)" stroke="#0af0ff" stroke-width="0.8" stroke-opacity="0.6"/>
      <!-- Iris -->
      <circle cx="20" cy="20" r="8" fill="url(#irisGrad)" filter="url(#glow)">
        <animate attributeName="r" values="8;8.5;8" dur="4s" repeatCount="indefinite"/>
      </circle>
      <!-- Pupil -->
      <circle cx="20" cy="20" r="3.5" fill="#000814"/>
      <circle cx="20" cy="20" r="3.5" fill="none" stroke="#0af0ff" stroke-width="0.5" stroke-opacity="0.8"/>
      <!-- Iris lines -->
      <line x1="12" y1="20" x2="28" y2="20" stroke="#0af0ff" stroke-width="0.3" stroke-opacity="0.3"/>
      <line x1="20" y1="12" x2="20" y2="28" stroke="#0af0ff" stroke-width="0.3" stroke-opacity="0.3"/>
      <!-- Glint -->
      <circle cx="17" cy="17.5" r="1.2" fill="white" opacity="0.5"/>
    </svg>

    <div class="logo-text">
      <h1>EYECORE</h1>
      <span>NEURAL VISION SYSTEM v2.0</span>
    </div>
  </div>

  <div class="hud-stats">
    <div class="hud-stat">
      <span class="lbl">UPTIME</span>
      <span class="val" id="uptime">00:00:00</span>
    </div>
    <div class="hud-stat">
      <span class="lbl">RETINAL FPS</span>
      <span class="val warn" id="fps">—</span>
    </div>
    <div class="hud-stat">
      <span class="lbl">OBJECTS</span>
      <span class="val" id="total">0</span>
    </div>
    <div class="hud-stat">
      <span class="lbl">IRIS</span>
      <span class="val online" id="status">CALIBRATING</span>
    </div>
  </div>

  <div class="hdr-actions">
    <button class="btn" onclick="toggleExport()">⬇ EXTRACT</button>
    <button class="btn danger" id="alert-mute" onclick="toggleMute()">◉ ALERTS ON</button>
  </div>
</header>

<!-- EXPORT MENU -->
<div id="export-menu">
  <!-- Exports the YOLOv8 detection history logged by Flask to CSV or JSON -->
  <button class="btn" onclick="exportCSV()">Export CSV</button>
  <button class="btn" onclick="exportJSON()">Export JSON</button>
</div>

<!-- MAIN -->
<div class="main">

  <!-- LEFT -->
  <div class="left-col">

    <!-- VIDEO PANEL — streams YOLOv8-annotated MJPEG from Flask /video -->
    <div class="panel">
      <div class="panel-hdr">
        <span class="panel-title">◉ RETINAL FEED — NODE 01</span>
        <span class="panel-tag" style="background:rgba(255,45,85,0.1);color:#ff2d55;border:1px solid rgba(255,45,85,0.3);">● LIVE</span>
      </div>
      <div class="video-wrap">
        <img id="feed" src="/video" alt="feed">
        <div class="eye-frame">
          <div class="iris-ring"></div>
          <div class="scan-sweep"></div>
        </div>
        <div class="video-badges" id="video-badges"></div>
      </div>
    </div>

    <!-- TILES — per-category counts from YOLOv8 via Flask /counts -->
    <div class="tiles-bar" id="tiles-bar"></div>

  </div>

  <!-- RIGHT -->
  <div class="right-col">

    <!-- CHART — plots YOLOv8 detection counts over time using Chart.js -->
    <div class="panel chart-panel">
      <div class="panel-hdr">
        <span class="panel-title">NEURAL ACTIVITY</span>
        <span class="panel-tag" style="background:rgba(10,240,255,0.05);color:var(--iris);border:1px solid rgba(10,240,255,0.15);">LIVE GRAPH</span>
      </div>
      <div class="chart-inner">
        <canvas id="chart"></canvas>
      </div>
    </div>

    <!-- CONFIDENCE TABLE — shows per-object YOLOv8 scores from Flask /confidences -->
    <div class="panel conf-panel">
      <div class="panel-hdr">
        <span class="panel-title">OBJECT RECOGNITION</span>
        <span id="obj-count" class="panel-tag" style="background:rgba(10,240,255,0.05);color:var(--iris);border:1px solid rgba(10,240,255,0.15);">0 TARGETS</span>
      </div>
      <div class="conf-body" id="conf-body">
        <div style="color:var(--dim);font-family:var(--font-eye);font-size:8px;padding:10px;letter-spacing:2px;">AWAITING VISUAL INPUT...</div>
      </div>
    </div>

    <!-- HISTORY — scrolling log of YOLOv8 detection events and alerts from Flask /history -->
    <div class="panel hist-panel">
      <div class="panel-hdr">
        <span class="panel-title">EVENT MEMBRANE</span>
        <button class="btn" onclick="clearHistory()" style="font-size:7px;padding:2px 8px;">FLUSH</button>
      </div>
      <div class="hist-body" id="hist-body">
        <div style="color:var(--dim);font-family:var(--font-eye);font-size:8px;padding:10px;letter-spacing:2px;">NO EVENTS RECORDED...</div>
      </div>
    </div>

  </div>

</div>

<script>
// ── CONFIG ──
// Category and color config mirrors the YOLOv8 CATEGORY_MAP defined in Python
const CATS = ["person","vehicle","animal","bicycle","bag","phone","weapon","umbrella","infra","furniture","object","others"];
const COLORS = {
  person:"#00ffcc",  vehicle:"#0af0ff",  animal:"#ff9f0a",  bicycle:"#ff66cc",
  bag:"#bf5af2",     phone:"#ffd60a",    weapon:"#ff2d55",  umbrella:"#64d2ff",
  infra:"#98989e",   furniture:"#6e6e73", object:"#d1b97a", others:"#3a5a7a"
};
// Must match ALERT_THRESHOLDS in Python so the UI reflects the same logic
const ALERT_CATS = { person:5, weapon:1, vehicle:4 };
// Subset of categories plotted on the Chart.js live graph
const CHART_CATS = ['person','vehicle','weapon','animal'];

let muted = false;
let uptime = 0;
let lastCounts = {};
let localHistory = [];   // Client-side cache of YOLOv8 detection history
let lastHistTime = 0;    // Timestamp cursor for incremental Flask /history fetches

// ── AUDIO ──
// Plays an alert tone when YOLOv8 detects an above-threshold category
let audioCtx;
function beep(freq, dur, type='sine') {
  if (muted) return;
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext||window.webkitAudioContext)();
    const o = audioCtx.createOscillator();
    const g = audioCtx.createGain();
    o.connect(g); g.connect(audioCtx.destination);
    o.type = type; o.frequency.value = freq;
    g.gain.setValueAtTime(0.25, audioCtx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + dur);
    o.start(); o.stop(audioCtx.currentTime + dur);
  } catch(e){}
}
function alertBeep() {
  beep(1400,0.08,'sawtooth');
  setTimeout(()=>beep(1000,0.08,'sawtooth'),150);
  setTimeout(()=>beep(1400,0.12,'sawtooth'),300);
}

function toggleMute() {
  muted = !muted;
  document.getElementById('alert-mute').textContent = muted ? '○ ALERTS OFF' : '◉ ALERTS ON';
}

// ── EXPORT ──
// Exports the locally cached YOLOv8 detection history to CSV or JSON
function toggleExport() {
  document.getElementById('export-menu').classList.toggle('open');
}
document.addEventListener('click', e => {
  if (!e.target.closest('#export-menu') && !e.target.closest('.btn'))
    document.getElementById('export-menu').classList.remove('open');
});
function exportCSV() {
  const rows = [['time','person','vehicle','animal','bicycle','bag','phone','weapon','umbrella','infra','furniture','object','others','total']];
  localHistory.forEach(e => {
    if (!e.counts) return;
    rows.push([e.time, ...CATS.map(c=>e.counts[c]||0), e.total||0]);
  });
  dl('eyecore_log.csv', rows.map(r=>r.join(',')).join('\n'), 'text/csv');
}
function exportJSON() {
  dl('eyecore_log.json', JSON.stringify(localHistory, null, 2), 'application/json');
}
function dl(name, content, type) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([content],{type}));
  a.download = name; a.click();
}

// ── CHART ──
// Chart.js line chart showing live YOLOv8 detection counts for key categories
const chart = new Chart(document.getElementById('chart'), {
  type: 'line',
  data: {
    labels: [],
    datasets: CHART_CATS.map(cat => ({
      label: cat.toUpperCase(),
      data: [],
      borderColor: COLORS[cat],
      backgroundColor: COLORS[cat] + '10',
      borderWidth: 1.5,
      tension: 0.5,
      pointRadius: 0,
      fill: false,
    }))
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 150 },
    plugins: {
      legend: { labels: { color:'rgba(10,240,255,0.4)', font:{size:8, family:'Orbitron'}, boxWidth:8, padding:10 } }
    },
    scales: {
      x: { display: false },
      y: {
        beginAtZero: true,
        ticks: { color:'rgba(10,240,255,0.25)', font:{size:8, family:'Orbitron'}, stepSize:1 },
        grid: { color:'rgba(10,240,255,0.04)' }
      }
    }
  }
});

// ── BUILD TILES ──
// Dynamically generates one tile per YOLOv8 category; counts updated by poll()
function buildTiles() {
  const bar = document.getElementById('tiles-bar');
  bar.innerHTML = '';
  CATS.forEach(cat => {
    const d = document.createElement('div');
    d.className = 'tile'; d.id = 'tile-'+cat;
    d.style.setProperty('--tcolor', COLORS[cat]);
    d.innerHTML = `
      <div class="tile-name">${cat.toUpperCase().slice(0,6)}</div>
      <div class="tile-val" id="tv-${cat}" style="color:${COLORS[cat]};text-shadow:0 0 10px ${COLORS[cat]}66">0</div>
      <div class="tile-delta" id="td-${cat}"></div>
    `;
    d.style.setProperty('--bc', COLORS[cat]);
    d.querySelector('.tile-val');
    const bar2 = document.createElement('div');
    bar2.style.cssText = `position:absolute;bottom:0;left:0;right:0;height:2px;background:${COLORS[cat]};opacity:0.3;border-radius:0 0 4px 4px;`;
    d.appendChild(bar2);
    bar.appendChild(d);
  });
}
buildTiles();

// ── BUILD BADGES ──
// Overlaid on the YOLOv8 video feed; shows live per-category counts
function buildBadges() {
  const wrap = document.getElementById('video-badges');
  wrap.innerHTML = '';
  ['person','vehicle','weapon','animal','bag','phone'].forEach(cat => {
    const b = document.createElement('div');
    b.className = `vbadge ${cat} zero`; b.id = 'vb-'+cat;
    b.innerHTML = `<span style="font-size:7px;letter-spacing:1px;">${cat.toUpperCase()}</span><span class="vb-count" id="vbv-${cat}">0</span>`;
    wrap.appendChild(b);
  });
}
buildBadges();

// ── HISTORY ──
const histBody = document.getElementById('hist-body');

// Prepends new YOLOv8 detection/alert events returned by Flask /history
function appendHistory(events) {
  if (!events.length) return;
  if (histBody.querySelector('div[style]')) histBody.innerHTML = '';
  events.forEach(ev => {
    const row = document.createElement('div');
    row.className = 'h-row';
    const type = ev.type || 'DETECTION';
    const msg = ev.message ||
      Object.entries(ev.counts||{}).filter(([,v])=>v>0).map(([k,v])=>`${k}:${v}`).join(' ') || '—';
    row.innerHTML = `
      <span class="h-time">${ev.time}</span>
      <span class="h-badge ${type}">${type}</span>
      <span class="h-msg">${msg}</span>
    `;
    histBody.prepend(row);
  });
  while (histBody.children.length > 80) histBody.removeChild(histBody.lastChild);
}

// ── CONFIDENCE ──
// Renders per-object YOLOv8 confidence scores from Flask /confidences
function updateConf(objs) {
  document.getElementById('obj-count').textContent = objs.length + ' TARGETS';
  const body = document.getElementById('conf-body');
  if (!objs.length) {
    body.innerHTML = '<div style="color:var(--dim);font-family:var(--font-eye);font-size:8px;padding:10px;letter-spacing:2px;">NO TARGETS ACQUIRED...</div>';
    return;
  }
  // Sort by descending YOLOv8 confidence, show top 20
  const sorted = [...objs].sort((a,b)=>b.confidence-a.confidence).slice(0,20);
  body.innerHTML = sorted.map(o => {
    const pct = Math.round(o.confidence*100);
    const col = COLORS[o.category]||'#888';
    return `
      <div class="conf-row">
        <div>
          <div class="conf-label">${o.label}</div>
          <div class="conf-cat">${o.category.toUpperCase()}</div>
        </div>
        <div class="conf-bar-track">
          <div class="conf-bar-fill" style="width:${pct}%;background:linear-gradient(90deg,${col}22,${col}66);border-right:1px solid ${col};"></div>
        </div>
        <div class="conf-pct" style="color:${col};">${pct}%</div>
      </div>`;
  }).join('');
}

// ── ALERTS ──
// Client-side mirror of the server-side YOLOv8 alert threshold logic
let activeAlerts = new Set();
function checkAlerts(counts) {
  let any = false;
  for (const [cat,thresh] of Object.entries(ALERT_CATS)) {
    if ((counts[cat]||0) >= thresh) {
      any = true;
      if (!activeAlerts.has(cat)) { activeAlerts.add(cat); alertBeep(); }
    } else {
      activeAlerts.delete(cat);
    }
  }
  const banner = document.getElementById('alert-banner');
  if (any) {
    banner.textContent = `◈ NEURAL ALERT — ${[...activeAlerts].map(c=>c.toUpperCase()).join(' + ')} THRESHOLD EXCEEDED ◈`;
    banner.style.display = 'block';
  } else {
    banner.style.display = 'none';
  }
}

// ── POLL ──
// Hits three Flask endpoints every 800ms to keep the dashboard live:
//   /counts      — YOLOv8 per-category detection counts + FPS
//   /confidences — per-object YOLOv8 confidence scores and bounding boxes
//   /history     — new detection/alert events since last poll
async function poll() {
  try {
    const [cRes, confRes, hRes] = await Promise.all([
      fetch('/counts'),
      fetch('/confidences'),
      fetch(`/history?since=${lastHistTime}`)
    ]);
    const counts = await cRes.json();
    const conf   = await confRes.json();
    const hist   = await hRes.json();

    // Update HUD with YOLOv8 FPS and total object count
    document.getElementById('status').textContent = 'ACTIVE';
    document.getElementById('status').className = 'val online';
    document.getElementById('fps').textContent = counts.fps || '—';
    document.getElementById('total').textContent = counts.total_detections || 0;

    // Update category tiles with latest YOLOv8 counts and delta indicators
    CATS.forEach(cat => {
      const val = counts[cat]||0;
      const prev = lastCounts[cat]||0;
      const delta = val - prev;
      document.getElementById('tv-'+cat).textContent = val;
      const de = document.getElementById('td-'+cat);
      de.textContent = delta>0?`▲${delta}`:delta<0?`▼${Math.abs(delta)}`:'';
      de.style.color = delta>0?'var(--pupil)':delta<0?'var(--alert)':'var(--dim)';
      const tile = document.getElementById('tile-'+cat);
      const isAlert = ALERT_CATS[cat] && val >= ALERT_CATS[cat];
      tile.className = 'tile'+(isAlert?' alert':'');
    });

    // Update video overlay badges with latest YOLOv8 counts
    ['person','vehicle','weapon','animal','bag','phone'].forEach(cat => {
      const val = counts[cat]||0;
      document.getElementById('vbv-'+cat).textContent = val;
      document.getElementById('vb-'+cat).className = `vbadge ${cat}${val===0?' zero':''}`;
    });

    // Append new YOLOv8 data points to the Chart.js rolling window (last 30 frames)
    chart.data.labels.push('');
    CHART_CATS.forEach((cat,i) => {
      chart.data.datasets[i].data.push(counts[cat]||0);
      if (chart.data.datasets[i].data.length > 30) chart.data.datasets[i].data.shift();
    });
    if (chart.data.labels.length > 30) chart.data.labels.shift();
    chart.update('none');

    // Render per-object YOLOv8 confidence scores
    updateConf(conf.objects||[]);

    // Append new Flask history events (detections + alerts) to the log panel
    if (hist.events?.length) {
      localHistory.push(...hist.events);
      appendHistory(hist.events);
      if (hist.last_time) lastHistTime = hist.last_time;
    }

    checkAlerts(counts);
    lastCounts = {...counts};

  } catch(e) {
    // Flask server unreachable — show OFFLINE status
    document.getElementById('status').textContent = 'OFFLINE';
    document.getElementById('status').className = 'val offline';
  }
}

function clearHistory() {
  histBody.innerHTML = '<div style="color:var(--dim);font-family:var(--font-eye);font-size:8px;padding:10px;letter-spacing:2px;">MEMBRANE FLUSHED...</div>';
  localHistory = []; lastHistTime = 0;
}

// ── UPTIME ──
setInterval(() => {
  uptime++;
  document.getElementById('uptime').textContent =
    [Math.floor(uptime/3600),Math.floor(uptime%3600/60),uptime%60]
    .map(n=>String(n).padStart(2,'0')).join(':');
}, 1000);

// Poll Flask endpoints every 800ms for fresh YOLOv8 detection data
setInterval(poll, 800);
poll();
</script>
</body>
</html>
"""


# =========================
# FLASK ROUTES
# All routes are served by Flask; data originates from the YOLOv8 worker thread
# =========================

@app.route('/')
def index():
    # Serves the single-page dashboard HTML (polls YOLOv8 data via JSON endpoints)
    return render_template_string(HTML)

@app.route('/video')
def video():
    # Streams YOLOv8-annotated frames as a multipart MJPEG response
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/counts')
def counts():
    # Returns latest per-category YOLOv8 detection counts and FPS as JSON
    with data_lock:
        return jsonify(shared_data)

@app.route('/confidences')
def confidences():
    # Returns per-object YOLOv8 detection details (label, category, confidence, bbox)
    with conf_lock:
        return jsonify({"objects": list(confidence_list)})

@app.route('/history')
def history():
    # Returns YOLOv8 detection/alert events newer than the `since` timestamp
    since = float(request.args.get('since', 0))
    with history_lock:
        events = [e for e in detection_history if e.get('timestamp', 0) > since]
        last_time = events[-1]['timestamp'] if events else since
    return jsonify({"events": events, "last_time": last_time})

@app.route('/export/csv')
def export_csv():
    # Server-side CSV export of the full YOLOv8 detection history (Flask route)
    with history_lock:
        rows = list(detection_history)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['time','timestamp'] + ALL_CATEGORIES + ['total'])
    for row in rows:
        if 'counts' not in row:
            continue
        writer.writerow([row.get('time',''), row.get('timestamp','')] +
                        [row['counts'].get(c,0) for c in ALL_CATEGORIES] +
                        [row.get('total',0)])
    return Response(output.getvalue(), mimetype='text/csv',
                    headers={"Content-Disposition": "attachment;filename=eyecore_log.csv"})

@app.route('/export/json')
def export_json():
    # Server-side JSON export of the full YOLOv8 detection history (Flask route)
    with history_lock:
        rows = list(detection_history)
    return Response(json.dumps(rows, indent=2), mimetype='application/json',
                    headers={"Content-Disposition": "attachment;filename=eyecore_log.json"})


# =========================
# START
# Launches background threads then starts the Flask dev server.
# Thread 1: yolo_worker  — runs YOLOv8 inference and writes results to shared state
# Thread 2: detection_loop — pulls MJPEG frames from the camera and feeds the queue
# =========================
if __name__ == '__main__':
    threading.Thread(target=yolo_worker, daemon=True).start()
    threading.Thread(target=detection_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
