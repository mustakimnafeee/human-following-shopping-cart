"""
app.py - Human Following Shopping Cart V5
Pi Camera v1.3 | NCNN YOLOv8n | BoT-SORT | PID+F | Steer-Priority
"""
import cv2
import time
import traceback
import threading
from flask import Flask, Response, jsonify
from ultralytics import YOLO

from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CENTER_X,
    STEER_KP, STEER_KI, STEER_KD, STEER_KF,
    THROTTLE_KP, THROTTLE_KI, THROTTLE_KD, THROTTLE_KF,
    MAX_SPEED, TARGET_HEIGHT_RATIO, HEIGHT_TOLERANCE, MIN_HEIGHT_RATIO,
    OUTPUT_SMOOTHING_ALPHA, LOST_BLEED_RATE,
    SERVER_HOST, SERVER_PORT, JPEG_QUALITY,
    YOLO_MODEL, TRACKER_TYPE,
    DETECTION_CONFIDENCE, DETECTION_IOU,
    LOCK_AFTER_FRAMES, LOST_TIMEOUT,
)
from camera import Camera
from pid_controller import PIDController
from motor_driver import drive, stop
from hud import HUD

print("=" * 60)
print("   🛒  HUMAN FOLLOWING SHOPPING CART  v5.0")
print("   Pi Camera v1.3 | NCNN YOLOv8n | PID+F")
print("=" * 60)

camera  = Camera()
display = HUD()

print(f"[AI] Loading NCNN model: {YOLO_MODEL} ...")
model = YOLO(YOLO_MODEL, task='detect')

steer_pid = PIDController(
    STEER_KP, STEER_KI, STEER_KD, kf=STEER_KF,
    output_min=-MAX_SPEED, output_max=MAX_SPEED,
)
throttle_pid = PIDController(
    THROTTLE_KP, THROTTLE_KI, THROTTLE_KD, kf=THROTTLE_KF,
    output_min=-MAX_SPEED, output_max=MAX_SPEED,
)

app          = Flask(__name__)
latest_frame = None
frame_lock   = threading.Lock()
robot_running = True

telemetry = {
    "status":       "STARTING",
    "fps":          0.0,
    "locked_id":    None,
    "steer":        0.0,
    "throttle":     0.0,
    "height_ratio": 0.0,
    "persons":      0,
}
telem_lock = threading.Lock()


# ──────────────────────────────────────────────────────────────
def _best_candidate(detected_persons, locked_id):
    """
    Pick the best target:
      1. Locked ID if still visible.
      2. Largest person above MIN_HEIGHT_RATIO.
    """
    if not detected_persons:
        return None

    if locked_id is not None:
        for p in detected_persons:
            if p['id'] == locked_id:
                return p

    candidates = [
        p for p in detected_persons
        if (p['bbox'][3] / CAMERA_HEIGHT) >= MIN_HEIGHT_RATIO
    ]
    if not candidates:
        return None

    return max(candidates, key=lambda p: p['bbox'][2] * p['bbox'][3])


# ──────────────────────────────────────────────────────────────
def control_loop():
    global latest_frame, robot_running

    locked_track_id  = None
    lock_countdown   = LOCK_AFTER_FRAMES
    time_target_lost = None
    smoothed_steer    = 0.0
    smoothed_throttle = 0.0

    while robot_running:
        try:
            frame  = camera.capture()
            output = frame.copy()

            # ── 1. DETECTION & TRACKING ────────────────────
            results = model.track(
                frame,
                persist=True,
                classes=[0],
                tracker=TRACKER_TYPE,
                conf=DETECTION_CONFIDENCE,
                iou=DETECTION_IOU,
                verbose=False,
            )

            detected_persons = []

            if (results[0].boxes is not None
                    and results[0].boxes.id is not None):
                boxes     = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confs     = results[0].boxes.conf.cpu().tolist()

                for box, tid, conf in zip(boxes, track_ids, confs):
                    x1, y1, x2, y2 = map(int, box)
                    bw, bh = x2 - x1, y2 - y1
                    detected_persons.append({
                        'id':   tid,
                        'bbox': (x1, y1, bw, bh),
                        'conf': conf,
                    })
                    if tid != locked_track_id:
                        display.draw_person_box(output, (x1, y1, bw, bh), track_id=tid)

            # ── 2. TARGET LOCKING ──────────────────────────
            target = _best_candidate(detected_persons, locked_track_id)

            if locked_track_id is not None:
                if target is not None and target['id'] == locked_track_id:
                    time_target_lost = None      # Still tracking fine
                else:
                    if time_target_lost is None:
                        time_target_lost = time.time()
                    elif (time.time() - time_target_lost) > LOST_TIMEOUT:
                        locked_track_id  = None
                        lock_countdown   = LOCK_AFTER_FRAMES
                        time_target_lost = None
                    target = None                # Don't jump to another person
            else:
                if target is not None:
                    lock_countdown -= 1
                    if lock_countdown <= 0:
                        locked_track_id = target['id']
                        print(f"[LOCK] Locked onto ID {locked_track_id}")
                else:
                    lock_countdown = LOCK_AFTER_FRAMES

            # ── 3. CONTROL ─────────────────────────────────
            steer_cmd    = 0.0
            throttle_cmd = 0.0
            status_text  = ""
            status_color = (200, 200, 200)
            height_ratio = 0.0

            if target is not None:
                tx, ty, tw, th = target['bbox']
                cx = tx + tw // 2
                height_ratio = th / CAMERA_HEIGHT

                steer_error    = (cx - CENTER_X) / (CAMERA_WIDTH / 2.0)
                throttle_error = (TARGET_HEIGHT_RATIO - height_ratio) / TARGET_HEIGHT_RATIO

                steer_cmd    = steer_pid.compute(steer_error)
                throttle_cmd = throttle_pid.compute(throttle_error)

                

                # Distance deadband
                if abs(throttle_error) < (HEIGHT_TOLERANCE / TARGET_HEIGHT_RATIO):
                    throttle_cmd = 0.0
                    throttle_pid.reset_integral()

                # Don't move until locked
                if locked_track_id is None:
                    steer_cmd    = 0.0
                    throttle_cmd = 0.0

                # EMA smoothing
                a = OUTPUT_SMOOTHING_ALPHA
                smoothed_steer    = a * steer_cmd    + (1 - a) * smoothed_steer
                smoothed_throttle = a * throttle_cmd + (1 - a) * smoothed_throttle

                drive(smoothed_steer, smoothed_throttle)

                display.draw_target_box(
                    output, target['bbox'],
                    locked=(locked_track_id is not None),
                    track_id=target['id'],
                )
                if locked_track_id is None:
                    display.draw_locking_bar(output, target['bbox'], lock_countdown)

                if locked_track_id is None:
                    status_text  = "ACQUIRING TARGET..."
                    status_color = (0, 220, 220)
                elif abs(throttle_error) < (HEIGHT_TOLERANCE / TARGET_HEIGHT_RATIO):
                    status_text  = "HOLDING DISTANCE"
                    status_color = (0, 255, 0)
                elif throttle_error > 0:
                    status_text  = "FOLLOWING  >>"
                    status_color = (0, 200, 80)
                else:
                    status_text  = "<<  BACKING OFF"
                    status_color = (0, 60, 255)

            else:
                steer_pid.reset()
                throttle_pid.reset()

                smoothed_steer    *= LOST_BLEED_RATE
                smoothed_throttle *= LOST_BLEED_RATE

                if abs(smoothed_steer) < 0.04 and abs(smoothed_throttle) < 0.04:
                    stop()
                    smoothed_steer    = 0.0
                    smoothed_throttle = 0.0
                else:
                    drive(smoothed_steer, smoothed_throttle)

                if locked_track_id is not None and time_target_lost is not None:
                    elapsed      = time.time() - time_target_lost
                    status_text  = f"TARGET LOST ({elapsed:.1f}s / {LOST_TIMEOUT:.1f}s)"
                    status_color = (0, 60, 255)
                else:
                    status_text  = "SEARCHING..."
                    status_color = (120, 120, 120)

            # ── 4. HUD ─────────────────────────────────────
            display.draw_crosshair(output)
            display.draw_status_bar(output, status_text, status_color)
            display.draw_motor_status(output)
            display.draw_person_count(output, len(detected_persons))
            display.draw_drive_bars(output, smoothed_steer, smoothed_throttle)
            display.draw_fps(output)

            with frame_lock:
                latest_frame = output.copy()

            # ── 5. TELEMETRY ───────────────────────────────
            with telem_lock:
                telemetry["status"]       = status_text
                telemetry["fps"]          = round(display.current_fps, 1)
                telemetry["locked_id"]    = locked_track_id
                telemetry["steer"]        = round(float(smoothed_steer),    3)
                telemetry["throttle"]     = round(float(smoothed_throttle), 3)
                telemetry["height_ratio"] = round(float(height_ratio),      3)
                telemetry["persons"]      = len(detected_persons)

            time.sleep(0.008)

        except Exception as e:
            stop()
            print(f"[ERROR] {e}")
            traceback.print_exc()
            time.sleep(0.1)


# ──────────────────────────────────────────────────────────────
def generate_web_frames():
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is not None:
            ret, buffer = cv2.imencode(
                '.jpg', frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n'
                       + buffer.tobytes() + b'\r\n')
        time.sleep(0.04)


# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Cart Cam</title>
  <style>
    body { background:#111; color:#eee; font-family:monospace;
           display:flex; flex-direction:column; align-items:center; padding:20px; margin:0; }
    h2   { margin:0 0 10px; color:#0f0; letter-spacing:2px; }
    img  { border:2px solid #333; border-radius:4px; max-width:100%; }
    #telem { margin-top:10px; font-size:13px; line-height:2;
             background:#1a1a1a; padding:10px 20px; border-radius:6px; min-width:300px; }
    .k { color:#888; } .v { color:#0f0; font-weight:bold; }
  </style>
</head>
<body>
  <h2>CART CAM v5</h2>
  <img src="/video_feed" width="640">
  <div id="telem">Loading...</div>
  <script>
    async function poll() {
      try {
        const d = await (await fetch('/status')).json();
        document.getElementById('telem').innerHTML =
          `<span class="k">STATUS   </span><span class="v">${d.status}</span><br>` +
          `<span class="k">FPS      </span><span class="v">${d.fps}</span><br>` +
          `<span class="k">LOCKED ID</span><span class="v">${d.locked_id ?? '—'}</span><br>` +
          `<span class="k">PERSONS  </span><span class="v">${d.persons}</span><br>` +
          `<span class="k">STEER    </span><span class="v">${d.steer.toFixed(3)}</span><br>` +
          `<span class="k">THROTTLE </span><span class="v">${d.throttle.toFixed(3)}</span><br>` +
          `<span class="k">HEIGHT   </span><span class="v">${d.height_ratio.toFixed(3)}</span>`;
      } catch(e) {}
    }
    setInterval(poll, 300);
    poll();
  </script>
</body>
</html>'''


@app.route('/video_feed')
def video_feed():
    return Response(generate_web_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    with telem_lock:
        return jsonify(dict(telemetry))


# ── Entry point ───────────────────────────────────────────────
if __name__ == '__main__':
    worker = threading.Thread(target=control_loop, daemon=True)
    worker.start()
    try:
        app.run(host=SERVER_HOST, port=SERVER_PORT, threaded=True)
    finally:
        robot_running = False
        stop()
        camera.close()
        print("[Main] Shutdown complete.")