"""
config.py - Human Following Shopping Cart V5
Pi Camera v1.3 | NCNN YOLOv8n | BoT-SORT | PID+F
"""

# ==============================================================
#  MOTOR HARDWARE
# ==============================================================
MOTORS_ENABLED = True

LEFT_MOTOR_FORWARD   = 27
LEFT_MOTOR_BACKWARD  = 17
RIGHT_MOTOR_FORWARD  = 24
RIGHT_MOTOR_BACKWARD = 25

# ==============================================================
#  SPEED LIMITS & DEADZONES
# ==============================================================
MAX_SPEED = 1
MIN_SPEED = 0.22

# ==============================================================
#  CAMERA SETTINGS  (Pi Camera v1.3 = OV5647)
# ==============================================================
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CENTER_X = CAMERA_WIDTH  // 2
CENTER_Y = CAMERA_HEIGHT // 2

CAMERA_SHARPNESS  = 2.0
CAMERA_CONTRAST   = 1.3
CAMERA_SATURATION = 1.0
CAMERA_BRIGHTNESS = 0.0
SWAP_RED_BLUE     = True   # OV5647 always needs R/B swap in XBGR mode

# ==============================================================
#  DEEP LEARNING TRACKING SETTINGS
# ==============================================================
YOLO_MODEL   = "yolov8n_ncnn_model"
TRACKER_TYPE = "botsort.yaml"

DETECTION_CONFIDENCE = 0.45
DETECTION_IOU        = 0.45

# ==============================================================
#  TARGET LOCKING LOGIC
# ==============================================================
LOCK_AFTER_FRAMES = 4      # Frames before locking onto a target
LOST_TIMEOUT      = 1    # Seconds before releasing a lost track

# ==============================================================
#  TARGET FOLLOWING DISTANCE
# ==============================================================
TARGET_HEIGHT_RATIO = 0.55  # Person fills 60% of frame height at ideal distance
HEIGHT_TOLERANCE    = 0.02  # UPGRADED: React to smaller distance changes
MIN_HEIGHT_RATIO    = 0.10
# ==============================================================
#  PID+F GAINS
# ==============================================================
STEER_KP = 1.0      # UPGRADED: Massive boost to turning speed
STEER_KI = 0.008
STEER_KD = 0.65
STEER_KF = 0.12      # UPGRADED: Harder initial kick to break rotational friction


THROTTLE_KP = 2.5
THROTTLE_KI = 0.015
THROTTLE_KD = 0.40
THROTTLE_KF = 0.18

PID_INTEGRAL_LIMIT   = 0.4
PID_DERIVATIVE_ALPHA = 0.15

# Reduce throttle during hard turns so cart centres before surging forward
STEER_PRIORITY_THRESHOLD      = 0.4
STEER_PRIORITY_THROTTLE_SCALE = 0.5    

# ==============================================================
#  SIGNAL PROCESSING
# ==============================================================
OUTPUT_SMOOTHING_ALPHA = .25  # UPGRADED: 1.0 = No delay, instant raw power
LOST_BLEED_RATE        = 0.45


# ==============================================================
#  SERVER
# ==============================================================
SERVER_HOST  = '0.0.0.0'
SERVER_PORT  = 5000
JPEG_QUALITY = 75