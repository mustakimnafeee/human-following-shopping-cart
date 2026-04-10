"""
camera.py - PiCamera2 wrapper (V5.1 - Full Field of View Fix)
"""
import cv2
import time
from picamera2 import Picamera2
from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT,
    CAMERA_SHARPNESS, CAMERA_CONTRAST,
    CAMERA_SATURATION, CAMERA_BRIGHTNESS,
    SWAP_RED_BLUE
)

class Camera:
    def __init__(self):
        print("[Camera] Initializing PiCamera2 for FULL FOV...")
        self._picam2 = Picamera2()

        # FIX: Ask for a high-res image (1280x960) to force the wide-angle lens view.
        # If we ask for 640x480 directly, the Pi hardware center-crops it like a telescope.
        config = self._picam2.create_preview_configuration(
            main={
                "size":   (1280, 960), 
                "format": "XBGR8888"
            }
        )
        self._picam2.configure(config)
        self._picam2.start()
        time.sleep(2.0)   # OV5647 needs ~2s for AE/AWB to settle

        try:
            self._picam2.set_controls({
                "Sharpness":  CAMERA_SHARPNESS,
                "Contrast":   CAMERA_CONTRAST,
                "Saturation": CAMERA_SATURATION,
                "Brightness": CAMERA_BRIGHTNESS,
                "AeEnable":   True,
                "AwbEnable":  True,
            })
        except Exception as e:
            print(f"[Camera] Warning — could not set controls: {e}")

        self._swap_rb = bool(SWAP_RED_BLUE)
        status = "ON" if self._swap_rb else "OFF"
        print(f"[Camera] R/B swap: {status}  |  AI Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")

    def capture(self):
        """Return a clean BGR frame ready for OpenCV / YOLO."""
        # Capture the massive wide-angle frame
        frame = self._picam2.capture_array("main")

        # FIX: Instantly shrink it back down to our target AI size
        frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]

        if self._swap_rb:
            frame = frame[:, :, ::-1].copy()

        return frame

    def close(self):
        self._picam2.close()
        print("[Camera] Closed.")