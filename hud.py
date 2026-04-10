"""
hud.py - On-screen display V5

Improvements over V4:
  - FPS smoothed with EMA
  - draw_drive_bars() — visual steer/throttle bar gauges
  - draw_locking_bar() uses actual LOCK_AFTER_FRAMES constant
  - draw_target_box() shows track ID
"""
import cv2
import time
from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT,
    CENTER_X, CENTER_Y,
    MOTORS_ENABLED,
    LOCK_AFTER_FRAMES,
)

_FPS_EMA_ALPHA = 0.2

class HUD:
    def __init__(self):
        self._frame_count = 0
        self._fps_time    = time.time()
        self.current_fps  = 0.0

    # ── FPS ──────────────────────────────────────────────────
    def update_fps(self):
        self._frame_count += 1
        elapsed = time.time() - self._fps_time
        if elapsed >= 0.5:
            instant = self._frame_count / elapsed
            self.current_fps = (_FPS_EMA_ALPHA * instant
                                + (1 - _FPS_EMA_ALPHA) * self.current_fps)
            self._frame_count = 0
            self._fps_time    = time.time()

    def draw_fps(self, frame):
        self.update_fps()
        fps   = self.current_fps
        color = (0, 255, 0) if fps >= 10 else (0, 165, 255) if fps >= 5 else (0, 0, 255)
        cv2.putText(frame, f"FPS:{fps:.1f}",
                    (CAMERA_WIDTH - 90, CAMERA_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    # ── Crosshair ────────────────────────────────────────────
    def draw_crosshair(self, frame):
        cv2.line(frame, (CENTER_X - 22, CENTER_Y), (CENTER_X + 22, CENTER_Y), (80, 80, 80), 1)
        cv2.line(frame, (CENTER_X, CENTER_Y - 22), (CENTER_X, CENTER_Y + 22), (80, 80, 80), 1)
        cv2.circle(frame, (CENTER_X, CENTER_Y), 4, (80, 80, 80), 1)

    # ── Non-target person boxes ───────────────────────────────
    def draw_person_box(self, frame, bbox, track_id=None):
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
        if track_id is not None:
            cv2.putText(frame, f"ID:{track_id}", (x, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    # ── Target box ───────────────────────────────────────────
    def draw_target_box(self, frame, bbox, locked=True, track_id=None):
        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2
        color  = (0, 255, 0) if locked else (0, 200, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Corner accent lines
        clen = min(18, w // 4, h // 4)
        for px, py, dx, dy in [
            (x,     y,      1,  1),
            (x + w, y,     -1,  1),
            (x,     y + h,  1, -1),
            (x + w, y + h, -1, -1),
        ]:
            cv2.line(frame, (px, py), (px + dx * clen, py), color, 3)
            cv2.line(frame, (px, py), (px, py + dy * clen), color, 3)

        cv2.circle(frame, (cx, cy), 4, color, -1)

        label = "LOCKED" if locked else "ACQUIRING"
        if track_id is not None:
            label += f"  ID:{track_id}"
        cv2.putText(frame, label, (x, max(y - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    # ── Lock progress bar ─────────────────────────────────────
    def draw_locking_bar(self, frame, bbox, countdown):
        x, y, w, _ = bbox
        progress = 1.0 - (countdown / LOCK_AFTER_FRAMES)
        bar_w    = max(0, int(w * progress))
        y0, y1   = y - 18, y - 8
        cv2.rectangle(frame, (x, y0), (x + w,    y1), (40,  40,  40),  -1)
        cv2.rectangle(frame, (x, y0), (x + bar_w, y1), (0, 220, 220), -1)
        cv2.putText(frame, f"LOCKING {int(progress * 100)}%",
                    (x, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 220), 1)

    # ── Status bar (top strip) ────────────────────────────────
    def draw_status_bar(self, frame, text, color=(255, 255, 255)):
        cv2.rectangle(frame, (0, 0), (CAMERA_WIDTH, 28), (0, 0, 0), -1)
        cv2.putText(frame, text, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

    # ── Motor badge ───────────────────────────────────────────
    def draw_motor_status(self, frame):
        text  = "MOTORS:ON"  if MOTORS_ENABLED else "MOTORS:OFF"
        color = (0, 255, 0)  if MOTORS_ENABLED else (0, 0, 255)
        cv2.putText(frame, text, (CAMERA_WIDTH - 115, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

    # ── Person count ──────────────────────────────────────────
    def draw_person_count(self, frame, count):
        cv2.putText(frame, f"Persons:{count}",
                    (CAMERA_WIDTH - 110, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    # ── Steer / Throttle bar gauges ───────────────────────────
    def draw_drive_bars(self, frame, steer, throttle):
        bar_w = 100
        bar_h = 10
        x0    = 8
        y0    = CAMERA_HEIGHT - 40

        for label, val, y in [("STR", steer, y0), ("THR", throttle, y0 + 18)]:
            cv2.rectangle(frame, (x0, y), (x0 + bar_w, y + bar_h), (40, 40, 40), -1)
            mid  = x0 + bar_w // 2
            fill = int(abs(val) * (bar_w // 2))
            col  = (0, 200, 100) if val >= 0 else (0, 100, 220)
            if val >= 0:
                cv2.rectangle(frame, (mid, y), (mid + fill, y + bar_h), col, -1)
            else:
                cv2.rectangle(frame, (mid - fill, y), (mid, y + bar_h), col, -1)
            cv2.rectangle(frame, (x0, y), (x0 + bar_w, y + bar_h), (80, 80, 80), 1)
            cv2.line(frame, (mid, y), (mid, y + bar_h), (120, 120, 120), 1)
            cv2.putText(frame, f"{label}:{val:+.2f}", (x0 + bar_w + 4, y + bar_h - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)