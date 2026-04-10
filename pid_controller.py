"""
pid_controller.py - Discrete PID+F Controller V5

Improvements over V4:
  - dt clamped to 200ms — prevents huge derivative spike after a stall
  - Proper conditional anti-windup — integral only accumulates when
    output is NOT already saturated (fixes overshoot lag)
  - Returns last output if dt is too small (avoids divide-by-zero)
"""
import time
import numpy as np
from config import PID_INTEGRAL_LIMIT, PID_DERIVATIVE_ALPHA

_MAX_DT = 0.20   # Cap dt to avoid D-term spike after a pause


class PIDController:
    def __init__(self, kp, ki, kd, kf=0.0, output_min=-1.0, output_max=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.kf = kf
        self.output_min = output_min
        self.output_max = output_max

        self._integral        = 0.0
        self._prev_error      = 0.0
        self._prev_derivative = 0.0
        self._prev_time       = None
        self._last_output     = 0.0

    def reset(self):
        self._integral        = 0.0
        self._prev_error      = 0.0
        self._prev_derivative = 0.0
        self._prev_time       = None
        self._last_output     = 0.0

    def reset_integral(self):
        self._integral = 0.0

    def compute(self, error):
        now = time.time()

        if self._prev_time is None:
            self._prev_time  = now
            self._prev_error = error
            return 0.0

        dt = now - self._prev_time
        if dt <= 1e-6:
            return self._last_output

        dt = min(dt, _MAX_DT)

        # P
        p_term = self.kp * error

        # I with conditional anti-windup
        tentative = p_term + self.ki * self._integral
        saturated = (tentative >= self.output_max and error > 0) or \
                    (tentative <= self.output_min and error < 0)
        if not saturated:
            self._integral += error * dt
            self._integral  = np.clip(self._integral,
                                      -PID_INTEGRAL_LIMIT, PID_INTEGRAL_LIMIT)
        i_term = self.ki * self._integral

        # D (low-pass filtered)
        raw_d = (error - self._prev_error) / dt
        self._prev_derivative = (PID_DERIVATIVE_ALPHA * raw_d
                                 + (1 - PID_DERIVATIVE_ALPHA) * self._prev_derivative)
        d_term = self.kd * self._prev_derivative

        # F (feedforward for static friction)
        ff_term = 0.0
        if abs(error) > 0.01:
            ff_term = self.kf * np.sign(error)

        output = np.clip(p_term + i_term + d_term + ff_term,
                         self.output_min, self.output_max)

        self._prev_error  = error
        self._prev_time   = now
        self._last_output = output
        return output