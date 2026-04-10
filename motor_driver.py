"""
motor_driver.py - Differential drive motor control V5

Improvements over V4:
  - Direct Motor objects (no _robot.left_motor deprecated path)
  - Steer-priority: throttle scaled down during hard turns so the
    cart rotates onto target before surging forward
  - stop() is safe to call multiple times
"""
import numpy as np
from config import (
    MOTORS_ENABLED,
    LEFT_MOTOR_FORWARD, LEFT_MOTOR_BACKWARD,
    RIGHT_MOTOR_FORWARD, RIGHT_MOTOR_BACKWARD,
    MAX_SPEED, MIN_SPEED,
    STEER_PRIORITY_THRESHOLD, STEER_PRIORITY_THROTTLE_SCALE,
)

if MOTORS_ENABLED:
    from gpiozero import Motor
    _left_motor  = Motor(forward=LEFT_MOTOR_FORWARD,  backward=LEFT_MOTOR_BACKWARD)
    _right_motor = Motor(forward=RIGHT_MOTOR_FORWARD, backward=RIGHT_MOTOR_BACKWARD)
    print("[Motor] Motors ENABLED.")
else:
    _left_motor = _right_motor = None
    print("[Motor] Motors DISABLED (test mode).")


def _apply_deadzone(speed):
    return 0.0 if abs(speed) < MIN_SPEED else float(speed)


def drive(steer_output, throttle_output):
    if not MOTORS_ENABLED:
        return

    # Steer-priority: rotate onto target before moving forward
    if abs(steer_output) > STEER_PRIORITY_THRESHOLD:
        throttle_output *= STEER_PRIORITY_THROTTLE_SCALE

    left_speed  = throttle_output + steer_output
    right_speed = throttle_output - steer_output

    left_speed  = float(np.clip(left_speed,  -MAX_SPEED, MAX_SPEED))
    right_speed = float(np.clip(right_speed, -MAX_SPEED, MAX_SPEED))

    left_speed  = _apply_deadzone(left_speed)
    right_speed = _apply_deadzone(right_speed)

    _command_motor(_left_motor,  left_speed)
    _command_motor(_right_motor, right_speed)


def _command_motor(motor, speed):
    if speed > 0:
        motor.forward(speed)
    elif speed < 0:
        motor.backward(abs(speed))
    else:
        motor.stop()


def stop():
    if MOTORS_ENABLED and _left_motor is not None:
        _left_motor.stop()
        _right_motor.stop()