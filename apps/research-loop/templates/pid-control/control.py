"""Tune a controller that holds a motor at a target speed: fast, without overshoot, despite a load change."""
import json

import numpy as np

DT = 0.01
T_END = 10.0
SETPOINT = 1.0
U_MAX = 5.0


def make_controller():
    """Return a function (error, dt) -> command. Baseline: proportional-only control."""
    kp = 1.0

    def controller(error, dt):
        return kp * error

    return controller


# Plant: a DC motor with inertia and a 50 ms actuator lag; at t = 5 s a load torque is applied.
controller = make_controller()
speed, current, cost, overshoot = 0.0, 0.0, 0.0, 0.0
for i in range(int(T_END / DT)):
    t = i * DT
    error = SETPOINT - speed
    command = float(np.clip(controller(error, DT), -U_MAX, U_MAX))
    current += DT * (command - current) / 0.05
    load = 0.4 if t >= 5.0 else 0.0
    speed += DT * (current - 0.5 * speed - load) / 0.8
    cost += DT * abs(error) * (1 + t)
    overshoot = max(overshoot, speed - SETPOINT)
    if i % 10 == 0:
        print(json.dumps({"t": round(t, 2), "speed": round(speed, 4), "target": SETPOINT}), flush=True)
score = cost + 10 * max(0.0, overshoot - 0.05)
print("RESULT " + json.dumps({"score": round(score, 4), "itae": round(cost, 4), "overshoot": round(overshoot, 4)}), flush=True)
