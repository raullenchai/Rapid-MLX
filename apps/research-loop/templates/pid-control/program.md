# Tune a motor speed controller

A small DC motor must reach and hold a target speed. The actuator is limited to ±5 and responds with a 50 ms lag, and at
t = 5 s a load is applied that slows the motor down. The baseline is a proportional-only controller.

Goal: minimise `score` = time-weighted absolute error (ITAE) plus a penalty of 10 per unit of overshoot above 5% (lower is better).

Rules:
- Edit only `make_controller()` (and helpers): it returns a function of (error, dt) that may keep internal state.
- Keep DT, T_END, SETPOINT, U_MAX, the plant model, the load and the scoring unchanged. The controller sees only the error.
- One idea per experiment. The simulation is deterministic; differences under 0.05 are noise.
