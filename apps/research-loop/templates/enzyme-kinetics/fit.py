"""Estimate Michaelis-Menten parameters for an enzyme from measured reaction rates, judged on held-out points."""
import json

import numpy as np

# Treloar (1974), puromycin-treated enzyme: substrate concentration (ppm) and initial velocity (counts/min^2).
CONC = np.array([0.02, 0.02, 0.06, 0.06, 0.11, 0.11, 0.22, 0.22, 0.56, 0.56, 1.10, 1.10])
RATE = np.array([76, 47, 97, 107, 123, 139, 159, 152, 191, 201, 207, 200], dtype=float)


def fit(conc, rate):
    """Return (vmax, km). Baseline: Lineweaver-Burk, a straight line through 1/rate versus 1/conc."""
    slope, intercept = np.polyfit(1 / conc, 1 / rate, 1)
    vmax = 1 / intercept
    km = slope * vmax
    return vmax, km


def predict(conc, vmax, km):
    return vmax * conc / (km + conc)


# Leave-one-out evaluation: fit on 11 points, predict the 12th, for every point.
errors = []
for i in range(len(CONC)):
    keep = np.arange(len(CONC)) != i
    vmax, km = fit(CONC[keep], RATE[keep])
    errors.append(float(predict(CONC[i], vmax, km) - RATE[i]))
vmax, km = fit(CONC, RATE)
for c, r in zip(CONC, RATE):
    print(json.dumps({"conc": float(c), "observed": float(r), "fitted": round(float(predict(c, vmax, km)), 2)}), flush=True)
rmse = float(np.sqrt(np.mean(np.square(errors))))
print("RESULT " + json.dumps({"loo_rmse": round(rmse, 3), "vmax": round(float(vmax), 2), "km": round(float(km), 4)}), flush=True)
