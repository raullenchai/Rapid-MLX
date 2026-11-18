"""Model how a drug's blood concentration rises and falls after an oral dose, judged on held-out samples."""
import json

import numpy as np

# Theophylline, subject 1 (Boeckmann, Sheiner and Beal): oral dose 4.02 mg/kg; time (h) and plasma concentration (mg/L).
DOSE = 4.02
TIME = np.array([0.0, 0.25, 0.57, 1.12, 2.02, 3.82, 5.10, 7.03, 9.05, 12.12, 24.37])
CONC = np.array([0.74, 2.84, 6.57, 10.50, 9.66, 8.58, 8.36, 7.47, 6.89, 5.94, 3.28])


def fit(time, conc, dose):
    """Return a function t -> predicted concentration. Baseline: one exponential decay fitted on log concentrations."""
    slope, intercept = np.polyfit(time, np.log(conc), 1)
    return lambda t: np.exp(intercept + slope * np.asarray(t))


errors = []
for i in range(1, len(TIME)):
    keep = np.arange(len(TIME)) != i
    model = fit(TIME[keep], CONC[keep], DOSE)
    errors.append(float(model(TIME[i]) - CONC[i]))
model = fit(TIME, CONC, DOSE)
for t in np.linspace(0, 24.37, 50):
    print(json.dumps({"hours": round(float(t), 2), "model": round(float(model(t)), 3)}), flush=True)
for t, c in zip(TIME, CONC):
    print(json.dumps({"hours": float(t), "measured": float(c)}), flush=True)
rmse = float(np.sqrt(np.mean(np.square(errors))))
print("RESULT " + json.dumps({"loo_rmse": round(rmse, 3)}), flush=True)
