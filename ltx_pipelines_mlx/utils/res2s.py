"""Res2s second-order ODE solver coefficients.

Ported from ltx-pipelines/src/ltx_pipelines/utils/res2s.py
"""

from __future__ import annotations

import math


def phi(j: int, neg_h: float) -> float:
    """Compute phi_j(z) for the res_2s ODE solver.

    These functions appear in exponential integrators for ODEs of the form
    dx/dt = A*x + g(x,t).

    Args:
        j: Order of the phi function.
        neg_h: Negative step size in log-space.

    Returns:
        phi_j(neg_h).
    """
    if abs(neg_h) < 1e-10:
        return 1.0 / math.factorial(j)
    remainder = sum(neg_h**k / math.factorial(k) for k in range(j))
    return (math.exp(neg_h) - remainder) / (neg_h**j)


def get_res2s_coefficients(h: float, phi_cache: dict, c2: float = 0.5) -> tuple[float, float, float]:
    """Compute res_2s Runge-Kutta coefficients for a given step size.

    Args:
        h: Step size in log-space = log(sigma / sigma_next).
        phi_cache: Dictionary to cache phi function results.
        c2: Substep position (default 0.5 = midpoint).

    Returns:
        (a21, b1, b2) coefficients.
    """

    def get_phi(j: int, neg_h: float) -> float:
        cache_key = (j, neg_h)
        if cache_key in phi_cache:
            return phi_cache[cache_key]
        result = phi(j, neg_h)
        phi_cache[cache_key] = result
        return result

    a21 = c2 * get_phi(1, -h * c2)
    b2 = get_phi(2, -h) / c2
    b1 = get_phi(1, -h) - b2
    return a21, b1, b2
