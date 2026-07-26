"""Synthetic data generators for validation tests.

Used to verify estimators recover known parameters and (in Phase 3) that
backtests and comparison tests have the size/power reported in the original
papers. Kept in the package, not the test tree, because notebooks reuse it.
"""

from __future__ import annotations

import numpy as np


def simulate_garch(
    n: int,
    omega: float = 0.05,
    alpha: float = 0.08,
    beta: float = 0.90,
    gamma: float = 0.0,
    nu: float | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Zero-mean (GJR-)GARCH(1,1) log-returns in decimal units.

    Parameters are in percent^2 / percent conventions (omega=0.05 with
    daily percent returns); output is divided by 100 to match the decimal
    return convention used across the package. ``nu`` switches innovations
    from standard normal to standardized Student-t.
    """
    if alpha + beta + 0.5 * gamma >= 1.0:
        raise ValueError("non-stationary parameter set")
    rng = np.random.default_rng(seed)
    burn = 1000
    r = np.empty(n + burn)
    s2 = omega / (1.0 - alpha - beta - 0.5 * gamma)
    for t in range(n + burn):
        if nu is None:
            z = rng.standard_normal()
        else:
            z = rng.standard_t(nu) * np.sqrt((nu - 2.0) / nu)
        r[t] = np.sqrt(s2) * z
        s2 = omega + (alpha + (gamma if r[t] < 0 else 0.0)) * r[t] ** 2 + beta * s2
    return r[burn:] / 100.0
