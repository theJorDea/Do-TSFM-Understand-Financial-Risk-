"""Formal forecast comparison: Diebold-Mariano and the Model Confidence Set.

Diebold & Mariano (1995), with the Harvey, Leybourne & Newbold (1997)
small-sample correction. The long-run variance of the loss differential uses
a Newey-West (Bartlett) kernel; for h-step forecasts the default truncation
lag is h - 1 (the MA(h-1) structure induced by overlapping forecast errors),
extendable for extra serial dependence.

Model Confidence Set (Hansen, Lunde & Nason 2011), T_max statistic with a
moving-block bootstrap. Sequential elimination: at each step the worst model
is removed if the equal-predictive-ability hypothesis is rejected; a model's
MCS p-value is the running maximum of elimination p-values, and the returned
set contains the models surviving at the requested confidence level.

Both are validated on synthetic data in tests: size under equal ability,
power against a dominated model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

# --------------------------------------------------------------------- DM


@dataclass(frozen=True)
class DMResult:
    statistic: float
    p_value: float
    mean_diff: float  # mean(loss_1 - loss_2); negative favors model 1
    n_obs: int


def newey_west_variance(d: np.ndarray, max_lag: int) -> float:
    """Long-run variance of the mean of ``d`` (Bartlett kernel)."""
    d = d - d.mean()
    n = d.size
    gamma0 = float(d @ d) / n
    lrv = gamma0
    for lag in range(1, min(max_lag, n - 1) + 1):
        w = 1.0 - lag / (max_lag + 1.0)
        gamma = float(d[lag:] @ d[:-lag]) / n
        lrv += 2.0 * w * gamma
    return max(lrv, 1e-16)


def diebold_mariano(
    loss_1: np.ndarray,
    loss_2: np.ndarray,
    horizon: int = 1,
    extra_lags: int = 0,
) -> DMResult:
    """H0: equal expected loss. Negative statistic favors model 1."""
    l1 = np.asarray(loss_1, dtype=float)
    l2 = np.asarray(loss_2, dtype=float)
    if l1.shape != l2.shape or l1.ndim != 1:
        raise ValueError("loss series must be 1-D and aligned")
    mask = np.isfinite(l1) & np.isfinite(l2)
    d = (l1 - l2)[mask]
    n = d.size
    if n < 100:
        raise ValueError("dm: need >= 100 aligned observations")

    max_lag = max(horizon - 1 + extra_lags, 0)
    lrv = newey_west_variance(d, max_lag)
    dm = float(d.mean() / np.sqrt(lrv / n))

    # Harvey-Leybourne-Newbold small-sample correction
    k = horizon
    c = np.sqrt((n + 1 - 2 * k + k * (k - 1) / n) / n)
    dm_adj = dm * c
    p = 2.0 * float(stats.t.sf(abs(dm_adj), df=n - 1))
    return DMResult(statistic=dm_adj, p_value=p, mean_diff=float(d.mean()), n_obs=n)


# -------------------------------------------------------------------- MCS


@dataclass(frozen=True)
class MCSResult:
    included: list[str]  # models in the MCS at the given confidence level
    p_values: dict[str, float]  # MCS p-value per model (order of elimination)
    confidence: float


def _block_bootstrap_indices(
    rng: np.random.Generator, n: int, block: int, n_boot: int
) -> np.ndarray:
    """Moving-block bootstrap index matrix of shape (n_boot, n)."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, -1)
    return idx[:, :n]


def model_confidence_set(
    losses: dict[str, np.ndarray],
    confidence: float = 0.90,
    n_boot: int = 2000,
    block: int | None = None,
    seed: int = 2026,
) -> MCSResult:
    """MCS over aligned per-observation loss series (lower loss = better)."""
    names = list(losses)
    if len(names) < 2:
        raise ValueError("mcs: need at least two models")
    mat = np.column_stack([np.asarray(losses[k], dtype=float) for k in names])
    mask = np.all(np.isfinite(mat), axis=1)
    mat = mat[mask]
    n = mat.shape[0]
    if n < 100:
        raise ValueError("mcs: need >= 100 aligned observations")
    if block is None:
        block = max(int(round(n ** (1 / 3))), 1)

    rng = np.random.default_rng(seed)
    boot_idx = _block_bootstrap_indices(rng, n, block, n_boot)

    active = list(range(len(names)))
    p_values: dict[str, float] = {}
    p_running = 0.0

    while len(active) > 1:
        sub = mat[:, active]
        dbar_i = sub.mean(axis=0) - sub.mean()  # loss vs. cross-model average

        # bootstrap distribution of centered per-model means
        boot_means = sub[boot_idx].mean(axis=1)  # (n_boot, m)
        boot_dbar = boot_means - boot_means.mean(axis=1, keepdims=True)
        se = np.sqrt(np.mean((boot_dbar - dbar_i) ** 2, axis=0))
        se = np.maximum(se, 1e-16)

        t_obs = dbar_i / se
        t_max_obs = float(t_obs.max())
        t_boot = (boot_dbar - dbar_i) / se
        t_max_boot = t_boot.max(axis=1)
        p_step = float(np.mean(t_max_boot >= t_max_obs))

        p_running = max(p_running, p_step)
        worst = int(np.argmax(t_obs))
        worst_name = names[active[worst]]

        if p_step >= 1.0 - confidence:
            # cannot reject equal ability: everyone still active is in the set
            for i in active:
                p_values.setdefault(names[i], max(p_running, p_step))
            return MCSResult(
                included=[names[i] for i in active],
                p_values=p_values,
                confidence=confidence,
            )
        p_values[worst_name] = p_running
        active.pop(worst)

    p_values[names[active[0]]] = 1.0
    return MCSResult(included=[names[active[0]]], p_values=p_values, confidence=confidence)
