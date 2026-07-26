"""GARCH(1,1) family with custom maximum likelihood, zero-mean returns.

Models (Bollerslev 1986; Glosten, Jagannathan & Runkle 1993):

    GARCH(1,1):      s2_t = omega + alpha * r_{t-1}^2 + beta * s2_{t-1}
    GJR-GARCH(1,1):  s2_t = omega + (alpha + gamma * 1[r_{t-1} < 0]) * r_{t-1}^2
                            + beta * s2_{t-1}

Innovations: standard normal or standardized Student-t (unit variance,
nu > 2). Estimation is MLE over the estimation window with the variance
recursion initialized at the window sample variance; returns are scaled to
percent internally for optimizer conditioning (results are scale-invariant
up to that constant and converted back).

Multi-step forecasts use the analytic recursion
    E[s2_{t+k}] = s2_bar + p^{k-1} * (s2_{t+1} - s2_bar),
with persistence p = alpha + beta (+ gamma/2 for GJR under symmetric
innovations) and s2_bar = omega / (1 - p); the h-day aggregate variance is
the sum over k = 1..h. Tail measures at horizon h reuse the one-day
innovation family — an approximation shared by all parametric baselines and
stated in the paper.

Estimates are validated in tests against the `arch` package on simulated
data (parameter recovery and forecast agreement).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special

from tsfm_risk.models.base import RiskForecast, RiskModel
from tsfm_risk.risk.measures import var_es_normal, var_es_student_t

_PCT = 100.0  # internal return scaling
_MIN_S2 = 1e-12
_MAX_PERSISTENCE = 0.9995


@dataclass(frozen=True)
class GarchParams:
    omega: float  # percent^2 units
    alpha: float
    beta: float
    gamma: float = 0.0
    nu: float | None = None
    loglik: float = np.nan
    converged: bool = False

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta + 0.5 * self.gamma


def _filter_variance(
    r: np.ndarray, omega: float, alpha: float, beta: float, gamma: float, s2_init: float
) -> tuple[np.ndarray, float]:
    """Run the variance recursion.

    Returns ``s2`` aligned with ``r`` (``s2[t]`` conditions on info up to
    t-1) and the one-step-ahead forecast ``s2_next`` for the day after the
    last observation.
    """
    n = r.size
    s2 = np.empty(n)
    s2_prev = max(s2_init, _MIN_S2)
    for t in range(n):
        s2[t] = s2_prev
        shock = (alpha + (gamma if r[t] < 0.0 else 0.0)) * r[t] * r[t]
        s2_prev = max(omega + shock + beta * s2[t], _MIN_S2)
    return s2, s2_prev


def _nll_normal(s2: np.ndarray, r: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.log(2.0 * np.pi) + np.log(s2) + r * r / s2))


def _nll_student_t(s2: np.ndarray, r: np.ndarray, nu: float) -> float:
    c = (
        special.gammaln((nu + 1.0) / 2.0)
        - special.gammaln(nu / 2.0)
        - 0.5 * np.log(np.pi * (nu - 2.0))
    )
    z2 = r * r / s2
    ll = c - 0.5 * np.log(s2) - (nu + 1.0) / 2.0 * np.log1p(z2 / (nu - 2.0))
    return -float(np.sum(ll))


class Garch(RiskModel):
    """GARCH(1,1) / GJR-GARCH(1,1) with normal or Student-t innovations."""

    def __init__(self, dist: str = "normal", leverage: bool = False):
        if dist not in ("normal", "t"):
            raise ValueError("dist must be 'normal' or 't'")
        self.dist = dist
        self.leverage = leverage
        self.params: GarchParams | None = None
        base = "gjr-garch11" if leverage else "garch11"
        self.name = f"{base}-{'t' if dist == 't' else 'n'}"

    # ------------------------------------------------------------------ fit

    def fit(self, returns: np.ndarray) -> None:
        r = self._validate_window(returns) * _PCT
        v = float(np.var(r))

        def unpack(x: np.ndarray) -> tuple[float, float, float, float, float | None]:
            omega, alpha, beta = x[0], x[1], x[2]
            i = 3
            gamma = 0.0
            if self.leverage:
                gamma = x[i]
                i += 1
            nu = x[i] if self.dist == "t" else None
            return omega, alpha, beta, gamma, nu

        def nll(x: np.ndarray) -> float:
            omega, alpha, beta, gamma, nu = unpack(x)
            if alpha + beta + 0.5 * gamma >= _MAX_PERSISTENCE:
                return 1e10
            s2, _ = _filter_variance(r, omega, alpha, beta, gamma, v)
            if self.dist == "t":
                return _nll_student_t(s2, r, nu)  # type: ignore[arg-type]
            return _nll_normal(s2, r)

        starts, bounds = self._starts_and_bounds(v)
        best: optimize.OptimizeResult | None = None
        for x0 in starts:
            res = optimize.minimize(nll, np.asarray(x0), method="L-BFGS-B", bounds=bounds)
            if (
                best is None
                or (res.success and not best.success)
                or (res.success == best.success and res.fun < best.fun)
            ):
                best = res
        assert best is not None

        if not best.success:
            # L-BFGS-B occasionally stalls on flat likelihoods (small alpha or
            # gamma near a bound); polish with derivative-free Nelder-Mead,
            # rejecting any excursion outside the box.
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])

            def nll_boxed(x: np.ndarray) -> float:
                if np.any(x < lo) or np.any(x > hi):
                    return 1e10
                return nll(x)

            polish = optimize.minimize(
                nll_boxed,
                best.x,
                method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10},
            )
            if polish.fun <= best.fun and np.isfinite(polish.fun):
                best = polish
        omega, alpha, beta, gamma, nu = unpack(best.x)
        self.params = GarchParams(
            omega=omega,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            nu=nu,
            loglik=-float(best.fun),
            converged=bool(best.success),
        )
        if not best.success:
            raise RuntimeError(f"{self.name}: MLE failed to converge")

    def _starts_and_bounds(self, v: float):
        base_starts = [(0.05, 0.90), (0.10, 0.80), (0.02, 0.95)]
        starts = []
        for a, b in base_starts:
            g = 0.05 if self.leverage else None
            p = a + b + (0.5 * g if g else 0.0)
            omega0 = max(v * (1.0 - p), 1e-6)
            x0 = [omega0, a, b]
            if self.leverage:
                x0.append(g)
            if self.dist == "t":
                x0.append(8.0)
            starts.append(x0)
        bounds = [(1e-8, 10.0 * v + 1e-6), (0.0, 0.5), (0.0, _MAX_PERSISTENCE)]
        if self.leverage:
            bounds.append((0.0, 0.5))
        if self.dist == "t":
            bounds.append((2.1, 100.0))
        return starts, bounds

    # ------------------------------------------------------------- forecast

    def forecast(
        self,
        returns: np.ndarray,
        horizons: tuple[int, ...],
        alphas: tuple[float, ...],
    ) -> list[RiskForecast]:
        if self.params is None:
            raise RuntimeError(f"{self.name}: call fit() before forecast()")
        p = self.params
        r = self._validate_window(returns) * _PCT
        _, s2_next = _filter_variance(
            r, p.omega, p.alpha, p.beta, p.gamma, float(np.var(r))
        )

        pers = min(p.persistence, _MAX_PERSISTENCE - 1e-6)
        s2_bar = p.omega / (1.0 - pers)
        out: list[RiskForecast] = []
        for h in horizons:
            steps = s2_bar + pers ** np.arange(h) * (s2_next - s2_bar)
            agg_pct2 = float(np.sum(steps))
            agg = agg_pct2 / (_PCT * _PCT)  # back to decimal^2
            sigma = float(np.sqrt(agg))
            var_d: dict[float, float] = {}
            es_d: dict[float, float] = {}
            for a in alphas:
                if self.dist == "t":
                    var_d[a], es_d[a] = var_es_student_t(sigma, p.nu, a)  # type: ignore[arg-type]
                else:
                    var_d[a], es_d[a] = var_es_normal(sigma, a)
            out.append(RiskForecast(horizon=h, variance=agg, var=var_d, es=es_d))
        return out
