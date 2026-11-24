"""Discretely monitored Asian options under GBM with a geometric-Asian control variate.

The arithmetic-average Asian option has no closed form, but the *geometric*
average of a lognormal path is itself lognormal, so the geometric Asian option
is priced exactly. Its discounted payoff is highly correlated with the
arithmetic payoff and therefore makes an excellent control variate.
"""
from __future__ import annotations

import math

import numpy as np

from src.black_scholes import _norm_cdf


def geometric_asian_price(
    S0: float, K: float, T: float, r: float, sigma: float,
    n_steps: int, option_type: str = "call",
) -> float:
    """Closed-form price of a discretely monitored geometric-average Asian option.

    With monitoring dates ``t_i = iT/n`` (i = 1..n), ``ln G = mean(ln S_{t_i})`` is
    normal with mean ``m = ln S0 + (r - sigma^2/2) T (n+1)/(2n)`` and variance
    ``v = sigma^2 T (n+1)(2n+1)/(6 n^2)``.
    """
    _check_type(option_type)
    n = n_steps
    m = math.log(S0) + (r - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    v = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    sd = math.sqrt(v)
    d2 = (m - math.log(K)) / sd
    d1 = d2 + sd
    disc = math.exp(-r * T)
    fwd = math.exp(m + 0.5 * v)
    if option_type == "call":
        return disc * (fwd * _norm_cdf(d1) - K * _norm_cdf(d2))
    return disc * (K * _norm_cdf(-d2) - fwd * _norm_cdf(-d1))


def _check_type(option_type: str) -> None:
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")


def _paths(S0: float, T: float, r: float, sigma: float, Z: np.ndarray) -> np.ndarray:
    """GBM prices at the monitoring dates from shocks ``Z`` of shape (n_paths, n_steps)."""
    dt = T / Z.shape[1]
    log_inc = (r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z
    return S0 * np.exp(np.cumsum(log_inc, axis=1))


def asian_option_price(
    S0: float, K: float, T: float, r: float, sigma: float,
    n_steps: int = 50, n_sims: int = 100_000, option_type: str = "call",
    seed: int | None = 42, antithetic: bool = True, control_variate: bool = True,
) -> tuple[float, float]:
    """Monte Carlo price of a discretely monitored arithmetic-average Asian option.

    Args:
        S0, K, T, r, sigma: spot, strike, maturity, rate, volatility.
        n_steps: number of equally spaced monitoring dates.
        n_sims: simulated paths (rounded down to even when ``antithetic``).
        option_type: ``"call"`` or ``"put"``.
        antithetic: simulate shock pairs ``(Z, -Z)`` and average each pair.
        control_variate: use the discounted geometric-Asian payoff, whose exact
            price is known, with a sample-estimated optimal coefficient.

    Returns:
        ``(price, standard_error)``; the error is taken over independent pair
        means when antithetic sampling is on.
    """
    _check_type(option_type)
    if n_steps < 1 or n_sims < 2:
        raise ValueError("n_steps must be >= 1 and n_sims >= 2")

    rng = np.random.default_rng(seed)
    disc = math.exp(-r * T)
    sign = 1.0 if option_type == "call" else -1.0

    def sample(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        S = _paths(S0, T, r, sigma, Z)
        arith = np.maximum(sign * (S.mean(axis=1) - K), 0.0)
        geo = np.maximum(sign * (np.exp(np.log(S).mean(axis=1)) - K), 0.0)
        return disc * arith, disc * geo

    if antithetic:
        Z = rng.standard_normal((n_sims // 2, n_steps))
        ya, xa = sample(Z)
        yb, xb = sample(-Z)
        Y, X = 0.5 * (ya + yb), 0.5 * (xa + xb)
    else:
        Y, X = sample(rng.standard_normal((n_sims, n_steps)))

    if control_variate:
        beta = np.cov(Y, X, ddof=1)[0, 1] / np.var(X, ddof=1)
        Y = Y - beta * (X - geometric_asian_price(S0, K, T, r, sigma, n_steps, option_type))

    return float(np.mean(Y)), float(np.std(Y, ddof=1) / np.sqrt(Y.size))
