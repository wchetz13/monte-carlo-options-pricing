"""Monte Carlo Greeks under GBM: pathwise, likelihood-ratio and bump-and-revalue.

All estimators use the exact terminal law
``S_T = S0 exp((r - sigma^2/2) T + sigma sqrt(T) Z)`` with antithetic pairs
``(Z, -Z)``. Each estimator returns a point estimate and a standard error
computed from the per-pair averages, so the error is valid for the antithetic
sample (pairs are independent, individual paths are not).

Payoffs supported: ``"call"``, ``"put"`` and ``"digital"`` (cash-or-nothing call,
pays 1 if ``S_T > K``). The pathwise method requires a Lipschitz payoff, so it is
not available for the digital.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

PAYOFFS = ("call", "put", "digital")


@dataclass(frozen=True)
class Estimate:
    """A Monte Carlo estimate with its standard error."""

    value: float
    se: float


def _payoff(ST: np.ndarray, K: float, kind: str) -> np.ndarray:
    if kind == "call":
        return np.maximum(ST - K, 0.0)
    if kind == "put":
        return np.maximum(K - ST, 0.0)
    if kind == "digital":
        return (ST > K).astype(float)
    raise ValueError(f"payoff must be one of {PAYOFFS}, got {kind!r}")


def _shocks(n_pairs: int, seed: int | None) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n_pairs)


def _terminal(S0: float, T: float, r: float, sigma: float, Z: np.ndarray) -> np.ndarray:
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)


def _summarise(f_plus: np.ndarray, f_minus: np.ndarray) -> Estimate:
    """Mean and SE of per-path values for antithetic shocks (Z, -Z)."""
    pair = 0.5 * (f_plus + f_minus)
    return Estimate(float(pair.mean()), float(pair.std(ddof=1) / math.sqrt(pair.size)))


def _per_path(fn, Z: np.ndarray) -> Estimate:
    return _summarise(fn(Z), fn(-Z))


def bs_greeks(S0: float, K: float, T: float, r: float, sigma: float, kind: str = "call") -> dict[str, float]:
    """Closed-form Black-Scholes price, delta, gamma and vega (per unit vol)."""
    st = sigma * math.sqrt(T)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / st
    d2 = d1 - st
    disc = math.exp(-r * T)
    if kind == "call":
        price = S0 * norm.cdf(d1) - K * disc * norm.cdf(d2)
        delta = norm.cdf(d1)
    elif kind == "put":
        price = K * disc * norm.cdf(-d2) - S0 * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
    elif kind == "digital":
        price = disc * norm.cdf(d2)
        delta = disc * norm.pdf(d2) / (S0 * st)
    else:
        raise ValueError(f"payoff must be one of {PAYOFFS}, got {kind!r}")
    if kind == "digital":
        gamma = -disc * norm.pdf(d2) * d1 / (S0**2 * sigma**2 * T)
        vega = -disc * norm.pdf(d2) * d1 / sigma
    else:
        gamma = norm.pdf(d1) / (S0 * st)
        vega = S0 * norm.pdf(d1) * math.sqrt(T)
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega}


def pathwise(
    greek: str, S0: float, K: float, T: float, r: float, sigma: float,
    n_pairs: int = 100_000, kind: str = "call", seed: int | None = 0,
) -> Estimate:
    """Pathwise (infinitesimal perturbation) estimator of delta or vega.

    Differentiates the payoff along each path: ``dS_T/dS0 = S_T/S0`` and
    ``dS_T/dsigma = S_T (sqrt(T) Z - sigma T)``. Unbiased for call/put; gamma and
    discontinuous payoffs are not supported (the second derivative of the payoff
    is a Dirac mass, so the naive estimator is identically zero).
    """
    if kind not in ("call", "put"):
        raise ValueError("pathwise estimator needs a Lipschitz payoff: 'call' or 'put'")
    if greek not in ("delta", "vega"):
        raise ValueError("pathwise supports 'delta' and 'vega'")
    disc = math.exp(-r * T)
    sign = 1.0 if kind == "call" else -1.0

    def fn(Z: np.ndarray) -> np.ndarray:
        ST = _terminal(S0, T, r, sigma, Z)
        itm = (ST > K) if kind == "call" else (ST < K)
        if greek == "delta":
            return sign * disc * itm * ST / S0
        return sign * disc * itm * ST * (math.sqrt(T) * Z - sigma * T)

    return _per_path(fn, _shocks(n_pairs, seed))


def likelihood_ratio(
    greek: str, S0: float, K: float, T: float, r: float, sigma: float,
    n_pairs: int = 100_000, kind: str = "call", seed: int | None = 0,
) -> Estimate:
    """Likelihood-ratio (score function) estimator of delta, gamma or vega.

    Differentiates the lognormal density instead of the payoff, so it works for
    any payoff, including digitals, at the price of higher variance for smooth ones:
    ``greek = E[disc * payoff * score]`` with score
    ``Z/(S0 sigma sqrt T)`` (delta), ``(Z^2 - 1 - sigma sqrt(T) Z)/(S0^2 sigma^2 T)``
    (gamma) and ``(Z^2 - 1)/sigma - sqrt(T) Z`` (vega).
    """
    disc, sq = math.exp(-r * T), math.sqrt(T)

    def score(Z: np.ndarray) -> np.ndarray:
        if greek == "delta":
            return Z / (S0 * sigma * sq)
        if greek == "gamma":
            return (Z**2 - 1.0 - sigma * sq * Z) / (S0**2 * sigma**2 * T)
        if greek == "vega":
            return (Z**2 - 1.0) / sigma - sq * Z
        raise ValueError("likelihood ratio supports 'delta', 'gamma' and 'vega'")

    def fn(Z: np.ndarray) -> np.ndarray:
        return disc * _payoff(_terminal(S0, T, r, sigma, Z), K, kind) * score(Z)

    return _per_path(fn, _shocks(n_pairs, seed))


def pathwise_lr_gamma(
    S0: float, K: float, T: float, r: float, sigma: float,
    n_pairs: int = 100_000, kind: str = "call", seed: int | None = 0,
) -> Estimate:
    """Mixed gamma: likelihood ratio applied to the pathwise delta.

    ``Gamma = E[disc * 1{ITM} * (S_T/S0) * (Z/(S0 sigma sqrt T) - 1/S0)]`` for a call
    (sign flipped for a put). Much lower variance than the pure LR gamma.
    """
    if kind not in ("call", "put"):
        raise ValueError("mixed estimator needs 'call' or 'put'")
    disc, sq = math.exp(-r * T), math.sqrt(T)
    sign = 1.0 if kind == "call" else -1.0

    def fn(Z: np.ndarray) -> np.ndarray:
        ST = _terminal(S0, T, r, sigma, Z)
        itm = (ST > K) if kind == "call" else (ST < K)
        return sign * disc * itm * (ST / S0) * (Z / (S0 * sigma * sq) - 1.0 / S0)

    return _per_path(fn, _shocks(n_pairs, seed))


def bump_and_revalue(
    greek: str, S0: float, K: float, T: float, r: float, sigma: float,
    n_pairs: int = 100_000, kind: str = "call", seed: int | None = 0,
    bump: float | None = None, common_random_numbers: bool = True,
) -> Estimate:
    """Central finite-difference delta, gamma or vega.

    With ``common_random_numbers=True`` the up/down/base prices share the same shocks
    (the standard practice); with ``False`` each leg gets independent shocks and the
    variance explodes like ``1/bump^2``. ``bump`` is relative for the spot (default
    1% of S0) and absolute for vol (default 0.01).
    """
    if greek not in ("delta", "gamma", "vega"):
        raise ValueError("bump-and-revalue supports 'delta', 'gamma' and 'vega'")
    disc = math.exp(-r * T)
    h = bump if bump is not None else (0.01 * S0 if greek != "vega" else 0.01)

    def price(s0: float, sig: float, Z: np.ndarray) -> np.ndarray:
        return disc * _payoff(_terminal(s0, T, r, sig, Z), K, kind)

    Z = _shocks(n_pairs, seed)
    Zs = [Z, Z, Z] if common_random_numbers else [
        _shocks(n_pairs, None if seed is None else seed + 1 + i) for i in range(3)
    ]

    def fn(sign: float) -> np.ndarray:
        up_args = (S0 + h, sigma) if greek != "vega" else (S0, sigma + h)
        dn_args = (S0 - h, sigma) if greek != "vega" else (S0, sigma - h)
        up = price(*up_args, sign * Zs[0])
        dn = price(*dn_args, sign * Zs[1])
        if greek == "gamma":
            return (up - 2.0 * price(S0, sigma, sign * Zs[2]) + dn) / h**2
        return (up - dn) / (2.0 * h)

    return _summarise(fn(1.0), fn(-1.0))
