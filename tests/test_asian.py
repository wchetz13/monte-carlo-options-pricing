import math

import numpy as np
import pytest

from src.asian import asian_option_price, geometric_asian_price
from src.black_scholes import bs_call_price, bs_put_price

P = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


def test_geometric_closed_form_matches_brute_force():
    """Exact lognormal price vs. very large plain simulation of the geometric payoff."""
    n = 12
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((400_000, n))
    dt = P["T"] / n
    S = P["S0"] * np.exp(np.cumsum((P["r"] - 0.5 * P["sigma"] ** 2) * dt
                                   + P["sigma"] * math.sqrt(dt) * Z, axis=1))
    pay = np.maximum(np.exp(np.log(S).mean(axis=1)) - P["K"], 0.0) * math.exp(-P["r"] * P["T"])
    se = pay.std(ddof=1) / math.sqrt(pay.size)
    assert abs(pay.mean() - geometric_asian_price(**P, n_steps=n)) < 4 * se


def test_single_fixing_reduces_to_black_scholes():
    assert geometric_asian_price(**P, n_steps=1) == pytest.approx(
        bs_call_price(P["S0"], P["K"], P["r"], P["sigma"], P["T"]), rel=1e-12)
    assert geometric_asian_price(**P, n_steps=1, option_type="put") == pytest.approx(
        bs_put_price(P["S0"], P["K"], P["r"], P["sigma"], P["T"]), rel=1e-12)


def test_asian_cheaper_than_vanilla_and_arith_above_geo():
    mc, _ = asian_option_price(**P, n_steps=50, n_sims=100_000)
    assert mc < bs_call_price(P["S0"], P["K"], P["r"], P["sigma"], P["T"])
    assert mc > geometric_asian_price(**P, n_steps=50)  # AM-GM


def test_one_fixing_mc_matches_black_scholes():
    mc, se = asian_option_price(**P, n_steps=1, n_sims=200_000,
                              option_type="put", control_variate=False)
    assert abs(mc - bs_put_price(P["S0"], P["K"], P["r"], P["sigma"], P["T"])) < 4 * se


def test_put_call_parity_for_arithmetic_asian():
    """C - P = e^{-rT}(E[A] - K), with E[A] = S0 * mean(e^{r t_i})."""
    n = 20
    c, sc = asian_option_price(**P, n_steps=n, n_sims=200_000, seed=1)
    p, sp = asian_option_price(**P, n_steps=n, n_sims=200_000, option_type="put", seed=2)
    ea = P["S0"] * np.mean(np.exp(P["r"] * P["T"] * np.arange(1, n + 1) / n))
    parity = math.exp(-P["r"] * P["T"]) * (ea - P["K"])
    assert abs((c - p) - parity) < 4 * math.hypot(sc, sp)


def test_control_variate_is_unbiased_and_cuts_error():
    plain = asian_option_price(**P, n_sims=50_000, antithetic=False, control_variate=False)
    cv = asian_option_price(**P, n_sims=50_000, antithetic=False, control_variate=True)
    assert abs(plain[0] - cv[0]) < 4 * math.hypot(plain[1], cv[1])
    assert cv[1] < plain[1] / 4


def test_reproducible_and_validation():
    assert asian_option_price(**P, n_sims=2_000) == asian_option_price(**P, n_sims=2_000)
    with pytest.raises(ValueError):
        asian_option_price(**P, option_type="straddle")
    with pytest.raises(ValueError):
        asian_option_price(**P, n_steps=0)
