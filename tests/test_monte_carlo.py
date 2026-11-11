import numpy as np
import pytest

from src.black_scholes import bs_call_price, bs_put_price
from src.monte_carlo import monte_carlo_option_price

P = dict(S0=100.0, K=105.0, T=1.0, r=0.05, sigma=0.20)
BS = {
    "call": bs_call_price(P["S0"], P["K"], P["r"], P["sigma"], P["T"]),
    "put": bs_put_price(P["S0"], P["K"], P["r"], P["sigma"], P["T"]),
}


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("antithetic", [False, True])
@pytest.mark.parametrize("control_variate", [False, True])
def test_within_four_standard_errors_of_black_scholes(option_type, antithetic, control_variate):
    price, se = monte_carlo_option_price(
        **P, n_sims=200_000, option_type=option_type, seed=7,
        antithetic=antithetic, control_variate=control_variate,
    )
    assert se > 0
    assert abs(price - BS[option_type]) < 4 * se


def test_seed_reproducible():
    a = monte_carlo_option_price(**P, n_sims=10_000, seed=1)
    b = monte_carlo_option_price(**P, n_sims=10_000, seed=1)
    assert a == b


def test_invalid_inputs():
    with pytest.raises(ValueError):
        monte_carlo_option_price(**P, option_type="straddle")
    with pytest.raises(ValueError):
        monte_carlo_option_price(**P, n_sims=1)


def test_variance_reduction_lowers_standard_error():
    def se(**kw):
        return monte_carlo_option_price(**P, n_sims=100_000, seed=3, **kw)[1]

    plain = se(antithetic=False, control_variate=False)
    assert se(antithetic=True, control_variate=False) < plain
    assert se(antithetic=False, control_variate=True) < plain
    assert se(antithetic=True, control_variate=True) < plain


def test_reported_se_matches_empirical_spread():
    """The reported SE must track the real run-to-run std of the estimator
    (this fails if antithetic pairs are treated as i.i.d. draws)."""
    prices, ses = [], []
    for seed in range(200):
        p, s = monte_carlo_option_price(
            **P, n_sims=2_000, seed=seed, antithetic=True, control_variate=False
        )
        prices.append(p)
        ses.append(s)
    assert np.mean(ses) == pytest.approx(np.std(prices, ddof=1), rel=0.15)
