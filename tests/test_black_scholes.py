import math

import pytest

from src.black_scholes import bs_call_price, bs_put_price

S0, K, T, R, SIGMA = 100.0, 105.0, 1.0, 0.05, 0.20


def test_known_value():
    # Reference value for S=100, K=105, T=1, r=5%, sigma=20%
    assert bs_call_price(S0, K, R, SIGMA, T) == pytest.approx(8.0214, abs=1e-3)


@pytest.mark.parametrize("K_", [80.0, 100.0, 105.0, 130.0])
def test_put_call_parity(K_):
    c = bs_call_price(S0, K_, R, SIGMA, T)
    p = bs_put_price(S0, K_, R, SIGMA, T)
    assert c - p == pytest.approx(S0 - K_ * math.exp(-R * T), abs=1e-10)


def test_expired_and_zero_vol_limits():
    assert bs_call_price(110.0, 100.0, R, SIGMA, 0.0) == 10.0
    assert bs_put_price(90.0, 100.0, R, SIGMA, 0.0) == 10.0
    fwd = S0 * math.exp(R * T)
    assert bs_call_price(S0, 90.0, R, 0.0, T) == pytest.approx(math.exp(-R * T) * (fwd - 90.0))
    assert bs_put_price(S0, 90.0, R, 0.0, T) == 0.0


def test_call_monotone_in_vol():
    prices = [bs_call_price(S0, K, R, s, T) for s in (0.1, 0.2, 0.3, 0.4)]
    assert prices == sorted(prices)
