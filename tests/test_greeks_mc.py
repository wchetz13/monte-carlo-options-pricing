import math

import pytest

from src.greeks_mc import (
    bs_greeks, bump_and_revalue, likelihood_ratio, pathwise, pathwise_lr_gamma,
)

P = dict(S0=100.0, K=105.0, T=1.0, r=0.05, sigma=0.2)
N = 200_000


def _z(est, truth):
    return abs(est.value - truth) / est.se


def test_bs_greeks_match_finite_differences():
    h = 1e-4
    for kind in ("call", "put", "digital"):
        g = bs_greeks(**P, kind=kind)
        up = bs_greeks(**{**P, "S0": P["S0"] + h}, kind=kind)
        dn = bs_greeks(**{**P, "S0": P["S0"] - h}, kind=kind)
        assert (up["price"] - dn["price"]) / (2 * h) == pytest.approx(g["delta"], rel=1e-5)
        assert (up["price"] - 2 * g["price"] + dn["price"]) / h**2 == pytest.approx(g["gamma"], rel=1e-3)
        vu = bs_greeks(**{**P, "sigma": P["sigma"] + h}, kind=kind)
        vd = bs_greeks(**{**P, "sigma": P["sigma"] - h}, kind=kind)
        assert (vu["price"] - vd["price"]) / (2 * h) == pytest.approx(g["vega"], rel=1e-5)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("greek", ["delta", "vega"])
def test_pathwise_unbiased(kind, greek):
    est = pathwise(greek, **P, n_pairs=N, kind=kind)
    assert _z(est, bs_greeks(**P, kind=kind)[greek]) < 4


@pytest.mark.parametrize("kind", ["call", "put", "digital"])
@pytest.mark.parametrize("greek", ["delta", "gamma", "vega"])
def test_likelihood_ratio_unbiased(kind, greek):
    est = likelihood_ratio(greek, **P, n_pairs=N, kind=kind)
    assert _z(est, bs_greeks(**P, kind=kind)[greek]) < 4


@pytest.mark.parametrize("kind", ["call", "put"])
def test_mixed_gamma_unbiased_and_lower_variance(kind):
    mixed = pathwise_lr_gamma(**P, n_pairs=N, kind=kind)
    pure = likelihood_ratio("gamma", **P, n_pairs=N, kind=kind)
    assert _z(mixed, bs_greeks(**P, kind=kind)["gamma"]) < 4
    assert mixed.se < pure.se


def test_bump_crn_close_to_analytic():
    for greek in ("delta", "gamma", "vega"):
        est = bump_and_revalue(greek, **P, n_pairs=N, bump=0.5 if greek != "vega" else 0.005)
        truth = bs_greeks(**P)[greek]
        # finite-difference bias is O(h^2); allow it on top of 4 SE
        assert abs(est.value - truth) < 4 * est.se + 0.02 * abs(truth)


def test_crn_reduces_bump_variance():
    crn = bump_and_revalue("delta", **P, n_pairs=50_000)
    indep = bump_and_revalue("delta", **P, n_pairs=50_000, common_random_numbers=False)
    assert indep.se > 5 * crn.se


def test_digital_bump_crn_variance_dominates_lr():
    lr = likelihood_ratio("delta", **P, n_pairs=50_000, kind="digital")
    bump = bump_and_revalue("delta", **P, n_pairs=50_000, kind="digital")
    assert bump.se > lr.se


def test_pathwise_rejects_digital_and_gamma():
    with pytest.raises(ValueError):
        pathwise("delta", **P, kind="digital")
    with pytest.raises(ValueError):
        pathwise("gamma", **P)


def test_antithetic_se_is_finite_and_positive():
    est = likelihood_ratio("delta", **P, n_pairs=1000)
    assert math.isfinite(est.se) and est.se > 0
