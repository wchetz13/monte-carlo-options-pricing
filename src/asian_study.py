"""Variance-reduction study for the Asian option pricer (``python -m src.asian_study``)."""
from __future__ import annotations

import numpy as np

from src.asian import asian_option_price

SCHEMES: dict[str, dict[str, bool]] = {
    "Plain MC": dict(antithetic=False, control_variate=False),
    "Antithetic": dict(antithetic=True, control_variate=False),
    "Geometric CV": dict(antithetic=False, control_variate=True),
    "Antithetic + geometric CV": dict(antithetic=True, control_variate=True),
}


def compare_schemes(
    S0: float = 100.0, K: float = 100.0, T: float = 1.0, r: float = 0.05,
    sigma: float = 0.20, n_steps: int = 50, option_type: str = "call",
    n_sims: int = 10_000, n_reps: int = 200,
) -> dict[str, dict[str, float]]:
    """Empirical std, mean reported SE, mean price and variance ratio per scheme."""
    out: dict[str, dict[str, float]] = {}
    for name, flags in SCHEMES.items():
        runs = np.array([
            asian_option_price(S0, K, T, r, sigma, n_steps, n_sims, option_type, seed=s, **flags)
            for s in range(n_reps)
        ])
        out[name] = dict(mean=runs[:, 0].mean(), std=runs[:, 0].std(ddof=1), se=runs[:, 1].mean())
    base = out["Plain MC"]["std"] ** 2
    for v in out.values():
        v["ratio"] = base / v["std"] ** 2
    return out


def main() -> None:
    res = compare_schemes()
    print("Arithmetic Asian call, S0=K=100, T=1, r=5%, sigma=20%, 50 fixings, 10,000 sims, 200 reps")
    print(f"{'Scheme':<28}{'Mean price':>11}{'Emp. std':>10}{'Mean SE':>10}{'Var. red.':>10}")
    for name, v in res.items():
        print(f"{name:<28}{v['mean']:>11.4f}{v['std']:>10.5f}{v['se']:>10.5f}{v['ratio']:>9.1f}x")


if __name__ == "__main__":
    main()
