"""Compare variance-reduction schemes by repeated estimation.

For each scheme the estimator is run ``n_reps`` times with independent seeds;
the empirical std of the price estimates is the true sampling error, which is
compared with the average reported standard error.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from src.monte_carlo import monte_carlo_option_price

SCHEMES: dict[str, dict[str, bool]] = {
    "Plain MC": dict(antithetic=False, control_variate=False),
    "Antithetic": dict(antithetic=True, control_variate=False),
    "Control variate": dict(antithetic=False, control_variate=True),
    "Antithetic + CV": dict(antithetic=True, control_variate=True),
}


def compare_schemes(
    S0: float = 100.0, K: float = 105.0, T: float = 1.0, r: float = 0.05,
    sigma: float = 0.20, option_type: str = "call",
    n_sims: int = 10_000, n_reps: int = 300,
) -> dict[str, dict[str, float]]:
    """Return empirical std, mean reported SE and variance ratio per scheme."""
    out: dict[str, dict[str, float]] = {}
    for name, flags in SCHEMES.items():
        runs = np.array([
            monte_carlo_option_price(S0, K, T, r, sigma, n_sims, option_type,
                                     seed=1000 + i, **flags)
            for i in range(n_reps)
        ])
        out[name] = {"empirical_std": runs[:, 0].std(ddof=1), "mean_se": runs[:, 1].mean()}
    base = out["Plain MC"]["empirical_std"] ** 2
    for v in out.values():
        v["variance_ratio"] = base / v["empirical_std"] ** 2
    return out


def error_vs_paths(path: str = "images/error_vs_paths.png") -> None:
    """Log-log plot of standard error against path count for each scheme."""
    sizes = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
    plt.figure(figsize=(8, 5))
    for name, flags in SCHEMES.items():
        ses = [monte_carlo_option_price(100, 105, 1.0, 0.05, 0.2, n, "call", seed=5, **flags)[1]
               for n in sizes]
        plt.loglog(sizes, ses, marker="o", label=name)
    plt.xlabel("Number of simulations")
    plt.ylabel("Standard error")
    plt.title("Standard error vs. paths (European call)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)


if __name__ == "__main__":
    res = compare_schemes()
    print("| Scheme | Empirical std | Mean reported SE | Variance reduction |")
    print("|---|---|---|---|")
    for k, v in res.items():
        print(f"| {k} | {v['empirical_std']:.5f} | {v['mean_se']:.5f} | {v['variance_ratio']:.1f}x |")
    error_vs_paths()
