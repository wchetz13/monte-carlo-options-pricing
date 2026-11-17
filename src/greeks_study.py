"""Reproduces the Greeks tables and figure in the README (``python -m src.greeks_study``)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.greeks_mc import bs_greeks, bump_and_revalue, likelihood_ratio, pathwise, pathwise_lr_gamma

P = dict(S0=100.0, K=105.0, T=1.0, r=0.05, sigma=0.2)
N = 100_000  # antithetic pairs (200,000 paths)


def main_table() -> None:
    bs = bs_greeks(**P)
    rows = [
        ("delta", "pathwise", pathwise("delta", **P, n_pairs=N)),
        ("delta", "likelihood ratio", likelihood_ratio("delta", **P, n_pairs=N)),
        ("delta", "bump (CRN)", bump_and_revalue("delta", **P, n_pairs=N)),
        ("gamma", "likelihood ratio", likelihood_ratio("gamma", **P, n_pairs=N)),
        ("gamma", "pathwise-LR", pathwise_lr_gamma(**P, n_pairs=N)),
        ("gamma", "bump (CRN)", bump_and_revalue("gamma", **P, n_pairs=N)),
        ("vega", "pathwise", pathwise("vega", **P, n_pairs=N)),
        ("vega", "likelihood ratio", likelihood_ratio("vega", **P, n_pairs=N)),
        ("vega", "bump (CRN)", bump_and_revalue("vega", **P, n_pairs=N)),
    ]
    print("| Greek | Method | Estimate | Std. error | Black-Scholes |")
    print("|---|---|---:|---:|---:|")
    for g, m, e in rows:
        print(f"| {g} | {m} | {e.value:.5f} | {e.se:.5f} | {bs[g]:.5f} |")


def digital_table() -> None:
    bs = bs_greeks(**P, kind="digital")["delta"]
    print(f"\nDigital delta (Black-Scholes {bs:.5f}), {N:,} pairs:")
    print("| Method | Estimate | Std. error |")
    print("|---|---:|---:|")
    for name, e in [
        ("likelihood ratio", likelihood_ratio("delta", **P, n_pairs=N, kind="digital")),
        ("bump h=1 (CRN)", bump_and_revalue("delta", **P, n_pairs=N, kind="digital", bump=1.0)),
        ("bump h=1 (independent)", bump_and_revalue("delta", **P, n_pairs=N, kind="digital", bump=1.0, common_random_numbers=False)),
    ]:
        print(f"| {name} | {e.value:.5f} | {e.se:.5f} |")


def figure(path: str = "images/greeks_error.png") -> None:
    ns = np.array([1_000, 4_000, 16_000, 64_000, 256_000])
    series = {
        "Pathwise delta": lambda n: pathwise("delta", **P, n_pairs=n),
        "LR delta": lambda n: likelihood_ratio("delta", **P, n_pairs=n),
        "Bump delta (CRN)": lambda n: bump_and_revalue("delta", **P, n_pairs=n),
        "LR gamma": lambda n: likelihood_ratio("gamma", **P, n_pairs=n),
        "Pathwise-LR gamma": lambda n: pathwise_lr_gamma(**P, n_pairs=n),
    }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, fn in series.items():
        truth = bs_greeks(**P)[label.split()[-2 if "(" in label else -1].lower()]
        ax.loglog(ns, [fn(int(n)).se / abs(truth) for n in ns], marker="o", label=label)
    ax.loglog(ns, 1.0 / np.sqrt(ns) * 3, "k:", label=r"$\propto 1/\sqrt{n}$")
    ax.set_xlabel("antithetic pairs")
    ax.set_ylabel("standard error / true value")
    ax.set_title("Relative standard error of MC Greeks (call, K=105)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main_table()
    digital_table()
    figure()
