import time
import numpy as np
import matplotlib.pyplot as plt

from src.black_scholes import bs_call_price, bs_put_price


def monte_carlo_option_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_sims: int = 100_000,
    option_type: str = "call",
    seed: int | None = 42,
    antithetic: bool = True,
    control_variate: bool = True,
) -> tuple[float, float]:

    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    rng = np.random.default_rng(seed)

    # --- Antithetic variates ---
    half = n_sims // 2
    Z_half = rng.standard_normal(half)
    Z = np.concatenate([Z_half, -Z_half])

    if Z.size < n_sims:
        Z = np.concatenate([Z, rng.standard_normal(1)])

    # --- GBM terminal price ---
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    ST = S0 * np.exp(drift + diffusion)

    # --- Payoff ---
    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    disc = np.exp(-r * T)
    discounted_payoff = disc * payoff

    # --- Control variate ---
    if control_variate:
        X = disc * ST
        EX = S0

        Y = discounted_payoff
        beta = np.cov(Y, X)[0, 1] / np.var(X)

        Y_cv = Y - beta * (X - EX)

        price = np.mean(Y_cv)
        se = np.std(Y_cv, ddof=1) / np.sqrt(len(Y_cv))
        return price, se

    price = np.mean(discounted_payoff)
    se = np.std(discounted_payoff, ddof=1) / np.sqrt(len(discounted_payoff))
    return price, se


def _bs_price(S0, K, T, r, sigma, option_type):
    if option_type == "call":
        return bs_call_price(S0, K, r, sigma, T)
    return bs_put_price(S0, K, r, sigma, T)


def convergence_plot(S0, K, T, r, sigma, option_type="call"):
    sims = np.linspace(1_000, 200_000, 25, dtype=int)
    prices = []

    for n in sims:
        price, _ = monte_carlo_option_price(
            S0, K, T, r, sigma,
            n_sims=n,
            option_type=option_type,
        )
        prices.append(price)

    bs = _bs_price(S0, K, T, r, sigma, option_type)

    plt.figure(figsize=(10, 6))
    plt.plot(sims, prices, label="Monte Carlo Price")
    plt.axhline(bs, linestyle="--", label="Black–Scholes")
    plt.xlabel("Number of Simulations")
    plt.ylabel("Option Price")
    plt.title(f"Convergence of Monte Carlo ({option_type.title()})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    S0 = 100.0
    K = 105.0
    T = 1.0
    r = 0.05
    sigma = 0.20
    n_sims = 200_000

    # --- Call ---
    t0 = time.perf_counter()
    mc_call, se_call = monte_carlo_option_price(
        S0, K, T, r, sigma, n_sims, "call"
    )
    call_time = time.perf_counter() - t0
    bs_call = _bs_price(S0, K, T, r, sigma, "call")

    # --- Put ---
    t1 = time.perf_counter()
    mc_put, se_put = monte_carlo_option_price(
        S0, K, T, r, sigma, n_sims, "put"
    )
    put_time = time.perf_counter() - t1
    bs_put = _bs_price(S0, K, T, r, sigma, "put")

    print("\nMonte Carlo Validation vs Black–Scholes")
    print(f"Params: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}, sims={n_sims:,}")
    print("-" * 86)
    print(f"{'Option':<8}{'MC Price':>14}{'MC SE':>12}{'BS Price':>14}{'Abs Error':>14}{'Runtime(s)':>12}")
    print("-" * 86)
    print(f"{'Call':<8}{mc_call:>14.6f}{se_call:>12.6f}{bs_call:>14.6f}{abs(mc_call-bs_call):>14.6f}{call_time:>12.3f}")
    print(f"{'Put':<8}{mc_put:>14.6f}{se_put:>12.6f}{bs_put:>14.6f}{abs(mc_put-bs_put):>14.6f}{put_time:>12.3f}")
    print("-" * 86)

    # Convergence plot (the wow factor)
    convergence_plot(S0, K, T, r, sigma, option_type="call")


if __name__ == "__main__":
    main()
