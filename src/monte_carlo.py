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
    """Price a European option under GBM by Monte Carlo.

    Args:
        S0, K, T, r, sigma: spot, strike, maturity (years), risk-free rate, volatility.
        n_sims: number of simulated terminal prices (rounded down to an even
            number when ``antithetic`` is True).
        option_type: ``"call"`` or ``"put"``.
        seed: RNG seed (``None`` for non-deterministic draws).
        antithetic: simulate pairs ``(Z, -Z)``.
        control_variate: use ``exp(-rT) * S_T`` (known mean ``S0``) as control.

    Returns:
        ``(price, standard_error)``. With antithetic sampling the two members of
        a pair are negatively correlated, so the estimator averages each pair
        and the standard error is computed over the independent pair means.
        Treating all ``n_sims`` draws as i.i.d. would overstate the error.
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if n_sims < 2:
        raise ValueError("n_sims must be at least 2")

    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)
    drift = (r - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)

    def sample(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ST = S0 * np.exp(drift + vol * Z)
        payoff = np.maximum(ST - K, 0.0) if option_type == "call" else np.maximum(K - ST, 0.0)
        return disc * payoff, disc * ST

    if antithetic:
        Z = rng.standard_normal(n_sims // 2)
        y_a, x_a = sample(Z)
        y_b, x_b = sample(-Z)
        Y, X = 0.5 * (y_a + y_b), 0.5 * (x_a + x_b)  # independent pair means
    else:
        Y, X = sample(rng.standard_normal(n_sims))

    if control_variate:
        beta = np.cov(Y, X, ddof=1)[0, 1] / np.var(X, ddof=1)
        Y = Y - beta * (X - S0)  # E[X] = S0 under the risk-neutral measure

    return float(np.mean(Y)), float(np.std(Y, ddof=1) / np.sqrt(Y.size))


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
