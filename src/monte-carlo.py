import numpy as np

def monte_carlo_call_price(S0, K, T, r, sigma, n_paths=100000):
    """
    Monte Carlo pricing of a European Call Option under GBM

    Parameters
    ----------
    S0 : float
        Initial stock price
    K : float
        Strike price
    T : float
        Time to maturity (years)
    r : float
        Risk-free rate
    sigma : float
        Volatility
    n_paths : int
        Number of simulation paths

    Returns
    -------
    float
        Option price
    """

    Z = np.random.standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)

    price = np.exp(-r * T) * np.mean(payoff)
    return price


if __name__ == "__main__":
    price = monte_carlo_call_price(
        S0=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.2
    )
    print(f"Option Price: {price:.4f}")
