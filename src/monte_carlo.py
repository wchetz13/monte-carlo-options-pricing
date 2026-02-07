import numpy as np


def monte_carlo_option_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_sims: int = 100000,
    option_type: str = "call",
):
    """
    Monte Carlo pricing of European options under GBM.
    """

    # Generate random normal shocks
    Z = np.random.standard_normal(n_sims)

    # Simulate terminal prices under GBM
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # Compute payoff
    if option_type == "call":
        payoff = np.maximum(ST - K, 0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Discount back to present value
    price = np.exp(-r * T) * np.mean(payoff)

    return price


if __name__ == "__main__":
    price = monte_carlo_option_price(
        S0=100,
        K=105,
        T=1.0,
        r=0.05,
        sigma=0.2,
        n_sims=100000,
        option_type="call",
    )

    print("Estimated Call Price:", price)
