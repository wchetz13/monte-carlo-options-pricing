import math

def _norm_cdf(x: float) -> float:
    # Standard normal CDF using erf (no extra dependency)
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call_price(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return max(S0 - K, 0.0)
    if sigma <= 0:
        # Deterministic forward under risk-neutral measure
        forward = S0 * math.exp(r * T)
        return math.exp(-r * T) * max(forward - K, 0.0)

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)

def bs_put_price(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return max(K - S0, 0.0)
    if sigma <= 0:
        forward = S0 * math.exp(r * T)
        return math.exp(-r * T) * max(K - forward, 0.0)

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S0 * _norm_cdf(-d1)
