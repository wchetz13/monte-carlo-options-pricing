# Monte Carlo Options Pricing

A high-performance Monte Carlo engine for pricing European options under Geometric Brownian Motion (GBM).  
This project demonstrates numerical methods, stochastic simulation, and vectorized Python applied to quantitative finance.

---

## Overview

This model simulates thousands of asset price paths to estimate the fair value of European call and put options.  
It is built using fully vectorized NumPy operations for speed and numerical accuracy.

The project showcases:

- Stochastic differential equation simulation
- Risk-neutral valuation
- Monte Carlo estimation
- Vectorization for performance
- Clean quantitative implementation

---

## Model

We simulate asset paths under **Geometric Brownian Motion (GBM)**:

S(t + Δt) = S(t) · exp[(r − 0.5·σ²)·Δt + σ·√Δt·Z]

where:

- r = risk-free rate  
- σ = volatility  
- Z ~ N(0,1)

The option price is estimated as:

Price = e^(−rT) · average(payoff)

---

## Project Structure
