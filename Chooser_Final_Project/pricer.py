"""
Chooser Option Pricer
=====================
Black-Scholes-Merton chooser pricing (Rubinstein, 1991), validated in Week 3
against the reference paper (analytic vs 200k-path Monte Carlo bias ~0.011%).

Only the functions the pricing engine actually uses are kept; the Monte-Carlo
variants and the complex-chooser formula used purely for Week-3 validation were
removed.

Dependencies: numpy, scipy.
"""

import numpy as np
from scipy.stats import norm


def bs_call(S, K, T, r, q, sigma):
    """Black-Scholes-Merton European call price."""
    if T <= 0:
        return np.maximum(S - K, 0.0)
    sigma = np.maximum(sigma, 1e-10)
    d1 = (np.log(np.maximum(S / K, 1e-10)) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, q, sigma):
    """Black-Scholes-Merton European put price."""
    if T <= 0:
        return np.maximum(K - S, 0.0)
    sigma = np.maximum(sigma, 1e-10)
    d1 = (np.log(np.maximum(S / K, 1e-10)) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def simple_chooser(S, K, t1, T2, r, q, sigma):
    """
    Simple chooser option price (Rubinstein, 1991).

    The holder can choose at t1 whether the option becomes a call or a put,
    both with strike K and maturity T2 (t1 <= T2):

        V = S e^{-qT2}[N(d1) - N(-d3)] - K e^{-rT2}[N(d2) - N(-d4)]

    Edge cases: t1=0 -> max(call, put); t1=T2 -> call + put (straddle).

    S, r, q, sigma may be arrays (broadcast element-wise).
    """
    sigma = np.maximum(sigma, 1e-10)
    if t1 <= 0:
        return np.maximum(bs_call(S, K, T2, r, q, sigma),
                          bs_put(S, K, T2, r, q, sigma))
    if t1 >= T2:
        return bs_call(S, K, T2, r, q, sigma) + bs_put(S, K, T2, r, q, sigma)

    sqrt_T2, sqrt_t1 = np.sqrt(T2), np.sqrt(t1)
    log_sk = np.log(np.maximum(S / K, 1e-10))
    d1 = (log_sk + (r - q + 0.5 * sigma ** 2) * T2) / (sigma * sqrt_T2)
    d2 = d1 - sigma * sqrt_T2
    d3 = (log_sk + (r - q) * T2 + 0.5 * sigma ** 2 * t1) / (sigma * sqrt_t1)
    d4 = d3 - sigma * sqrt_t1

    return (S * np.exp(-q * T2) * (norm.cdf(d1) - norm.cdf(-d3))
            - K * np.exp(-r * T2) * (norm.cdf(d2) - norm.cdf(-d4)))


if __name__ == '__main__':
    # Quick self-check against the Week-3 validated value (paper parameters).
    p = simple_chooser(S=156.70, K=150.0, t1=0.5, T2=1.0, r=0.0015, q=0.0233, sigma=0.282)
    print(f'simple_chooser(paper params) = ${p:.4f}  (expected ~$29.13)')
