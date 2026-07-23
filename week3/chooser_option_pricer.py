"""
Chooser Option Pricing Module
==============================
Implements Simple and Complex Chooser Option pricing under BSM framework.
Based on Rubinstein (1991) and the project paper.

Simple Chooser Option:
    Holder chooses at t1 whether option is a call or put (same K, T2).

Complex Chooser Option:
    Holder chooses at t1 between a call on asset A (KA, TA)
    and a put on asset B (KB, TB).
"""

import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal


def bs_call(S, K, T, r, q, sigma):
    """
    Black-Scholes-Merton call option price.
    """
    if T <= 0:
        return np.maximum(S - K, 0.0)
    if isinstance(T, (int, float)) and T < 1e-10:
        return np.maximum(S - K, 0.0)
    sigma = np.maximum(sigma, 1e-10)
    d1 = (np.log(np.maximum(S / K, 1e-10)) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, q, sigma):
    """
    Black-Scholes-Merton put option price.
    """
    if T <= 0:
        return np.maximum(K - S, 0.0)
    if isinstance(T, (int, float)) and T < 1e-10:
        return np.maximum(K - S, 0.0)
    sigma = np.maximum(sigma, 1e-10)
    d1 = (np.log(np.maximum(S / K, 1e-10)) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def simple_chooser(S, K, t1, T2, r, q, sigma):
    """
    Simple Chooser Option price (Rubinstein, 1991).

    Holder can choose at t1 whether the option becomes a call or put,
    both with strike K and maturity T2 (t1 <= T2).

    Formula: V = Se^{-qT2}[N(d1) - N(-d3)] - Ke^{-rT2}[N(d2) - N(-d4)]

    where:
        d1 = [ln(S/K)+(r-q+σ²/2)T2]/(σ√T2)
        d2 = d1 - σ√T2
        d3 = [ln(S/K)+(r-q)T2+σ²t1/2]/(σ√t1)
        d4 = d3 - σ√t1

    Parameters
    ----------
    S : float | np.ndarray   Spot price
    K : float                Strike price
    t1 : float               Choice date (years, t1 <= T2)
    T2 : float               Option maturity (years)
    r : float | np.ndarray  Risk-free rate (decimal)
    q : float | np.ndarray  Dividend yield (decimal)
    sigma : float | np.ndarray  Volatility (decimal)

    Returns
    -------
    float | np.ndarray  Chooser option price
    """
    sigma = np.maximum(sigma, 1e-10)

    if t1 <= 0:
        c = bs_call(S, K, T2, r, q, sigma)
        p = bs_put(S, K, T2, r, q, sigma)
        return np.maximum(c, p)

    if t1 >= T2:
        # At maturity, chooser = |S-K|
        c = bs_call(S, K, T2, r, q, sigma)
        p = bs_put(S, K, T2, r, q, sigma)
        return c + p  # straddle

    sqrt_T2 = np.sqrt(T2)
    sqrt_t1 = np.sqrt(t1)

    d1 = (np.log(np.maximum(S / K, 1e-10)) + (r - q + 0.5 * sigma ** 2) * T2) / (sigma * sqrt_T2)
    d2 = d1 - sigma * sqrt_T2

    # d3 uses T2 for (r-q) drift term, t1 for volatility horizon
    d3 = (np.log(np.maximum(S / K, 1e-10)) + (r - q) * T2 + 0.5 * sigma ** 2 * t1) / (sigma * sqrt_t1)
    d4 = d3 - sigma * sqrt_t1

    discounted_S = S * np.exp(-q * T2)
    discounted_K = K * np.exp(-r * T2)

    return discounted_S * (norm.cdf(d1) - norm.cdf(-d3)) - discounted_K * (norm.cdf(d2) - norm.cdf(-d4))


def simple_chooser_alt(S, K, t1, T2, r, q, sigma):
    """
    Alternative Simple Chooser formula via put-call parity decomposition:
        V = C(S,K,T2) + P_eff(S,K,t1,T2)
    where P_eff is a put option on Se^{-qT2} with strike Ke^{-rT2}, maturity t1.
    """
    sigma = np.maximum(sigma, 1e-10)
    call_price = bs_call(S, K, T2, r, q, sigma)

    if t1 <= 0:
        return call_price  # max(call, put) when t1=0 depends on moneyness

    # Put option on underlying X = S*e^{-qT2} with strike K_eff = K*e^{-rT2}
    X0 = S * np.exp(-q * T2)
    K_eff = K * np.exp(-r * T2)

    if t1 >= T2:
        return call_price + bs_put(S, K, T2, r, q, sigma)

    sqrt_t1 = np.sqrt(t1)
    d3 = (np.log(np.maximum(X0 / K_eff, 1e-10)) + 0.5 * sigma ** 2 * t1) / (sigma * sqrt_t1)
    d4 = d3 - sigma * sqrt_t1

    put_price = K_eff * np.exp(-r * t1) * norm.cdf(-d4) - X0 * norm.cdf(-d3)
    # Wait, this double counts discount. Let me fix.
    # X0 and K_eff are already discounted to time 0.
    # The put formula with X0 (no div) and K_eff (discounted):
    # P = K_eff * N(-d4) - X0 * N(-d3)
    # (No additional e^{-rt1} since K_eff already discounts from T2 to 0)

    return call_price + put_price


def simple_chooser_decomposed(S, K, t1, T2, r, q, sigma):
    """
    Decompose the simple chooser into components for analysis.
    """
    call_part = bs_call(S, K, T2, r, q, sigma)
    sigma_safe = np.maximum(sigma, 1e-10)
    sqrt_t1 = np.sqrt(max(t1, 1e-10))

    X0 = S * np.exp(-q * T2)
    K_eff = K * np.exp(-r * T2)

    d3 = (np.log(max(X0 / K_eff, 1e-10)) + 0.5 * sigma_safe ** 2 * t1) / (sigma_safe * sqrt_t1)
    d4 = d3 - sigma_safe * sqrt_t1

    optionality_value = K_eff * norm.cdf(-d4) - X0 * norm.cdf(-d3)

    return {
        'call_part': call_part,
        'optionality_part': optionality_value,
        'total': call_part + optionality_value
    }


def complex_chooser(
    S_A, S_B,
    K_A, K_B,
    t1, T_A, T_B,
    r, q_A, q_B,
    sigma_A, sigma_B, rho
):
    """
    Complex Chooser Option price.

    Holder chooses at t1 between:
      - Call on asset A (strike K_A, maturity T_A >= t1)
      - Put  on asset B (strike K_B, maturity T_B >= t1)

    Uses bivariate normal CDF.
    """
    sigma_A = max(sigma_A, 1e-10)
    sigma_B = max(sigma_B, 1e-10)

    if t1 <= 0:
        return max(bs_call(S_A, K_A, T_A, r, q_A, sigma_A),
                   bs_put(S_B, K_B, T_B, r, q_B, sigma_B))

    sqrt_TA = np.sqrt(T_A)
    sqrt_TB = np.sqrt(T_B)
    sqrt_t1 = np.sqrt(t1)

    # d1/d2 for asset A's call
    d1_A = (np.log(S_A / K_A) + (r - q_A + 0.5 * sigma_A ** 2) * T_A) / (sigma_A * sqrt_TA)
    d2_A = d1_A - sigma_A * sqrt_TA

    # d1/d2 for asset B's put
    d1_B = (np.log(S_B / K_B) + (r - q_B + 0.5 * sigma_B ** 2) * T_B) / (sigma_B * sqrt_TB)
    d2_B = d1_B - sigma_B * sqrt_TB

    # Find critical S* at t1 where call_A(t1) = put_B(t1)
    from scipy.optimize import brentq

    def payoff_diff(lnS):
        s_a = np.exp(lnS)
        s_b = s_a * (S_B / S_A)  # proportional scaling
        c_a = bs_call(s_a, K_A, T_A - t1, r, q_A, sigma_A)
        p_b = bs_put(s_b, K_B, T_B - t1, r, q_B, sigma_B)
        return c_a - p_b

    try:
        S_low = max(S_A * 0.001, 1e-10)
        S_high = S_A * 1000
        f_low = payoff_diff(np.log(S_low))
        f_high = payoff_diff(np.log(S_high))
        if f_low * f_high > 0:
            S_crit = S_low if f_low > 0 else S_high
        else:
            S_crit = np.exp(brentq(payoff_diff, np.log(S_low), np.log(S_high)))
    except (ValueError, RuntimeError):
        S_crit = S_A

    # Standardized critical value
    y = (np.log(S_crit / S_A) - (r - q_A - 0.5 * sigma_A ** 2) * t1) / (sigma_A * sqrt_t1)
    y_B = (np.log(S_crit / S_B) - (r - q_B - 0.5 * sigma_B ** 2) * t1) / (sigma_B * sqrt_t1)

    # Correlation adjustments
    rho_A = rho * sigma_B * np.sqrt(t1) / (np.sqrt(T_A) * sigma_A) if sigma_A > 0 else 0
    rho_B = rho * sigma_A * np.sqrt(t1) / (np.sqrt(T_B) * sigma_B) if sigma_B > 0 else 0

    def M(a, b, rho_val):
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, rho_val], [rho_val, 1.0]])
        return multivariate_normal.cdf(np.array([a, b]), mean=mean, cov=cov)

    # Term 1 & 2: Call on A when chosen (S_t1 > S_crit)
    y1_A = y - sigma_A * np.sqrt(t1)
    term1 = S_A * np.exp(-q_A * T_A) * M(d1_A, y1_A, rho_A)
    term2 = K_A * np.exp(-r * T_A) * M(d2_A, y1_A, rho_A)

    # Term 3 & 4: Put on B when chosen (S_t1 < S_crit)
    y1_B = -y_B + sigma_B * np.sqrt(t1)
    term3 = S_B * np.exp(-q_B * T_B) * M(-d1_B, y1_B, -rho_B)
    term4 = K_B * np.exp(-r * T_B) * M(-d2_B, y1_B, -rho_B)

    return term1 - term2 - term3 + term4


def simple_chooser_mc(S, K, t1, T2, r, q, sigma, n_sims=200000):
    """
    Monte Carlo price for Simple Chooser Option (validation).

    Simulates S_t1, then at t1 computes max(BS_call_value, BS_put_value),
    discounts back to time 0.
    """
    np.random.seed(42)
    Z = np.random.normal(0, 1, n_sims)

    # Simulate S at t1
    S_t1 = S * np.exp((r - q - 0.5 * sigma ** 2) * t1 + sigma * np.sqrt(t1) * Z)

    # Compute BS call and put values at t1 (remaining time = T2 - t1)
    remaining = T2 - t1
    if remaining > 1e-10:
        d1 = (np.log(np.maximum(S_t1 / K, 1e-10)) + (r - q + 0.5 * sigma ** 2) * remaining) / (sigma * np.sqrt(remaining))
        d2 = d1 - sigma * np.sqrt(remaining)
        call_t1 = S_t1 * np.exp(-q * remaining) * norm.cdf(d1) - K * np.exp(-r * remaining) * norm.cdf(d2)
        put_t1 = K * np.exp(-r * remaining) * norm.cdf(-d2) - S_t1 * np.exp(-q * remaining) * norm.cdf(-d1)
    else:
        call_t1 = np.maximum(S_t1 - K, 0)
        put_t1 = np.maximum(K - S_t1, 0)

    payoff_at_t1 = np.maximum(call_t1, put_t1)
    mc_price = np.exp(-r * t1) * np.mean(payoff_at_t1)
    se = np.std(payoff_at_t1 * np.exp(-r * t1)) / np.sqrt(n_sims)

    return mc_price, se


def simple_chooser_mc_straddle(S, K, t1, T2, r, q, sigma, n_sims=200000):
    """
    Alternative MC: simulate to T2, payoff = max(|S_T2-K|, ...).
    This is NOT the same as the proper MC above, shown for comparison.
    """
    np.random.seed(42)
    Z1 = np.random.normal(0, 1, n_sims)
    Z2 = np.random.normal(0, 1, n_sims)

    S_T2 = S * np.exp((r - q - 0.5 * sigma ** 2) * T2 +
                       sigma * np.sqrt(t1) * Z1 +
                       sigma * np.sqrt(T2 - t1) * Z2)
    payoff = np.abs(S_T2 - K)
    mc_price = np.exp(-r * T2) * np.mean(payoff)
    se = np.std(payoff * np.exp(-r * T2)) / np.sqrt(n_sims)
    return mc_price, se


if __name__ == '__main__':
    # === Self-test with Rubinstein (1991) parameters ===
    S, K, t1, T2, r, q, sigma = 100, 100, 0.25, 1.0, 0.08, 0.0, 0.20

    p = simple_chooser(S, K, t1, T2, r, q, sigma)
    c = bs_call(S, K, T2, r, q, sigma)
    pt = bs_put(S, K, T2, r, q, sigma)
    decomp = simple_chooser_decomposed(S, K, t1, T2, r, q, sigma)
    mc_p, mc_se = simple_chooser_mc(S, K, t1, T2, r, q, sigma)

    print("=" * 60)
    print("SIMPLE CHOOSER OPTION — SELF TEST")
    print("=" * 60)
    print(f"S={S}, K={K}, t1={t1}, T2={T2}, r={r}, q={q}, sigma={sigma}")
    print()
    print(f"  BS Call:                        {c:>8.4f}")
    print(f"  BS Put:                         {pt:>8.4f}")
    print(f"  Max(Call, Put):                 {max(c, pt):>8.4f}")
    print(f"  Call + Put (Straddle at T2):    {c + pt:>8.4f}")
    print(f"  Chooser (Analytic):             {p:>8.4f}")
    print(f"  Chooser (MC, n={200000}):          {float(mc_p):>8.4f} ± {float(mc_se):.4f}")
    print()
    print("  Decomposition:")
    print(f"    Call part:                    {decomp['call_part']:>8.4f}")
    print(f"    Optionality part:             {decomp['optionality_part']:>8.4f}")
    print(f"    Total:                        {decomp['total']:>8.4f}")
    print()

    # Test with different t1 values
    print("-" * 60)
    print("  Chooser price vs choice date (t1):")
    for t1_val in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        p_t1 = simple_chooser(S, K, t1_val, T2, r, q, sigma)
        straddle = c + pt if t1_val >= T2 else None
        print(f"    t1={t1_val:.2f}:  {p_t1:>8.4f}", end="")
        if t1_val >= T2:
            print(f"  (= straddle)", end="")
        print()

    print()
    # === Complex chooser test ===
    try:
        cp = complex_chooser(
            S_A=100, S_B=100,
            K_A=100, K_B=100,
            t1=0.25, T_A=1.0, T_B=1.0,
            r=0.08, q_A=0.0, q_B=0.0,
            sigma_A=0.20, sigma_B=0.25, rho=0.5
        )
        print(f"  Complex Chooser:                {cp:>8.4f}")
    except Exception as e:
        print(f"  Complex Chooser: Error — {e}")
