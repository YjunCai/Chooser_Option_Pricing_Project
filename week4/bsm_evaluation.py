"""
BSM Baseline Model Performance Evaluation — Week 4
====================================================
Evaluates the BSM Chooser Option pricing model along four dimensions:

  1. Numerical Accuracy — Analytic vs MC convergence (MAE, RMSE, convergence rate)
  2. Greeks & Risk Sensitivity — Delta, Vega, Rho, Theta, Gamma for Chooser
  3. Parameter Impact Quantification — Tornado / sensitivity decomposition
  4. Market Regime Error Analysis — Pricing errors across bull/bear/high-vol/calm regimes
  5. BSM Assumption Violations — Real-data diagnostics (normality, vol stationarity)

All functions work with the week3 chooser_option_pricer module and the
feature_dataset.csv produced by week2.
"""

import sys
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm, kstest, jarque_bera

warnings.filterwarnings('ignore')

# ── Add week3 to path to reuse pricer ────────────────────────────────────────
WEEK3_PATH = Path(__file__).resolve().parent.parent / 'week3'
if str(WEEK3_PATH) not in sys.path:
    sys.path.insert(0, str(WEEK3_PATH))

from chooser_option_pricer import (
    bs_call, bs_put, simple_chooser, simple_chooser_mc, simple_chooser_decomposed
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Convergence Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def convergence_analysis(
    S: float = 100.0, K: float = 100.0, t1: float = 0.25, T2: float = 1.0,
    r: float = 0.08, q: float = 0.0, sigma: float = 0.20,
    sim_levels: Optional[List[int]] = None, n_trials: int = 10
) -> pd.DataFrame:
    """
    Compare analytic vs MC prices at multiple simulation sizes.

    For each n_sims level, runs n_trials independent MC simulations and
    computes MAE, RMSE, bias, and convergence rate relative to the analytic price.

    Parameters
    ----------
    sim_levels : list of int, optional
        Number of MC paths to test (default: [1K, 5K, 10K, 50K, 100K, 200K, 500K])
    n_trials : int
        Independent MC trials per level for statistical significance

    Returns
    -------
    pd.DataFrame with columns: n_sims, trial, mc_price, mc_se, analytic,
                                abs_error, squared_error, bias_pct
    """
    if sim_levels is None:
        sim_levels = [1_000, 5_000, 10_000, 50_000, 100_000, 200_000, 500_000]

    analytic = simple_chooser(S, K, t1, T2, r, q, sigma)
    rows = []

    for n_sims in sim_levels:
        for trial in range(n_trials):
            mc_price, mc_se = simple_chooser_mc(S, K, t1, T2, r, q, sigma, n_sims=n_sims)
            rows.append({
                'n_sims': n_sims,
                'trial': trial,
                'mc_price': float(mc_price),
                'mc_se': float(mc_se),
                'analytic': float(analytic),
                'abs_error': float(abs(mc_price - analytic)),
                'squared_error': float((mc_price - analytic) ** 2),
                'bias_pct': float((mc_price - analytic) / analytic * 100),
            })

    df = pd.DataFrame(rows)

    # Aggregate per n_sims level
    summary = df.groupby('n_sims').agg(
        mean_mc_price=('mc_price', 'mean'),
        mean_se=('mc_se', 'mean'),
        analytic=('analytic', 'first'),
        mae=('abs_error', 'mean'),
        rmse=('squared_error', lambda x: np.sqrt(x.mean())),
        mean_bias_pct=('bias_pct', 'mean'),
        std_bias_pct=('bias_pct', 'std'),
    ).reset_index()

    # Convergence rate: how RMSE scales with n_sims
    if len(summary) >= 3:
        log_n = np.log(summary['n_sims'].values)
        log_rmse = np.log(summary['rmse'].values)
        # Fit: log(RMSE) ~ beta * log(n_sims) + C, expected beta ≈ -0.5
        A = np.vstack([log_n, np.ones_like(log_n)]).T
        beta, intercept = np.linalg.lstsq(A, log_rmse, rcond=None)[0]
        summary['convergence_rate_beta'] = beta
    else:
        summary['convergence_rate_beta'] = np.nan

    return summary


def convergence_grid(
    S_values: Optional[List[float]] = None,
    sigma_values: Optional[List[float]] = None,
    t1_values: Optional[List[float]] = None,
    K: float = 100.0, T2: float = 1.0, r: float = 0.08, q: float = 0.0,
    n_sims: int = 100_000
) -> pd.DataFrame:
    """
    Sweep over (S, sigma, t1) grid and compute analytic vs MC discrepancy.
    Identifies parameter regions where the analytic model is least accurate.

    Returns
    -------
    pd.DataFrame with columns: S, sigma, t1, analytic, mc_price, mc_se, abs_error, bias_pct
    """
    if S_values is None:
        S_values = [80, 90, 100, 110, 120]
    if sigma_values is None:
        sigma_values = [0.10, 0.20, 0.30, 0.40, 0.50]
    if t1_values is None:
        t1_values = [0.1, 0.25, 0.5, 0.75]

    rows = []
    for S in S_values:
        for sigma in sigma_values:
            for t1 in t1_values:
                if t1 > T2:
                    continue
                analytic = simple_chooser(S, K, t1, T2, r, q, sigma)
                mc_price, mc_se = simple_chooser_mc(S, K, t1, T2, r, q, sigma, n_sims=n_sims)
                rows.append({
                    'S': S,
                    'sigma': sigma,
                    't1': t1,
                    'analytic': float(analytic),
                    'mc_price': float(mc_price),
                    'mc_se': float(mc_se),
                    'abs_error': float(abs(mc_price - analytic)),
                    'bias_pct': float((mc_price - analytic) / analytic * 100) if analytic > 1e-8 else 0.0,
                })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Greeks for Chooser Option
# ═══════════════════════════════════════════════════════════════════════════════

def chooser_delta(S, K, t1, T2, r, q, sigma, eps=1e-4):
    """∂V/∂S — First derivative of Chooser price wrt spot price."""
    v_up = simple_chooser(S * (1 + eps), K, t1, T2, r, q, sigma)
    v_dn = simple_chooser(S * (1 - eps), K, t1, T2, r, q, sigma)
    return (v_up - v_dn) / (2 * S * eps)


def chooser_gamma(S, K, t1, T2, r, q, sigma, eps=1e-4):
    """∂²V/∂S² — Second derivative of Chooser price wrt spot price."""
    v_up = simple_chooser(S * (1 + eps), K, t1, T2, r, q, sigma)
    v_mid = simple_chooser(S, K, t1, T2, r, q, sigma)
    v_dn = simple_chooser(S * (1 - eps), K, t1, T2, r, q, sigma)
    return (v_up - 2 * v_mid + v_dn) / (S * eps) ** 2


def chooser_vega(S, K, t1, T2, r, q, sigma, eps=1e-4):
    """∂V/∂σ — Sensitivity to volatility."""
    v_up = simple_chooser(S, K, t1, T2, r, q, sigma + eps)
    v_dn = simple_chooser(S, K, t1, T2, r, q, sigma - eps)
    return (v_up - v_dn) / (2 * eps)


def chooser_rho(S, K, t1, T2, r, q, sigma, eps=1e-4):
    """∂V/∂r — Sensitivity to risk-free rate."""
    v_up = simple_chooser(S, K, t1, T2, r + eps, q, sigma)
    v_dn = simple_chooser(S, K, t1, T2, r - eps, q, sigma)
    return (v_up - v_dn) / (2 * eps)


def chooser_theta(S, K, t1, T2, r, q, sigma, eps=1e-4):
    """∂V/∂t — Time decay (wrt t1, the choice date)."""
    # Theta here = sensitivity to the choice date t1
    v_up = simple_chooser(S, K, t1 + eps, T2, r, q, sigma)
    v_dn = simple_chooser(S, K, t1 - eps, T2, r, q, sigma)
    return (v_up - v_dn) / (2 * eps)


def compute_all_greeks(S, K, t1, T2, r, q, sigma):
    """
    Compute all major Greeks for the Simple Chooser Option.

    Returns dict with: delta, gamma, vega, rho, theta
    """
    return {
        'delta': chooser_delta(S, K, t1, T2, r, q, sigma),
        'gamma': chooser_gamma(S, K, t1, T2, r, q, sigma),
        'vega': chooser_vega(S, K, t1, T2, r, q, sigma),
        'rho': chooser_rho(S, K, t1, T2, r, q, sigma),
        'theta_t1': chooser_theta(S, K, t1, T2, r, q, sigma),
    }


def greeks_surface_scan(
    param_name: str,
    param_range: np.ndarray,
    S: float = 100.0, K: float = 100.0, t1: float = 0.25, T2: float = 1.0,
    r: float = 0.08, q: float = 0.0, sigma: float = 0.20
) -> pd.DataFrame:
    """
    Scan how a given parameter affects all Greeks.
    param_name: one of 'S', 'K', 't1', 'sigma', 'r', 'q'
    """
    rows = []
    for val in param_range:
        kwargs = {'S': S, 'K': K, 't1': t1, 'T2': T2, 'r': r, 'q': q, 'sigma': sigma}
        kwargs[param_name] = val
        price = simple_chooser(**kwargs)
        greeks = compute_all_greeks(**kwargs)
        greeks['price'] = price
        greeks[param_name] = val
        rows.append(greeks)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Parameter Sensitivity / Impact Quantification
# ═══════════════════════════════════════════════════════════════════════════════

def parameter_impact_analysis(
    base_params: Optional[Dict[str, float]] = None,
    shock_pct: float = 10.0
) -> pd.DataFrame:
    """
    Quantify how each parameter ±shock_pct changes the Chooser price.

    Parameters
    ----------
    base_params : dict, optional
        Baseline parameter set. Defaults to reference paper values.
    shock_pct : float
        Percentage shock to apply (±).

    Returns
    -------
    pd.DataFrame with columns: param, base_value, low_value, high_value,
                               price_low, price_high, price_range, range_pct
    """
    if base_params is None:
        base_params = {'S': 156.7, 'K': 150.0, 't1': 0.5, 'T2': 1.0,
                       'r': 0.0015, 'q': 0.0233, 'sigma': 0.282}

    base_price = simple_chooser(**base_params)

    rows = []
    for param in ['S', 'sigma', 'r', 'q', 't1', 'K']:
        base_val = base_params[param]
        shock = base_val * shock_pct / 100.0

        # Ensure non-negative for valid params
        low_val = max(base_val - shock, 1e-6) if param != 't1' else max(base_val - shock, 0.0)
        low_val = min(low_val, base_params['T2']) if param == 't1' else low_val
        high_val = base_val + shock

        p_low = dict(base_params)
        p_high = dict(base_params)
        p_low[param] = low_val
        p_high[param] = high_val

        price_low = simple_chooser(**p_low)
        price_high = simple_chooser(**p_high)

        rows.append({
            'param': param,
            'base_value': base_val,
            'low_value': low_val,
            'high_value': high_val,
            'price_low': float(price_low),
            'price_high': float(price_high),
            'price_range': float(abs(price_high - price_low)),
            'range_pct': float(abs(price_high - price_low) / base_price * 100),
            'direction': 'up' if price_high > price_low else 'down',
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Market Regime Classification & Error Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def classify_market_regime(df: pd.DataFrame) -> pd.Series:
    """
    Classify each trading day into a market regime.

    Regimes:
      - BULL:    S > SMA_252 AND vol_21d < median(vol_21d)
      - BEAR:    S < SMA_252 AND vol_21d > median(vol_21d)
      - HIGH_VOL: vol_21d > 75th percentile
      - CALM:    vol_21d < 25th percentile AND sentiment_score > 0.5
      - NORMAL:  everything else

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: 'close_jpm', 'vol_21d', 'sentiment_score'

    Returns
    -------
    pd.Series of regime labels (same index as df)
    """
    sma_252 = df['close_jpm'].rolling(252, min_periods=60).mean()
    vol_median = df['vol_21d'].median()
    vol_q75 = df['vol_21d'].quantile(0.75)
    vol_q25 = df['vol_21d'].quantile(0.25)

    conditions = [
        (df['close_jpm'] > sma_252) & (df['vol_21d'] <= vol_median),
        (df['close_jpm'] < sma_252) & (df['vol_21d'] > vol_median),
        (df['vol_21d'] > vol_q75),
        (df['vol_21d'] < vol_q25) & (df['sentiment_score'] > 0.5),
    ]
    labels = ['BULL', 'BEAR', 'HIGH_VOL', 'CALM']

    regime = pd.Series('NORMAL', index=df.index)
    for cond, label in zip(conditions, labels):
        regime[cond] = label

    return regime


def regime_error_analysis(
    df_prices: pd.DataFrame,
    K: float = 150.0, t1: float = 0.5, T2: float = 1.0,
    mc_nsims: int = 100_000
) -> pd.DataFrame:
    """
    Compare analytic Chooser prices vs MC across market regimes.

    Parameters
    ----------
    df_prices : pd.DataFrame
        Must contain columns: 'date', 'S', 'r', 'q', 'sigma'

    Returns
    -------
    pd.DataFrame with regime-level error statistics
    """
    results = []
    for _, row in df_prices.iterrows():
        S = row['S']
        r_val = row['r']
        q_val = row['q']
        sigma_val = max(row['sigma'], 1e-6)

        analytic = simple_chooser(S, K, t1, T2, r_val, q_val, sigma_val)
        mc_price, mc_se = simple_chooser_mc(S, K, t1, T2, r_val, q_val, sigma_val, n_sims=mc_nsims)

        results.append({
            'date': row['date'],
            'analytic': float(analytic),
            'mc_price': float(mc_price),
            'mc_se': float(mc_se),
            'abs_error': float(abs(mc_price - analytic)),
            'bias_pct': float((mc_price - analytic) / max(analytic, 1e-8) * 100),
            'S': float(S),
            'sigma': float(sigma_val),
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BSM Assumption Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def test_return_normality(returns: np.ndarray) -> Dict:
    """
    Test whether daily returns follow a normal distribution (BSM assumption).

    Returns dict with:
      - jarque_bera: (statistic, pvalue)
      - ks_test: (statistic, pvalue)
      - skewness, kurtosis
      - is_normal: True if both tests fail to reject normality at alpha=0.01
    """
    returns = returns[~np.isnan(returns)]

    if len(returns) < 10:
        return {'error': 'Insufficient data', 'is_normal': False}

    jb_stat, jb_p = jarque_bera(returns)
    ks_stat, ks_p = kstest(returns, 'norm', args=(returns.mean(), returns.std()))
    skew = float(pd.Series(returns).skew())
    kurt = float(pd.Series(returns).kurtosis())  # excess kurtosis

    return {
        'jarque_bera_stat': jb_stat,
        'jarque_bera_p': jb_p,
        'ks_stat': ks_stat,
        'ks_p': ks_p,
        'skewness': skew,
        'excess_kurtosis': kurt,
        'is_normal': (jb_p > 0.01) and (ks_p > 0.01),
        'n_obs': len(returns),
    }


def test_vol_stationarity(vol_series: pd.Series) -> Dict:
    """
    Test whether volatility is constant (BSM assumption).

    Returns:
      - cv: coefficient of variation (high = non-constant)
      - range_ratio: (max - min) / mean
      - pct_days_above_2x_median: proportion of days with vol > 2x median
    """
    vol = vol_series.dropna().values
    if len(vol) < 10:
        return {'error': 'Insufficient data'}

    median_vol = np.median(vol)
    mean_vol = np.mean(vol)
    std_vol = np.std(vol)

    return {
        'mean_vol': float(mean_vol),
        'median_vol': float(median_vol),
        'std_vol': float(std_vol),
        'cv': float(std_vol / mean_vol) if mean_vol > 0 else np.nan,
        'range_ratio': float((vol.max() - vol.min()) / mean_vol) if mean_vol > 0 else np.nan,
        'pct_days_above_2x_median': float(np.mean(vol > 2 * median_vol) * 100),
        'vol_min': float(vol.min()),
        'vol_max': float(vol.max()),
    }


def test_jump_detection(returns: np.ndarray, z_threshold: float = 3.0) -> Dict:
    """
    Detect return jumps (BSM assumes continuous paths).

    Returns:
      - n_jumps: count of returns exceeding z_threshold standard deviations
      - jump_pct: percentage of days with jumps
      - max_abs_return: maximum absolute return observed
    """
    returns = returns[~np.isnan(returns)]
    if len(returns) < 10:
        return {'error': 'Insufficient data'}

    std_ret = returns.std()
    mean_ret = returns.mean()
    z_scores = (returns - mean_ret) / std_ret
    n_jumps = int(np.sum(np.abs(z_scores) > z_threshold))

    return {
        'n_jumps': n_jumps,
        'jump_pct': float(n_jumps / len(returns) * 100),
        'max_abs_return': float(np.max(np.abs(returns))),
        'std_return': float(std_ret),
        'z_threshold': z_threshold,
        'n_obs': len(returns),
    }


def full_bsm_assessment(df: pd.DataFrame) -> Dict:
    """
    Run all BSM assumption diagnostics on real market data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'daily_return', 'vol_21d'

    Returns
    -------
    Dict with all diagnostic results.
    """
    returns = df['daily_return'].dropna().values
    vol_21d = df['vol_21d']

    normality = test_return_normality(returns)
    vol_test = test_vol_stationarity(vol_21d)
    jump_results = test_jump_detection(returns)

    # Determine overall model suitability score (0-100)
    scores = []
    # Normality: penalize if non-normal
    if not normality['is_normal']:
        # How severe? Based on excess kurtosis
        ek = abs(normality['excess_kurtosis'])
        if ek > 5:
            scores.append(20)
        elif ek > 2:
            scores.append(40)
        else:
            scores.append(60)
    else:
        scores.append(90)

    # Vol constancy: penalize high CV
    cv = vol_test.get('cv', 0)
    if cv > 1.0:
        scores.append(20)
    elif cv > 0.5:
        scores.append(40)
    else:
        scores.append(70)

    # Jump frequency
    jp = jump_results.get('jump_pct', 0)
    if jp > 5:
        scores.append(30)
    elif jp > 1:
        scores.append(50)
    else:
        scores.append(80)

    overall_score = np.mean(scores)

    return {
        'normality_test': normality,
        'volatility_test': vol_test,
        'jump_test': jump_results,
        'bsm_suitability_score': overall_score,
        'assessment': (
            'BSM assumptions are REASONABLY met' if overall_score >= 60
            else 'BSM assumptions are PARTIALLY violated'
            if overall_score >= 40
            else 'BSM assumptions are STRONGLY violated'
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Load Prepared Data from Week3
# ═══════════════════════════════════════════════════════════════════════════════

def load_week3_data() -> pd.DataFrame:
    """
    Load the feature dataset produced by week2 and add dividend yield.

    Returns
    -------
    pd.DataFrame with columns for pricing (S, r, q, sigma, date, ...)
    """
    data_path = WEEK3_PATH.parent / 'week2' / 'data' / 'feature_dataset.csv'
    if not data_path.exists():
        raise FileNotFoundError(f"Feature dataset not found at {data_path}")

    df = pd.read_csv(data_path, parse_dates=['date'])

    # Forward-fill NaN values for BSM inputs
    for col in ['vol_21d', 'vol_63d']:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Estimate dividend yield using the same approach as week3
    df = _estimate_dividend_yield(df)

    return df


def _estimate_dividend_yield(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback dividend yield estimation using known JPM dividend schedule.
    Mirrors the week3 notebook logic.
    """
    div_schedule = {
        '2018-01-01': 0.56, '2018-04-01': 0.56, '2018-07-01': 0.56, '2018-10-01': 0.56,
        '2019-01-01': 0.80, '2019-04-01': 0.80, '2019-07-01': 0.80, '2019-10-01': 0.80,
        '2020-01-01': 0.90, '2020-04-01': 0.90, '2020-07-01': 0.90, '2020-10-01': 0.90,
        '2021-01-01': 1.00, '2021-04-01': 1.00, '2021-07-01': 1.00, '2021-10-01': 1.00,
        '2022-01-01': 1.00, '2022-04-01': 1.00, '2022-07-01': 1.00, '2022-10-01': 1.00,
        '2023-01-01': 1.00, '2023-04-01': 1.00, '2023-07-01': 1.00, '2023-10-01': 1.00,
        '2024-01-01': 1.15, '2024-04-01': 1.15, '2024-07-01': 1.15, '2024-10-01': 1.25,
        '2025-01-01': 1.25, '2025-04-01': 1.25,
    }
    div_df = pd.DataFrame(list(div_schedule.items()), columns=['date', 'dps'])
    div_df['date'] = pd.to_datetime(div_df['date'])

    df_out = df.copy()
    df_out['dps_annual'] = 0.0
    for i, row in df_out.iterrows():
        dt = row['date']
        past_divs = div_df[div_df['date'] <= dt]
        if len(past_divs) >= 4:
            last4 = past_divs.tail(4)['dps'].sum()
            df_out.at[i, 'dps_annual'] = last4
        elif len(past_divs) > 0:
            df_out.at[i, 'dps_annual'] = past_divs['dps'].iloc[-1] * 4
        else:
            df_out.at[i, 'dps_annual'] = 2.24

    df_out['dividend_yield'] = df_out['dps_annual'] / df_out['close_jpm']
    return df_out


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Self-test / Demo
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("  BSM Baseline Evaluation — Self Test")
    print("=" * 65)
    print()

    # ── 1. Convergence Analysis ──
    print("[1] Convergence Analysis (Analytic vs MC)")
    conv = convergence_analysis(n_trials=5)
    print(f"  {'n_sims':>8s}  {'MAE':>8s}  {'RMSE':>8s}  {'Bias%':>8s}  {'Beta':>8s}")
    print("  " + "-" * 44)
    for _, row in conv.iterrows():
        beta_str = f"{row['convergence_rate_beta']:.3f}" if not np.isnan(row['convergence_rate_beta']) else 'N/A'
        print(f"  {int(row['n_sims']):>8,d}  {row['mae']:>8.4f}  {row['rmse']:>8.4f}  {row['mean_bias_pct']:>7.3f}%  {beta_str:>8s}")
    print(f"  Expected convergence rate beta ≈ -0.5 (MC standard error)")
    print()

    # ── 2. Greeks ──
    print("[2] Greeks at Reference Parameters")
    greeks = compute_all_greeks(S=156.7, K=150.0, t1=0.5, T2=1.0, r=0.0015, q=0.0233, sigma=0.282)
    for k, v in greeks.items():
        print(f"  {k:>10s}: {v:>10.4f}")
    print()

    # ── 3. Parameter Impact ──
    print("[3] Parameter Impact (10% shock)")
    impact = parameter_impact_analysis(shock_pct=10)
    print(f"  {'Param':>8s}  {'Base':>8s}  {'Price Low':>10s}  {'Price High':>10s}  {'Range':>8s}  {'Range%':>8s}")
    print("  " + "-" * 56)
    for _, row in impact.iterrows():
        print(f"  {row['param']:>8s}  {row['base_value']:>8.3f}  ${row['price_low']:>8.2f}  ${row['price_high']:>8.2f}  ${row['price_range']:>7.2f}  {row['range_pct']:>7.2f}%")
    print()

    # ── 4. BSM Assumptions ──
    print("[4] BSM Assumption Diagnostics")
    try:
        df_data = load_week3_data()
        assessment = full_bsm_assessment(df_data)
        print(f"  Normality test (JB p-value): {assessment['normality_test']['jarque_bera_p']:.4f}")
        print(f"  Excess kurtosis:             {assessment['normality_test']['excess_kurtosis']:.4f}")
        print(f"  Volatility CV:               {assessment['volatility_test']['cv']:.4f}")
        print(f"  Jump days (%):               {assessment['jump_test']['jump_pct']:.2f}%")
        print(f"  BSM Suitability Score:       {assessment['bsm_suitability_score']:.1f}/100")
        print(f"  Assessment:                  {assessment['assessment']}")
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
    print()

    print("=" * 65)
