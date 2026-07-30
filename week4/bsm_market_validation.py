"""
BSM Market Validation — Week 4
================================
Compares BSM model prices against real JPM vanilla option market prices
from Yahoo Finance option chains.

Approach:
  1. Fetch current JPM market data (price, vol_21d, r, q)
  2. Fetch JPM option chain for multiple expirations
  3. Compute BSM theoretical price for each option
  4. Compare BSM price vs market lastPrice
  5. Compute MAE, RMSE, MAPE across moneyness/tenor buckets
  6. Also back out implied volatility from market prices and compare with vol_21d

This provides a real-world benchmark for BSM model accuracy, even though
chooser options themselves are OTC products without public market prices.
"""

import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# Add week3 pricer
WEEK3_PATH = Path(__file__).resolve().parent.parent / 'week3'
if str(WEEK3_PATH) not in sys.path:
    sys.path.insert(0, str(WEEK3_PATH))

from chooser_option_pricer import bs_call, bs_put
from scipy.stats import norm


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Market Data Fetching
# ═══════════════════════════════════════════════════════════════════════════════

def get_current_market_params() -> Dict:
    """
    Fetch current JPM market parameters using yfinance.

    Returns
    -------
    dict with: S, r, q, vol_21d, vol_63d, date
    """
    import yfinance as yf

    jpm = yf.Ticker('JPM')

    # Stock price & volatility
    hist = jpm.history(period='6mo')
    S = float(hist['Close'].iloc[-1])
    hist['return'] = np.log(hist['Close'] / hist['Close'].shift(1))
    vol_21d = float(hist['return'].rolling(21).std().iloc[-1] * np.sqrt(252))
    vol_63d = float(hist['return'].rolling(63).std().iloc[-1] * np.sqrt(252))

    # Risk-free rate (13-week T-bill)
    try:
        tnx = yf.Ticker('^IRX')
        irx_hist = tnx.history(period='5d')
        r = float(irx_hist['Close'].iloc[-1] / 100.0)
    except Exception:
        try:
            # Fallback: ^TNX (10yr) if IRX fails
            tnx = yf.Ticker('^TNX')
            irx_hist = tnx.history(period='5d')
            r = float(irx_hist['Close'].iloc[-1] / 100.0)
        except Exception:
            r = 0.05

    # Dividend yield (last 4 quarters / current price)
    try:
        div = jpm.dividends
        if len(div) >= 4:
            annual_div = div.tail(4).sum()
            q = float(annual_div / S)
        elif len(div) > 0:
            annual_div = div.tail(1).iloc[0] * 4
            q = float(annual_div / S)
        else:
            q = 0.02
    except Exception:
        q = 0.02

    return {
        'S': S,
        'r': r,
        'q': q,
        'vol_21d': vol_21d,
        'vol_63d': vol_63d,
        'date': date.today().isoformat(),
    }


def fetch_option_chain(ticker: str = 'JPM', expiry: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch option chain for a given ticker and expiry.

    If expiry is None, fetches all available monthly expirations.
    Returns combined DataFrame with both calls and puts.
    """
    import yfinance as yf

    stock = yf.Ticker(ticker)

    if expiry is None:
        # Fetch multiple expirations
        all_expiries = stock.options
        # Pick monthly expirations (skip weekly: indices 0-5 are weekly)
        monthly_idx = [i for i in range(len(all_expiries)) if i >= 5][:6]
        selected = [all_expiries[i] for i in monthly_idx]
    else:
        selected = [expiry]

    rows = []
    today = datetime.now()

    for exp in selected:
        opt = stock.option_chain(exp)
        expiry_date = datetime.strptime(exp, '%Y-%m-%d')
        T = (expiry_date - today).days / 365.0
        if T <= 0:
            continue

        for opt_type, df in [('call', opt.calls), ('put', opt.puts)]:
            for _, row in df.iterrows():
                last_price = row.get('lastPrice', np.nan)
                if pd.isna(last_price) or last_price <= 0:
                    continue
                rows.append({
                    'expiry': exp,
                    'T': T,
                    'strike': row['strike'],
                    'type': opt_type,
                    'lastPrice': last_price,
                    'bid': row.get('bid', 0) or 0,
                    'ask': row.get('ask', 0) or 0,
                    'impliedVolatility': row.get('impliedVolatility', np.nan),
                    'volume': row.get('volume', 0) or 0,
                    'openInterest': row.get('openInterest', 0) or 0,
                })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BSM vs Market Price Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def compute_bsm_vs_market(
    options_df: pd.DataFrame,
    S: float, r: float, q: float, sigma: float
) -> pd.DataFrame:
    """
    For each option, compute BSM theoretical price and compare with market lastPrice.

    Parameters
    ----------
    options_df : pd.DataFrame
        Must have columns: strike, T, type (call/put), lastPrice
    S, r, q, sigma : float
        Current market parameters

    Returns
    -------
    pd.DataFrame with added columns: bsm_price, abs_error, squared_error, mape, mid_price
    """
    df = options_df.copy()

    bsm_prices = []
    for _, row in df.iterrows():
        K = row['strike']
        T = row['T']
        if row['type'] == 'call':
            price = bs_call(S, K, max(T, 1e-6), r, q, sigma)
        else:
            price = bs_put(S, K, max(T, 1e-6), r, q, sigma)
        bsm_prices.append(float(price))

    df['bsm_price'] = bsm_prices
    df['mid_price'] = np.where((df['bid'] > 0) & (df['ask'] > 0),
                                (df['bid'] + df['ask']) / 2, df['lastPrice'])
    df['abs_error'] = abs(df['bsm_price'] - df['lastPrice'])
    df['squared_error'] = (df['bsm_price'] - df['lastPrice']) ** 2
    df['bias'] = df['bsm_price'] - df['lastPrice']
    df['bias_pct'] = df['bias'] / df['lastPrice'] * 100
    df['mape'] = df['abs_error'] / df['lastPrice'] * 100
    df['moneyness'] = df.apply(
        lambda r: 'ITM' if (r['type'] == 'call' and r['strike'] < S) or
                          (r['type'] == 'put' and r['strike'] > S)
                  else 'OTM' if (r['type'] == 'call' and r['strike'] > S) or
                                (r['type'] == 'put' and r['strike'] < S)
                  else 'ATM',
        axis=1
    )
    df['log_moneyness'] = np.log(df['strike'] / S)
    df['tenor_bucket'] = pd.cut(df['T'], bins=[0, 0.25, 0.5, 1.0, 2.0, 5.0],
                                 labels=['<3m', '3-6m', '6-12m', '1-2y', '2y+'])

    return df


def compute_error_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute error metrics (MAE, RMSE, MAPE, bias) overall and by subgroups.

    Returns DataFrame of metrics.
    """
    metrics = []

    # Overall
    all_data = df[df['lastPrice'] > 0]
    metrics.append({
        'group': 'ALL',
        'count': len(all_data),
        'MAE': all_data['abs_error'].mean(),
        'RMSE': np.sqrt(all_data['squared_error'].mean()),
        'MAPE': all_data['mape'].mean(),
        'MeanBias': all_data['bias'].mean(),
        'MeanBiasPct': all_data['bias_pct'].mean(),
    })

    # By option type
    for opt_type in ['call', 'put']:
        subset = all_data[all_data['type'] == opt_type]
        if len(subset) > 0:
            metrics.append({
                'group': f'{opt_type.upper()}',
                'count': len(subset),
                'MAE': subset['abs_error'].mean(),
                'RMSE': np.sqrt(subset['squared_error'].mean()),
                'MAPE': subset['mape'].mean(),
                'MeanBias': subset['bias'].mean(),
                'MeanBiasPct': subset['bias_pct'].mean(),
            })

    # By moneyness
    for mn in ['ITM', 'ATM', 'OTM']:
        subset = all_data[all_data['moneyness'] == mn]
        if len(subset) > 0:
            metrics.append({
                'group': mn,
                'count': len(subset),
                'MAE': subset['abs_error'].mean(),
                'RMSE': np.sqrt(subset['squared_error'].mean()),
                'MAPE': subset['mape'].mean(),
                'MeanBias': subset['bias'].mean(),
                'MeanBiasPct': subset['bias_pct'].mean(),
            })

    # By tenor
    for tenor in all_data['tenor_bucket'].cat.categories:
        subset = all_data[all_data['tenor_bucket'] == tenor]
        if len(subset) > 0:
            metrics.append({
                'group': str(tenor),
                'count': len(subset),
                'MAE': subset['abs_error'].mean(),
                'RMSE': np.sqrt(subset['squared_error'].mean()),
                'MAPE': subset['mape'].mean(),
                'MeanBias': subset['bias'].mean(),
                'MeanBiasPct': subset['bias_pct'].mean(),
            })

    return pd.DataFrame(metrics)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Implied Volatility Back-out (for verification)
# ═══════════════════════════════════════════════════════════════════════════════

def bs_price(S, K, T, r, q, sigma, opt_type='call'):
    """Wrapper for BSM pricing."""
    if opt_type == 'call':
        return bs_call(S, K, T, r, q, sigma)
    return bs_put(S, K, T, r, q, sigma)


def implied_volatility(
    market_price: float, S: float, K: float, T: float,
    r: float, q: float, opt_type: str = 'call',
    tol: float = 1e-6, max_iter: int = 100
) -> float:
    """
    Back out implied volatility from market price using Newton-Raphson.

    Uses a bisection pre-warm to get close, then Newton-Raphson refinement.
    """
    if market_price <= 0 or T <= 0:
        return 0.0

    # Check for arbitrage violations
    intrinsic = max(S - K, 0.0) if opt_type == 'call' else max(K - S, 0.0)
    time_value = market_price - intrinsic
    if time_value < -0.01:
        return 0.0  # Arbitrage: ignore this option
    if time_value <= 0:
        return 0.001  # Deep ITM, near-zero vol

    # Bisection pre-warm
    lo, hi = 0.001, 3.0
    for _ in range(30):
        mid = (lo + hi) / 2
        p = bs_price(S, K, T, r, q, mid, opt_type)
        if p > market_price:
            hi = mid
        else:
            lo = mid
        if abs(p - market_price) < tol * 10:
            break

    sigma = (lo + hi) / 2

    # Newton-Raphson refinement
    for i in range(max_iter):
        price = bs_price(S, K, T, r, q, sigma, opt_type)
        diff = price - market_price
        if abs(diff) < tol:
            return max(sigma, 0.001)

        eps = sigma * 0.01 + 1e-8
        vega = (bs_price(S, K, T, r, q, sigma + eps, opt_type) -
                bs_price(S, K, T, r, q, sigma - eps, opt_type)) / (2 * eps)

        if abs(vega) < 1e-12:
            break
        sigma = sigma - diff / vega
        sigma = max(min(sigma, 3.0), 0.001)

    return sigma


def compute_implied_volatilities(df: pd.DataFrame, S: float, r: float, q: float) -> pd.DataFrame:
    """
    Compute implied volatility for each option from market price.
    Then compare with the BSM vol_21d input.

    Returns df with added columns: implied_vol, vol_error (implied_vol - vol_21d)
    """
    df = df.copy()
    ivs = []
    for _, row in df.iterrows():
        iv = implied_volatility(
            row['lastPrice'], S, row['strike'], max(row['T'], 1e-6),
            r, q, row['type']
        )
        ivs.append(iv)
    df['implied_vol'] = ivs
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def filter_otm_results(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to OTM options and add helpful analysis columns."""
    otm = df[df['moneyness'] == 'OTM'].copy()
    otm['strike_bucket'] = pd.cut(otm['strike'], bins=8)
    otm['expiry_group'] = otm['expiry']
    return otm


def run_full_validation(
    ticker: str = 'JPM',
    expiries: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run the complete BSM vs market validation pipeline.

    Returns
    -------
    (options_df, metrics_df, market_params)
    """
    print("=" * 65)
    print("  BSM Market Validation Pipeline")
    print("=" * 65)
    print()

    # Step 1: Get current market params
    print("[1] Fetching current market parameters...")
    params = get_current_market_params()
    print(f"    S=${params['S']:.2f}, r={params['r']*100:.2f}%, q={params['q']*100:.2f}%")
    print(f"    vol_21d={params['vol_21d']*100:.1f}%, vol_63d={params['vol_63d']*100:.1f}%")
    print()

    # Step 2: Fetch option chain
    print("[2] Fetching JPM option chains...")
    options = fetch_option_chain(ticker, expiries)
    print(f"    Total options: {len(options)} ({options['type'].value_counts().to_dict()})")
    print(f"    Expirations: {options['expiry'].nunique()}")
    print(f"    T range: {options['T'].min():.4f} - {options['T'].max():.4f} yr")
    print()

    # Step 3: Compute BSM vs market
    print("[3] Computing BSM prices vs market prices...")
    results = compute_bsm_vs_market(options, params['S'], params['r'], params['q'], params['vol_21d'])
    print(f"    Options with BSM price: {len(results)}")
    print()

    # Step 4: Compute implied vols
    print("[4] Backing out implied volatilities...")
    results = compute_implied_volatilities(results, params['S'], params['r'], params['q'])
    results['vol_error'] = results['implied_vol'] - params['vol_21d']
    print(f"    Mean implied vol: {results['implied_vol'].mean()*100:.1f}%")
    print(f"    vol_21d: {params['vol_21d']*100:.1f}%")
    print()

    # Step 5: Compute error metrics (overall and filtered)
    print("[5] Computing error metrics...")
    metrics = compute_error_metrics(results)

    # ATM+OTM only metrics
    valid = results[results['moneyness'] != 'ITM']
    atm = results[abs(results['log_moneyness']) < 0.05]

    print("  --- ALL Options ---")
    print(metrics[['group', 'count', 'MAE', 'RMSE', 'MAPE', 'MeanBiasPct']].to_string(index=False))
    print()
    print(f"  --- ATM+OTM Only ({len(valid)} options) ---")
    print(f"    MAE=${valid['abs_error'].mean():.4f}  RMSE=${np.sqrt((valid['squared_error']).mean()):.4f}")
    print(f"    MAPE={valid['mape'].mean():.1f}%  Bias={valid['bias_pct'].mean():.1f}%")
    print()
    print(f"  --- ATM Only |log-mny|<0.05 ({len(atm)} options) ---")
    print(f"    MAE=${atm['abs_error'].mean():.4f}  RMSE=${np.sqrt((atm['squared_error']).mean()):.4f}")
    print(f"    Mean IV={atm['implied_vol'].mean()*100:.1f}%  (vol_21d={params['vol_21d']*100:.1f}%)")
    print(f"    IV Premium={(atm['implied_vol'] - params['vol_21d']).mean()*100:.1f}%")
    print()

    return results, metrics, params


if __name__ == '__main__':
    options, metrics, params = run_full_validation()
