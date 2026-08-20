"""
Week 7 Pricing-Tool Engine
==========================
Shared engine behind both the Streamlit prototype and the offline run script.
It wraps the Week 6 trained models (models/*.pkl + JSON sidecar) with a
production-facing API:

  * load a model artifact and the exact feature list it was trained on;
  * build the 16-dim market feature frame from raw OHLC/VIX/rate data
    (mirrors week2 engineering so the tool can price "today" in real time);
  * predict a volatility input from any approach-1 model (incl. the anchored
    ratio model and the VIX-proxy VIX/100 convention);
  * price a simple chooser with the Week-3 validated Rubinstein formula;
  * dual-track pricing (BSM(vol_21d) baseline vs best ML) with the Week-6
    test-set error margins attached.

All functions are pure / deterministic so they can be unit-tested offline.
"""

import json
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import w7config as cfg

# week3 pricer (sys.path is wired by w7config)
from chooser_option_pricer import bs_call, bs_put, simple_chooser

# week5 helpers
from data_preparation import load_dataset
from targets import add_fwd_vol_targets

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Model artifact loading (from the Week 6 models/ directory)
# ═══════════════════════════════════════════════════════════════════════════════

def load_artifact(name: str) -> Tuple[object, dict]:
    """Return (pickled payload, JSON metadata) for a Week-6 model artifact."""
    import joblib
    payload = joblib.load(cfg.MODELS_DIR / f'{name}.pkl')
    with open(cfg.MODELS_DIR / f'{name}.json', encoding='utf-8') as f:
        meta = json.load(f)
    return payload, meta


def load_vol_model(family: str) -> Tuple[object, List[str], dict]:
    """Load an approach-1 vol model; return (payload, features, metadata)."""
    name = cfg.VOL_MODEL_ARTIFACTS[family]
    payload, meta = load_artifact(name)
    return payload, list(meta['features']), meta


def load_price_model(family: str) -> Tuple[object, List[str], dict]:
    """Load an approach-2 end-to-end pricing model."""
    name = cfg.PRICE_MODEL_ARTIFACTS[family]
    payload, meta = load_artifact(name)
    return payload, list(meta['features']), meta


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Feature engineering from raw market data
# ═══════════════════════════════════════════════════════════════════════════════

def build_features(market: pd.DataFrame) -> pd.DataFrame:
    """
    Build the week-2 16-dim feature frame from a raw market frame.

    `market` must be indexed by datetime and contain columns:
        close, high, low, volume, vix, rate
    (e.g. the output of `fetch_market_history`). The computation mirrors
    week6/live_metrics.snapshot_features and the original week-2 pipeline so
    the model inputs are exactly the same quantities the models were trained
    on. Returns the full feature frame (close_jpm / close_vix / treasury +
    16 features); NaN rows are left intact for the caller to drop.
    """
    a = market.copy().sort_index()
    close = a['close']
    r = np.log(close / close.shift(1))
    v = (a['high'] - a['low']) / close

    f = pd.DataFrame(index=a.index)
    f[cfg.SPOT_COL] = close                 # close_jpm
    f[cfg.VIX_COL] = a['vix']               # close_vix
    f[cfg.RATE_COL] = a['rate']             # value_treasury_3mo
    f['daily_return'] = r
    for w in (5, 21, 63):
        f[f'vol_{w}d'] = r.rolling(w).std() * np.sqrt(cfg.TRADING_DAYS_PER_YEAR)
    f['high_low_spread'] = v
    f['volume_change_1d'] = np.log(a['volume'] / a['volume'].shift(1)).replace([np.inf, -np.inf], 0)
    f['sma_ratio_21'] = close / close.rolling(21).mean()
    f['vix_change_1d'] = a['vix'].diff()
    f['vix_jpm_corr_21d'] = r.rolling(21, min_periods=16).corr(a['vix'].diff())
    f['vix_jpm_cross_1d'] = -r * a['vix'].diff()
    f['rate_change_1d_bps'] = a['rate'].diff() * 100
    f['rate_momentum_5d_bps'] = a['rate'].diff(5) * 100
    roll_min = a['vix'].rolling(252, min_periods=20).min()
    roll_max = a['vix'].rolling(252, min_periods=20).max()
    f['sentiment_score'] = (1 - (a['vix'] - roll_min) / (roll_max - roll_min).replace(0, np.nan)).fillna(0.5).clip(0, 1)
    f['jpm_vol_ratio'] = f['vol_5d'] / f['vol_21d']
    f['vix_ratio'] = a['vix'] / (f['vol_21d'] * 100)

    try:                       # dividend growth rate (annual YoY of trailing dividends)
        import yfinance as yf
        div = yf.Ticker('JPM').dividends
        if hasattr(div.index, 'tz'):
            div.index = div.index.tz_localize(None)
        div = div[div.index.normalize() <= a.index.max()]
        if len(div) >= 8:
            yoy = div.groupby(div.index.year).sum().pct_change().iloc[-1]
            f['dps_growth_rate'] = float(yoy) if np.isfinite(yoy) else 0.0
        else:
            f['dps_growth_rate'] = 0.0
    except Exception:
        f['dps_growth_rate'] = 0.0
    f['dps_growth_rate'] = f['dps_growth_rate'].fillna(0.0)

    return f


def regime_dummies(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot regime dummies for every row of a feature frame, using the SAME
    thresholds as Week 4/6 (quantiles over the full training dataset, SMA_252
    backward-looking). Reference level NORMAL is dropped.
    """
    train = load_dataset()          # week2 feature dataset (2018-2024)
    vol_median = float(train['vol_21d'].median())
    vol_q75 = float(train['vol_21d'].quantile(0.75))
    vol_q25 = float(train['vol_21d'].quantile(0.25))
    sma = feature_frame[cfg.SPOT_COL].rolling(252, min_periods=60).mean()

    s = feature_frame[cfg.SPOT_COL]
    v = feature_frame['vol_21d']
    sent = feature_frame['sentiment_score']
    label = pd.Series('NORMAL', index=feature_frame.index)
    label[(s > sma) & (v <= vol_median)] = 'BULL'
    label[(s < sma) & (v > vol_median)] = 'BEAR'
    label[v > vol_q75] = 'HIGH_VOL'
    label[(v < vol_q25) & (sent > 0.5)] = 'CALM'

    out = pd.DataFrame(0.0, index=feature_frame.index, columns=cfg.REGIME_DUMMY_FEATURES)
    for col in cfg.REGIME_DUMMY_FEATURES:
        out[col] = (label == col.split('_', 1)[1]).astype(float)
    return out


def prepare_model_input(feature_frame: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Align a feature frame to the exact feature list a model was trained on."""
    missing = [c for c in features if c not in feature_frame.columns]
    if missing:
        raise KeyError(f'missing features for model input: {missing}')
    return feature_frame[features].astype(float)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Real-time market history
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_market_history(period: str = '2y') -> pd.DataFrame:
    """
    Fetch JPM / VIX / 3-mo Treasury daily data and align them into one frame:
        columns = close, high, low, volume, vix, rate
    Uses yfinance with graceful fallback to the cached d:/tmp/*.csv files
    (used by Week 4-6 snapshots) so the tool works offline too.
    """
    try:
        import yfinance as yf
        jpm = yf.Ticker('JPM').history(period=period, auto_adjust=True)
        vix = yf.Ticker('^VIX').history(period=period, auto_adjust=True)
        irx = yf.Ticker('^IRX').history(period=period, auto_adjust=True)
        if len(jpm) < 30 or len(vix) < 30 or len(irx) < 30:
            raise RuntimeError('yfinance returned too few rows')
        idx = jpm.index.tz_localize(None)
        out = pd.DataFrame(index=idx)
        out['close'] = jpm['Close'].values
        out['high'] = jpm['High'].values
        out['low'] = jpm['Low'].values
        out['volume'] = jpm['Volume'].values
        out['vix'] = vix['Close'].reindex(jpm.index).values
        out['rate'] = irx['Close'].reindex(jpm.index).values
        return out.dropna(subset=['close', 'vix', 'rate'])
    except Exception:
        # offline fallback: d:/tmp raw CSVs (cache written by week1/week4)
        return _load_cached_market()


def _load_cached_market() -> pd.DataFrame:
    def _load_clean(path: str) -> pd.DataFrame:
        d = pd.read_csv(path, index_col=0, parse_dates=False)
        d.index = pd.to_datetime(d.index.str.split(' ').str[0])
        return d

    jpm = _load_clean('d:/tmp/jpm_daily.csv')
    vix = _load_clean('d:/tmp/vix_daily.csv')
    irx = _load_clean('d:/tmp/irx_daily.csv')

    a = pd.DataFrame(index=jpm.index)
    a['close'] = jpm['j_Close']; a['high'] = jpm['j_High']; a['low'] = jpm['j_Low']
    a['volume'] = jpm['j_Volume']
    a['vix'] = vix['v_Close'].reindex(jpm.index)
    a['rate'] = irx['r_Close'].reindex(jpm.index)
    return a.dropna()


def latest_market_row(feature_frame: pd.DataFrame, features: List[str]) -> pd.Series:
    """The most recent complete feature row aligned to a model's feature list."""
    f = feature_frame.copy()
    for col in cfg.REGIME_DUMMY_FEATURES:
        f[col] = regime_dummies(feature_frame)[col]
    row = f.iloc[-1]
    return row[features].astype(float)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Volatility prediction
# ═══════════════════════════════════════════════════════════════════════════════

def _inner(payload):
    """Return the sklearn Pipeline (or dict pipeline) from an artifact payload."""
    if isinstance(payload, dict):            # vix_proxy artifact = {'pipeline': ...}
        return payload['pipeline']
    return payload


def _is_vix_proxy(payload: object) -> bool:
    """VIX-proxy artifacts are stored as {'pipeline': ..., 'params': ...} dicts;
    their pipeline predicts the raw VIX level (points), so sigma = VIX / 100."""
    return isinstance(payload, dict) and 'pipeline' in payload


def predict_sigma_from_pipe(payload: object, X: pd.DataFrame) -> np.ndarray:
    """
    Predict annualized volatility from a feature matrix, applying the correct
    output convention per model:
      * direct vol models (rf / gbdt / xgb)  -> forward vol directly
      * vix_proxy (dict artifact)            -> predicts VIX at t+1, VIX/100
    """
    raw = np.asarray(_inner(payload).predict(X), dtype=float)
    if _is_vix_proxy(payload):
        raw = raw / 100.0
    return np.clip(raw, 1e-4, 2.0)


def predict_sigma(payload: object, feature_row: pd.Series) -> float:
    """Predict the annualized volatility input for a single feature row."""
    X = pd.DataFrame([feature_row.values], columns=feature_row.index).astype(float)
    return float(predict_sigma_from_pipe(payload, X)[0])


def predict_sigma_anchored(payload: object, feature_row: pd.Series, vol_21d: float) -> float:
    pipe = _inner(payload)
    X = pd.DataFrame([feature_row.values], columns=feature_row.index).astype(float)
    ratio = float(np.clip(pipe.predict(X)[0], 0.1, 5.0))
    return vol_21d * ratio


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Chooser pricing
# ═══════════════════════════════════════════════════════════════════════════════

def price_chooser(S, K, t1, T2, r, q, sigma) -> float:
    """Rubinstein simple-chooser price (Week-3 validated formula)."""
    return float(simple_chooser(S, K, t1, T2, r, q, np.maximum(sigma, 1e-6)))


def price_chooser_vec(S, K, t1, T2, r, q, sigma) -> np.ndarray:
    """Vectorized chooser price; S and/or sigma may be arrays (broadcast)."""
    return np.asarray(simple_chooser(S, K, t1, T2, r, q, np.maximum(sigma, 1e-6)),
                      dtype=float)


def price_dual(S, K, t1, T2, r, q, sigma_ml, sigma_base=None) -> Dict[str, float]:
    """
    Dual-track chooser price: BSM(vol_21d) baseline vs ML-enhanced volatility.
    Returns a dict with both prices and the dollar spread.
    """
    if sigma_base is None:
        sigma_base = sigma_ml
    p_ml = price_chooser(S, K, t1, T2, r, q, sigma_ml)
    p_base = price_chooser(S, K, t1, T2, r, q, sigma_base)
    return {'price_ML': p_ml, 'price_BSM': p_base, 'spread_$': p_ml - p_base,
            'sigma_ML': sigma_ml, 'sigma_base': sigma_base}


def price_vega(S, K, t1, T2, r, q, sigma, eps: float = 1e-4) -> float:
    """∂P/∂σ for the simple chooser via central finite difference (in $ per 1.0 vol)."""
    p_hi = price_chooser(S, K, t1, T2, r, q, sigma + eps)
    p_lo = price_chooser(S, K, t1, T2, r, q, sigma - eps)
    return (p_hi - p_lo) / (2 * eps)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Error-margin metadata (Week-6 test-set metrics, shown in the tool)
# ═══════════════════════════════════════════════════════════════════════════════

def error_margins() -> pd.DataFrame:
    """
    Week-6 held-out test-set metrics per model (vol MAE and chooser-price
    MAE/RMSE), read from the Week-6 output CSVs so the tool displays the same
    numbers the reports quote.
    """
    vol = pd.read_csv(cfg.WEEK6_DIR / 'output' / 'vol_test_comparison.csv')
    cho = pd.read_csv(cfg.WEEK6_DIR / 'output' / 'chooser_test_comparison.csv')
    rows = []
    for _, r in cho.iterrows():
        name = str(r['model'])
        family = None
        for fam in cfg.VOL_MODEL_ARTIFACTS:
            if name.startswith(fam) or fam.split('_', 1)[-1] == name:
                family = fam
        if name == 'vix_proxy':
            family = 'vol_vix_proxy'
        if name == 'BSM(vol_21d)':
            family = 'vol_21d'
        rows.append({'family': family or name, 'label': name,
                     'price_MAE_$': float(r['MAE']), 'price_RMSE_$': float(r['RMSE']),
                     'price_R2': float(r['R2'])})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Self-check
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Week 7 tool engine self-check')
    print('-' * 50)
    for fam in ('vol_xgb', 'vol_vix_proxy', 'vol_gbdt_anchored'):
        payload, feats, meta = load_vol_model(fam)
        print(f'{fam:<20} feats={len(feats)} test_MAE={meta.get("test_MAE")}')
    print('error margins:')
    print(error_margins().to_string(index=False))
    p = price_chooser(150, 150, 0.5, 1.0, 0.05, 0.017, 0.20)
    print('chooser(150,150,0.5,1.0,5%,1.7%,20%) =', round(p, 4))
