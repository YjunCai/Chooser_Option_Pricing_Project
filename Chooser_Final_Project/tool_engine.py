"""
Pricing-Tool Engine
===================
Production-facing engine behind the Streamlit tool, the dashboard and the
offline scripts. Wraps the trained models (models/*.pkl + JSON sidecars) with:

  * load a model artifact and the exact feature list it was trained on;
  * build the 16-dim market feature frame from raw OHLC/VIX/rate data
    (mirrors training-time feature engineering so the tool can price "today");
  * predict a volatility input from any Approach-1 model (incl. the anchored
    ratio model and the VIX-proxy VIX/100 convention);
  * price a simple chooser with the validated Rubinstein formula;
  * dual-track pricing (BSM(vol_21d) baseline vs best ML) with the Week-6
    test-set error margins attached.

Dependencies: numpy, pandas, scikit-learn, joblib, xgboost, yfinance.
"""

import json
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import config
from pricer import simple_chooser
from data_preparation import load_dataset
from targets import add_fwd_vol_targets

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Model artifact loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_artifact(name: str) -> Tuple[object, dict]:
    """Return (pickled payload, JSON metadata) for a trained model artifact."""
    import joblib
    payload = joblib.load(config.MODELS_DIR / f'{name}.pkl')
    with open(config.MODELS_DIR / f'{name}.json', encoding='utf-8') as f:
        meta = json.load(f)
    return payload, meta


def load_vol_model(family: str) -> Tuple[object, List[str], dict]:
    """Load an Approach-1 vol model; return (payload, features, metadata)."""
    payload, meta = load_artifact(config.VOL_MODEL_ARTIFACTS[family])
    return payload, list(meta['features']), meta


def load_price_model(family: str) -> Tuple[object, List[str], dict]:
    """Load an Approach-2 end-to-end pricing model."""
    payload, meta = load_artifact(config.PRICE_MODEL_ARTIFACTS[family])
    return payload, list(meta['features']), meta


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Feature engineering from raw market data (identical to training)
# ═══════════════════════════════════════════════════════════════════════════════

def build_features(market: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 16-dim feature frame from a raw market frame.

    `market` must be indexed by datetime with columns close/high/low/volume/vix/rate
    (e.g. fetch_market_history output). Produces the exact quantities the models
    were trained on (close_jpm / close_vix / value_treasury_3mo + 16 features).
    """
    a = market.copy().sort_index()
    close = a['close']
    r = np.log(close / close.shift(1))
    v = (a['high'] - a['low']) / close

    f = pd.DataFrame(index=a.index)
    f[config.SPOT_COL] = close
    f[config.VIX_COL] = a['vix']
    f[config.RATE_COL] = a['rate']
    f['daily_return'] = r
    for w in (5, 21, 63):
        f[f'vol_{w}d'] = r.rolling(w).std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
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

    f['dps_growth_rate'] = 0.0   # dividend growth; recomputed from yfinance if reachable
    try:
        import yfinance as yf
        div = yf.Ticker('JPM').dividends
        if hasattr(div.index, 'tz'):
            div.index = div.index.tz_localize(None)
        div = div[div.index.normalize() <= a.index.max()]
        if len(div) >= 8:
            yoy = div.groupby(div.index.year).sum().pct_change().iloc[-1]
            f['dps_growth_rate'] = float(yoy) if np.isfinite(yoy) else 0.0
    except Exception:
        pass
    return f


def regime_dummies(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot regime dummies for every row using the same thresholds as training
    (quantiles of vol_21d over the full feature dataset, SMA_252 backward).
    Reference level NORMAL is dropped.
    """
    train = load_dataset()
    vol_median = float(train['vol_21d'].median())
    vol_q75 = float(train['vol_21d'].quantile(0.75))
    vol_q25 = float(train['vol_21d'].quantile(0.25))
    sma = feature_frame[config.SPOT_COL].rolling(252, min_periods=60).mean()

    s = feature_frame[config.SPOT_COL]
    v = feature_frame['vol_21d']
    sent = feature_frame['sentiment_score']
    label = pd.Series('NORMAL', index=feature_frame.index)
    label[(s > sma) & (v <= vol_median)] = 'BULL'
    label[(s < sma) & (v > vol_median)] = 'BEAR'
    label[v > vol_q75] = 'HIGH_VOL'
    label[(v < vol_q25) & (sent > 0.5)] = 'CALM'

    out = pd.DataFrame(0.0, index=feature_frame.index, columns=config.REGIME_DUMMY_FEATURES)
    for col in config.REGIME_DUMMY_FEATURES:
        out[col] = (label == col.split('_', 1)[1]).astype(float)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Real-time market history
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_market_history(period: str = '5y') -> pd.DataFrame:
    """
    Fetch JPM / VIX / 3-mo Treasury daily data and align into one frame:
        columns = close, high, low, volume, vix, rate
    Uses yfinance with graceful fallback to the local snapshot cache so the
    tool works offline too.
    """
    try:
        import yfinance as yf
        jpm = yf.Ticker('JPM').history(period=period, auto_adjust=True)
        vix = yf.Ticker('^VIX').history(period=period, auto_adjust=True)
        irx = yf.Ticker('^IRX').history(period=period, auto_adjust=True)
        if len(jpm) < 30 or len(vix) < 30 or len(irx) < 30:
            raise RuntimeError('yfinance returned too few rows')
        # Normalize every source to a naive (tz-free) date index before aligning,
        # otherwise reindex against a tz-aware vs naive mix returns all-NaN.
        idx = jpm.index.tz_localize(None)
        out = pd.DataFrame(index=idx)
        out['close'] = jpm['Close'].values
        out['high'] = jpm['High'].values
        out['low'] = jpm['Low'].values
        out['volume'] = jpm['Volume'].values
        vix_s = vix['Close'].copy(); vix_s.index = vix.index.tz_localize(None)
        irx_s = irx['Close'].copy(); irx_s.index = irx.index.tz_localize(None)
        out['vix'] = vix_s.reindex(idx).values
        out['rate'] = irx_s.reindex(idx).values
        out = out.dropna(subset=['close', 'vix', 'rate'])
        if len(out) < 30:
            raise RuntimeError('yfinance returned too few rows after alignment')
        return out
    except Exception:
        return _load_cached_market()


def _load_cached_market() -> pd.DataFrame:
    """Offline fallback: local snapshot CSVs (written by the initial data pull)."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Volatility prediction
# ═══════════════════════════════════════════════════════════════════════════════

def _inner(payload):
    """Return the sklearn Pipeline (or dict pipeline) from an artifact payload."""
    return payload['pipeline'] if isinstance(payload, dict) else payload


def _is_vix_proxy(payload: object) -> bool:
    """VIX-proxy artifacts are stored as {'pipeline': ..., 'params': ...} dicts;
    their pipeline predicts the raw VIX level (points), so sigma = VIX / 100."""
    return isinstance(payload, dict) and 'pipeline' in payload


def predict_sigma_from_pipe(payload: object, X: pd.DataFrame) -> np.ndarray:
    """
    Predict annualized volatility from a feature matrix, applying the correct
    output convention per model:
      * direct vol models (rf / gbdt / xgb) -> forward vol directly
      * vix_proxy (dict artifact)           -> predicts VIX at t+1, VIX/100
    """
    raw = np.asarray(_inner(payload).predict(X), dtype=float)
    if _is_vix_proxy(payload):
        raw = raw / 100.0
    return np.clip(raw, 1e-4, 2.0)


def predict_sigma(payload: object, feature_row: pd.Series) -> float:
    """Predict the annualized volatility input for a single feature row."""
    X = pd.DataFrame([feature_row.values], columns=feature_row.index).astype(float)
    return float(predict_sigma_from_pipe(payload, X)[0])


def predict_sigma_family(family: str, payload: object, feature_row: pd.Series,
                         vol_21d: float = None) -> float:
    """
    Family-aware volatility prediction that applies the correct output
    convention:
      * 'vix'   models -> sigma = VIX/100 (handled by predict_sigma_from_pipe)
      * 'ratio' models (gbdt_anchored) -> sigma = ratio * vol_21d
      * 'vol'   models -> sigma as-is
    """
    raw = predict_sigma(payload, feature_row)
    if config.VOL_MODEL_CONVENTIONS.get(family) == 'ratio':
        vol21 = vol_21d if vol_21d is not None else float(feature_row['vol_21d'])
        return vol21 * raw
    return raw


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Chooser pricing
# ═══════════════════════════════════════════════════════════════════════════════

def price_chooser(S, K, t1, T2, r, q, sigma) -> float:
    """Rubinstein simple-chooser price."""
    return float(simple_chooser(S, K, t1, T2, r, q, np.maximum(sigma, 1e-6)))


def price_chooser_vec(S, K, t1, T2, r, q, sigma) -> np.ndarray:
    """Vectorized chooser price; S and/or sigma may be arrays (broadcast)."""
    return np.asarray(simple_chooser(S, K, t1, T2, r, q, np.maximum(sigma, 1e-6)),
                      dtype=float)


def price_dual(S, K, t1, T2, r, q, sigma_ml, sigma_base=None) -> Dict[str, float]:
    """Dual-track chooser price: BSM(vol_21d) baseline vs ML volatility."""
    if sigma_base is None:
        sigma_base = sigma_ml
    p_ml = price_chooser(S, K, t1, T2, r, q, sigma_ml)
    p_base = price_chooser(S, K, t1, T2, r, q, sigma_base)
    return {'price_ML': p_ml, 'price_BSM': p_base, 'spread_$': p_ml - p_base,
            'sigma_ML': sigma_ml, 'sigma_base': sigma_base}


def price_vega(S, K, t1, T2, r, q, sigma, eps: float = 1e-4) -> float:
    """dP/dsigma for the simple chooser via central finite difference ($ per 1.0 vol)."""
    return (price_chooser(S, K, t1, T2, r, q, sigma + eps)
            - price_chooser(S, K, t1, T2, r, q, sigma - eps)) / (2 * eps)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Error margins (Week-6 test-set metrics, shown as error bars in the tool)
# ═══════════════════════════════════════════════════════════════════════════════

def error_margins() -> pd.DataFrame:
    """Week-6 held-out test-set metrics per model (chooser-price MAE/RMSE/R2)."""
    rows = []
    for family, m in config.ERROR_MARGINS.items():
        rows.append({'family': family, 'label': m['label'],
                     'price_MAE_$': m['MAE'], 'price_RMSE_$': m['RMSE'],
                     'price_R2': m['R2']})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print('Pricing-tool engine self-check')
    print('-' * 50)
    for fam in ('vol_xgb', 'vol_vix_proxy', 'vol_gbdt_anchored'):
        payload, feats, meta = load_vol_model(fam)
        print(f'{fam:<20} feats={len(feats)}')
    print('error margins:')
    print(error_margins().to_string(index=False))
    p = price_chooser(150, 150, 0.5, 1.0, 0.05, 0.017, 0.20)
    print('chooser(150,150,0.5,1.0,5%,1.7%,20%) =', round(p, 4))
