"""
Sensitivity Analysis
====================
Extreme-scenario testing and univariate price-perturbation curves used by the
pricing tool's interactive panels. Priced against the BSM(vol_21d) baseline and
the best ML volatility models (XGBoost = live pick, VIX-proxy = IV-aligned).

Scenario mechanics:
  * vol_factor  scales every volatility input by 1.5;
  * rate_shift  adds 200 bps to the risk-free rate;
  * vix_factor  re-engineers the VIX-linked features and re-predicts the ML
                volatilities so the models "feel" the VIX jump.

Dependencies: numpy, pandas, shap, xgboost (via tool_engine).
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

import config
import tool_engine as te
from data_preparation import load_dataset


def build_base_state() -> Tuple[pd.Series, float, float, float, float]:
    """
    Live market state used as the scenario / perturbation baseline:
        (feature_row_with_regime, spot, rate, q, vol_21d)
    """
    market = te.fetch_market_history(period='2y')
    feat = te.build_features(market)
    feat = pd.concat([feat, te.regime_dummies(feat)], axis=1).dropna()
    row = feat.iloc[-1]
    return (row, float(row[config.SPOT_COL]), float(row[config.RATE_COL]) / 100.0,
            config.Q_YIELD, float(row['vol_21d']))


def vix_shocked_features(base_row: pd.Series, vix_factor: float = 1.5) -> pd.Series:
    """
    A +50% VIX jump re-engineered through the feature set: the raw VIX level is
    scaled, which moves sentiment_score, vix_ratio, vix_change_1d and
    vix_jpm_cross_1d. Correlation and rate features are untouched.
    """
    out = base_row.copy()
    vix_new = float(base_row[config.VIX_COL]) * vix_factor
    out[config.VIX_COL] = vix_new

    train = load_dataset()
    lo, hi = float(train[config.VIX_COL].min()), float(train[config.VIX_COL].max())
    span = max(hi - lo, 1e-6)
    out['sentiment_score'] = float(np.clip(1 - (vix_new - lo) / span, 0.0, 1.0))
    out['vix_ratio'] = float(base_row['vix_ratio']) * vix_factor
    out['vix_change_1d'] = float(base_row['vix_change_1d']) * vix_factor
    out['vix_jpm_cross_1d'] = float(base_row['vix_jpm_cross_1d']) * vix_factor
    return out


def perturb_price_curve(
    feature: str,
    base_row: pd.Series,
    model_family: str = 'vol_vix_proxy',
    step: Optional[float] = None,
    n_steps: int = 3,
    spot_override: Optional[float] = None,
) -> pd.DataFrame:
    """
    Price the chooser as a single feature is moved away from its base value.
    BSM(vol_21d) baseline is kept fixed at the base vol (except for the vol_21d
    feature itself, where it is re-anchored to the perturbed vol).
    """
    payload, feats, _ = te.load_vol_model(model_family)
    base = base_row.copy()
    x0 = float(base[feature])
    step = step or (abs(x0) * 0.1 + 1e-6)

    K, t1, T2 = config.CHOOSER_PARAMS['K'], config.CHOOSER_PARAMS['t1'], config.CHOOSER_PARAMS['T2']
    spot = spot_override if spot_override is not None else float(base[config.SPOT_COL])
    rate, q = float(base[config.RATE_COL]) / 100.0, config.Q_YIELD
    vol21_base = float(base['vol_21d'])

    rows = []
    for k in range(-n_steps, n_steps + 1):
        row = base.copy()
        row[feature] = x0 + k * step
        X = pd.DataFrame([row[feats].values], columns=feats).astype(float)
        sig_ml = float(te.predict_sigma_from_pipe(payload, X)[0])
        sig_base = float(row['vol_21d']) if feature == 'vol_21d' else vol21_base
        rows.append({
            'feature': feature, 'feature_value': x0 + k * step,
            'sigma_ML': sig_ml, 'sigma_base': sig_base,
            'price_ML_$': te.price_chooser(spot, K, t1, T2, rate, q, sig_ml),
            'price_BSM_$': te.price_chooser(spot, K, t1, T2, rate, q, sig_base),
        })
    return pd.DataFrame(rows)


def extreme_scenarios(
    base_row: pd.Series, spot: float, rate: float, q: float,
    vol21: float, verbose: bool = True,
) -> pd.DataFrame:
    """
    Price the chooser under every extreme scenario for BSM(vol_21d), XGBoost
    (live pick) and the VIX-proxy model. Returns a per-scenario frame with
    absolute prices and % change vs the base scenario.
    """
    K, t1, T2 = config.CHOOSER_PARAMS['K'], config.CHOOSER_PARAMS['t1'], config.CHOOSER_PARAMS['T2']
    payload_xgb, feats_xgb, _ = te.load_vol_model('vol_xgb')
    payload_vp, feats_vp, _ = te.load_vol_model('vol_vix_proxy')

    def _predict(payload, feats, row):
        X = pd.DataFrame([row[feats].values], columns=feats).astype(float)
        return float(te.predict_sigma_from_pipe(payload, X)[0])

    base_sig_xgb = _predict(payload_xgb, feats_xgb, base_row)
    base_sig_vp = _predict(payload_vp, feats_vp, base_row)

    rows = []
    for key, sc in config.SCENARIOS.items():
        if sc['vix_factor'] > 1.0:
            row = vix_shocked_features(base_row, sc['vix_factor'])
            sig_xgb = _predict(payload_xgb, feats_xgb, row) * sc['vol_factor']
            sig_vp = _predict(payload_vp, feats_vp, row) * sc['vol_factor']
        else:
            row = base_row
            sig_xgb = base_sig_xgb * sc['vol_factor']
            sig_vp = base_sig_vp * sc['vol_factor']
        sig_base = vol21 * sc['vol_factor']
        r_eff = rate + sc['rate_shift']

        rows.append({
            'scenario': key, 'label': sc['label'],
            'r_%': round(r_eff * 100, 3),
            'sigma_base_%': round(sig_base * 100, 2),
            'sigma_xgb_%': round(sig_xgb * 100, 2),
            'sigma_vixproxy_%': round(sig_vp * 100, 2),
            'price_BSM_$': te.price_chooser(spot, K, t1, T2, r_eff, q, sig_base),
            'price_XGB_$': te.price_chooser(spot, K, t1, T2, r_eff, q, sig_xgb),
            'price_VIXproxy_$': te.price_chooser(spot, K, t1, T2, r_eff, q, sig_vp),
        })
    out = pd.DataFrame(rows)
    for col in ('price_BSM_$', 'price_XGB_$', 'price_VIXproxy_$'):
        out[f'{col}_pct'] = (out[col] - out.iloc[0][col]) / out.iloc[0][col] * 100
    if verbose:
        print(f'Extreme-scenario table (base S={spot:.2f}, r={rate*100:.2f}%, q={q*100:.2f}%)')
        print(out.round(3).to_string(index=False))
    return out


def t1_sweep(base_row: pd.Series, spot: float, rate: float, q: float,
             vol21: float) -> pd.DataFrame:
    """Price vs choice date t1 under the base and the +50% vol-spike scenario."""
    payload_xgb, feats_xgb, _ = te.load_vol_model('vol_xgb')
    X = pd.DataFrame([base_row[feats_xgb].values], columns=feats_xgb).astype(float)
    sig_ml = float(np.clip(_inner(payload_xgb).predict(X)[0], 1e-4, 1.0))
    K, T2 = config.CHOOSER_PARAMS['K'], config.CHOOSER_PARAMS['T2']
    rows = []
    for t1 in np.arange(0.1, 1.01, 0.1):
        rows.append({'t1_years': round(float(t1), 2),
                     'price_BSM_base': te.price_chooser(spot, K, t1, T2, rate, q, vol21),
                     'price_ML_base': te.price_chooser(spot, K, t1, T2, rate, q, sig_ml),
                     'price_BSM_spike': te.price_chooser(spot, K, t1, T2, rate, q, vol21 * 1.5),
                     'price_ML_spike': te.price_chooser(spot, K, t1, T2, rate, q, sig_ml * 1.5)})
    return pd.DataFrame(rows)


def _inner(payload):
    return payload['pipeline'] if isinstance(payload, dict) else payload
