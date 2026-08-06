"""
Evaluation Metrics & Baseline Comparison -- Week 5
==================================================
Standard regression metrics (MAE, RMSE, MAPE, R^2, bias) plus baseline-aware
comparison helpers:

  - evaluate_predictions(y_true, y_pred)            -> metric dict
  - compare_vol_models(test, model_predictions)     -> volatility comparison table
  - price_errors(chooser_true, chooser_model)       -> pricing MAE / RMSE / R^2

The BSM baseline always refers to the static vol_21d input established in
Week 4 (MAE = $1.44 on OTM/ATM vanilla, ATM IV premium 4.9%).
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(y_true, y_pred, mape_min_abs: float = 0.0) -> Dict[str, float]:
    """
    Compute the standard metric set for a regression prediction.

    mape_min_abs : MAPE is only computed over samples with |y_true| >= this
                   value, because percentage error is unstable for tiny prices
                   (e.g. deep-OTM options quoted in cents).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(yt, yp)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mask = np.abs(yt) >= mape_min_abs
    if mask.sum() == 0:
        mape = float('nan')
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = np.abs(yt[mask] - yp[mask]) / np.abs(yt[mask])
            rel = rel[np.isfinite(rel)]          # drop near-zero-denominator points
            mape = float(np.mean(rel) * 100) if len(rel) else float('nan')
    bias = float(np.mean(yp - yt))
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE%': mape,
        'R2': float(r2_score(yt, yp)),
        'bias': bias,
    }


def format_metric_table(rows: List[Dict], cols: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if cols:
        df = df[cols]
    return df


def compare_vol_models(
    y_true: Sequence[float],
    predictions: Dict[str, Sequence[float]],
    baselines: Optional[Dict[str, Sequence[float]]] = None,
    baseline_name: str = 'BSM(vol_21d) persistence',
    baseline: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """
    Build a comparison table of volatility-prediction models against baselines.

    baselines : dict {name: prediction} of reference forecasts, e.g.
                {'BSM(vol_21d) persistence': vol_21d,
                 'historical mean vol': mean_vol}. Every model is measured as
                MAE/RMSE improvement over the FIRST (primary) baseline.
    """
    yt = np.asarray(y_true, dtype=float)
    if baselines is None:
        if baseline is None:
            raise ValueError('a baseline (persistence / vol_21d) is required')
        baselines = {baseline_name: baseline}

    rows = []
    for bname, bpred in baselines.items():
        rows.append({**{'model': bname}, **evaluate_predictions(yt, bpred)})
    for name, pred in predictions.items():
        rows.append({**{'model': name}, **evaluate_predictions(yt, pred)})

    df = pd.DataFrame(rows)
    # improvement of each model over the primary (first) baseline
    primary_mae = float(df['MAE'].iloc[0])
    df['MAE_improve_vs_base_%'] = (primary_mae - df['MAE']) / primary_mae * 100
    df = df.sort_values('MAE').reset_index(drop=True)
    return df


def price_errors(
    chooser_true: Sequence[float],
    chooser_pred: Sequence[float],
) -> Dict[str, float]:
    """Pricing error metrics for a chooser-price series (dollar terms)."""
    return evaluate_predictions(chooser_true, chooser_pred)


def summarize_pricing_comparison(
    chooser_true: pd.Series,
    price_dict: Dict[str, pd.Series],
    baseline_name: str = 'BSM(vol_21d)',
    baseline: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Comparison of chooser-price error vs the fair (forward-vol) price."""
    yt = np.asarray(chooser_true, dtype=float)
    rows = []

    def _add(name, series):
        rows.append({**{'model': name},
                     **evaluate_predictions(yt, np.asarray(series, dtype=float))})

    if baseline is not None:
        _add(baseline_name, baseline)
    for name, s in price_dict.items():
        _add(name, s)

    df = pd.DataFrame(rows).sort_values('MAE').reset_index(drop=True)
    return df
