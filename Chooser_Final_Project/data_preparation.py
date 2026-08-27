"""
Data Preparation
================
Chronological 70/15/15 split and LSTM sequence building designed to prevent
look-ahead bias (validated by Week-5's anti-leakage test suite):

  1. Split is strictly chronological -- never shuffled.
  2. All features at date t are backward-looking (rolling windows built on <= t).
  3. Target at row t describes the FUTURE window [t+1, t+h].
  4. A purge gap (PURGE_GAP_DAYS) is dropped between train->val and val->test so
     target windows straddling the boundary do not leak.
  5. Any scaler is fit on train only and only *transforms* val/test.

Dependencies: numpy, pandas, scikit-learn.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config


def load_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """Load the feature dataset, parse dates and sort chronologically."""
    p = pd.read_csv(path or config.FEATURE_DATASET)
    p[config.DATE_COL] = pd.to_datetime(p[config.DATE_COL])
    return p.sort_values(config.DATE_COL).reset_index(drop=True)


def make_chronological_split(
    n: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    purge_gap: int = 21,
) -> Tuple[slice, slice, slice]:
    """
    Return (train, val, test) index slices over 0..n-1 with purge gaps:

        [----- train -----] [GAP] [--- val ---] [GAP] [--- test ---]
    """
    n_train = int(np.floor(n * train_ratio))
    n_val = int(np.floor(n * val_ratio))
    n_test = n - n_train - n_val - 2 * purge_gap
    assert n_test > 0, 'not enough rows for split + purge gaps'

    val_start = n_train + purge_gap
    test_start = val_start + n_val + purge_gap
    return (
        slice(0, n_train),
        slice(val_start, val_start + n_val),
        slice(test_start, test_start + n_test),
    )


def build_splits(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    target: Optional[str] = None,
    scale: bool = True,
) -> Dict[str, dict]:
    """
    Build 70/15/15 train/val/test splits for a (features, target) task.

    Returns a dict with 'train'/'val'/'test' entries {X, y, idx} where idx are
    positions into the original df (used by make_sequences), plus 'scaler' (fit
    on train only) and 'X_full'/'y_full' (whole frame scaled by that scaler).
    """
    feats = list(features or config.BASE_FEATURES)
    train_sl, val_sl, test_sl = make_chronological_split(len(df))

    if scale:
        scaler = StandardScaler().fit(df.iloc[train_sl][feats])
        full = df.copy()
        full[feats] = scaler.transform(full[feats])
    else:
        scaler, full = None, df.copy()

    out = {}
    for name, sl in (('train', train_sl), ('val', val_sl), ('test', test_sl)):
        sub = full.iloc[sl].reset_index(drop=True)
        entry = {'idx': np.arange(sl.start, sl.stop, dtype=int),
                 'X': sub[feats].astype(np.float64), 'n': len(sub)}
        if target and target in sub.columns:
            entry['y'] = sub[target].astype(np.float64)
        out[name] = entry

    out['scaler'] = scaler
    out['X_full'] = full[feats].astype(np.float64)
    out['y_full'] = df[target].astype(np.float64) if target and target in df.columns else None
    return out


def make_sequences(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    seq_len: int,
    idx: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Build sliding-window sequences for LSTM-style models.

    For every anchor j the sequence covers rows [j-seq_len+1 .. j] -- strictly
    <= j, so no future data enters the input. Anchors without a full lookback
    window are dropped.

    Returns (X_seq (n, seq_len, n_feat), y_seq, valid_anchors).
    """
    arr = X.values
    anchors = np.asarray(idx, dtype=int)
    valid = anchors[anchors >= seq_len - 1]
    Xs = np.empty((len(valid), seq_len, arr.shape[1]), dtype=np.float64)
    for k, j in enumerate(valid):
        Xs[k] = arr[j - seq_len + 1: j + 1]
    ys = None
    if y is not None:
        yv = y.values
        ys = np.asarray([yv[j] for j in valid], dtype=np.float64)
    return Xs, ys, valid
