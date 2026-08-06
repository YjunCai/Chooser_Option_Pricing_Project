"""
Time-Series Data Preparation -- Week 5
======================================
Chronological 70/15/15 split designed to prevent look-ahead bias.

Key anti-leakage rules:
  1. Split is strictly chronological -- never shuffle.
  2. All features at date t are backward-looking (rolling windows built on t<=t).
  3. Target alignment: the label at row t describes the FUTURE window [t+1, t+h],
     so no feature ever encodes information that was not yet available.
  4. Purge/embargo: a gap of PURGE_GAP_DAYS is dropped between train->val and
     val->test so that target windows straddling the boundary do not leak.
  5. Any scaler (StandardScaler etc.) must be fit on train and only *transform*
     val/test -- handled by build_splits() via the transform callbacks.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config


def load_dataset(path=None) -> pd.DataFrame:
    """Load the week2 feature dataset, parse dates and sort chronologically."""
    p = pd.read_csv(path or config.FEATURE_DATASET)
    p[config.DATE_COL] = pd.to_datetime(p[config.DATE_COL])
    p = p.sort_values(config.DATE_COL).reset_index(drop=True)
    return p


def make_chronological_split(
    n: int,
    train_ratio: float = config.SPLIT['train'],
    val_ratio: float = config.SPLIT['val'],
    purge_gap: int = config.PURGE_GAP_DAYS,
) -> Tuple[slice, slice, slice]:
    """
    Return (train_idx, val_idx, test_idx) as pandas index slices over 0..n-1.

    Chronological split with purge gaps removed at the boundaries:
        [----- train -----] [GAP] [--- val ---] [GAP] [--- test ---]
    The gap rows are dropped from all sets to avoid target-window leakage.
    """
    n_train = int(np.floor(n * train_ratio))
    n_val = int(np.floor(n * val_ratio))
    n_test = n - n_train - n_val - 2 * purge_gap

    assert n_test > 0, 'not enough rows for split + purge gaps'

    train_end = n_train
    val_start = train_end + purge_gap
    val_end = val_start + n_val
    test_start = val_end + purge_gap
    test_end = test_start + n_test

    return (
        slice(0, train_end),
        slice(val_start, val_end),
        slice(test_start, test_end),
    )


def _scaled(df: pd.DataFrame, train_rows: pd.DataFrame, feats: List[str]) -> pd.DataFrame:
    """Return a copy of df with feats standardized by a scaler fit on train_rows."""
    out = df.copy()
    sc = StandardScaler().fit(train_rows[feats])
    out[feats] = sc.transform(out[feats])
    return out


def build_splits(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    target: Optional[str] = None,
    scale: bool = True,
    purge_gap: int = config.PURGE_GAP_DAYS,
) -> Dict[str, dict]:
    """
    Build the 70/15/15 train/val/test splits for a single (features, target) task.

    Parameters
    ----------
    df        : chronological feature frame with a target column already added.
    features  : feature columns (default config.BASE_FEATURES).
    target    : label column name (default None -> only X splits are produced).
    scale     : standardize features with a train-fit scaler.

    Returns
    -------
    dict with:
      'train'/'val'/'test' : {'X', 'y', 'idx'} where idx are positions into the
                             ORIGINAL df (used by the LSTM window builder) and
                             X is the *subset* feature frame (reset index).
      'scaler'             : the StandardScaler fit on train (or None).
      'X_full'             : full feature frame scaled by the train scaler.
      'y_full'             : full target series (aligned with X_full).
    """
    feats = list(features or config.BASE_FEATURES)
    train_sl, val_sl, test_sl = make_chronological_split(len(df), purge_gap=purge_gap)

    if scale:
        scaler = StandardScaler().fit(df.iloc[train_sl][feats])
        full = df.copy()
        full[feats] = scaler.transform(full[feats])
    else:
        scaler = None
        full = df.copy()

    out = {}
    for name, sl in (('train', train_sl), ('val', val_sl), ('test', test_sl)):
        sub = full.iloc[sl].reset_index(drop=True)
        entry = {
            'idx': np.arange(sl.start, sl.stop, dtype=int),
            'X': sub[feats].astype(np.float64),
            'n': len(sub),
        }
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
    Build sliding-window sequences for the LSTM / windowed models.

    `X` and `y` must be the FULL chronological frame (e.g. split['X_full'] /
    split['y_full']); `idx` holds the anchor row positions in that frame. For
    every anchor j the sequence covers rows [j-seq_len+1 .. j] -- strictly
    <= j, so no future data enters the input.

    Returns
    -------
    (X_seq, y_seq, valid_anchors): X_seq shape (n_valid, seq_len, n_features),
    y_seq aligned to X_seq, valid_anchors = idx filtered to anchors with a full
    lookback window.
    """
    feats = list(X.columns)
    arr = X.values
    rows, n_feat = arr.shape
    anchors = np.asarray(idx, dtype=int)

    valid = anchors[anchors >= seq_len - 1]
    Xs = np.empty((len(valid), seq_len, n_feat), dtype=np.float64)
    for k, j in enumerate(valid):
        Xs[k] = arr[j - seq_len + 1: j + 1]

    ys = None
    if y is not None:
        yv = y.values
        ys = np.asarray([yv[j] for j in valid], dtype=np.float64)
    return Xs, ys, valid


def report_split(df: pd.DataFrame, train_sl, val_sl, test_sl) -> str:
    """Human-readable split report (dates + counts)."""
    dates = pd.to_datetime(df[config.DATE_COL])
    lines = [
        'Time-series split (chronological, purged)',
        f'  rows         : {len(df)}',
        f'  train {config.SPLIT["train"]:.0%}  : {train_sl.stop - train_sl.start:5d}  '
        f'({dates[train_sl.start]:%Y-%m-%d} -> {dates[train_sl.stop-1]:%Y-%m-%d})',
        f'  val   {config.SPLIT["val"]:.0%}  : {val_sl.stop - val_sl.start:5d}  '
        f'({dates[val_sl.start]:%Y-%m-%d} -> {dates[val_sl.stop-1]:%Y-%m-%d})',
        f'  test  {config.SPLIT["test"]:.0%}  : {test_sl.stop - test_sl.start:5d}  '
        f'({dates[test_sl.start]:%Y-%m-%d} -> {dates[test_sl.stop-1]:%Y-%m-%d})',
    ]
    return '\n'.join(lines)


if __name__ == '__main__':
    d = load_dataset()
    tr, va, te = make_chronological_split(len(d))
    print(report_split(d, tr, va, te))
