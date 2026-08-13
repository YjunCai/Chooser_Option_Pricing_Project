"""
Purged Time-Series Cross-Validation -- Week 6
=============================================
Expanding-window chronological CV with purge/embargo, used by the hyper-parameter
search. The searchable region (train + val rows of the outer 70/15/15 split) is
passed here as a *contiguous* array; each fold is

    fold k :  train = rows [0, train_end_k)          (expanding)
              val   = rows [train_end_k + purge, train_end_k + purge + val_size)

so validation always sits strictly *after* the training block, and the purge gap
prevents target windows [t, t+h] that straddle the boundary from leaking into
the validation block.

Guarantees (asserted by test_week6.py):
  * strictly chronological -- training rows are always earlier than val rows
  * purge respected -- >= purge_gap rows dropped at every train->val boundary
  * no overlap between train and val
"""

from typing import Iterator, List, Tuple

import numpy as np

import w6config


def purged_cv_indices(
    n: int,
    n_splits: int = w6config.CV['n_splits'],
    val_frac: float = w6config.CV['val_frac'],
    purge_gap: int = w6config.CV['purge_gap'],
    min_train_frac: float = w6config.CV['min_train_frac'],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Return a list of (train_idx, val_idx) pairs over rows 0..n-1.

    Parameters
    ----------
    n            : number of rows in the (contiguous) searchable region.
    n_splits     : number of folds.
    val_frac     : per-fold validation block length as a fraction of n.
    purge_gap    : rows embargoed between train end and val start.
    min_train_frac : the first fold's train block must span at least this
                   fraction of n (expanding windows thereafter).
    """
    if n <= 0:
        raise ValueError('n must be positive')
    val_size = max(int(np.floor(n * val_frac)), 10)
    # latest allowed train end so the last val block still fits inside n
    last_train_end = n - purge_gap - val_size
    if last_train_end < 10:
        raise ValueError(
            f'n={n} too small for {n_splits} folds with val_size={val_size}, '
            f'purge={purge_gap}')
    min_train = int(np.floor(n * min_train_frac))
    train_ends = np.linspace(min_train, last_train_end, n_splits).astype(int)
    train_ends = np.unique(train_ends)

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for te in train_ends:
        va_start = te + purge_gap
        va_end = min(va_start + val_size, n)
        if va_end <= va_start:
            continue
        folds.append((np.arange(0, te), np.arange(va_start, va_end)))
    if not folds:
        raise ValueError('no valid fold could be constructed')
    return folds


class PurgedTimeSeriesSplit:
    """sklearn-compatible cross-validator wrapping :func:`purged_cv_indices`."""

    def __init__(self, n_splits=None, val_frac=None, purge_gap=None):
        self.n_splits_ = n_splits or w6config.CV['n_splits']
        self.val_frac_ = val_frac or w6config.CV['val_frac']
        self.purge_gap_ = purge_gap if purge_gap is not None else w6config.CV['purge_gap']

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits_

    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = X.shape[0]
        for tr, va in purged_cv_indices(n, n_splits=self.n_splits_,
                                        val_frac=self.val_frac_,
                                        purge_gap=self.purge_gap_):
            yield tr, va


if __name__ == '__main__':
    for k, (tr, va) in enumerate(purged_cv_indices(1000, n_splits=4)):
        print(f'fold {k}: train {len(tr):4d} rows [0,{tr[-1]}] | val {len(va):4d} '
              f'rows [{va[0]},{va[-1]}] | min_gap={va[0]-tr[-1]-1}')
