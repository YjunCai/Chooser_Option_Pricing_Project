"""
Week 7 Real-Time Data Integration -- Market Data Auto-Update Module
==================================================================
Keeps the training feature dataset fresh so the pricing tool always prices
against the latest market state.

Pipeline
--------
    fetch raw history (yfinance: JPM / ^VIX / ^IRX)
        -> build the week-2 feature frame (identical quantities to training)
        -> merge rows AFTER the last dataset date (idempotent, incremental)
        -> write a JSON status (last updated, rows added, source)

CLI
---
    python data_updater.py --check      # report freshness, no writes
    python data_updater.py --dry-run    # fetch + build, report delta, no writes
    python data_updater.py --commit     # fetch + build + merge into the dataset
                                        # (refuses the offline/cache fallback
                                        #  unless --force, to keep the training
                                        #  dataset free of simulated data)

The GitHub Actions workflow `.github/workflows/week7_data_update.yml` schedules
this module on weekdays so the dataset auto-refreshes in CI.
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import w7config as cfg
import tool_engine as te

# standard feature-dataset columns produced by week-2 engineering
FEATURE_COLUMNS = [cfg.DATE_COL, cfg.SPOT_COL, cfg.VIX_COL, cfg.RATE_COL] + list(cfg.BASE_FEATURES)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Fresh raw market history
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_raw_history(period: str = '2y') -> Tuple[pd.DataFrame, str]:
    """
    Return (raw_market_frame, source) where source is 'yfinance' (live) or
    'cache' (offline fallback to the d:/tmp snapshot CSVs).
    """
    try:
        import yfinance as yf
        jpm = yf.Ticker('JPM').history(period=period, auto_adjust=True)
        vix = yf.Ticker('^VIX').history(period=period, auto_adjust=True)
        irx = yf.Ticker('^IRX').history(period=period, auto_adjust=True)
        if len(jpm) < 30 or len(vix) < 30 or len(irx) < 30:
            raise RuntimeError('yfinance returned too few rows')
        # Normalize every source to a naive (tz-free) date index before
        # aligning (see tool_engine.fetch_market_history).
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
        return out, 'yfinance'
    except Exception as exc:
        sys.stderr.write(f'[data_updater] yfinance unavailable ({exc}); '
                         f'falling back to cached snapshot data\n')
        return te._load_cached_market(), 'cache'


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Incremental merge into the training dataset
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_rows(market: pd.DataFrame) -> pd.DataFrame:
    """Full feature frame (all dates) from raw market data."""
    feat = te.build_features(market)
    feat.index.name = cfg.DATE_COL
    return feat.reset_index()


def load_dataset() -> pd.DataFrame:
    if not cfg.WEEK2_FEATURE_DATASET.exists():
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    df = pd.read_csv(cfg.WEEK2_FEATURE_DATASET)
    df[cfg.DATE_COL] = pd.to_datetime(df[cfg.DATE_COL])
    return df.sort_values(cfg.DATE_COL).reset_index(drop=True)


def incremental_rows(new_feat: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """
    Rows in `new_feat` whose date is AFTER the last date of `existing`
    (idempotent incremental update). Overlapping dates keep the existing rows.
    """
    if len(existing) == 0:
        return new_feat.copy()
    last_date = pd.to_datetime(existing[cfg.DATE_COL]).max()
    add = new_feat[pd.to_datetime(new_feat[cfg.DATE_COL]) > last_date].copy()
    return add.reset_index(drop=True)


def merge_into_dataset(new_feat: pd.DataFrame) -> Dict[str, object]:
    """Append new rows to week2/data/feature_dataset.csv; return a status dict."""
    existing = load_dataset()
    add = incremental_rows(new_feat, existing)
    cols = [c for c in FEATURE_COLUMNS if c in new_feat.columns]
    add = add[cols].copy()
    for c in cols:
        if c == cfg.DATE_COL:
            continue                     # keep date as datetime; pd.to_numeric
                                          # would turn it into int nanoseconds
        add[c] = pd.to_numeric(add[c], errors='coerce')
    merged = pd.concat([existing, add], ignore_index=True)
    merged = merged.drop_duplicates(subset=[cfg.DATE_COL], keep='last')
    merged = merged.sort_values(cfg.DATE_COL).reset_index(drop=True)
    cfg.WEEK2_FEATURE_DATASET.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cfg.WEEK2_FEATURE_DATASET, index=False)
    return {'rows_added': int(len(add)), 'rows_total': int(len(merged))}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Status reporting
# ═══════════════════════════════════════════════════════════════════════════════

def write_status(status: Dict[str, object]):
    with open(cfg.UPDATER_STATUS, 'w', encoding='utf-8') as f:
        json.dump({**status, 'as_of': datetime.now().isoformat(timespec='seconds')},
                  f, ensure_ascii=False, indent=2)


def read_status() -> Dict[str, object]:
    if not cfg.UPDATER_STATUS.exists():
        return {}
    with open(cfg.UPDATER_STATUS, encoding='utf-8') as f:
        return json.load(f)


def run(mode: str = 'check', force: bool = False, verbose: bool = True) -> Dict:
    market, source = fetch_raw_history()
    new_feat = build_feature_rows(market)
    existing = load_dataset()
    add = incremental_rows(new_feat, existing)

    status = {
        'mode': mode,
        'source': source,
        'latest_raw_date': str(market.index.max().date()),
        'latest_feature_date': str(new_feat[cfg.DATE_COL].max()),
        'dataset_last_date': str(pd.to_datetime(existing[cfg.DATE_COL]).max())
        if len(existing) else None,
        'dataset_rows': int(len(existing)),
        'rows_to_add': int(len(add)),
        'fetched_rows': int(len(market)),
        'fetched': True,
    }

    if mode == 'check':
        pass
    elif mode in ('dry_run', 'commit'):
        latest_spot = float(market['close'].iloc[-1])
        latest_vix = float(market['vix'].iloc[-1])
        latest_rate = float(market['rate'].iloc[-1])
        status.update(latest_spot=latest_spot, latest_vix=latest_vix,
                      latest_rate=latest_rate)
        if mode == 'commit':
            if source == 'cache' and not force:
                status['committed'] = False
                status['error'] = ('refusing to commit offline cache fallback '
                                   '(simulated data); use --force to override')
            else:
                status['committed'] = True
                status.update(merge_into_dataset(new_feat))

    if verbose:
        _print(status)
    if mode != 'check':
        write_status(status)
    return status


def _print(s: Dict):
    print('=' * 60)
    print('  WEEK 7 -- REAL-TIME DATA INTEGRATION')
    print('=' * 60)
    print(f"  source             : {s.get('source')}")
    print(f"  latest raw date    : {s.get('latest_raw_date')}")
    print(f"  latest feature date: {s.get('latest_feature_date')}")
    print(f"  dataset last date  : {s.get('dataset_last_date')}")
    print(f"  dataset rows       : {s.get('dataset_rows')}")
    print(f"  rows to add        : {s.get('rows_to_add')}")
    print(f"  latest spot        : {s.get('latest_spot', '--')}")
    print(f"  latest VIX         : {s.get('latest_vix', '--')}")
    print(f"  latest 3M rate     : {s.get('latest_rate', '--')}")
    if s.get('committed') is True:
        print(f"  committed          : yes ({s['rows_added']} rows added, "
              f"{s['rows_total']} total)")
    if s.get('error'):
        print(f"  error              : {s['error']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Week 7 real-time data auto-update')
    ap.add_argument('--check', action='store_true', help='report freshness only')
    ap.add_argument('--dry-run', action='store_true', help='fetch+build, no write')
    ap.add_argument('--commit', action='store_true', help='merge into dataset')
    ap.add_argument('--force', action='store_true', help='allow cache-fallback commit')
    args = ap.parse_args()
    mode = 'commit' if args.commit else ('dry_run' if args.dry_run else 'check')
    run(mode=mode, force=args.force)
