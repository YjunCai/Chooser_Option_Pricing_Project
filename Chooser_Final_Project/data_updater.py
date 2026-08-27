"""
Real-Time Data Updater
======================
Keeps the training feature dataset fresh so the pricing tool always prices
against the latest market state.

Pipeline:
    fetch raw history (yfinance: JPM / ^VIX / ^IRX)
      -> build the feature frame (identical quantities to training)
      -> merge rows AFTER the last dataset date (idempotent, incremental)
      -> write a JSON status (last updated, rows added, source)

CLI:
    python data_updater.py --check      # report freshness, no writes
    python data_updater.py --dry-run    # fetch + build, report delta, no writes
    python data_updater.py --commit     # fetch + build + merge into the dataset
                                        # (refuses the offline/cache fallback
                                        #  unless --force)

Dependencies: numpy, pandas, yfinance.
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

import config
import tool_engine as te

FEATURE_COLUMNS = [config.DATE_COL, config.SPOT_COL, config.VIX_COL, config.RATE_COL] + list(config.BASE_FEATURES)


def fetch_raw_history(period: str = '2y') -> Tuple[pd.DataFrame, str]:
    """Return (raw_market_frame, source) where source is 'yfinance' or 'cache'."""
    try:
        market = te.fetch_market_history(period=period)
        if len(market) < 30:
            raise RuntimeError('too few rows')
        return market, 'yfinance'
    except Exception as exc:
        sys.stderr.write(f'[data_updater] yfinance unavailable ({exc}); '
                         f'falling back to cached snapshot data\n')
        return te._load_cached_market(), 'cache'


def build_feature_rows(market: pd.DataFrame) -> pd.DataFrame:
    """Full feature frame (all dates) from raw market data."""
    feat = te.build_features(market)
    feat.index.name = config.DATE_COL
    return feat.reset_index()


def load_dataset() -> pd.DataFrame:
    if not config.FEATURE_DATASET.exists():
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    df = pd.read_csv(config.FEATURE_DATASET)
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    return df.sort_values(config.DATE_COL).reset_index(drop=True)


def incremental_rows(new_feat: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Rows in `new_feat` whose date is AFTER the last date of `existing`."""
    if len(existing) == 0:
        return new_feat.copy()
    last_date = pd.to_datetime(existing[config.DATE_COL]).max()
    add = new_feat[pd.to_datetime(new_feat[config.DATE_COL]) > last_date].copy()
    return add.reset_index(drop=True)


def merge_into_dataset(new_feat: pd.DataFrame) -> Dict[str, object]:
    """Append new rows to the feature dataset; return a status dict."""
    existing = load_dataset()
    add = incremental_rows(new_feat, existing)
    cols = [c for c in FEATURE_COLUMNS if c in new_feat.columns]
    add = add[cols].copy()
    for c in cols:
        if c == config.DATE_COL:
            continue                     # keep date as datetime; pd.to_numeric
                                          # would turn it into int nanoseconds
        add[c] = pd.to_numeric(add[c], errors='coerce')
    merged = pd.concat([existing, add], ignore_index=True)
    merged = merged.drop_duplicates(subset=[config.DATE_COL], keep='last')
    merged = merged.sort_values(config.DATE_COL).reset_index(drop=True)
    config.FEATURE_DATASET.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(config.FEATURE_DATASET, index=False)
    return {'rows_added': int(len(add)), 'rows_total': int(len(merged))}


def write_status(status: Dict[str, object]):
    path = config.OUTPUT_DIR / 'updater_status.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({**status, 'as_of': datetime.now().isoformat(timespec='seconds')},
                  f, ensure_ascii=False, indent=2)


def read_status() -> Dict[str, object]:
    path = config.OUTPUT_DIR / 'updater_status.json'
    if not path.exists():
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def run(mode: str = 'check', force: bool = False, verbose: bool = True) -> Dict:
    market, source = fetch_raw_history()
    new_feat = build_feature_rows(market)
    existing = load_dataset()
    add = incremental_rows(new_feat, existing)

    status = {
        'mode': mode, 'source': source,
        'latest_raw_date': str(market.index.max().date()),
        'latest_feature_date': str(new_feat[config.DATE_COL].max()),
        'dataset_last_date': str(pd.to_datetime(existing[config.DATE_COL]).max())
        if len(existing) else None,
        'dataset_rows': int(len(existing)),
        'rows_to_add': int(len(add)),
        'fetched_rows': int(len(market)),
    }

    if mode == 'check':
        pass
    elif mode in ('dry_run', 'commit'):
        status.update(latest_spot=float(market['close'].iloc[-1]),
                      latest_vix=float(market['vix'].iloc[-1]),
                      latest_rate=float(market['rate'].iloc[-1]))
        if mode == 'commit':
            if source == 'cache' and not force:
                status['committed'] = False
                status['error'] = ('refusing to commit offline cache fallback; '
                                   'use --force to override')
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
    print('  REAL-TIME DATA INTEGRATION')
    print('=' * 60)
    for key in ('source', 'latest_raw_date', 'latest_feature_date',
                'dataset_last_date', 'dataset_rows', 'rows_to_add'):
        print(f'  {key:20s}: {s.get(key)}')
    if s.get('committed') is True:
        print(f"  committed          : yes ({s['rows_added']} rows added, "
              f"{s['rows_total']} total)")
    if s.get('error'):
        print(f"  error              : {s['error']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Real-time data auto-update')
    ap.add_argument('--check', action='store_true', help='report freshness only')
    ap.add_argument('--dry-run', action='store_true', help='fetch+build, no write')
    ap.add_argument('--commit', action='store_true', help='merge into dataset')
    ap.add_argument('--force', action='store_true', help='allow cache-fallback commit')
    args = ap.parse_args()
    mode = 'commit' if args.commit else ('dry_run' if args.dry_run else 'check')
    run(mode=mode, force=args.force)
