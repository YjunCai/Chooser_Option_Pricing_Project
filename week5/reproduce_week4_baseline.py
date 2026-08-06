# -*- coding: utf-8 -*-
"""Reproduce Week-4 live-market baseline metrics from the 595-contract snapshot
and print them, so the Week-5 ML comparison in the report can be validated.

Columns are renamed to ASCII by POSITION to avoid any source-encoding issue
with the Chinese header (the CSV itself is UTF-8 with BOM).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

WEEK5 = Path(__file__).resolve().parent
WEEK4 = WEEK5.parent / 'week4'
WEEK3 = WEEK5.parent / 'week3'
sys.path.insert(0, str(WEEK3))

from chooser_option_pricer import bs_call, bs_put  # noqa: E402

COL_NAMES = [
    'expiry', 'T', 'strike', 'otype', 'mkt_price', 'bid', 'ask', 'YahooIV',
    'volume', 'open_int', 'bsm_price', 'mid', 'abs_err', 'sq_err', 'bias',
    'bias_pct', 'mape_pct', 'moneyness', 'log_moneyness', 'tenor_bucket',
    'implied_iv', 'iv_vol21_gap', 'par_S', 'par_r', 'par_q', 'par_vol_21d',
]

df = pd.read_csv(WEEK4 / 'data' / 'jpm_options_595contracts.csv',
                 encoding='utf-8-sig', header=0)
df.columns = COL_NAMES[:len(df.columns)]

S = float(df['par_S'].iloc[0])
r = float(df['par_r'].iloc[0])
q = float(df['par_q'].iloc[0])
vol21 = float(df['par_vol_21d'].iloc[0])
print(f'market params: S={S:.2f} r={r:.4f} q={q:.4f} vol_21d={vol21:.4f}')

# moneyness buckets, matching week4's exact grouping.
# log_moneyness = ln(K/S). ITM = call K<S  OR  put K>S.  ATM band = |ln(K/S)|<5%.
df['atm'] = df['log_moneyness'].abs() < 0.05
df['itm'] = ((df['otype'] == 'call') & (df['log_moneyness'] < 0)) | \
            ((df['otype'] == 'put') & (df['log_moneyness'] > 0))
atm_otm = df[~df['itm']]                     # 323 contracts, matches week4 "ATM+OTM"

def bias_pct(grp):
    return np.mean((grp['bsm_price'] - grp['mkt_price']) / grp['mkt_price']) * 100

mae = np.mean(np.abs(atm_otm['bsm_price'] - atm_otm['mkt_price']))
rmse = np.sqrt(np.mean((atm_otm['bsm_price'] - atm_otm['mkt_price']) ** 2))
atm_iv = df.loc[df['atm'], 'implied_iv'].mean()
otm_puts = df[(df['otype'] == 'put') & (~df['itm'])]

print('\n=== Week-4 baseline reproduction (BSM vol_21d) ===')
print(f'  N (ATM+OTM)        : {len(atm_otm)}   (week4: 323)')
print(f'  N (deep ITM)       : {len(df[df["itm"]])}   (week4: 272)')
print(f'  MAE                : ${mae:.2f}   (week4 report: $1.44)')
print(f'  RMSE               : ${rmse:.2f}   (week4 report: $2.10)')
print(f'  ATM implied vol    : {atm_iv*100:.1f}%   (week4 report: 25.2%)')
print(f'  ATM IV premium     : {(atm_iv-vol21)*100:.1f}%  (week4 report: 4.9%)')
print(f'  OTM put bias (ATM+OTM group) : {bias_pct(atm_otm):.1f}%  (week4: -65.7%)')
print(f'  OTM put bias (puts only)     : {bias_pct(otm_puts):.1f}%')
