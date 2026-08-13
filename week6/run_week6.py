"""
Week 6 Main Orchestration
=========================
Runs the complete Week 6 pipeline end-to-end:

  1. Hyper-parameter optimization (purged time-series CV + random search)
     -- Approach 1 (rf/gbdt/xgb/gbdt_anchored/lstm) & Approach 2
        (pricing_gbdt/pricing_nn), with regime-adaptive features.
  2. Final model training on train+val, one-shot evaluation on the held-out
     test set (vol + chooser price + end-to-end price).
  3. Performance comparison vs the BSM baseline (MAE / RMSE / R2) and
     before/after-tuning + regime-stratified analysis.
  4. Pickle export of every final model.
  5. SHAP / LIME interpretability (feature-importance plots).
  6. Live-market metrics on the 2026-07-27 snapshot vs the Week 6 targets.

Usage:
    python run_week6.py --search   # re-run hyper-parameter search
    python run_week6.py            # reuse cached search results (default)
"""

import argparse
import pickle
import sys
import time

import w6config
import hp_search as hp
import train_final as tf
import performance as perf
import interpretability as itp
import live_metrics as lm
import regime as regime_mod
from data_preparation import load_dataset
from targets import add_fwd_vol_targets


def _load_or_search(cache, do_search, fn, verbose=True):
    if do_search:
        res = fn()
        with open(w6config.OUTPUT_DIR / cache, 'wb') as f:
            pickle.dump(res, f)
        return res
    with open(w6config.OUTPUT_DIR / cache, 'rb') as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--search', action='store_true',
                    help='re-run hyper-parameter search (default: reuse cache)')
    args = ap.parse_args()

    t0 = time.time()
    print('=' * 72)
    print('  WEEK 6 -- MODEL TRAINING, TUNING & COMPARISON')
    print('=' * 72)

    # 1. hyper-parameter search (Approach 1 + 2)
    print('\n[1/6] Hyper-parameter search (purged time-series CV)...')
    vol_search = _load_or_search('vol_search_results.pkl', args.search,
                                 lambda: hp.run_vol_search(regime=True))
    price_search = _load_or_search('price_search_results.pkl', args.search,
                                   lambda: hp.search_price_models(*hp.load_approach2_frame()))
    print(hp.summarize_search(vol_search).to_string(index=False))

    # 2. final training + test evaluation
    print('\n[2/6] Final model training & test evaluation...')
    a1 = tf.run_approach1_final(vol_search)
    a2 = tf.run_approach2_final(price_search)

    # 3. performance comparison
    print('\n[3/6] Performance comparison vs BSM baseline...')
    perf.before_after_vol_table(a1)
    perf.chooser_price_table(a1)
    perf.approach2_price_table(a2)
    regime_series = regime_mod.add_regime_features(
        add_fwd_vol_targets(load_dataset()))['regime']
    a1['_te'] = a1['te']
    regime_err = perf.regime_error_analysis(a1, regime_series, best_key=_best(a1))
    perf.plot_regime_errors(regime_err)
    perf.consolidated_summary(a1, a2)
    perf.plot_comparisons(a1, a2)
    print('    -> output/*.csv , assets/fig_w6_*.png')

    # 4. pickle export
    print('\n[4/6] Exporting final models (pickle)...')
    tf.export_models(a1, a2)

    # 5. interpretability
    print('\n[5/6] SHAP / LIME interpretability...')
    itp.run_interpretability(a1, a2)

    # 6. live-market metrics vs Week 6 targets
    print('\n[6/6] Live-market metrics (2026-07-27 snapshot)...')
    live = lm.compute_live_metrics()
    lm.plot_live_metrics(live)

    print(f'\nTotal runtime: {time.time()-t0:.1f}s')
    print('\nArtifacts -> output/, assets/, models/')


def _best(a1: dict) -> str:
    pc = a1['price_comparison']
    if 'XGB-VIX proxy (IV)' in pc['model'].values:
        return 'vix_proxy'
    return a1['metrics'].set_index('family')['MAE'].idxmin()


if __name__ == '__main__':
    main()
