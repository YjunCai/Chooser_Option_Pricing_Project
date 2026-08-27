"""
Project Orchestration
=====================
One-command entry point for the tool's pipeline:

  1. generate dashboard figures (price trends + performance metrics);
  2. pricing-engine self-check (dual-track pricing on the latest market state);
  3. real-time data freshness check (data_updater --check).

Usage:
    python run.py          # full pipeline
    python run.py --tests  # run the smoke tests only
"""

import argparse
import time

import config
import dashboard as db
import tool_engine as te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tests', action='store_true', help='run smoke tests only')
    args = ap.parse_args()

    t0 = time.time()
    print('=' * 60)
    print('  CHOOSER OPTION PRICING -- PROJECT PIPELINE')
    print('=' * 60)

    if args.tests:
        import test_smoke
        test_smoke.main()
        return

    print('\n[1/3] Dashboard figures (price trends + performance metrics)...')
    db.run_dashboard()

    print('\n[2/3] Pricing-engine self-check (dual-track on latest market)...')
    from sensitivity import build_base_state
    base_row, spot, rate, q, vol21 = build_base_state()
    payload_xgb, feats_xgb, _ = te.load_vol_model(config.TOOL_LIVE_MODEL)
    sub = base_row[feats_xgb]
    sigma_ml = te.predict_sigma_family(config.TOOL_LIVE_MODEL, payload_xgb, sub, vol21)
    p = te.price_dual(spot, config.CHOOSER_PARAMS['K'], config.CHOOSER_PARAMS['t1'],
                      config.CHOOSER_PARAMS['T2'], rate, q, sigma_ml, vol21)
    print(f'  live dual-track: S={spot:.2f} K={config.CHOOSER_PARAMS["K"]:.0f}')
    print(f'    BSM(vol_21d)={p["price_BSM"]:.4f}$  ML({config.TOOL_LIVE_MODEL})={p["price_ML"]:.4f}$  '
          f'spread={p["spread_$"]:+.4f}$')

    print('\n[3/3] Real-time data freshness check...')
    import data_updater as du
    du.run(mode='check', verbose=False)

    print(f'\nTotal runtime: {time.time() - t0:.1f}s')
    print('Artifacts -> output/, assets/')


if __name__ == '__main__':
    main()
