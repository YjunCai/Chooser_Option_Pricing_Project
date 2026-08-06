"""
Week 5 Experiment Runner -- CLI
===============================
Runs the full dual-track ML framework and prints a summary.

Usage:
    python run_experiments.py            # full run (both approaches)
    python run_experiments.py --feats    # also dump the feature-engineering
                                         # optimization report first
"""

import argparse
import sys

sys.path.insert(0, '.')

import ml_framework


def main():
    ap = argparse.ArgumentParser(description='Week 5 ML framework experiment runner')
    ap.add_argument('--feats', action='store_true',
                    help='run the feature-engineering optimization report first')
    ap.add_argument('--quick', action='store_true',
                    help='skip the LSTM (fast smoke run)')
    args = ap.parse_args()

    if args.feats:
        import logging
        import feature_engineering_optimization as feo
        logging.basicConfig(level=logging.INFO)
        enhanced, analysis, selected = feo.optimize_pipeline()
        print(f'\nSelected {len(selected)} optimized features -> '
              f'output/selected_features.csv\n')

    results = ml_framework.run_framework(verbose=True)

    print('\n' + '=' * 70)
    print('  WEEK 5 SUMMARY')
    print('=' * 70)
    for set_name, res in results['approach1'].items():
        best = res['vol_comparison'].iloc[0]
        print(f'  [A1 {set_name}] best vol model {best["model"]}: MAE={best["MAE"]*100:.2f}%')
    best_p = results['approach2']['comparison'].iloc[0]
    print(f'  [A2] best pricing model {best_p["model"]}: MAE=${best_p["MAE"]:.3f}')
    print('\n  artifacts -> output/  figures -> assets/')


if __name__ == '__main__':
    main()
