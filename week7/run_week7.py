"""
Week 7 Main Orchestration
=========================
Runs the complete Week 7 deliverable pipeline:

  1. Extended sensitivity analysis:
       - SHAP x vega price-impact decomposition of the new features
         (sentiment, VIX) on the chooser price;
       - univariate price-perturbation curves (ATM base);
       - extreme-scenario testing (vol +50%, rate +2%, combined, VIX shock)
         at the live base and at the ATM canonical base;
       - choice-date (t1) sweep.
  2. Pricing-tool engine self-check (dual-track pricing + error margins).
  3. Real-time data freshness report (data_updater --check).
  4. Report compilation (Week_7_实验报告.tex -> Week_7_实验报告.pdf).

Usage:
    python run_week7.py            # full pipeline (default)
    python run_week7.py --no-tex   # skip LaTeX compilation
    python run_week7.py --tests    # run unit tests then exit
"""

import argparse
import subprocess
import sys
import time

import w7config as cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-tex', action='store_true', help='skip LaTeX compilation')
    ap.add_argument('--tests', action='store_true', help='run unit tests only')
    args = ap.parse_args()

    t0 = time.time()
    print('=' * 72)
    print('  WEEK 7 -- ADVANCED SENSITIVITY ANALYSIS & TOOL DEVELOPMENT')
    print('=' * 72)

    if args.tests:
        import test_week7
        test_week7.test_tool_engine()
        test_week7.test_sensitivity()
        test_week7.test_data_updater()
        test_week7.test_streamlit_app()
        print('\nAll Week-7 tests passed.')
        return

    # 1. sensitivity analysis
    print('\n[1/4] Extended sensitivity analysis...')
    import sensitivity as sens
    res = sens.run_sensitivity()
    impact = res['impact_vp']
    sc_atm = res['scenarios_atm']
    print('    -> output/sensitivity_*.csv, assets/fig_w7_*.png')

    # 2. tool engine self-check
    print('\n[2/4] Pricing-tool engine self-check...')
    import tool_engine as te
    base_row, spot, rate, q, vol21 = res['base_state']
    payload_xgb, feats_xgb, _ = te.load_vol_model(cfg.TOOL_LIVE_MODEL)
    sigma_ml = te.predict_sigma(payload_xgb, base_row[feats_xgb])
    p = te.price_dual(spot, cfg.CHOOSER_PARAMS['K'], cfg.CHOOSER_PARAMS['t1'],
                      cfg.CHOOSER_PARAMS['T2'], rate, q, sigma_ml, vol21)
    print(f'    live dual-track: S={spot:.2f} K={cfg.CHOOSER_PARAMS["K"]:.0f} '
          f't1={cfg.CHOOSER_PARAMS["t1"]} T2={cfg.CHOOSER_PARAMS["T2"]}')
    print(f'      BSM(vol_21d)={p["price_BSM"]:.4f}$  ML(XGB)={p["price_ML"]:.4f}$  '
          f'spread={p["spread_$"]:+.4f}$')

    # 3. real-time data freshness
    print('\n[3/4] Real-time data freshness check...')
    import data_updater as du
    du.run(mode='check', verbose=False)

    # 4. report
    print('\n[4/4] Compiling report...')
    if not args.no_tex:
        ok = compile_report()
        print(f'    -> Week_7_实验报告.pdf  {"OK" if ok else "FAILED"}')
    else:
        print('    skipped (--no-tex)')

    print(f'\nTotal runtime: {time.time()-t0:.1f}s')
    print('Artifacts -> output/, assets/, Week_7_实验报告.pdf')


def compile_report(runs: int = 2) -> bool:
    """
    Compile the Week-7 LaTeX report to PDF (xelatex for CJK).

    The report filename contains CJK characters; passing it directly to
    xelatex on Windows mangles the bytes (mojibake output file). We therefore
    compile an ASCII-named copy (Week7_build.tex) and rename the result to
    Week_7_实验报告.pdf. `runs` passes are used to resolve cross-references.
    """
    if not cfg.REPORT_TEX.exists():
        print('    report .tex not found; skipping')
        return False
    import shutil
    exe = shutil.which('xelatex') or shutil.which('pdflatex')
    if exe is None:
        print('    no LaTeX engine found; skipping')
        return False
    tmp_tex = cfg.W7_DIR / 'Week7_build.tex'
    tmp_pdf = cfg.W7_DIR / 'Week7_build.pdf'
    shutil.copy(cfg.REPORT_TEX, tmp_tex)

    try:
        for _ in range(runs):
            cmd = [exe, '-interaction=nonstopmode', '-halt-on-error', tmp_tex.name]
            r = subprocess.run(cmd, cwd=str(cfg.W7_DIR), capture_output=True, text=True,
                               encoding='utf-8', errors='replace')
            if r.returncode != 0:
                print('    LaTeX error (see below)')
                print(r.stdout[-1500:])
                return False
        if not tmp_pdf.exists():
            print('    build did not produce Week7_build.pdf')
            return False
        try:
            tmp_pdf.replace(cfg.REPORT_PDF)          # -> Week_7_实验报告.pdf
        except PermissionError:
            # target is open in a PDF viewer; keep a versioned copy instead
            fallback = cfg.W7_DIR / 'Week_7_实验报告_v2.pdf'
            tmp_pdf.replace(fallback)
            print(f'    Week_7_实验报告.pdf is locked (open in a viewer); '
                  f'wrote {fallback.name} instead')
        return cfg.REPORT_PDF.exists() or (cfg.W7_DIR / 'Week_7_实验报告_v2.pdf').exists()
    finally:
        for suffix in ('.tex', '.pdf', '.aux', '.log', '.out', '.toc'):
            p = cfg.W7_DIR / f'Week7_build{suffix}'
            if p.exists():
                p.unlink()


if __name__ == '__main__':
    main()
