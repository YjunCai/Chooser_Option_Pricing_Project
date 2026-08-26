"""
Week 8 Main Orchestration -- Tool Finalization & Project Delivery
=================================================================
Runs the complete Week-8 deliverable pipeline:

  1. Dashboard figures (price trends + performance metrics) and CSVs.
  2. Pricing-tool engine self-check (dual pricing + error margins).
  3. (Optional) re-run the Week-7 sensitivity analysis if assets are missing.
  4. Report compilation (Week_8_实验报告.tex -> E:/实习交付/实验报告/Week_8_实验报告.pdf).
  5. Presentation deck (Week_8_Final_Presentation.pptx).

Usage:
    python run_week8.py             # full pipeline (default)
    python run_week8.py --no-tex    # skip LaTeX compilation
    python run_week8.py --tests     # run unit tests then exit
    python run_week8.py --no-pptx   # skip the presentation build
"""

import argparse
import shutil
import subprocess
import sys
import time

import w8config as cfg


def compile_report(runs: int = 2) -> bool:
    """
    Compile the Week-8 LaTeX report to PDF (xelatex for CJK). The report
    filename contains CJK characters; we compile an ASCII-named copy
    (Week8_build.tex) and rename the result to Week_8_实验报告.pdf.
    """
    if not cfg.REPORT_TEX.exists():
        print('    report .tex not found; skipping')
        return False
    exe = shutil.which('xelatex') or shutil.which('pdflatex')
    if exe is None:
        print('    no LaTeX engine found; skipping')
        return False
    tmp_tex = cfg.W8_DIR / 'Week8_build.tex'
    tmp_pdf = cfg.W8_DIR / 'Week8_build.pdf'
    shutil.copy(cfg.REPORT_TEX, tmp_tex)
    try:
        for _ in range(runs):
            cmd = [exe, '-interaction=nonstopmode', '-halt-on-error', tmp_tex.name]
            r = subprocess.run(cmd, cwd=str(cfg.W8_DIR), capture_output=True, text=True,
                               encoding='utf-8', errors='replace')
            if r.returncode != 0:
                print('    LaTeX error (see below)')
                print(r.stdout[-1500:])
                return False
        if not tmp_pdf.exists():
            print('    build did not produce Week8_build.pdf')
            return False
        cfg.REPORT_PDF.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp_pdf.replace(cfg.REPORT_PDF)
        except PermissionError:
            fallback = cfg.W8_DIR / 'Week_8_实验报告_v2.pdf'
            tmp_pdf.replace(fallback)
            print(f'    {cfg.REPORT_PDF.name} locked (open in a viewer); '
                  f'wrote {fallback.name} instead')
        return cfg.REPORT_PDF.exists() or (cfg.W8_DIR / 'Week_8_实验报告_v2.pdf').exists()
    finally:
        for suffix in ('.tex', '.pdf', '.aux', '.log', '.out', '.toc'):
            p = cfg.W8_DIR / f'Week8_build{suffix}'
            if p.exists():
                p.unlink()


def build_presentation() -> bool:
    """Build the final presentation deck (Week_8_Final_Presentation.pptx)."""
    script = cfg.W8_DIR / 'make_presentation.py'
    if not script.exists():
        print('    make_presentation.py not found; skipping')
        return False
    r = subprocess.run([sys.executable, str(script)], cwd=str(cfg.W8_DIR),
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode != 0:
        print('    presentation build failed:')
        print(r.stdout[-1200:])
        print(r.stderr[-1200:])
        return False
    return cfg.PRESENTATION_PPTX.exists()


def ensure_week7_assets() -> bool:
    """Copy the Week-7 figures referenced by the final report into week8/assets
    so the report / repo are self-contained. Regenerate only if missing."""
    need = ['fig_w7_scenarios_atm.png', 'fig_w7_scenarios_live.png',
            'fig_w7_shap_price_impact_vixproxy.png', 'fig_w7_perturb_curves.png']
    missing = [f for f in need if not (cfg.ASSETS_DIR / f).exists()]
    if missing:
        w7_assets = cfg.W7_DIR / 'assets'
        for f in missing:
            src = w7_assets / f
            if src.exists():
                shutil.copy(src, cfg.ASSETS_DIR / f)
                print(f'    copied {f} from week7/assets')
            else:
                print(f'    WARN: {f} missing in week7/assets too; report will skip it')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-tex', action='store_true', help='skip LaTeX compilation')
    ap.add_argument('--no-pptx', action='store_true', help='skip presentation build')
    ap.add_argument('--tests', action='store_true', help='run unit tests only')
    args = ap.parse_args()

    t0 = time.time()
    print('=' * 72)
    print('  WEEK 8 -- TOOL FINALIZATION & PROJECT DELIVERY')
    print('=' * 72)

    if args.tests:
        import test_week8
        test_week8.main()
        return

    # 1. dashboard figures
    print('\n[1/5] Dashboard figures (price trends + performance metrics)...')
    import dashboard as db
    res = db.run_dashboard()
    print('    -> assets/fig_w8_*.png, output/price_trend_series.csv')

    # copy week7 figures needed by the report
    print('\n[1b/5] Ensuring Week-7 figures for the report...')
    ensure_week7_assets()

    # 2. tool engine self-check (dual pricing on the latest live market state)
    print('\n[2/5] Pricing-tool engine self-check...')
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
    print('\n[3/5] Real-time data freshness check...')
    import data_updater as du
    du.run(mode='check', verbose=False)

    # 4. report
    print('\n[4/5] Compiling final report...')
    if not args.no_tex:
        ok = compile_report()
        print(f'    -> {cfg.REPORT_PDF}  {"OK" if ok else "FAILED"}')
    else:
        print('    skipped (--no-tex)')

    # 5. presentation
    print('\n[5/5] Building final presentation...')
    if not args.no_pptx:
        ok = build_presentation()
        print(f'    -> {cfg.PRESENTATION_PPTX}  {"OK" if ok else "FAILED"}')
    else:
        print('    skipped (--no-pptx)')

    print(f'\nTotal runtime: {time.time()-t0:.1f}s')
    print('Artifacts -> output/, assets/, Week_8_实验报告.pdf, presentation/')


if __name__ == '__main__':
    main()
