"""
Preprocessing Pipeline - Week 2 Main Orchestrator

Runs the full data preprocessing & feature engineering pipeline:
  1. Load raw data (Week 1 outputs)
  2. Clean: missing value interpolation, IQR outlier capping
  3. Time-align all sources via date
  4. Feature engineering (>=10 derived features)
  5. Save structured dataset & summary report

Usage:
    python preprocessing_pipeline.py                    # full pipeline
    python preprocessing_pipeline.py --format parquet   # override output format
    python preprocessing_pipeline.py --output ./custom  # custom output dir
"""

import sys
import argparse
import logging
from pathlib import Path

import pandas as pd

import data_cleaner
import feature_engineering
from config import WEEK2_OUTPUT_DIR, OUTPUT_FORMAT

logger = logging.getLogger(__name__)


FEATURE_DESCRIPTIONS = {
    "daily_return": "JPM close-to-close log return (1d)",
    "vol_5d": "5-day rolling annualised volatility (JPM returns)",
    "vol_21d": "21-day rolling annualised volatility (JPM returns)",
    "vol_63d": "63-day rolling annualised volatility (JPM returns)",
    "high_low_spread": "Daily (high - low) / close (JPM) - intraday volatility",
    "dps_growth_rate": "YoY dividend per share growth rate (%) - real JPM data",
    "volume_change_1d": "1-day log change in JPM trading volume",
    "sma_ratio_21": "JPM close / 21-day SMA - mean reversion indicator",
    "vix_change_1d": "VIX daily absolute change (points)",
    "vix_jpm_corr_21d": "21-day rolling correlation: JPM return vs VIX change",
    "vix_jpm_cross_1d": "1-day co-movement: -return * VIX_chg (risk-off indicator)",
    "rate_change_1d_bps": "3-month Treasury rate daily change (basis points)",
    "rate_momentum_5d_bps": "5-day change in 3-month Treasury rate (bps)",
    "sentiment_score": "VIX min-max sentiment score [0,1] - formula from Week 1 report",
    "jpm_vol_ratio": "vol_5d / vol_21d - volatility term structure regime",
    "vix_ratio": "VIX / (vol_21d x 100) - VIX premium vs realised vol",
}

INPUT_COLUMNS = {
    "close_jpm": "JPM close price (USD) - underlying asset price",
    "close_vix": "VIX index level (points) - market fear gauge",
    "value_treasury_3mo": "3-month Treasury constant maturity rate (%)",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 2 - Data Preprocessing & Feature Engineering Pipeline",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default=OUTPUT_FORMAT,
        help="Output format (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=str(WEEK2_OUTPUT_DIR),
        help="Output directory (default: %(default)s)",
    )
    return parser.parse_args(argv)


def print_summary(df: pd.DataFrame, feature_df: pd.DataFrame, output_dir: str):
    """Print pipeline summary report."""
    print("\n" + "=" * 80)
    print("WEEK 2 - DATA PREPROCESSING PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Total rows (aligned): {len(df)}")
    print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
    print(f"\nInput columns: {len([c for c in INPUT_COLUMNS if c in df.columns])}")
    for col, desc in INPUT_COLUMNS.items():
        if col in df.columns:
            print(f"  - {col} -- {desc}")

    feature_cols = [c for c in feature_df.columns if c in FEATURE_DESCRIPTIONS]
    print(f"\nDerived features: {len(feature_cols)}")
    for col in feature_cols:
        desc = FEATURE_DESCRIPTIONS.get(col, "")
        non_null = feature_df[col].notna().sum()
        print(f"  + {col}: {desc} ({non_null} non-null values)")

    remaining = [c for c in feature_df.columns if c not in FEATURE_DESCRIPTIONS
                 and not c.startswith("_") and c not in ("date",)]
    if remaining:
        print(f"\nOther columns: {', '.join(remaining)}")

    print("\n" + "=" * 80)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("Week 2: Data Preprocessing & Feature Engineering Pipeline")
    logger.info("=" * 60)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load raw data
    # ------------------------------------------------------------------
    logger.info("Step 1/4: Loading raw data...")
    raw = data_cleaner.load_raw_data()
    if not raw:
        logger.error("No raw data loaded. Aborting.")
        return 1

    # ------------------------------------------------------------------
    # Step 2: Clean, interpolate, align
    # ------------------------------------------------------------------
    logger.info("Step 2/4: Cleaning & aligning...")
    # IQR capping applied to all numeric columns including price and VIX.
    # Note: IQR on trending price levels may flag bull-market highs as outliers,
    # and IQR on VIX will cap extreme fear signals (e.g. COVID VIX=82 → 35.5).
    aligned = data_cleaner.clean_all(raw)
    logger.info("  -> Aligned shape: %d rows x %d cols", *aligned.shape)

    # ------------------------------------------------------------------
    # Step 3: Feature engineering
    # ------------------------------------------------------------------
    logger.info("Step 3/4: Engineering features...")
    feature_dataset = feature_engineering.build_feature_set(aligned)
    logger.info("  -> Feature dataset shape: %d rows x %d cols", *feature_dataset.shape)

    # ------------------------------------------------------------------
    # Step 4: Save & report
    # ------------------------------------------------------------------
    logger.info("Step 4/4: Saving output...")
    ext = "parquet" if args.format == "parquet" else "csv"
    output_path = output_dir / f"feature_dataset.{ext}"
    if ext == "parquet":
        feature_dataset.to_parquet(output_path, index=False)
    else:
        feature_dataset.to_csv(output_path, index=False)
    logger.info("Saved feature dataset -> %s", output_path)

    # Also save the aligned (pre-feature) data for inspection
    aligned_path = output_dir / f"aligned_clean.{ext}"
    if ext == "parquet":
        aligned[["date"] + [c for c in aligned.columns if c != "date"]].to_parquet(aligned_path, index=False)
    else:
        aligned[["date"] + [c for c in aligned.columns if c != "date"]].to_csv(aligned_path, index=False)
    logger.info("Saved aligned clean data -> %s", aligned_path)

    # Summary report
    print_summary(aligned, feature_dataset, str(output_dir))
    logger.info("Week 2 pipeline complete - feature dataset ready")

    return 0


if __name__ == "__main__":
    sys.exit(main())
