"""
Main orchestrator for Week 1 data collection.

Runs all three data sources sequentially:
  1. Yahoo Finance  — JPM, VIX, S&P 500, TNX
  2. Alpha Vantage  — JPM daily + news sentiment (requires API key)
  3. FRED           — Treasury yields, Fed Funds, CPI (requires API key)

Usage:
    python data_collection.py              # full pipeline
    python data_collection.py --sources yahoo,fred   # selected sources only
    python data_collection.py --format csv           # override output format
"""

import sys
import argparse
import logging
from pathlib import Path

import pandas as pd

from config import OUTPUT_DIR, OUTPUT_FORMAT

# Source modules
import yahoo_finance
import alpha_vantage
import fred_api

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chooser Option Pricing — Week 1 Data Collection",
    )
    parser.add_argument(
        "--sources",
        default="yahoo,alphavantage,fred",
        help="Comma-separated list of sources to run (default: all)",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default=OUTPUT_FORMAT,
        help="Output format (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="Output directory (default: %(default)s)",
    )
    return parser.parse_args(argv)


def verify_output(output_dir: str | Path) -> pd.DataFrame:
    """Load and display summary of all collected data files."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        logger.warning("Output directory does not exist: %s", output_dir)
        return pd.DataFrame()

    files = list(output_dir.glob("*.parquet")) + list(output_dir.glob("*.csv"))
    if not files:
        logger.warning("No data files found in %s", output_dir)
        return pd.DataFrame()

    summary = []
    for f in sorted(files):
        try:
            if f.suffix == ".parquet":
                df = pd.read_parquet(f)
            else:
                df = pd.read_csv(f)
            summary.append({
                "file": f.name,
                "rows": len(df),
                "columns": len(df.columns),
                "cols": list(df.columns),
                "date_range": f"{df.iloc[0]['date']} → {df.iloc[-1]['date']}" if len(df) > 0 else "empty",
            })
        except Exception as e:
            summary.append({"file": f.name, "rows": -1, "columns": -1, "cols": [], "date_range": str(e)})

    report = pd.DataFrame(summary)
    print("\n" + "=" * 80)
    print("DATA COLLECTION SUMMARY")
    print("=" * 80)
    print(report.to_string(index=False))
    print("=" * 80)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sources = [s.strip() for s in args.sources.split(",")]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Starting Week 1 data collection — sources: %s", sources)

    # ------------------------------------------------------------------
    # 1. Yahoo Finance (no API key required)
    # ------------------------------------------------------------------
    if "yahoo" in sources:
        logger.info("--- Yahoo Finance ---")
        try:
            yahoo_data = yahoo_finance.collect_all()
            if yahoo_data:
                yahoo_finance.save_to_disk(yahoo_data, args.output)
            else:
                logger.warning("Yahoo Finance returned no data")
        except Exception as e:
            logger.error("Yahoo Finance failed: %s", e)
    else:
        logger.info("Skipping Yahoo Finance")

    # ------------------------------------------------------------------
    # 2. Alpha Vantage (API key required)
    # ------------------------------------------------------------------
    if "alphavantage" in sources:
        logger.info("--- Alpha Vantage ---")
        try:
            client = alpha_vantage.AlphaVantageClient()
            av_results = {}
            df_stock = client.fetch_daily_stock("JPM")
            if not df_stock.empty:
                av_results["jpm_daily"] = df_stock
            df_sent = client.fetch_news_sentiment("JPM", limit=100)
            if not df_sent.empty:
                av_results["news_sentiment"] = df_sent
            if av_results:
                alpha_vantage.save_to_disk(av_results, args.output)
            else:
                logger.warning("Alpha Vantage returned no data")
        except ValueError as e:
            logger.warning("Alpha Vantage skipped: %s", e)
        except Exception as e:
            logger.error("Alpha Vantage failed: %s", e)
    else:
        logger.info("Skipping Alpha Vantage")

    # ------------------------------------------------------------------
    # 3. FRED (API key required)
    # ------------------------------------------------------------------
    if "fred" in sources:
        logger.info("--- FRED ---")
        try:
            client = fred_api.FredClient()
            fred_results = client.fetch_all()
            if fred_results:
                combined = client.pivot_series(fred_results)
                fred_api.save_to_disk(fred_results, combined, args.output)
            else:
                logger.warning("FRED returned no data")
        except ValueError as e:
            logger.warning("FRED skipped: %s", e)
        except Exception as e:
            logger.error("FRED failed: %s", e)
    else:
        logger.info("Skipping FRED")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    verify_output(args.output)
    logger.info("Week 1 data collection complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
