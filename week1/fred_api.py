"""
FRED (Federal Reserve Economic Data) collector.

Fetches macroeconomic indicators:
  - DGS10 : 10-Year Treasury Constant Maturity Rate
  - DGS3MO : 3-Month Treasury Bill
  - FEDFUNDS : Federal Funds Effective Rate
  - VIXCLS : CBOE Volatility Index (FRED series)
  - CPIAUCSL : Consumer Price Index (monthly)

Requires FRED_API_KEY environment variable (free registration).
"""

import os
import logging
from pathlib import Path
from urllib.parse import urlencode

import requests
import pandas as pd

from config import (
    FRED_API_KEY,
    FRED_BASE_URL,
    FRED_SERIES,
    START_DATE,
    END_DATE,
    OUTPUT_DIR,
    OUTPUT_FORMAT,
)

logger = logging.getLogger(__name__)


class FredClient:
    """Client for the FRED API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or FRED_API_KEY
        if not self.api_key:
            raise ValueError(
                "FRED API key not set. "
                "Set FRED_API_KEY environment variable "
                "(register at https://fred.stlouisfed.org/docs/api/)."
            )

    def fetch_series(self, series_id: str) -> pd.DataFrame:
        """Fetch observations for a single FRED series."""
        logger.info("FRED: fetching %s", series_id)
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": START_DATE,
            "observation_end": END_DATE,
            "sort_order": "asc",
        }

        url = f"{FRED_BASE_URL}?{urlencode(params)}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        observations = data.get("observations", [])
        if not observations:
            logger.warning("No observations for %s", series_id)
            return pd.DataFrame()

        rows = []
        for obs in observations:
            val = obs["value"]
            rows.append({
                "series_id": series_id,
                "date": pd.to_datetime(obs["date"]).date(),
                "value": float(val) if val != "." else None,
            })

        df = pd.DataFrame(rows)
        # Drop rows with missing values
        before = len(df)
        df.dropna(subset=["value"], inplace=True)
        if len(df) < before:
            logger.info("  Dropped %d missing-value rows", before - len(df))

        logger.info("  → %s rows for %s", len(df), series_id)
        return df

    def fetch_all(self) -> dict[str, pd.DataFrame]:
        """Fetch all configured FRED series."""
        results = {}
        for name, series_id in FRED_SERIES.items():
            df = self.fetch_series(series_id)
            if not df.empty:
                results[name] = df
        return results

    @staticmethod
    def pivot_series(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple series into a single wide-format dataframe."""
        if not data:
            return pd.DataFrame()

        # Start with the first series
        items = list(data.items())
        combined = items[0][1].rename(columns={"value": items[0][0]})
        combined = combined[["date", items[0][0]]]

        for name, df in items[1:]:
            sub = df[["date", "value"]].rename(columns={"value": name})
            combined = pd.merge(combined, sub, on="date", how="outer")

        combined.sort_values("date", inplace=True)
        combined.reset_index(drop=True, inplace=True)
        return combined


def save_to_disk(
    data: dict[str, pd.DataFrame],
    combined: pd.DataFrame,
    output_dir: str | Path = OUTPUT_DIR,
):
    """Save individual and combined FRED data to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ext = "parquet" if OUTPUT_FORMAT == "parquet" else "csv"
    saved = []

    for name, df in data.items():
        path = output_dir / f"fred_{name.lower()}.{ext}"
        if ext == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        logger.info("Saved → %s", path)
        saved.append(path)

    if not combined.empty:
        path = output_dir / f"fred_macro_combined.{ext}"
        if ext == "parquet":
            combined.to_parquet(path, index=False)
        else:
            combined.to_csv(path, index=False)
        logger.info("Combined macro data → %s", path)
        saved.append(path)

    return saved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        client = FredClient()
        results = client.fetch_all()
        if results:
            combined = client.pivot_series(results)
            save_to_disk(results, combined)
            logger.info("FRED collection complete — %d series saved", len(results))
        else:
            logger.warning("No data collected from FRED")

    except ValueError as e:
        logger.error(e)
        logger.info("Skipping FRED — set FRED_API_KEY to enable.")
    except Exception as e:
        logger.error("FRED collection failed: %s", e)
