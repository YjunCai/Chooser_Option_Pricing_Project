"""
Yahoo Finance data collector.

Uses direct HTTP calls to Yahoo Finance's v8 chart API (undocumented but widely used).
No API key required — implements retry with exponential backoff and browser-like headers.
"""

import time
import random
import logging
from pathlib import Path
from datetime import datetime, timezone

import requests
import pandas as pd

from config import YAHOO_TICKERS, START_DATE, END_DATE, OUTPUT_DIR, OUTPUT_FORMAT

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

_SESSION = requests.Session()
_SESSION.headers.update(_HEADERS)


def _date_to_unix(date_str: str) -> int:
    """Convert YYYY-MM-DD string to Unix timestamp."""
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def _fetch_chart(ticker: str, start: str, end: str) -> dict | None:
    """Call Yahoo Finance v8 chart API with retry."""
    url = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": _date_to_unix(start),
        "period2": _date_to_unix(end),
        "interval": "1d",
        "includePrePost": "false",
    }

    max_retries = 5
    for attempt in range(max_retries):
        try:
            resp = _SESSION.get(url.format(ticker=ticker), params=params, timeout=30)
            if resp.status_code == 429:
                delay = (2 ** attempt) + random.uniform(0, 2)
                logger.warning("HTTP 429 on %s, retry %d/%d after %.0fs", ticker, attempt + 1, max_retries, delay)
                time.sleep(delay)
                continue

            resp.raise_for_status()
            data = resp.json()

            error = data.get("chart", {}).get("error")
            if error:
                logger.warning("API error for %s: %s", ticker, error.get("description", error))
                return None

            return data

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429 and attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 2)
                logger.warning("HTTP 429 on %s, retry %d/%d after %.0fs", ticker, attempt + 1, max_retries, delay)
                time.sleep(delay)
            else:
                logger.error("HTTP error for %s: %s", ticker, e)
                return None
        except requests.exceptions.RequestException as e:
            logger.error("Request failed for %s: %s", ticker, e)
            return None

    logger.error("Exhausted retries for %s", ticker)
    return None


def _parse_chart(data: dict, name: str) -> pd.DataFrame:
    """Parse Yahoo Finance v8 chart API response into a DataFrame."""
    result = data.get("chart", {}).get("result", [None])[0]
    if result is None:
        return pd.DataFrame()

    meta = result.get("meta", {})
    timestamps = result.get("timestamp", [])
    indicators = result.get("indicators", {})
    quote = (indicators.get("quote") or [{}])[0]
    adjclose = (indicators.get("adjclose") or [{}])[0].get("adjclose", [])

    if not timestamps or not quote.get("close"):
        return pd.DataFrame()

    rows = []
    for i, ts in enumerate(timestamps):
        close = quote["close"][i]
        if close is None:
            continue
        rows.append({
            "ticker": name,
            "date": pd.to_datetime(ts, unit="s").date(),
            "open": quote.get("open", [None])[i],
            "high": quote.get("high", [None])[i],
            "low": quote.get("low", [None])[i],
            "close": close,
            "volume": quote.get("volume", [None])[i] or 0,
            "adjusted_close": adjclose[i] if i < len(adjclose) and adjclose[i] is not None else close,
        })

    df = pd.DataFrame(rows)
    if "currency" in meta:
        df["currency"] = meta["currency"]
    return df


def fetch_yahoo_data(ticker: str, name: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily data for a single ticker using Yahoo Finance chart API."""
    logger.info("Fetching %s (%s) from %s to %s", name, ticker, start, end)
    data = _fetch_chart(ticker, start, end)
    if data is None:
        return pd.DataFrame()

    df = _parse_chart(data, name)
    if df.empty:
        logger.warning("No data returned for %s (%s)", name, ticker)
    else:
        logger.info("  → %s rows for %s", len(df), name)
    return df


def collect_all(start: str = START_DATE, end: str = END_DATE) -> dict[str, pd.DataFrame]:
    """Fetch all configured Yahoo Finance tickers."""
    results = {}
    for name, ticker in YAHOO_TICKERS.items():
        df = fetch_yahoo_data(ticker, name, start, end)
        if not df.empty:
            results[name] = df
        time.sleep(2)
    return results


def save_to_disk(data: dict[str, pd.DataFrame], output_dir: str | Path = OUTPUT_DIR):
    """Save each dataframe to disk in the configured format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ext = "parquet" if OUTPUT_FORMAT == "parquet" else "csv"
    saved = []
    for name, df in data.items():
        path = output_dir / f"yahoo_{name.lower()}.{ext}"
        if ext == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        logger.info("Saved %s → %s", name, path)
        saved.append(path)
    return saved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Starting Yahoo Finance data collection")

    data = collect_all()
    if data:
        save_to_disk(data)
        combined = pd.concat(data.values(), ignore_index=True)
        combined_path = Path(OUTPUT_DIR) / f"yahoo_all.{'parquet' if OUTPUT_FORMAT == 'parquet' else 'csv'}"
        if OUTPUT_FORMAT == "parquet":
            combined.to_parquet(combined_path, index=False)
        else:
            combined.to_csv(combined_path, index=False)
        logger.info("Combined data saved → %s", combined_path)
    else:
        logger.warning("No data collected from Yahoo Finance")

    logger.info("Yahoo Finance collection complete — %d tickers saved", len(data))
