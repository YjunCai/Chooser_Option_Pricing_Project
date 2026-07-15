"""
Alpha Vantage data collector.

Provides alternative data sources:
  - TIME_SERIES_DAILY: Stock prices (JPM)
  - NEWS_SENTIMENT: Market news sentiment scores
  - Put/Call ratio data (if available via Alpha Vantage)

Requires ALPHA_VANTAGE_API_KEY environment variable.
Free tier: 5 calls/min, 500 calls/day.
"""

import os
import time
import logging
from pathlib import Path
from urllib.parse import urlencode

import requests
import pandas as pd

from config import (
    ALPHA_VANTAGE_API_KEY,
    ALPHA_VANTAGE_BASE_URL,
    ALPHA_VANTAGE_SLEEP_SEC,
    START_DATE,
    END_DATE,
    OUTPUT_DIR,
    OUTPUT_FORMAT,
)

logger = logging.getLogger(__name__)


class AlphaVantageClient:
    """Thin wrapper around Alpha Vantage REST API with rate-limit handling."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or ALPHA_VANTAGE_API_KEY
        if not self.api_key:
            raise ValueError(
                "Alpha Vantage API key not set. "
                "Set ALPHA_VANTAGE_API_KEY environment variable."
            )
        self._last_call = 0.0

    def _rate_limited_request(self, params: dict) -> dict:
        """Make a rate-limited request to the Alpha Vantage API."""
        elapsed = time.time() - self._last_call
        if elapsed < ALPHA_VANTAGE_SLEEP_SEC:
            time.sleep(ALPHA_VANTAGE_SLEEP_SEC - elapsed)

        params["apikey"] = self.api_key
        url = f"{ALPHA_VANTAGE_BASE_URL}?{urlencode(params)}"
        self._last_call = time.time()

        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Alpha Vantage returns error info in the JSON body
        if "Error Message" in data:
            raise RuntimeError(f"Alpha Vantage API error: {data['Error Message']}")
        if "Note" in data:
            logger.warning("API notice: %s", data["Note"])
        return data

    def fetch_daily_stock(self, symbol: str = "JPM") -> pd.DataFrame:
        """Fetch daily time-series for a stock symbol.

        Tries TIME_SERIES_DAILY_ADJUSTED (full) first for premium users.
        Falls back to TIME_SERIES_DAILY (compact) for free tier.
        """
        logger.info("Alpha Vantage: fetching daily stock %s", symbol)

        # Try premium endpoint first
        data = self._rate_limited_request({
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "datatype": "json",
        })

        # Check if endpoint is premium-locked
        if "Information" in data and "premium" in str(data.get("Information", "")).lower():
            logger.warning(
                "TIME_SERIES_DAILY_ADJUSTED is a premium endpoint (free tier only gets ~100 days). "
                "Falling back to TIME_SERIES_DAILY compact."
            )
            data = self._rate_limited_request({
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "datatype": "json",
            })

        series = data.get("Time Series (Daily)", {})
        if not series:
            logger.warning("No daily series for %s", symbol)
            return pd.DataFrame()

        rows = []
        for date_str, vals in series.items():
            row = {
                "date": pd.to_datetime(date_str).date(),
                "ticker": symbol,
                "open": float(vals["1. open"]),
                "high": float(vals["2. high"]),
                "low": float(vals["3. low"]),
                "close": float(vals["4. close"]),
                "volume": int(vals.get("6. volume", 0)),
            }
            # These fields exist only in the ADJUSTED endpoint
            if "5. adjusted close" in vals:
                row["adjusted_close"] = float(vals["5. adjusted close"])
            if "7. dividend amount" in vals:
                row["dividend"] = float(vals["7. dividend amount"])
            if "8. split coefficient" in vals:
                row["split_coefficient"] = float(vals["8. split coefficient"])
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df[(df["date"] >= pd.to_datetime(START_DATE).date()) &
                (df["date"] <= pd.to_datetime(END_DATE).date())]
        df.sort_values("date", inplace=True)
        logger.info("  → %s rows for %s", len(df), symbol)
        return df

    def fetch_news_sentiment(self, tickers: str = "JPM", limit: int = 100) -> pd.DataFrame:
        """Fetch news sentiment scores for given tickers."""
        logger.info("Alpha Vantage: fetching news sentiment for %s", tickers)
        data = self._rate_limited_request({
            "function": "NEWS_SENTIMENT",
            "tickers": tickers,
            "limit": limit,
            "sort": "RELEVANCE",
        })

        articles = data.get("feed", [])
        if not articles:
            logger.warning("No news sentiment data returned")
            return pd.DataFrame()

        rows = []
        for art in articles:
            rows.append({
                "date": pd.to_datetime(art["time_published"], format="%Y%m%dT%H%M%S").date(),
                "title": art["title"],
                "source": art.get("source", ""),
                "overall_sentiment_score": float(art.get("overall_sentiment_score", 0)),
                "overall_sentiment_label": art.get("overall_sentiment_label", ""),
                "url": art.get("url", ""),
            })

        df = pd.DataFrame(rows)
        df = df[(df["date"] >= pd.to_datetime(START_DATE).date()) &
                (df["date"] <= pd.to_datetime(END_DATE).date())]
        logger.info("  → %s sentiment articles", len(df))
        return df


def save_to_disk(data: dict[str, pd.DataFrame], output_dir: str | Path = OUTPUT_DIR):
    """Save each dataframe to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ext = "parquet" if OUTPUT_FORMAT == "parquet" else "csv"
    saved = []
    for name, df in data.items():
        if df.empty:
            continue
        path = output_dir / f"alphavantage_{name}.{ext}"
        if ext == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        logger.info("Saved %s → %s", name, path)
        saved.append(path)
    return saved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        client = AlphaVantageClient()
        results = {}

        # 1. Daily stock data
        df_stock = client.fetch_daily_stock("JPM")
        if not df_stock.empty:
            results["jpm_daily"] = df_stock

        # 2. News sentiment
        df_sentiment = client.fetch_news_sentiment("JPM", limit=100)
        if not df_sentiment.empty:
            results["news_sentiment"] = df_sentiment

        if results:
            save_to_disk(results)
            logger.info("Alpha Vantage collection complete — %d datasets saved", len(results))
        else:
            logger.warning("No data collected from Alpha Vantage")

    except ValueError as e:
        logger.error(e)
        logger.info("Skipping Alpha Vantage — set ALPHA_VANTAGE_API_KEY to enable.")
    except Exception as e:
        logger.error("Alpha Vantage collection failed: %s", e)
