#!/usr/bin/env python3
"""
Fetch 3 years of daily OHLC + earnings for each ticker in tickers.csv.
Writes one CSV per ticker to the data/ folder.
"""

import os
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
TICKERS_CSV = PROJECT_ROOT / "tickers.csv"
DATA_DIR = PROJECT_ROOT / "data"

# Fetch window
PERIOD = "3y"
INTERVAL = "1d"
EARNINGS_LIMIT = 25  # quarters (~6 years) so we cover the 3y window

# Throttle (seconds between tickers) to reduce rate limits
SLEEP_BETWEEN_TICKERS = 0.15
# When Yahoo returns 429, wait this long before retrying (seconds)
RATE_LIMIT_WAIT = 120
MAX_RATE_LIMIT_RETRIES = 3

def replace_dots_with_dash(s: str) -> str:
        """Replace all '.' with '-' in the given string."""
        return s.replace('.', '-')

def fetch_ticker_data(ticker: str) -> pd.DataFrame | None:
    """Fetch OHLC for 3y and merge earnings. Returns None on failure."""
    
    ticker = replace_dots_with_dash(ticker)
    t = yf.Ticker(ticker)

    # Daily OHLC
    hist = t.history(period=PERIOD, interval=INTERVAL, auto_adjust=True)
    if hist is None or hist.empty:
        return None

    # Flatten multi-level columns if present (yfinance sometimes returns MultiIndex)
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    # Keep standard OHLC + Volume
    ohlc_cols = ["Open", "High", "Low", "Close"]
    if "Volume" in hist.columns:
        ohlc_cols = ohlc_cols + ["Volume"]
    hist = hist[[c for c in ohlc_cols if c in hist.columns]].copy()

    # Normalize index to date for merging
    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
    hist = hist[~hist.index.duplicated(keep="first")]

    # Earnings: get_earnings_dates can return None or raise
    earnings = None
    try:
        earnings = t.get_earnings_dates(limit=EARNINGS_LIMIT)
    except Exception:
        pass

    if earnings is not None and not earnings.empty:
        # Normalize earnings index to date
        earn_idx = pd.to_datetime(earnings.index).tz_localize(None).normalize()
        earnings = earnings.set_index(earn_idx)
        earnings = earnings[~earnings.index.duplicated(keep="first")]

        # Map column names (yfinance may use "Surprise (%)" or "Surprise(%)")
        rename = {}
        for c in earnings.columns:
            if "Surprise" in c and c != "Surprise(%)":
                rename[c] = "Surprise(%)"
        earnings = earnings.rename(columns=rename)

        # For each OHLC date, mark if earnings day and attach EPS info
        hist["is_earnings_day"] = hist.index.isin(earnings.index)
        for col in ["EPS Estimate", "Reported EPS", "Surprise(%)"]:
            if col in earnings.columns:
                hist[col] = hist.index.map(
                    lambda d: earnings.loc[d, col] if d in earnings.index else None
                )
    else:
        hist["is_earnings_day"] = False

    hist = hist.sort_index()
    return hist


def main():
    import argparse
    p = argparse.ArgumentParser(description="Fetch OHLC + earnings for tickers in tickers.csv")
    p.add_argument("--limit", type=int, default=None, help="Max number of tickers to process (for testing)")
    p.add_argument("--no-resume", action="store_true", help="Re-fetch all tickers (ignore existing data/*.csv)")
    p.add_argument("--verbose", action="store_true", help="Enable yfinance debug logging (see 429/rate-limit in logs)")
    args = p.parse_args()

    if args.verbose:
        yf.config.debug.logging = True

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TICKERS_CSV)
    tickers = df["ticker"].astype(str).str.strip().unique().tolist()
    if args.limit:
        tickers = tickers[: args.limit]
    total = len(tickers)
    if total == 0:
        print("Nothing to do (all tickers already have data, or list empty). Use --no-resume to re-fetch all.")
        return

    ok = 0
    failed = []
    rate_limit_hits = 0

    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{total}] {ticker} ... ", end="", flush=True)
        retries = 0
        while True:
            try:
                out = fetch_ticker_data(ticker)
                if out is None or out.empty:
                    print("no data")
                    failed.append(ticker)
                    break
                out_path = DATA_DIR / f"{ticker}.csv"
                out.to_csv(out_path, index=True)
                print(f"ok -> {out_path.name} ({len(out)} rows)")
                ok += 1
                break
            except YFRateLimitError:
                rate_limit_hits += 1
                retries += 1
                if retries > MAX_RATE_LIMIT_RETRIES:
                    print("RATE LIMITED (gave up after %d retries)" % MAX_RATE_LIMIT_RETRIES)
                    failed.append(ticker)
                    break
                print("RATE LIMITED - waiting %ds then retry %d/%d ... " % (RATE_LIMIT_WAIT, retries, MAX_RATE_LIMIT_RETRIES), end="", flush=True)
                time.sleep(RATE_LIMIT_WAIT)
            except Exception as e:
                print(f"error: {e}")
                failed.append(ticker)
                break

        if i < total - 1:
            time.sleep(SLEEP_BETWEEN_TICKERS)

    print()
    print(f"Done. Wrote {ok} tickers to {DATA_DIR}. Failed: {len(failed)}")
    if rate_limit_hits:
        print(f"Rate limit hit {rate_limit_hits} time(s); consider increasing SLEEP_BETWEEN_TICKERS or RATE_LIMIT_WAIT in the script.")
    if failed:
        print("Failed tickers:", failed[:50], "..." if len(failed) > 50 else "")


if __name__ == "__main__":
    main()
