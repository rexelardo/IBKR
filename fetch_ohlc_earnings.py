#!/usr/bin/env python3
"""
Fetch 3 years of daily OHLC + earnings for each ticker in tickers.csv.
- Writes one CSV per ticker to data/<TICKER>.csv (so you can import later if needed).
- Also writes to SQLite data/ohlc_earnings.db for querying.
"""

import sqlite3
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
TICKERS_CSV = PROJECT_ROOT / "tickers.csv"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "ohlc_earnings.db"

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
    try:
        ticker_clean = replace_dots_with_dash(ticker)
        t = yf.Ticker(ticker_clean)

        # Daily OHLC (can raise "Execution failed" or similar for bad/rate-limited symbols)
        hist = t.history(period=PERIOD, interval=INTERVAL, auto_adjust=True)
        if hist is None or hist.empty:
            return None

        # Flatten multi-level columns if present (yfinance sometimes returns MultiIndex)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        # Keep standard OHLC + daily Volume (one column)
        ohlc_cols = ["Open", "High", "Low", "Close", "Volume"]
        hist = hist[[c for c in ohlc_cols if c in hist.columns]].copy()
        if "Volume" not in hist.columns:
            hist["Volume"] = None  # ensure column exists for CSV/DB

        # Normalize index to date for merging
        hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
        hist = hist[~hist.index.duplicated(keep="first")]

        # Earnings: get_earnings_dates can return None or raise (e.g. "No earnings dates found")
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
    except YFRateLimitError:
        raise  # Let caller handle wait + retry
    except Exception:
        return None  # No data / execution failed / delisted etc.


def init_db(conn: sqlite3.Connection) -> None:
    """Create the daily and tickers tables and indexes if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            is_earnings_day INTEGER NOT NULL DEFAULT 0,
            eps_estimate REAL,
            reported_eps REAL,
            surprise_pct REAL,
            industry TEXT,
            PRIMARY KEY (ticker, date)
        )
    """)
    # Add industry column if it doesn't exist (migration for older DBs)
    try:
        conn.execute("ALTER TABLE daily ADD COLUMN industry TEXT")
    except sqlite3.OperationalError as e:
        if "duplicate column" not in str(e).lower():
            raise
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tickers (
            ticker TEXT PRIMARY KEY,
            name TEXT,
            industry TEXT,
            market_cap TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_ticker ON daily(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_date ON daily(date)")
    conn.commit()


def sync_tickers_from_csv(conn: sqlite3.Connection) -> None:
    """Load ticker metadata (name, industry, market_cap) from tickers.csv into the tickers table."""
    if not TICKERS_CSV.exists():
        return
    df = pd.read_csv(TICKERS_CSV)
    df = df.rename(columns={"marketCap": "market_cap"})
    df = df[["ticker", "name", "industry", "market_cap"]].dropna(subset=["ticker"])
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df = df[df["ticker"].str.upper() != "NAN"]
    df.to_sql("tickers", conn, if_exists="replace", index=False)
    conn.commit()


def tickers_in_db(conn: sqlite3.Connection) -> set[str]:
    """Return set of tickers that already have data in the DB."""
    r = conn.execute("SELECT DISTINCT ticker FROM daily")
    return {row[0] for row in r.fetchall()}


def write_ticker_to_db(conn: sqlite3.Connection, ticker_original: str, df: pd.DataFrame) -> None:
    """Replace rows for this ticker with the new dataframe."""
    df = df.copy()
    df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")
    df = df.reset_index()
    df = df.rename(columns={
        "index": "date",
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Volume": "volume",
        "EPS Estimate": "eps_estimate",
        "Reported EPS": "reported_eps",
        "Surprise(%)": "surprise_pct",
    })
    df["ticker"] = ticker_original
    # NOT NULL column: ensure no NaN (SQLite rejects NULL here)
    df["is_earnings_day"] = pd.to_numeric(df["is_earnings_day"], errors="coerce").fillna(0).astype(int)
    row = conn.execute("SELECT industry FROM tickers WHERE ticker = ?", (ticker_original,)).fetchone()
    df["industry"] = row[0] if row else None
    cols = ["ticker", "date", "open", "high", "low", "close", "volume",
            "is_earnings_day", "eps_estimate", "reported_eps", "surprise_pct", "industry"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    # Drop rows with null date/ticker so we don't hit NOT NULL
    df = df.dropna(subset=["ticker", "date"])
    df = df.drop_duplicates(subset=["ticker", "date"], keep="first")
    # Ensure date is string YYYY-MM-DD for SQLite
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    conn.execute("DELETE FROM daily WHERE ticker = ?", (ticker_original,))
    df.to_sql("daily", conn, if_exists="append", index=False)
    conn.commit()


def main():
    import argparse
    p = argparse.ArgumentParser(description="Fetch OHLC + earnings for tickers in tickers.csv")
    p.add_argument("--limit", type=int, default=None, help="Max number of tickers to process (for testing)")
    p.add_argument("--no-resume", action="store_true", help="Re-fetch all tickers (ignore existing data in DB)")
    p.add_argument("--verbose", action="store_true", help="Enable yfinance debug logging (see 429/rate-limit in logs)")
    args = p.parse_args()

    if args.verbose:
        yf.config.debug.logging = True

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    sync_tickers_from_csv(conn)

    df = pd.read_csv(TICKERS_CSV)
    tickers = df["ticker"].dropna().astype(str).str.strip().unique().tolist()
    tickers = [t for t in tickers if t and t.upper() != "NAN"]
    if args.limit:
        tickers = tickers[: args.limit]
    if not args.no_resume:
        existing = tickers_in_db(conn)
        before = len(tickers)
        tickers = [t for t in tickers if t not in existing]
        skipped = before - len(tickers)
        if skipped:
            print(f"Resuming: skipping {skipped} tickers already in DB.\n")

    total = len(tickers)
    if total == 0:
        print("Nothing to do (all tickers already in DB, or list empty). Use --no-resume to re-fetch all.")
        conn.close()
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
                # Add industry from tickers table (for CSV and DB)
                row = conn.execute("SELECT industry FROM tickers WHERE ticker = ?", (ticker,)).fetchone()
                out["industry"] = row[0] if row else None
                # Always write CSV first (so we have data even if DB fails)
                csv_path = DATA_DIR / f"{ticker}.csv"
                out.to_csv(csv_path, index=True)
                # Then write to DB
                write_ticker_to_db(conn, ticker, out)
                print(f"ok -> {csv_path.name} + DB ({len(out)} rows)")
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
                cause = getattr(e, "__cause__", None)
                if cause is not None:
                    print(f"error: {type(e).__name__}: {e}\n  -> {type(cause).__name__}: {cause}")
                else:
                    print(f"error: {type(e).__name__}: {e}")
                failed.append(ticker)
                break

        if i < total - 1:
            time.sleep(SLEEP_BETWEEN_TICKERS)

    conn.close()
    print()
    print(f"Done. Wrote {ok} tickers to CSV (data/*.csv) and {DB_PATH}. Failed: {len(failed)}")
    if rate_limit_hits:
        print(f"Rate limit hit {rate_limit_hits} time(s); consider increasing SLEEP_BETWEEN_TICKERS or RATE_LIMIT_WAIT in the script.")
    if failed:
        print("Failed tickers:", failed[:50], "..." if len(failed) > 50 else "")


if __name__ == "__main__":
    main()
