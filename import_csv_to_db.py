#!/usr/bin/env python3
"""
Import existing data/*.csv files into the SQLite DB (data/ohlc_earnings.db).
Also ensures the tickers table is populated from tickers.csv so industry/name/market_cap are available.
"""

import sqlite3
from pathlib import Path

import pandas as pd

from fetch_ohlc_earnings import DATA_DIR, DB_PATH, init_db, sync_tickers_from_csv


def import_csv(csv_path: Path, ticker: str) -> pd.DataFrame | None:
    """Read a single ticker CSV and return a dataframe in DB shape, or None on error."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  read error: {e}")
        return None
    if df.empty:
        return None
    # CSV columns: Date, Open, High, Low, Close, Volume, is_earnings_day, EPS Estimate, Reported EPS, Surprise(%)
    rename = {
        "Date": "date",
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Volume": "volume",
        "EPS Estimate": "eps_estimate",
        "Reported EPS": "reported_eps",
        "Surprise(%)": "surprise_pct",
    }
    for c in list(df.columns):
        if "Surprise" in c and c != "Surprise(%)":
            rename[c] = "surprise_pct"
    df = df.rename(columns=rename)
    # Normalize date
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    # is_earnings_day: CSV has True/False strings
    if "is_earnings_day" in df.columns:
        df["is_earnings_day"] = df["is_earnings_day"].map(
            lambda x: 1 if str(x).strip().lower() in ("true", "1", "1.0") else 0
        )
    else:
        df["is_earnings_day"] = 0
    df["ticker"] = ticker
    cols = ["ticker", "date", "open", "high", "low", "close", "volume",
            "is_earnings_day", "eps_estimate", "reported_eps", "surprise_pct"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    return df


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    sync_tickers_from_csv(conn)
    print("Synced tickers (name, industry, market_cap) from tickers.csv")

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in", DATA_DIR)
        conn.close()
        return

    industry_map = dict(conn.execute("SELECT ticker, industry FROM tickers").fetchall())

    total = 0
    failed = []
    n = len(csv_files)
    for i, csv_path in enumerate(csv_files):
        ticker = csv_path.stem
        df = import_csv(csv_path, ticker)
        if df is None or df.empty:
            failed.append(ticker)
            continue
        df["industry"] = industry_map.get(ticker)
        try:
            conn.execute("DELETE FROM daily WHERE ticker = ?", (ticker,))
            df.to_sql("daily", conn, if_exists="append", index=False)
            total += len(df)
            if (i + 1) % 500 == 0 or i == 0 or i == n - 1:
                print(f"  [{i+1}/{n}] {ticker}: {len(df)} rows ...")
        except Exception as e:
            print(f"  {ticker}: error {e}")
            failed.append(ticker)

    conn.commit()
    conn.close()
    print(f"\nDone. Imported {total} rows from {len(csv_files) - len(failed)} CSVs into {DB_PATH}")
    if failed:
        print("Failed:", failed[:20], "..." if len(failed) > 20 else "")


if __name__ == "__main__":
    main()
