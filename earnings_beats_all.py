#!/usr/bin/env python3
"""
Select all rows that have an earnings beat (reported_eps > eps_estimate).
Simple query: no time window, no consistency requirement.
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from fetch_ohlc_earnings import DB_PATH


def main():
    p = argparse.ArgumentParser(description="All rows with an earnings beat")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="Output CSV path (default: earnings_beats_all.csv)")
    p.add_argument("-n", type=int, default=None, help="Limit number of rows (default: all)")
    args = p.parse_args()

    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}. Run fetch_ohlc_earnings.py first.")
        return

    conn = sqlite3.connect(DB_PATH)

    # All earnings rows where reported > estimate (beat)
    # Cast to REAL in case they were stored as text
    q = """
    SELECT
        d.ticker,
        t.name,
        d.industry,
        d.date,
        d.close,
        d.eps_estimate,
        d.reported_eps,
        d.surprise_pct
    FROM daily d
    LEFT JOIN tickers t ON t.ticker = d.ticker
    WHERE d.is_earnings_day = 1
      AND d.eps_estimate IS NOT NULL
      AND d.reported_eps IS NOT NULL
      AND CAST(d.reported_eps AS REAL) > CAST(d.eps_estimate AS REAL)
    ORDER BY d.date DESC, d.ticker
    """
    df = pd.read_sql(q, conn)
    if args.n is not None:
        df = df.head(args.n)
    conn.close()

    out_path = Path(args.output) if args.output else Path(__file__).resolve().parent / "earnings_beats_all.csv"
    df.to_csv(out_path, index=False)
    print(f"Found {len(df)} rows with an earnings beat.")
    print(f"Wrote: {out_path}")
    if not df.empty:
        print()
        print(df.head(20).to_string(index=False))
        if len(df) > 20:
            print("...")


if __name__ == "__main__":
    main()
