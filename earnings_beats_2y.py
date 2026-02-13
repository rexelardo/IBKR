#!/usr/bin/env python3
"""
Select tickers that had an earnings beat on every report in the past 2 years.
Beat = reported_eps > eps_estimate. Requires at least 4 quarters (2 years) of data.
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from fetch_ohlc_earnings import DB_PATH

DEFAULT_MIN_QUARTERS = 4  # 2 years of quarterly reports


def main():
    p = argparse.ArgumentParser(description="Tickers with every earnings beat in past 2 years")
    p.add_argument("--min-quarters", type=int, default=DEFAULT_MIN_QUARTERS,
                   help=f"Minimum number of earnings reports (default {DEFAULT_MIN_QUARTERS})")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="Output CSV path (default: consistent_earnings_beats_2y.csv)")
    args = p.parse_args()

    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}. Run fetch_ohlc_earnings.py first.")
        return

    conn = sqlite3.connect(DB_PATH)

    # Earnings rows in last 2 years with both estimate and reported EPS
    # Beat = reported_eps > eps_estimate (both must be non-null)
    q = """
    WITH earnings AS (
        SELECT
            ticker,
            date,
            eps_estimate,
            reported_eps,
            surprise_pct,
            industry,
            CASE WHEN reported_eps > eps_estimate THEN 1 ELSE 0 END AS beat
        FROM daily
        WHERE is_earnings_day = 1
          AND date >= date('now', '-2 years')
          AND eps_estimate IS NOT NULL
          AND reported_eps IS NOT NULL
    ),
    by_ticker AS (
        SELECT
            ticker,
            industry,
            COUNT(*) AS reports,
            SUM(beat) AS beats
        FROM earnings
        GROUP BY ticker, industry
    )
    SELECT b.ticker, t.name, b.industry, b.reports, b.beats
    FROM by_ticker b
    LEFT JOIN tickers t ON t.ticker = b.ticker
    WHERE b.reports >= ?
      AND b.beats = b.reports
    ORDER BY b.reports DESC, b.ticker
    """
    df = pd.read_sql(q, conn, params=(args.min_quarters,))
    conn.close()

    out_path = Path(args.output) if args.output else Path(__file__).resolve().parent / "consistent_earnings_beats_2y.csv"
    df.to_csv(out_path, index=False)
    print(f"Found {len(df)} tickers with a beat on every report in the past 2 years (min {args.min_quarters} reports).")
    print(f"Wrote: {out_path}")
    print()
    if not df.empty:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
