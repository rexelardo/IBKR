#!/usr/bin/env python3
"""
Quick peek at what's in the SQLite DB: tables, row counts, and sample rows.
"""

import sqlite3
import sys
from pathlib import Path

from fetch_ohlc_earnings import DB_PATH


def main():
    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}")
        return 1

    conn = sqlite3.connect(DB_PATH)
    limit = 5
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            pass

    print(f"DB: {DB_PATH}\n")

    # Tables
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    print("Tables:", [t[0] for t in tables])

    for (tname,) in tables:
        n = conn.execute(f"SELECT COUNT(*) FROM [{tname}]").fetchone()[0]
        print(f"\n--- {tname} ({n} rows) ---")
        cols = [c[1] for c in conn.execute(f"PRAGMA table_info([{tname}])").fetchall()]
        print("Columns:", cols)
        if n > 0:
            q = f"SELECT * FROM [{tname}] LIMIT {limit}"
            rows = conn.execute(q).fetchall()
            print(f"Sample (first {limit} rows):")
            for i, row in enumerate(rows):
                print(f"  {i+1}: {row}")
        print()

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
