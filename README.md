# IBKR

Fetch and analyze US stock data using tickers from a CSV.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # or: venv\Scripts\activate on Windows
pip install pandas yfinance
```

## Usage

- **Fetch 3 years OHLC + earnings** for all tickers in `tickers.csv` (stores in SQLite for easy multi-ticker queries):

  ```bash
  python fetch_ohlc_earnings.py
  ```

  Options: `--limit N` (test on N tickers), `--no-resume` (re-fetch all), `--verbose` (yfinance debug).

- Re-run to resume (skips tickers already in the DB); rate limits are handled with wait + retry.

## Data

- `tickers.csv` — input list (ticker, name, industry, marketCap).
- `data/ohlc_earnings.db` — SQLite DB:
  - **daily**: one row per (ticker, date) — OHLC, volume, earnings-day flag, EPS columns, and **industry** (from tickers.csv).
  - **tickers**: one row per ticker — name, industry, market_cap (from tickers.csv; kept in sync by the fetcher and import script).

### Import existing CSVs into the DB

If you have CSV files from an earlier run:

```bash
python import_csv_to_db.py
```

This fills the `daily` table from `data/*.csv` and syncs the `tickers` table (name, industry, market_cap) from `tickers.csv`.

### Querying the DB

```bash
# SQLite CLI
sqlite3 data/ohlc_earnings.db
```

```sql
-- All days for a ticker (industry is a column in daily)
SELECT * FROM daily WHERE ticker = 'AAPL' ORDER BY date;

-- Multiple tickers
SELECT ticker, industry, date, close, volume, is_earnings_day
FROM daily
WHERE ticker IN ('AAPL', 'MSFT', 'GOOGL') AND date >= '2024-01-01'
ORDER BY ticker, date;

-- Filter by industry
SELECT ticker, date, close, industry FROM daily
WHERE industry = 'Consumer Electronics' AND date >= '2024-01-01';

-- Only earnings days (join tickers for company name if needed)
SELECT ticker, industry, date, close, eps_estimate, reported_eps, surprise_pct
FROM daily WHERE is_earnings_day = 1 ORDER BY date DESC;

-- Latest close per ticker
SELECT ticker, industry, date, close
FROM daily d
WHERE date = (SELECT MAX(date) FROM daily d2 WHERE d2.ticker = d.ticker);
```

From Python (pandas):

```python
import sqlite3
import pandas as pd
conn = sqlite3.connect("data/ohlc_earnings.db")
df = pd.read_sql("SELECT * FROM daily WHERE ticker IN ('AAPL','MSFT')", conn)
conn.close()
```
