#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

# ----------------------------
# CONFIG
# ----------------------------

OUT_CSV = "sec_quarterly_net_income.csv"
N_QUARTERS = 100
MAX_TICKERS: Optional[int] = None

PAUSE_S = 0.5
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 1.6

CACHE_DIR = "sec_cache"
USE_CACHE = True

TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
BASE_CONCEPT = "https://data.sec.gov/api/xbrl/companyconcept"

HEADERS = {
    "User-Agent": "edgar-net-income-downloader rex@memetica.sbs",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Accept-Encoding": "gzip, deflate, br",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# ----------------------------
# HTTP Helpers
# ----------------------------

def _cache_path(key: str) -> str:
    safe = key.replace("/", "_").replace(":", "_")
    return os.path.join(CACHE_DIR, safe + ".json")


def fetch_json(url: str, cache_key: Optional[str] = None) -> Dict[str, Any]:

    if USE_CACHE and cache_key:
        path = _cache_path(cache_key)
        if os.path.exists(path):
            return json.load(open(path))

    time.sleep(PAUSE_S)

    last_err: Optional[BaseException] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = SESSION.get(url, timeout=45)

            ctype = (resp.headers.get("Content-Type") or "").lower()
            text_head = resp.text[:200].strip().replace("\n", " ")

            if resp.status_code == 200:
                if "json" not in ctype:
                    raise RuntimeError(
                        f"Expected JSON but got {ctype} | First bytes: {text_head}"
                    )

                data = resp.json()

                if USE_CACHE and cache_key:
                    os.makedirs(CACHE_DIR, exist_ok=True)
                    json.dump(data, open(_cache_path(cache_key), "w"))

                return data

            if resp.status_code in (403, 429, 500, 502, 503, 504):
                print(f"Request failed ({resp.status_code}) — retrying... (attempt {attempt})")
                wait = RETRY_BACKOFF_BASE ** attempt
                time.sleep(wait)
                last_err = RuntimeError(text_head)
                continue

            raise RuntimeError(
                f"Request failed {resp.status_code} | First bytes: {text_head}"
            )

        except Exception as e:
            print(f"Request exception — retrying... (attempt {attempt})")
            last_err = e
            wait = RETRY_BACKOFF_BASE ** attempt
            time.sleep(wait)

    raise RuntimeError(f"Failed after retries: {url} (last_err={last_err})")


# ----------------------------
# SEC Helpers
# ----------------------------

def load_ticker_to_cik() -> Dict[str, str]:
    data = fetch_json(TICKER_CIK_URL, cache_key="company_tickers")

    out: Dict[str, str] = {}
    for _, row in data.items():
        ticker = str(row.get("ticker", "")).upper().strip()
        cik_str = row.get("cik_str")
        if ticker and cik_str is not None:
            out[ticker] = str(int(cik_str)).zfill(10)
    return out


def concept_url(cik10: str) -> str:
    return f"{BASE_CONCEPT}/CIK{cik10}/us-gaap/NetIncomeLoss.json"


def pick_unit_series(data: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    units_obj = data.get("units", {})
    if not units_obj:
        raise ValueError("No units found")

    if "USD" in units_obj:
        return "USD", units_obj["USD"]

    first_unit = next(iter(units_obj))
    return first_unit, units_obj[first_unit]


def extract_quarterly_latest(obs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in obs:
        if row.get("fp") not in {"Q1", "Q2", "Q3", "Q4"}:
            continue
        end = row.get("end")
        if not end:
            continue

        prev = out.get(end)
        if prev is None or (row.get("filed") or "") >= (prev.get("filed") or ""):
            out[end] = row

    return out


def most_recent_quarters(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    rows_sorted = sorted(rows, key=lambda r: r["quarter_end"])
    return rows_sorted[-n:] if len(rows_sorted) > n else rows_sorted


def rolling_3y_cagr(rows: List[Dict[str, Any]]) -> None:
    rows.sort(key=lambda r: r["quarter_end"])

    for i, r in enumerate(rows):
        r["net_income_cagr_3y"] = None
        if i < 12:
            continue

        ni_now = r["net_income"]
        ni_then = rows[i - 12]["net_income"]

        if ni_now and ni_then and ni_now > 0 and ni_then > 0:
            r["net_income_cagr_3y"] = (ni_now / ni_then) ** (1/3) - 1


# ----------------------------
# Main
# ----------------------------

def main():

    ticker_to_cik = load_ticker_to_cik()
    tickers = sorted(ticker_to_cik.keys())

    if MAX_TICKERS:
        tickers = tickers[:MAX_TICKERS]

    extracted = []
    failures = []

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ticker", "cik10", "quarter_end",
            "net_income", "net_income_cagr_3y"
        ])
        writer.writeheader()

        for ticker in tickers:
            cik10 = ticker_to_cik[ticker]

            try:
                print(f"Fetching ticker {ticker}...")

                data = fetch_json(
                    concept_url(cik10),
                    cache_key=f"{cik10}_NetIncomeLoss"
                )

                unit, obs = pick_unit_series(data)
                q = extract_quarterly_latest(obs)

                rows = []
                for end, row in q.items():
                    rows.append({
                        "quarter_end": end,
                        "net_income": float(row.get("val", 0))
                    })

                rows = most_recent_quarters(rows, N_QUARTERS)
                rolling_3y_cagr(rows)

                for r in rows:
                    writer.writerow({
                        "ticker": ticker,
                        "cik10": cik10,
                        **r
                    })

                print(f"Fetched ticker {ticker}")
                extracted.append(ticker)

            except Exception as e:
                print(f"FAILED ticker {ticker}")
                failures.append((ticker, str(e)))

    print("\nExtracted companies:")
    for t in extracted:
        print(t)

    if failures:
        print("\nFailures:")
        for t, err in failures:
            print(f"{t} -> {err}")

    print("\nDone.")
    print(f"CSV written to: {OUT_CSV}")


if __name__ == "__main__":
    main()
