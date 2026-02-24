from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yfinance as yf


# =========================
# Files (edit if needed)
# =========================
EARNINGS_CSV = Path("sec_quarterly_net_income.csv")
MARKETCAP_CSV = Path("capital_flows/tickers_2026-02-23.csv")

# =========================
# Synthetic Portfolio
# =========================
PORTFOLIO = [
    {"ticker": "AAPL", "weight": 0.15},
    {"ticker": "MSFT", "weight": 0.15},
    {"ticker": "JNJ",  "weight": 0.10},
    {"ticker": "PG",   "weight": 0.10},
    {"ticker": "XOM",  "weight": 0.10},
    {"ticker": "NVDA", "weight": 0.10},
    {"ticker": "TSLA", "weight": 0.10},
    {"ticker": "AMZN", "weight": 0.10},
    {"ticker": "JPM",  "weight": 0.05},
    {"ticker": "CASH", "weight": 0.05},  # special
]

BENCH_TICKER = "SPY"
HISTORY_YEARS = 3
RISK_FREE_ANNUAL = 0.0  # MVP: 0%; plug in T-bill later


# =========================
# Market metrics helpers
# =========================
def max_drawdown_from_returns(rets: pd.Series) -> float:
    if rets.dropna().empty:
        return float("nan")
    equity = (1 + rets.fillna(0)).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())  # negative


def annualize_vol(daily_vol: float) -> float:
    return float(daily_vol * math.sqrt(252))


def annualize_return(daily_rets: pd.Series) -> float:
    daily_rets = daily_rets.dropna()
    if daily_rets.empty:
        return float("nan")
    compounded = (1 + daily_rets).prod()
    years = len(daily_rets) / 252.0
    if years <= 0:
        return float("nan")
    return float(compounded ** (1 / years) - 1)


def sharpe_ratio(daily_rets: pd.Series, rf_annual: float = 0.0) -> float:
    daily_rets = daily_rets.dropna()
    if daily_rets.empty:
        return float("nan")
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    excess = daily_rets - rf_daily
    mu = excess.mean()
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float((mu / sd) * math.sqrt(252))


def beta_to_benchmark(asset_rets: pd.Series, bench_rets: pd.Series) -> float:
    df = pd.concat([asset_rets, bench_rets], axis=1).dropna()
    if df.shape[0] < 60:
        return float("nan")
    a = df.iloc[:, 0].values
    b = df.iloc[:, 1].values
    var_b = np.var(b, ddof=1)
    if var_b == 0 or np.isnan(var_b):
        return float("nan")
    cov = np.cov(a, b, ddof=1)[0, 1]
    return float(cov / var_b)


def minmax_norm(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo = np.nanmin(s.values)
    hi = np.nanmax(s.values)
    if np.isnan(lo) or np.isnan(hi) or hi == lo:
        return pd.Series([0.5] * len(s), index=s.index, dtype=float)
    return (s - lo) / (hi - lo)


# =========================
# CSV loaders
# =========================
def load_latest_earnings(earnings_csv: Path) -> pd.DataFrame:
    """
    Returns indexed by ticker:
      - latest_quarter_end
      - net_income_latest_q
      - net_income_cagr_3y (last non-null if present)
    """
    df = pd.read_csv(earnings_csv)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["quarter_end"] = pd.to_datetime(df["quarter_end"], errors="coerce")

    latest = (
        df.sort_values(["ticker", "quarter_end"])
          .groupby("ticker", as_index=False)
          .tail(1)
          .set_index("ticker")
    )

    cagr = (
        df.dropna(subset=["net_income_cagr_3y"])
          .sort_values(["ticker", "quarter_end"])
          .groupby("ticker", as_index=False)
          .tail(1)
          .set_index("ticker")[["net_income_cagr_3y"]]
    )

    out = pd.DataFrame(index=latest.index)
    out["latest_quarter_end"] = latest["quarter_end"].dt.date.astype(str)
    out["net_income_latest_q"] = pd.to_numeric(latest["net_income"], errors="coerce")
    out = out.join(cagr, how="left")
    out["net_income_cagr_3y"] = pd.to_numeric(out["net_income_cagr_3y"], errors="coerce")
    return out


def load_marketcap(marketcap_csv: Path) -> pd.DataFrame:
    """
    Returns indexed by ticker:
      - name
      - industry
      - marketCap
    """
    df = pd.read_csv(marketcap_csv)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.set_index("ticker")

    if "marketCap" in df.columns:
        df["marketCap"] = pd.to_numeric(df["marketCap"], errors="coerce")

    cols = [c for c in ["name", "industry", "marketCap"] if c in df.columns]
    return df[cols]


# =========================
# Bucket logic (overlapping)
# =========================
def bucket_splits(row: pd.Series) -> Dict[str, float]:
    """
    Overlapping buckets:
      Defensive / Quality / Speculative / Degen
    Returns splits that sum to 1.0.
    Uses normalized features in row.
    """
    vol = row["vol_norm"]
    dd = row["dd_norm"]
    beta = row["beta_norm"]
    pe = row["pe_norm"]
    sharpe = row["sharpe_norm"]
    ni_cagr = row["ni_cagr_norm"]
    divy = row["div_yield_norm"]

    defensive = (
        0.40 * (1 - vol) +
        0.30 * (1 - dd) +
        0.20 * (1 - beta) +
        0.10 * divy
    )

    quality = (
        0.45 * sharpe +
        0.25 * (1 - dd) +
        0.30 * ni_cagr
    )

    speculative = (
        0.40 * pe +
        0.30 * beta +
        0.30 * sharpe
    )

    degen = (
        0.40 * vol +
        0.35 * dd +
        0.20 * beta +
        0.05 * pe
    )

    raw = np.array([defensive, quality, speculative, degen], dtype=float)
    raw = np.where(np.isnan(raw), 0.0, raw)
    tot = raw.sum()
    if tot <= 0:
        return {"Defensive": 0.25, "Quality": 0.25, "Speculative": 0.25, "Degen": 0.25}

    s = raw / tot
    return {"Defensive": float(s[0]), "Quality": float(s[1]), "Speculative": float(s[2]), "Degen": float(s[3])}


def main():
    # ----- portfolio prep
    weights = {p["ticker"].upper(): float(p["weight"]) for p in PORTFOLIO}
    tickers = [t for t in weights.keys() if t != "CASH"]
    cash_weight = float(weights.get("CASH", 0.0))

    # ----- load fundamentals from files
    earnings = load_latest_earnings(EARNINGS_CSV) if EARNINGS_CSV.exists() else pd.DataFrame()
    mcap = load_marketcap(MARKETCAP_CSV) if MARKETCAP_CSV.exists() else pd.DataFrame()

    # ----- prices
    all_tickers = tickers + [BENCH_TICKER]
    hist = yf.download(
        tickers=all_tickers,
        period=f"{HISTORY_YEARS}y",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    def get_close(t: str) -> pd.Series:
        if isinstance(hist.columns, pd.MultiIndex):
            for col in ["Close", "Adj Close"]:
                if (t, col) in hist.columns:
                    return hist[(t, col)].dropna()
            raise KeyError(f"Missing Close for {t}")
        return hist["Close"].dropna()

    bench_rets = get_close(BENCH_TICKER).pct_change()

    # ----- compute per-stock market metrics
    rows: List[Dict[str, Any]] = []
    for t in tickers:
        px = get_close(t)
        rets = px.pct_change()

        trailing_pe = None
        dividend_yield = None
        try:
            info = yf.Ticker(t).info
            trailing_pe = info.get("trailingPE", None)
            dividend_yield = info.get("dividendYield", None)  # in your env: percent points
        except Exception:
            pass

        rows.append({
            "ticker": t,
            "weight": weights[t],
            "ann_return": annualize_return(rets),
            "ann_vol": annualize_vol(rets.std(ddof=1)),
            "sharpe": sharpe_ratio(rets, rf_annual=RISK_FREE_ANNUAL),
            "max_drawdown": max_drawdown_from_returns(rets),
            "beta": beta_to_benchmark(rets, bench_rets),
            "trailing_pe": trailing_pe,
            "dividend_yield_raw": dividend_yield,  # keep raw for debugging
        })

    df = pd.DataFrame(rows).set_index("ticker")

    # ----- join earnings + market cap
    if not earnings.empty:
        df = df.join(earnings[["latest_quarter_end", "net_income_latest_q", "net_income_cagr_3y"]], how="left")
    else:
        df["latest_quarter_end"] = None
        df["net_income_latest_q"] = np.nan
        df["net_income_cagr_3y"] = np.nan

    if not mcap.empty:
        df = df.join(mcap, how="left")
    else:
        df["name"] = df.index.to_series()
        df["industry"] = "Unknown"
        df["marketCap"] = np.nan

    # fill for grouping
    df["industry"] = df["industry"].fillna("Unknown")
    df["name"] = df["name"].fillna(df.index.to_series())

    # ----- dividend yield normalization (CRITICAL)
    # Your observed yfinance values are percent points (2.73 means 2.73%).
    # Convert to fraction: 0.0273.
    df["dividend_yield"] = pd.to_numeric(df["dividend_yield_raw"], errors="coerce").fillna(0.0) / 100.0
    df["dividend_yield"] = df["dividend_yield"].clip(lower=0.0, upper=0.20)

    # ----- prep for normalization
    df["dd_mag"] = df["max_drawdown"].abs()

    df["trailing_pe"] = pd.to_numeric(df["trailing_pe"], errors="coerce")
    df["net_income_cagr_3y"] = pd.to_numeric(df["net_income_cagr_3y"], errors="coerce")

    pe_med = df["trailing_pe"].median(skipna=True)
    df["pe_filled"] = df["trailing_pe"].fillna(pe_med if not np.isnan(pe_med) else 20.0)

    sharpe_med = df["sharpe"].median(skipna=True)
    df["sharpe_filled"] = df["sharpe"].fillna(sharpe_med if not np.isnan(sharpe_med) else 0.0)

    df["ni_cagr_filled"] = df["net_income_cagr_3y"].fillna(0.0)
    df["div_yield_filled"] = df["dividend_yield"].fillna(0.0)

    # ----- normalize features (within the holdings set)
    df["vol_norm"] = minmax_norm(df["ann_vol"])
    df["dd_norm"] = minmax_norm(df["dd_mag"])
    df["beta_norm"] = minmax_norm(df["beta"])
    df["pe_norm"] = minmax_norm(df["pe_filled"])
    df["sharpe_norm"] = minmax_norm(df["sharpe_filled"])
    df["ni_cagr_norm"] = minmax_norm(df["ni_cagr_filled"])
    df["div_yield_norm"] = minmax_norm(df["div_yield_filled"])

    # ----- bucket splits
    bucket_cols = ["Defensive", "Quality", "Speculative", "Degen"]
    splits = df.apply(bucket_splits, axis=1, result_type="expand")
    df = pd.concat([df, splits], axis=1)

    # =========================
    # Portfolio-level outputs
    # =========================

    # Portfolio dividend yield (weight-weighted)
    portfolio_div_yield = float((df["dividend_yield"] * df["weight"]).sum())  # cash assumed 0
    print("\n=== Portfolio Dividend Yield ===")
    print(f"Weighted dividend yield: {portfolio_div_yield*100:6.2f}%")

    # Portfolio bucket mix
    mix = {b: float((df[b] * df["weight"]).sum()) for b in bucket_cols}
    if cash_weight > 0:
        mix["Defensive"] += cash_weight  # cash is 100% Defensive
    tot = sum(mix.values())
    mix = {k: (v / tot if tot else 0.0) for k, v in mix.items()}

    print("\n=== Portfolio Bucket Mix ===")
    for k in bucket_cols:
        print(f"{k:12s}: {mix[k]*100:6.2f}%")

    # Most X (raw trait highest)
    print("\n=== Most X (raw trait highest within holdings) ===")
    print(f"Most Defensive : {df['Defensive'].idxmax()}")
    print(f"Most Degen     : {df['Degen'].idxmax()}")
    print(f"Most Quality   : {df['Quality'].idxmax()}")

    # Biggest contributors (trait * weight) + portfolio percentage for each
    contrib = pd.DataFrame(index=df.index)
    for b in bucket_cols:
        contrib[b] = df[b] * df["weight"]  # contribution to the portfolio's bucket %

    print("\n=== Biggest Contributors (trait * weight) ===")
    for b in ["Defensive", "Degen", "Quality"]:
        leader = contrib[b].idxmax()
        leader_pct = float(contrib.loc[leader, b]) * 100  # already in portfolio-weight terms
        print(f"{b:9s} leader: {leader:5s}  ({leader_pct:5.2f}% of portfolio in {b})")

    # =========================
    # Industry outputs (instead of per-stock printing)
    # =========================
    industry_df = df[["industry", "weight", "dividend_yield"] + bucket_cols].copy()

    # Add cash as its own industry
    if cash_weight > 0:
        cash_row = pd.DataFrame([{
            "industry": "Cash",
            "weight": cash_weight,
            "dividend_yield": 0.0,
            "Defensive": 1.0,
            "Quality": 0.0,
            "Speculative": 0.0,
            "Degen": 0.0,
        }], index=["CASH"])
        industry_df = pd.concat([industry_df, cash_row], axis=0)

    # 1) industry allocation
    industry_alloc = (
        industry_df.groupby("industry")["weight"]
        .sum()
        .sort_values(ascending=False)
    )

    # 2) industry dividend yield (weight-normalized within industry)
    industry_yield = (
        industry_df.assign(div_contrib=industry_df["weight"] * industry_df["dividend_yield"])
        .groupby("industry")[["weight", "div_contrib"]]
        .sum()
    )
    industry_yield["industry_dividend_yield"] = np.where(
        industry_yield["weight"] > 0,
        industry_yield["div_contrib"] / industry_yield["weight"],
        0.0
    )

    # 3) industry bucket mix (within each industry)
    for b in bucket_cols:
        industry_df[f"{b}_contrib"] = industry_df["weight"] * industry_df[b]

    industry_bucket = industry_df.groupby("industry")[[f"{b}_contrib" for b in bucket_cols]].sum()
    row_sums = industry_bucket.sum(axis=1).replace(0, np.nan)
    industry_bucket_norm = industry_bucket.div(row_sums, axis=0).fillna(0.0)
    industry_bucket_norm.columns = bucket_cols

    # Combine into a single industry summary table
    industry_summary = pd.DataFrame({
        "industry_weight": industry_alloc
    }).join(industry_yield["industry_dividend_yield"], how="left").join(industry_bucket_norm, how="left")

    industry_summary = industry_summary.sort_values("industry_weight", ascending=False)

    print("\n=== Industry Summary (weight, dividend yield, bucket mix) ===")
    N = 20
    view = industry_summary.head(N).copy()

    # Format for display
    view_fmt = view.copy()
    view_fmt["industry_weight"] = view_fmt["industry_weight"] * 100
    view_fmt["industry_dividend_yield"] = view_fmt["industry_dividend_yield"] * 100
    for b in bucket_cols:
        view_fmt[b] = view_fmt[b] * 100

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(view_fmt.to_string(float_format=lambda x: f"{x:6.2f}"))

    # Most X industry (composition-based)
    if not industry_bucket_norm.empty:
        print("\n=== Most X Industry (composition-based) ===")
        print(f"Most Defensive industry: {industry_bucket_norm['Defensive'].idxmax()}")
        print(f"Most Degen industry    : {industry_bucket_norm['Degen'].idxmax()}")
        print(f"Most Quality industry  : {industry_bucket_norm['Quality'].idxmax()}")


if __name__ == "__main__":
    main()