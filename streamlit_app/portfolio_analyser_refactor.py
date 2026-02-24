from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ====== helpers ======
def max_drawdown_from_returns(rets: pd.Series) -> float:
    rets = rets.dropna()
    if rets.empty:
        return float("nan")
    equity = (1 + rets).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


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
    s = pd.to_numeric(s, errors="coerce").astype(float)
    lo = np.nanmin(s.values)
    hi = np.nanmax(s.values)
    if np.isnan(lo) or np.isnan(hi) or hi == lo:
        return pd.Series([0.5] * len(s), index=s.index, dtype=float)
    return (s - lo) / (hi - lo)


def load_latest_earnings(earnings_csv: Path) -> pd.DataFrame:
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
    out["net_income_latest_q"] = pd.to_numeric(latest.get("net_income"), errors="coerce")
    out = out.join(cagr, how="left")
    out["net_income_cagr_3y"] = pd.to_numeric(out["net_income_cagr_3y"], errors="coerce")
    return out


def load_marketcap(marketcap_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(marketcap_csv)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.set_index("ticker")
    if "marketCap" in df.columns:
        df["marketCap"] = pd.to_numeric(df["marketCap"], errors="coerce")
    cols = [c for c in ["name", "industry", "marketCap"] if c in df.columns]
    return df[cols]


def bucket_splits(row: pd.Series) -> Dict[str, float]:
    vol = row["vol_norm"]
    dd = row["dd_norm"]
    beta = row["beta_norm"]
    pe = row["pe_norm"]
    sharpe = row["sharpe_norm"]
    ni_cagr = row["ni_cagr_norm"]
    divy = row["div_yield_norm"]

    defensive = (0.40 * (1 - vol) + 0.30 * (1 - dd) + 0.20 * (1 - beta) + 0.10 * divy)
    quality = (0.45 * sharpe + 0.25 * (1 - dd) + 0.30 * ni_cagr)
    speculative = (0.40 * pe + 0.30 * beta + 0.30 * sharpe)
    degen = (0.40 * vol + 0.35 * dd + 0.20 * beta + 0.05 * pe)

    raw = np.array([defensive, quality, speculative, degen], dtype=float)
    raw = np.where(np.isnan(raw), 0.0, raw)
    tot = raw.sum()
    if tot <= 0:
        return {"Defensive": 0.25, "Quality": 0.25, "Speculative": 0.25, "Degen": 0.25}
    s = raw / tot
    return {"Defensive": float(s[0]), "Quality": float(s[1]), "Speculative": float(s[2]), "Degen": float(s[3])}


def _safe_download_prices(tickers: List[str], period: str, interval: str = "1d") -> pd.DataFrame:
    """
    yfinance can return:
      - MultiIndex columns for multiple tickers
      - single-level columns for one ticker
    We keep it as a DataFrame and handle both cases later.
    """
    if not tickers:
        return pd.DataFrame()

    return yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )


def _get_close_series(hist: pd.DataFrame, ticker: str) -> pd.Series:
    if hist.empty:
        return pd.Series(dtype=float)

    # MultiIndex: (ticker, field)
    if isinstance(hist.columns, pd.MultiIndex):
        for col in ["Close", "Adj Close"]:
            if (ticker, col) in hist.columns:
                s = hist[(ticker, col)]
                return pd.to_numeric(s, errors="coerce").dropna()
        return pd.Series(dtype=float)

    # Single ticker case: columns like ["Open","High","Low","Close",...]
    if "Close" in hist.columns:
        return pd.to_numeric(hist["Close"], errors="coerce").dropna()

    return pd.Series(dtype=float)


def analyze_portfolio(
    portfolio: List[Dict[str, float]],
    earnings_csv: Optional[Path] = None,
    marketcap_csv: Optional[Path] = None,
    bench_ticker: str = "SPY",
    history_years: int = 3,
    risk_free_annual: float = 0.0,
) -> Dict[str, Any]:
    """
    portfolio: [{"ticker":"AAPL","weight":0.2}, ...] (weights should sum to 1)
    Supports "CASH" as special ticker (treated as 0 vol/0 dd and 100% Defensive in mixes).
    """
    if not portfolio:
        raise ValueError("Portfolio is empty.")

    weights = {p["ticker"].upper(): float(p["weight"]) for p in portfolio}
    tickers = [t for t in weights.keys() if t != "CASH"]
    cash_weight = float(weights.get("CASH", 0.0))

    # Handle all-cash
    if not tickers:
        industry_summary = pd.DataFrame(
            {
                "industry_weight": [1.0],
                "industry_dividend_yield": [0.0],
                "Defensive": [1.0],
                "Quality": [0.0],
                "Speculative": [0.0],
                "Degen": [0.0],
            },
            index=["Cash"],
        )
        per_stock = pd.DataFrame(
            {
                "weight": [cash_weight if cash_weight > 0 else 1.0],
                "industry": ["Cash"],
                "dividend_yield": [0.0],
                "Defensive": [1.0],
                "Quality": [0.0],
                "Speculative": [0.0],
                "Degen": [0.0],
            },
            index=["CASH"],
        )
        return {
            "per_stock": per_stock,
            "portfolio_dividend_yield": 0.0,
            "bucket_mix": {"Defensive": 1.0, "Quality": 0.0, "Speculative": 0.0, "Degen": 0.0},
            "most_x": {"Most Defensive": "CASH", "Most Degen": "CASH", "Most Quality": "CASH"},
            "leaders": {
                "Defensive": {"ticker": "CASH", "portfolio_pct_in_bucket": 1.0},
                "Degen": {"ticker": "CASH", "portfolio_pct_in_bucket": 0.0},
                "Quality": {"ticker": "CASH", "portfolio_pct_in_bucket": 0.0},
            },
            "industry_summary": industry_summary,
        }

    earnings = load_latest_earnings(earnings_csv) if (earnings_csv and earnings_csv.exists()) else pd.DataFrame()
    mcap = load_marketcap(marketcap_csv) if (marketcap_csv and marketcap_csv.exists()) else pd.DataFrame()

    all_tickers = list(dict.fromkeys(tickers + [bench_ticker]))  # de-dupe
    hist = _safe_download_prices(all_tickers, period=f"{history_years}y")

    bench_close = _get_close_series(hist, bench_ticker)
    if bench_close.empty:
        # fallback: beta becomes NaN for all
        bench_rets = pd.Series(dtype=float)
    else:
        bench_rets = bench_close.pct_change()

    rows: List[Dict[str, Any]] = []
    skipped: List[Tuple[str, str]] = []

    for t in tickers:
        close = _get_close_series(hist, t)
        if close.empty:
            skipped.append((t, "no_price_history"))
            continue

        rets = close.pct_change()

        trailing_pe = None
        dividend_yield = None
        try:
            info = yf.Ticker(t).info or {}
            trailing_pe = info.get("trailingPE", None)

            # IMPORTANT: yfinance dividendYield is typically already a fraction (0.015 = 1.5%)
            dividend_yield = info.get("dividendYield", None)
        except Exception:
            pass

        rows.append(
            {
                "ticker": t,
                "weight": weights[t],
                "ann_return": annualize_return(rets),
                "ann_vol": annualize_vol(rets.std(ddof=1)),
                "sharpe": sharpe_ratio(rets, rf_annual=risk_free_annual),
                "max_drawdown": max_drawdown_from_returns(rets),
                "beta": beta_to_benchmark(rets, bench_rets) if not bench_rets.empty else float("nan"),
                "trailing_pe": trailing_pe,
                "dividend_yield_raw": dividend_yield,
            }
        )

    if not rows:
        raise ValueError(f"No tickers had usable price history. Skipped: {skipped[:10]}")

    df = pd.DataFrame(rows).set_index("ticker")

    # joins
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

    df["industry"] = df["industry"].fillna("Unknown")
    df["name"] = df["name"].fillna(df.index.to_series())

    # dividend yield: keep as fraction (0.015) not /100
    dy = pd.to_numeric(df["dividend_yield_raw"], errors="coerce").fillna(0.0)

    # Sometimes data sources provide 1.5 for 1.5% (rare but happens); detect and fix
    # If median > 1, assume percent-points and convert to fraction
    if dy.median() > 1.0:
        dy = dy / 100.0

    df["dividend_yield"] = dy.clip(lower=0.0, upper=0.20)

    # feature prep
    df["dd_mag"] = pd.to_numeric(df["max_drawdown"], errors="coerce").abs()
    df["trailing_pe"] = pd.to_numeric(df["trailing_pe"], errors="coerce")
    df["net_income_cagr_3y"] = pd.to_numeric(df["net_income_cagr_3y"], errors="coerce")

    pe_med = df["trailing_pe"].median(skipna=True)
    df["pe_filled"] = df["trailing_pe"].fillna(pe_med if not np.isnan(pe_med) else 20.0)

    sharpe_med = df["sharpe"].median(skipna=True)
    df["sharpe_filled"] = df["sharpe"].fillna(sharpe_med if not np.isnan(sharpe_med) else 0.0)

    df["ni_cagr_filled"] = df["net_income_cagr_3y"].fillna(0.0)
    df["div_yield_filled"] = df["dividend_yield"].fillna(0.0)

    # norms
    df["vol_norm"] = minmax_norm(df["ann_vol"])
    df["dd_norm"] = minmax_norm(df["dd_mag"])
    df["beta_norm"] = minmax_norm(df["beta"])
    df["pe_norm"] = minmax_norm(df["pe_filled"])
    df["sharpe_norm"] = minmax_norm(df["sharpe_filled"])
    df["ni_cagr_norm"] = minmax_norm(df["ni_cagr_filled"])
    df["div_yield_norm"] = minmax_norm(df["div_yield_filled"])

    # bucket splits
    bucket_cols = ["Defensive", "Quality", "Speculative", "Degen"]
    splits = df.apply(bucket_splits, axis=1, result_type="expand")
    df = pd.concat([df, splits], axis=1)

    # portfolio outputs
    portfolio_div_yield = float((df["dividend_yield"] * df["weight"]).sum())

    mix = {b: float((df[b] * df["weight"]).sum()) for b in bucket_cols}
    if cash_weight > 0:
        mix["Defensive"] += cash_weight
    tot = sum(mix.values())
    mix = {k: (v / tot if tot else 0.0) for k, v in mix.items()}

    most_x = {
        "Most Defensive": df["Defensive"].idxmax(),
        "Most Degen": df["Degen"].idxmax(),
        "Most Quality": df["Quality"].idxmax(),
    }

    contrib = pd.DataFrame(index=df.index)
    for b in bucket_cols:
        contrib[b] = df[b] * df["weight"]

    leaders = {}
    for b in ["Defensive", "Degen", "Quality"]:
        leader = contrib[b].idxmax()
        leaders[b] = {"ticker": leader, "portfolio_pct_in_bucket": float(contrib.loc[leader, b])}

    # industry summary
    industry_df = df[["industry", "weight", "dividend_yield"] + bucket_cols].copy()
    if cash_weight > 0:
        cash_row = pd.DataFrame(
            [
                {
                    "industry": "Cash",
                    "weight": cash_weight,
                    "dividend_yield": 0.0,
                    "Defensive": 1.0,
                    "Quality": 0.0,
                    "Speculative": 0.0,
                    "Degen": 0.0,
                }
            ],
            index=["CASH"],
        )
        industry_df = pd.concat([industry_df, cash_row], axis=0)

    industry_alloc = industry_df.groupby("industry")["weight"].sum().sort_values(ascending=False)

    industry_yield = (
        industry_df.assign(div_contrib=industry_df["weight"] * industry_df["dividend_yield"])
        .groupby("industry")[["weight", "div_contrib"]]
        .sum()
    )
    industry_yield["industry_dividend_yield"] = np.where(
        industry_yield["weight"] > 0, industry_yield["div_contrib"] / industry_yield["weight"], 0.0
    )

    for b in bucket_cols:
        industry_df[f"{b}_contrib"] = industry_df["weight"] * industry_df[b]
    industry_bucket = industry_df.groupby("industry")[[f"{b}_contrib" for b in bucket_cols]].sum()
    row_sums = industry_bucket.sum(axis=1).replace(0, np.nan)
    industry_bucket_norm = industry_bucket.div(row_sums, axis=0).fillna(0.0)
    industry_bucket_norm.columns = bucket_cols

    industry_summary = (
        pd.DataFrame({"industry_weight": industry_alloc})
        .join(industry_yield["industry_dividend_yield"], how="left")
        .join(industry_bucket_norm, how="left")
        .sort_values("industry_weight", ascending=False)
    )

    # optionally expose skips for debugging
    df.attrs["skipped_tickers"] = skipped

    return {
        "per_stock": df,
        "portfolio_dividend_yield": portfolio_div_yield,
        "bucket_mix": mix,
        "most_x": most_x,
        "leaders": leaders,
        "industry_summary": industry_summary,
    }