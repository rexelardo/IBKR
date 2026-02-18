"""
tws_portfolio_value_plus_option_delta_usd_pies.py

WHAT THIS DOES
--------------
- Stocks: uses IBKR PortfolioItem.marketValue (no yfinance stock prices needed).
- Options: converts option positions into delta-equivalent UNDERLYING exposure using Black–Scholes delta,
           with IV sourced from yfinance option_chain (fallback to HV if IV missing),
           and underlying price sourced from yfinance (ONLY for underlyings that have options).
- Converts everything to USD using yfinance FX rates.
- Creates:
    1) Net LONG exposure pie (USD) with Other (<=1%) + Other breakdown (legend)
    2) Net SHORT exposure pie (abs USD) with Other (<=1%) + Other breakdown (legend)
- Saves PNGs with timestamps.
- FX snapshot + model stats placed in the bottom margin (won’t cover the pie).

INSTALL
-------
pip install ib_insync pandas matplotlib yfinance scipy

RUN
---
python tws_portfolio_value_plus_option_delta_usd_pies.py
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
import math
import os
from typing import Dict, Optional, Tuple, Set

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm

from ib_insync import IB
import logging


# ============================== CONFIG ==============================

HOST = "127.0.0.1"
PORT = 7496
CLIENT_ID = 31

THRESHOLD_PCT = 1.0

# Option delta model
RISK_FREE_RATE = 0.045
DIVIDEND_YIELD_FALLBACK = 0.0

# Vol fallback
HV_WINDOW_DAYS = 60
TRADING_DAYS = 252
YF_PRICE_LOOKBACK = "6mo"

# FX snapshot window
YF_FX_LOOKBACK_DAYS = 7

# Yahoo ticker overrides (only needed for OPTION underlyings or FX tickers)
YF_SYMBOL_MAP: Dict[str, str] = {
    # If you have options on HK numeric tickers:
    # "700": "0700.HK",
    # "3690": "3690.HK",
    # "AKRBP": "AKRBP.OL",
    # "POU": "POU.TO",
    # "ETL": "ETL.PA",
}

# Silence ib_insync chatter
SILENCE_IB_LOGS = True

FANNIE_GROUP_NAME = "FANNIE MAE"
FANNIE_TICKERS = {"FMCC", "FMCCS", "FMCKJ", "FNMA", "FNMAL", "FNMAS", "FNMAT"}

# ===================================================================


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def fmt_money(x: float) -> str:
    return f"{x:,.2f}"


@dataclass(frozen=True)
class FxQuote:
    pair: str
    px: float
    direction: str
    usd_per_ccy: float
    source: str


def yahoo_symbol(ib_symbol: str) -> str:
    return YF_SYMBOL_MAP.get(ib_symbol, ib_symbol)


# -------------------- FX (yfinance) --------------------

def yfinance_usd_per_ccy(ccy: str) -> Optional[FxQuote]:
    ccy = (ccy or "").upper().strip()
    if ccy in {"", "USD"}:
        return FxQuote(pair="USD", px=1.0, direction="direct", usd_per_ccy=1.0, source="BASE")

    sym_direct = f"{ccy}USD=X"
    try:
        hist = yf.Ticker(sym_direct).history(period=f"{YF_FX_LOOKBACK_DAYS}d", interval="1d")
        if not hist.empty:
            px = safe_float(hist["Close"].iloc[-1])
            if px and px > 0:
                return FxQuote(pair=sym_direct, px=px, direction="direct", usd_per_ccy=px, source="YF")
    except Exception:
        pass

    sym_invert = f"USD{ccy}=X"
    try:
        hist = yf.Ticker(sym_invert).history(period=f"{YF_FX_LOOKBACK_DAYS}d", interval="1d")
        if not hist.empty:
            px = safe_float(hist["Close"].iloc[-1])
            if px and px > 0:
                return FxQuote(pair=sym_invert, px=px, direction="invert", usd_per_ccy=1.0 / px, source="YF")
    except Exception:
        pass

    if ccy == "CNY":
        for sym in ("USDCNH=X", "CNHUSD=X"):
            try:
                hist = yf.Ticker(sym).history(period=f"{YF_FX_LOOKBACK_DAYS}d", interval="1d")
                if not hist.empty:
                    px = safe_float(hist["Close"].iloc[-1])
                    if px and px > 0:
                        if sym.startswith("USD"):
                            return FxQuote(pair=sym, px=px, direction="invert", usd_per_ccy=1.0 / px, source="YF")
                        else:
                            return FxQuote(pair=sym, px=px, direction="direct", usd_per_ccy=px, source="YF")
            except Exception:
                pass

    return None


def fx_snapshot_text(fx_quotes: Dict[str, FxQuote], max_lines: int = 16) -> str:
    lines = []
    for ccy in sorted(fx_quotes.keys()):
        q = fx_quotes[ccy]
        if ccy == "USD":
            lines.append("USD: 1.000000 (base)")
        else:
            lines.append(f"{ccy}: {q.usd_per_ccy:.6f} USD/{ccy} [{q.source}] ({q.pair}={q.px:.6f})")

    if len(lines) <= max_lines:
        return "FX snapshot used:\n" + "\n".join(lines)

    head = lines[: max_lines - 1]
    tail = len(lines) - (max_lines - 1)
    return "FX snapshot used:\n" + "\n".join(head + [f"... +{tail} more"])


# -------------------- Black-Scholes delta helpers --------------------

def annualized_hist_vol(closes: pd.Series, window_days: int) -> Optional[float]:
    closes = closes.dropna()
    if len(closes) < window_days + 2:
        return None
    logret = (closes / closes.shift(1)).apply(lambda x: math.log(x) if x and x > 0 else float("nan")).dropna()
    if len(logret) < window_days:
        return None
    sigma_daily = float(logret.tail(window_days).std(ddof=1))
    return sigma_daily * math.sqrt(TRADING_DAYS)


def yfinance_underlying_inputs_for_options(ib_symbol: str) -> Tuple[Optional[float], Optional[float], float]:
    """
    Used ONLY for OPTION underlyings.
    Returns: S_last_close, sigma_hv_annual, dividend_yield
    """
    ysym = yahoo_symbol(ib_symbol)
    t = yf.Ticker(ysym)

    hist = t.history(period=YF_PRICE_LOOKBACK, interval="1d")
    if hist.empty:
        return None, None, DIVIDEND_YIELD_FALLBACK

    closes = hist["Close"].dropna()
    if closes.empty:
        return None, None, DIVIDEND_YIELD_FALLBACK

    S = safe_float(closes.iloc[-1])
    sigma_hv = annualized_hist_vol(closes, HV_WINDOW_DAYS)

    q = DIVIDEND_YIELD_FALLBACK
    try:
        info = t.info or {}
        q_info = safe_float(info.get("dividendYield"))
        if q_info is not None:
            q = q_info
    except Exception:
        pass

    return S, sigma_hv, q


def bs_delta(S: float, K: float, T: float, r: float, q: float, sigma: float, right: str) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    if right.upper() == "C":
        return math.exp(-q * T) * norm.cdf(d1)
    else:
        return math.exp(-q * T) * (norm.cdf(d1) - 1.0)


def parse_ib_expiry(last_trade: str) -> Optional[date]:
    if not last_trade:
        return None
    s = last_trade.strip().split(" ")[0]
    if len(s) >= 8 and s[:8].isdigit():
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    return None


def get_chain_row(chain_df: pd.DataFrame, strike: float) -> Optional[pd.Series]:
    if chain_df is None or chain_df.empty:
        return None
    m = chain_df[chain_df["strike"] == strike]
    if not m.empty:
        return m.iloc[0]
    m = chain_df[(chain_df["strike"] - strike).abs() < 1e-6]
    if not m.empty:
        return m.iloc[0]
    return None


# -------------------- Charting --------------------

def add_info_box_bottom_margin(fig, text: str):
    """
    Put the box in the FIGURE margin area, not on top of the pie.
    We also allocate extra bottom space using subplots_adjust.
    """
    fig.text(
        0.01, 0.01,
        text,
        ha="left", va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.90)
    )


def plot_main_pie_with_other(df: pd.DataFrame, value_col: str, title: str, out_file: str, info_box_text: str):
    big = df[df["pct"] > THRESHOLD_PCT].copy()
    small = df[df["pct"] <= THRESHOLD_PCT].copy()

    main = big.copy()
    if not small.empty:
        main = pd.concat([main, pd.DataFrame([{
            "underlying": f"Other (≤{THRESHOLD_PCT:.0f}%)",
            "currency": "",
            value_col: small[value_col].sum(),
            "pct": small["pct"].sum(),
        }])], ignore_index=True)

    labels = []
    for r in main.itertuples(index=False):
        if str(r.underlying).startswith("Other"):
            labels.append(f"{r.underlying} ({r.pct:.1f}%)")
        else:
            labels.append(f"{r.underlying} ({r.currency}) ({r.pct:.1f}%)")

    fig = plt.figure(figsize=(10, 9))
    ax = plt.gca()

    ax.pie(
        main[value_col],
        labels=labels,
        autopct=lambda p: f"{p:.1f}%" if p >= 2 else "",
        startangle=90
    )
    ax.set_title(title)

    # Reserve bottom margin so info box never overlaps chart
    plt.subplots_adjust(bottom=0.22)
    add_info_box_bottom_margin(fig, info_box_text)

    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def pie_other_breakdown_with_legend_auto(df: pd.DataFrame, value_col: str, title: str, out_file: str, info_box_text: str):
    df = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    n = len(df)

    if n > 45:
        ncol = 3
    elif n > 20:
        ncol = 2
    else:
        ncol = 1

    width = 12 + max(0, n - 15) * 0.15
    if ncol == 2:
        width += 1.5
    elif ncol == 3:
        width += 3.0
    width = min(22, width)
    height = 9

    values = df[value_col].values
    legend_labels = [f"{r.underlying} ({r.currency}) — {r.pct:.2f}%" for r in df.itertuples(index=False)]

    fig = plt.figure(figsize=(width, height))
    ax = plt.gca()

    wedges, _, _ = ax.pie(
        values,
        labels=None,
        autopct=lambda p: f"{p:.1f}%" if p >= 2 else "",
        startangle=90,
        pctdistance=0.75
    )
    ax.set_title(title)

    ax.legend(
        wedges,
        legend_labels,
        title="Holdings (% of breakdown)",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        title_fontsize=10,
        frameon=True,
        ncol=ncol,
        columnspacing=1.2,
        handletextpad=0.6,
        borderaxespad=0.0,
    )

    plt.subplots_adjust(bottom=0.22)
    add_info_box_bottom_margin(fig, info_box_text)

    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================== MAIN ==============================

def main():
    if SILENCE_IB_LOGS:
        logging.getLogger("ib_insync").setLevel(logging.CRITICAL)
        logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    ib = IB()
    # Shorter timeout avoids long hangs; we don't care about orders here.
    ib.RequestTimeout = 5

    ib.connect(HOST, PORT, clientId=CLIENT_ID)

    try:
        portfolio = ib.portfolio()
        if not portfolio:
            print("No portfolio items returned.")
            return

        stk_positions = [p for p in portfolio if p.contract.secType == "STK" and (safe_float(p.position) or 0) != 0]
        opt_positions = [p for p in portfolio if p.contract.secType == "OPT" and (safe_float(p.position) or 0) != 0]

        # We only need yfinance underlying price/IV for underlyings that actually have options
        option_underlyings: Set[str] = {p.contract.symbol for p in opt_positions}

        # Collect currencies for FX
        currencies = set()
        for p in portfolio:
            currencies.add((p.contract.currency or "USD").upper())

        # Build FX map via yfinance
        fx_quotes: Dict[str, FxQuote] = {"USD": FxQuote(pair="USD", px=1.0, direction="direct", usd_per_ccy=1.0, source="BASE")}
        fx_map: Dict[str, float] = {"USD": 1.0}

        for ccy in sorted(currencies):
            if ccy == "USD":
                continue
            q = yfinance_usd_per_ccy(ccy)
            if q is None:
                print(f"WARNING: Missing FX for {ccy}; holdings in {ccy} will be skipped.")
                continue
            fx_quotes[ccy] = q
            fx_map[ccy] = q.usd_per_ccy

        # ---- Stocks: value from IB marketValue (no yfinance price) ----
        # We'll aggregate by underlying symbol (stock symbol)
        net_usd_by_underlying: Dict[Tuple[str, str], float] = {}  # (symbol, currency) -> USD value

        skipped_stock_fx = 0
        for p in stk_positions:
            sym = p.contract.symbol
            ccy = (p.contract.currency or "USD").upper()
            mv = safe_float(p.marketValue)  # in that currency
            if mv is None:
                continue
            if ccy not in fx_map:
                skipped_stock_fx += 1
                continue
            usd_mv = mv * fx_map[ccy]
            net_usd_by_underlying[(sym, ccy)] = net_usd_by_underlying.get((sym, ccy), 0.0) + usd_mv

        # ---- Options: delta-equivalent underlying exposure ----
        underlying_cache: Dict[str, Tuple[Optional[float], Optional[float], float]] = {}
        chain_cache: Dict[Tuple[str, str], Tuple[pd.DataFrame, pd.DataFrame]] = {}

        def get_underlying_inputs(sym: str):
            if sym not in underlying_cache:
                underlying_cache[sym] = yfinance_underlying_inputs_for_options(sym)
            return underlying_cache[sym]

        def get_chain(sym: str, expiry_str: str):
            key = (sym, expiry_str)
            if key in chain_cache:
                return chain_cache[key]
            t = yf.Ticker(yahoo_symbol(sym))
            try:
                oc = t.option_chain(expiry_str)
                chain_cache[key] = (oc.calls, oc.puts)
            except Exception:
                chain_cache[key] = (pd.DataFrame(), pd.DataFrame())
            return chain_cache[key]

        today = date.today()
        used_iv = 0
        used_hv = 0
        skipped_opts = 0
        skipped_opt_fx = 0
        missing_opt_underlying_price = 0

        for p in opt_positions:
            c = p.contract
            sym = c.symbol
            ccy = (c.currency or "USD").upper()

            if ccy not in fx_map:
                skipped_opt_fx += 1
                continue

            expiry = parse_ib_expiry(getattr(c, "lastTradeDateOrContractMonth", ""))
            if expiry is None:
                skipped_opts += 1
                continue

            days = (expiry - today).days
            T = max(days, 0) / 365.0
            if T <= 0:
                skipped_opts += 1
                continue

            K = safe_float(getattr(c, "strike", None))
            if K is None or K <= 0:
                skipped_opts += 1
                continue

            right = getattr(c, "right", "").upper()
            if right not in {"C", "P"}:
                skipped_opts += 1
                continue

            pos_contracts = safe_float(p.position) or 0.0
            mult = safe_float(getattr(c, "multiplier", None)) or 100.0

            # Underlying inputs from yfinance (ONLY for option underlyings)
            S, sigma_hv, q_div = get_underlying_inputs(sym)
            if S is None:
                missing_opt_underlying_price += 1
                skipped_opts += 1
                continue

            expiry_str = expiry.strftime("%Y-%m-%d")
            calls_df, puts_df = get_chain(sym, expiry_str)
            chain_df = calls_df if right == "C" else puts_df

            iv = None
            row = get_chain_row(chain_df, K)
            if row is not None:
                iv = safe_float(row.get("impliedVolatility"))

            if iv is not None and iv > 0:
                sigma = iv / 100.0 if iv > 5 else iv
                used_iv += 1
            elif sigma_hv is not None and sigma_hv > 0:
                sigma = sigma_hv
                used_hv += 1
            else:
                skipped_opts += 1
                continue

            d = bs_delta(S=S, K=K, T=T, r=RISK_FREE_RATE, q=q_div, sigma=sigma, right=right)

            # Delta-equivalent UNDERLYING shares
            shares_equiv = pos_contracts * d * mult

            # Convert to USD VALUE using underlying price
            usd_value = shares_equiv * S * fx_map[ccy]

            net_usd_by_underlying[(sym, ccy)] = net_usd_by_underlying.get((sym, ccy), 0.0) + usd_value

        # ---- Build DataFrame ----
        rows = []
        for (sym, ccy), usd_value in net_usd_by_underlying.items():
            if abs(usd_value) < 1e-6:
                continue
            rows.append({"underlying": sym, "currency": ccy, "usdValue": usd_value})

        if not rows:
            print("No exposures computed.")
            return
        raw_df = pd.DataFrame(rows)
        raw_df["underlying"] = raw_df["underlying"].astype(str).str.upper().replace(
            {t: FANNIE_GROUP_NAME for t in FANNIE_TICKERS}
)
        df = raw_df.groupby(["underlying", "currency"], as_index=False).agg(
            usdValue=("usdValue", "sum")
        )

        longs = df[df["usdValue"] > 0].copy().sort_values("usdValue", ascending=False)
        shorts = df[df["usdValue"] < 0].copy()
        shorts["absUsdValue"] = shorts["usdValue"].abs()
        shorts = shorts.sort_values("absUsdValue", ascending=False)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        fx_box = fx_snapshot_text(fx_quotes, max_lines=16)
        model_box = (
            "Stocks: IB marketValue (no yfinance prices)\n"
            "Options: Black–Scholes delta → delta-equivalent underlying value\n"
            f"Sigma: yfinance IV per contract (fallback HV {HV_WINDOW_DAYS}d)\n"
            f"r={RISK_FREE_RATE:.3%}, q=yfinance dividendYield (fallback {DIVIDEND_YIELD_FALLBACK:.1%})\n"
            f"Used IV={used_iv}, HV fallback={used_hv}, skipped opts={skipped_opts}\n"
            f"Skipped FX: stocks={skipped_stock_fx}, opts={skipped_opt_fx}\n"
            f"Missing option-underlying price={missing_opt_underlying_price}"
        )
        info_box = fx_box + "\n\n" + model_box

        # Print summary tables
        print("\n=== Net exposure (USD) — LONGS ===\n")
        if not longs.empty:
            out = longs.copy()
            out["usdValue"] = out["usdValue"].map(fmt_money)
            print(out[["underlying", "currency", "usdValue"]].to_string(index=False))
            print(f"\nTOTAL LONG USD: {fmt_money(longs['usdValue'].sum())}\n")

        print("\n=== Net exposure (USD) — SHORTS ===\n")
        if not shorts.empty:
            out = shorts.copy()
            out["usdValue"] = out["usdValue"].map(fmt_money)
            out["absUsdValue"] = out["absUsdValue"].map(fmt_money)
            print(out[["underlying", "currency", "usdValue", "absUsdValue"]].to_string(index=False))
            print(f"\nTOTAL SHORT (abs) USD: {fmt_money(shorts['absUsdValue'].sum())}\n")

        # ---- Charts ----
        if not longs.empty:
            longs["pct"] = longs["usdValue"] / longs["usdValue"].sum() * 100.0

            plot_main_pie_with_other(
                df=longs,
                value_col="usdValue",
                title="Net LONG exposure (USD)",
                out_file=f"exposure_long_main_{ts}.png",
                info_box_text=info_box
            )

            long_small = longs[longs["pct"] <= THRESHOLD_PCT].copy()
            if not long_small.empty:
                long_small["pct"] = long_small["usdValue"] / long_small["usdValue"].sum() * 100.0
                pie_other_breakdown_with_legend_auto(
                    df=long_small,
                    value_col="usdValue",
                    title=f"Net LONG exposure (USD) — Other breakdown (≤{THRESHOLD_PCT:.0f}%)",
                    out_file=f"exposure_long_other_breakdown_{ts}.png",
                    info_box_text=info_box
                )

        if not shorts.empty:
            shorts["pct"] = shorts["absUsdValue"] / shorts["absUsdValue"].sum() * 100.0

            plot_main_pie_with_other(
                df=shorts,
                value_col="absUsdValue",
                title="Net SHORT exposure (USD, abs)",
                out_file=f"exposure_short_main_{ts}.png",
                info_box_text=info_box
            )

            short_small = shorts[shorts["pct"] <= THRESHOLD_PCT].copy()
            if not short_small.empty:
                short_small["pct"] = short_small["absUsdValue"] / short_small["absUsdValue"].sum() * 100.0
                pie_other_breakdown_with_legend_auto(
                    df=short_small,
                    value_col="absUsdValue",
                    title=f"Net SHORT exposure (USD, abs) — Other breakdown (≤{THRESHOLD_PCT:.0f}%)",
                    out_file=f"exposure_short_other_breakdown_{ts}.png",
                    info_box_text=info_box
                )

        print("\nSaved charts with timestamp:", ts)

    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
