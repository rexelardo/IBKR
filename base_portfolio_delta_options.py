from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
import math
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm

from ib_insync import IB


# ---------------- CONFIG ----------------
HOST = "127.0.0.1"
PORT = 7496
CLIENT_ID = 21

THRESHOLD_PCT = 1.0
TRADING_DAYS = 252

# Delta model params
RISK_FREE_RATE = 0.045          # constant; good enough for this use
HV_WINDOW_DAYS = 60             # fallback sigma window if IV missing
YF_PRICE_LOOKBACK = "6mo"       # for HV calc
YF_FX_LOOKBACK_DAYS = 7

DIVIDEND_YIELD_FALLBACK = 0.0   # if yfinance dividendYield missing

# Yahoo ticker mapping (IB symbol -> Yahoo symbol)
# Fill if you have non-US underlyings that don't resolve by symbol alone.
YF_SYMBOL_MAP: Dict[str, str] = {
    # Examples:
    # "SHOP": "SHOP.TO",
    # "AKRBP": "AKRBP.OL",
}
# ---------------------------------------


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


def yfinance_usd_per_ccy(ccy: str) -> Optional[FxQuote]:
    ccy = (ccy or "").upper().strip()
    if ccy in {"", "USD"}:
        return FxQuote(pair="USD", px=1.0, direction="direct", usd_per_ccy=1.0, source="BASE")

    # direct: CCYUSD=X
    sym_direct = f"{ccy}USD=X"
    try:
        hist = yf.Ticker(sym_direct).history(period=f"{YF_FX_LOOKBACK_DAYS}d", interval="1d")
        if not hist.empty:
            px = safe_float(hist["Close"].iloc[-1])
            if px and px > 0:
                return FxQuote(pair=sym_direct, px=px, direction="direct", usd_per_ccy=px, source="YF")
    except Exception:
        pass

    # invert: USDCCY=X
    sym_invert = f"USD{ccy}=X"
    try:
        hist = yf.Ticker(sym_invert).history(period=f"{YF_FX_LOOKBACK_DAYS}d", interval="1d")
        if not hist.empty:
            px = safe_float(hist["Close"].iloc[-1])
            if px and px > 0:
                return FxQuote(pair=sym_invert, px=px, direction="invert", usd_per_ccy=1.0 / px, source="YF")
    except Exception:
        pass

    # CNY fallback
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


def add_info_box(fig, text: str):
    fig.text(
        0.01, 0.01,
        text,
        ha="left", va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85)
    )


def annualized_hist_vol(closes: pd.Series, window_days: int) -> Optional[float]:
    closes = closes.dropna()
    if len(closes) < window_days + 2:
        return None
    logret = (closes / closes.shift(1)).apply(lambda x: math.log(x) if x and x > 0 else float("nan")).dropna()
    if len(logret) < window_days:
        return None
    sigma_daily = float(logret.tail(window_days).std(ddof=1))
    return sigma_daily * math.sqrt(TRADING_DAYS)


def yfinance_underlying(symbol: str) -> Tuple[Optional[float], Optional[float], float]:
    """
    Returns (S_last_close, sigma_hv_annual, q_dividend_yield)
    """
    ysym = YF_SYMBOL_MAP.get(symbol, symbol)
    t = yf.Ticker(ysym)

    hist = t.history(period=YF_PRICE_LOOKBACK, interval="1d")
    if hist.empty:
        return None, None, DIVIDEND_YIELD_FALLBACK

    closes = hist["Close"].dropna()
    if closes.empty:
        return None, None, DIVIDEND_YIELD_FALLBACK

    S = safe_float(closes.iloc[-1])
    sigma = annualized_hist_vol(closes, HV_WINDOW_DAYS)

    # dividend yield (flaky)
    q = DIVIDEND_YIELD_FALLBACK
    try:
        info = t.info or {}
        q_info = safe_float(info.get("dividendYield"))
        if q_info is not None:
            q = q_info
    except Exception:
        pass

    return S, sigma, q


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
    """
    Find closest strike match (tolerant to float repr).
    """
    if chain_df is None or chain_df.empty:
        return None
    # exact match first
    m = chain_df[chain_df["strike"] == strike]
    if not m.empty:
        return m.iloc[0]
    # tolerant match
    m = chain_df[(chain_df["strike"] - strike).abs() < 1e-6]
    if not m.empty:
        return m.iloc[0]
    return None


def get_option_iv_from_yfinance(underlying: str, expiry: date, right: str, strike: float) -> Optional[float]:
    """
    Returns IV as a decimal (e.g. 0.25) or None.
    """
    ysym = YF_SYMBOL_MAP.get(underlying, underlying)
    t = yf.Ticker(ysym)

    exp_str = expiry.strftime("%Y-%m-%d")
    try:
        chain = t.option_chain(exp_str)
    except Exception:
        return None

    df = chain.calls if right.upper() == "C" else chain.puts
    row = get_chain_row(df, strike)
    if row is None:
        return None

    iv = safe_float(row.get("impliedVolatility"))
    if iv is None:
        return None

    # yfinance IV is usually already a decimal (e.g. 0.35)
    if iv > 5:  # sanity: if someone returns percent-like 35
        iv = iv / 100.0

    return iv


def plot_pie_with_other_breakdown(df: pd.DataFrame, value_col: str, title: str, prefix: str, box: str, ts: str):
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

    fig = plt.figure(figsize=(9, 9))
    plt.pie(main[value_col], labels=labels, autopct=lambda p: f"{p:.1f}%" if p >= 2 else "")
    plt.title(title)
    plt.tight_layout()
    add_info_box(fig, box)

    fn = f"{prefix}_main_{ts}.png"
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {os.path.abspath(fn)}")

    if not small.empty:
        other_total = small[value_col].sum()
        small = small.sort_values(value_col, ascending=False).reset_index(drop=True)

        other_labels = [
            f"{r.underlying} ({r.currency}) ({(getattr(r, value_col)/other_total*100):.1f}% of Other)"
            for r in small.itertuples(index=False)
        ]

        fig2 = plt.figure(figsize=(9, 9))
        plt.pie(small[value_col], labels=other_labels, autopct=lambda p: f"{p:.1f}%" if p >= 2 else "")
        plt.title(f"{title} — Other breakdown")
        plt.tight_layout()
        add_info_box(fig2, box)

        fn2 = f"{prefix}_other_breakdown_{ts}.png"
        plt.savefig(fn2, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved → {os.path.abspath(fn2)}")


def main():
    ib = IB()
    ib.connect(HOST, PORT, clientId=CLIENT_ID)

    try:
        portfolio = ib.portfolio()
        if not portfolio:
            print("No portfolio items returned.")
            return

        stk = [p for p in portfolio if p.contract.secType == "STK" and (safe_float(p.position) or 0) != 0]
        opt = [p for p in portfolio if p.contract.secType == "OPT" and (safe_float(p.position) or 0) != 0]

        # Base net shares from stocks
        net_shares: Dict[Tuple[str, str], float] = {}
        under_ccy: Dict[str, str] = {}

        for p in stk:
            sym = p.contract.symbol
            ccy = (p.contract.currency or "USD").upper()
            under_ccy[sym] = ccy
            net_shares[(sym, ccy)] = net_shares.get((sym, ccy), 0.0) + (safe_float(p.position) or 0.0)

        # Collect currencies for FX
        currencies = set((p.contract.currency or "USD").upper() for p in stk)
        currencies |= set((p.contract.currency or "USD").upper() for p in opt)

        fx_quotes: Dict[str, FxQuote] = {"USD": FxQuote(pair="USD", px=1.0, direction="direct", usd_per_ccy=1.0, source="BASE")}
        fx_map: Dict[str, float] = {"USD": 1.0}
        for ccy in sorted(currencies):
            if ccy == "USD":
                continue
            q = yfinance_usd_per_ccy(ccy)
            if q is None:
                print(f"WARNING: Missing FX for {ccy}; exposures in this currency will be skipped.")
                continue
            fx_quotes[ccy] = q
            fx_map[ccy] = q.usd_per_ccy

        # Cache underlying inputs + option chains per (underlying, expiry)
        underlying_cache: Dict[str, Tuple[Optional[float], Optional[float], float]] = {}
        chain_cache: Dict[Tuple[str, str], Tuple[pd.DataFrame, pd.DataFrame]] = {}

        def get_underlying_inputs(sym: str):
            if sym not in underlying_cache:
                underlying_cache[sym] = yfinance_underlying(sym)
            return underlying_cache[sym]

        def get_chain(underlying: str, expiry: date):
            key = (underlying, expiry.strftime("%Y-%m-%d"))
            if key in chain_cache:
                return chain_cache[key]
            ysym = YF_SYMBOL_MAP.get(underlying, underlying)
            t = yf.Ticker(ysym)
            try:
                oc = t.option_chain(key[1])
                chain_cache[key] = (oc.calls, oc.puts)
            except Exception:
                chain_cache[key] = (pd.DataFrame(), pd.DataFrame())
            return chain_cache[key]

        today = date.today()
        used_iv = 0
        used_hv_fallback = 0
        skipped_opts = 0

        # Add option delta-equivalent shares
        for p in opt:
            c = p.contract
            sym = c.symbol
            ccy = (c.currency or "USD").upper()
            under_ccy.setdefault(sym, ccy)

            if ccy not in fx_map:
                skipped_opts += 1
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

            S, sigma_hv, q_div = get_underlying_inputs(sym)
            if S is None:
                skipped_opts += 1
                continue

            # Prefer IV from yfinance option chain
            iv = None
            calls_df, puts_df = get_chain(sym, expiry)
            chain_df = calls_df if right == "C" else puts_df
            row = get_chain_row(chain_df, K)
            if row is not None:
                iv = safe_float(row.get("impliedVolatility"))

            sigma = None
            if iv is not None and iv > 0:
                if iv > 5:
                    iv = iv / 100.0
                sigma = iv
                used_iv += 1
            elif sigma_hv is not None and sigma_hv > 0:
                sigma = sigma_hv
                used_hv_fallback += 1
            else:
                skipped_opts += 1
                continue

            d = bs_delta(S=S, K=K, T=T, r=RISK_FREE_RATE, q=q_div, sigma=sigma, right=right)
            shares_equiv = pos_contracts * d * mult

            net_shares[(sym, ccy)] = net_shares.get((sym, ccy), 0.0) + shares_equiv

        # Build exposures in USD
        rows = []
        missing_underlying_price = 0
        missing_fx = 0

        for (sym, ccy), shares in net_shares.items():
            if ccy not in fx_map:
                missing_fx += 1
                continue

            S, _, _ = get_underlying_inputs(sym)
            if S is None:
                missing_underlying_price += 1
                continue

            usd_value = shares * S * fx_map[ccy]
            if abs(usd_value) < 1e-6:
                continue

            rows.append({
                "underlying": sym,
                "currency": ccy,
                "netShares": shares,
                "underlyingPrice": S,
                "usdValue": usd_value
            })

        if not rows:
            print("No exposures computed. Likely yfinance symbol mapping needed for your non-US underlyings.")
            return

        df = pd.DataFrame(rows).groupby(["underlying", "currency"], as_index=False).agg(
            netShares=("netShares", "sum"),
            underlyingPrice=("underlyingPrice", "first"),
            usdValue=("usdValue", "sum")
        )

        longs = df[df["usdValue"] > 0].copy().sort_values("usdValue", ascending=False)
        shorts = df[df["usdValue"] < 0].copy()
        shorts["absUsdValue"] = shorts["usdValue"].abs()
        shorts = shorts.sort_values("absUsdValue", ascending=False)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        fx_box = fx_snapshot_text(fx_quotes, max_lines=16)
        model_box = (
            "Delta model: Black–Scholes\n"
            f"Sigma: yfinance IV per contract (fallback HV {HV_WINDOW_DAYS}d)\n"
            f"r={RISK_FREE_RATE:.3%}, q=yfinance dividendYield (fallback {DIVIDEND_YIELD_FALLBACK:.1%})\n"
            f"Used IV={used_iv}, HV fallback={used_hv_fallback}, skipped opts={skipped_opts}\n"
            f"Missing underlying price={missing_underlying_price}, missing FX={missing_fx}"
        )
        info_box = fx_box + "\n\n" + model_box

        # Print summary tables
        print("\n=== Net delta-equivalent exposure (USD) — LONGS ===\n")
        if not longs.empty:
            out = longs.copy()
            out["usdValue"] = out["usdValue"].map(fmt_money)
            out["netShares"] = out["netShares"].map(lambda x: f"{x:,.2f}")
            out["underlyingPrice"] = out["underlyingPrice"].map(lambda x: f"{x:,.4f}")
            print(out[["underlying", "currency", "netShares", "underlyingPrice", "usdValue"]].to_string(index=False))
            print(f"\nTOTAL LONG USD: {fmt_money(longs['usdValue'].sum())}\n")
        else:
            print("No net long exposures.\n")

        print("\n=== Net delta-equivalent exposure (USD) — SHORTS ===\n")
        if not shorts.empty:
            out = shorts.copy()
            out["usdValue"] = out["usdValue"].map(fmt_money)
            out["absUsdValue"] = out["absUsdValue"].map(fmt_money)
            out["netShares"] = out["netShares"].map(lambda x: f"{x:,.2f}")
            out["underlyingPrice"] = out["underlyingPrice"].map(lambda x: f"{x:,.4f}")
            print(out[["underlying", "currency", "netShares", "underlyingPrice", "usdValue", "absUsdValue"]].to_string(index=False))
            print(f"\nTOTAL SHORT (abs) USD: {fmt_money(shorts['absUsdValue'].sum())}\n")
        else:
            print("No net short exposures.\n")

        # Charts
        if not longs.empty:
            longs["pct"] = longs["usdValue"] / longs["usdValue"].sum() * 100.0
            plot_pie_with_other_breakdown(
                df=longs,
                value_col="usdValue",
                title="Net LONG delta-equivalent exposure (USD)",
                prefix="exposure_long",
                box=info_box,
                ts=ts
            )

        if not shorts.empty:
            shorts["pct"] = shorts["absUsdValue"] / shorts["absUsdValue"].sum() * 100.0
            plot_pie_with_other_breakdown(
                df=shorts,
                value_col="absUsdValue",
                title="Net SHORT delta-equivalent exposure (USD, abs)",
                prefix="exposure_short",
                box=info_box,
                ts=ts
            )

    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
