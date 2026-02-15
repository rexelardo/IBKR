from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from ib_insync import IB, Forex

import yfinance as yf


# -------- CONFIG --------
HOST = "127.0.0.1"
PORT = 7496        # 7497 paper, 7496 live (commonly)
CLIENT_ID = 7

THRESHOLD_PCT = 1.0      # <= 1% goes into "Other"
USE_DELAYED_DATA = True  # if you lack live FX data permissions in IB
YF_LOOKBACK_DAYS = 7     # yfinance: fetch recent window and use last close
# ------------------------


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
    pair: str               # e.g. EURUSD, USDCAD, CADUSD=X
    px: float               # quote price
    direction: str          # "direct" or "invert"
    usd_per_ccy: float      # multiplier: native * usd_per_ccy => USD
    source: str             # "IB" or "YF"


def _ticker_price_best_effort(t) -> Optional[float]:
    px = safe_float(t.marketPrice())
    if px and px > 0:
        return px
    for attr in ("last", "close", "bid", "ask"):
        v = safe_float(getattr(t, attr, None))
        if v and v > 0:
            return v
    b = safe_float(getattr(t, "bid", None))
    a = safe_float(getattr(t, "ask", None))
    if b and a and b > 0 and a > 0:
        return (b + a) / 2
    return None


def try_fx_pair_ib(ib: IB, pair: str) -> Optional[float]:
    """Return price for an IB Forex pair, or None if not definable/price unavailable."""
    try:
        c = Forex(pair)
        ib.qualifyContracts(c)
        t = ib.reqTickers(c)[0]
        return _ticker_price_best_effort(t)
    except Exception:
        return None


def yfinance_usd_per_ccy(ccy: str) -> Optional[Tuple[str, float, str, float]]:
    """
    Returns (yf_symbol, px, direction, usd_per_ccy) or None.

    yfinance convention:
      - "EURUSD=X" gives USD per EUR (direct)
      - "USDJPY=X" gives JPY per USD (invert)
    Strategy:
      - Try CCYUSD=X (direct)
      - Else try USDCCY=X (invert)
    """
    ccy = (ccy or "").upper().strip()
    if ccy in {"", "USD"}:
        return ("USDUSD=X", 1.0, "direct", 1.0)

    # Try direct
    sym_direct = f"{ccy}USD=X"
    try:
        hist = yf.Ticker(sym_direct).history(period=f"{YF_LOOKBACK_DAYS}d", interval="1d")
        if not hist.empty:
            px = safe_float(hist["Close"].iloc[-1])
            if px and px > 0:
                return (sym_direct, px, "direct", px)
    except Exception:
        pass

    # Try invert
    sym_invert = f"USD{ccy}=X"
    try:
        hist = yf.Ticker(sym_invert).history(period=f"{YF_LOOKBACK_DAYS}d", interval="1d")
        if not hist.empty:
            px = safe_float(hist["Close"].iloc[-1])
            if px and px > 0:
                usd_per_ccy = 1.0 / px
                return (sym_invert, px, "invert", usd_per_ccy)
    except Exception:
        pass

    # Common special case: CNY vs CNH
    if ccy == "CNY":
        for sym in ("USDCNH=X", "CNHUSD=X"):
            try:
                hist = yf.Ticker(sym).history(period=f"{YF_LOOKBACK_DAYS}d", interval="1d")
                if not hist.empty:
                    px = safe_float(hist["Close"].iloc[-1])
                    if px and px > 0:
                        if sym.startswith("USD"):
                            return (sym, px, "invert", 1.0 / px)
                        else:
                            return (sym, px, "direct", px)
            except Exception:
                pass

    return None


def fetch_fx_snapshot_to_usd(
    ib: IB, currencies: List[str]
) -> Tuple[Dict[str, float], Dict[str, FxQuote], List[str]]:
    """
    Try IB first, fallback to yfinance.

    Returns:
      fx_map: ccy -> usd_per_ccy multiplier
      fx_quotes: ccy -> FxQuote (for annotation)
      missing: currencies without FX
    """
    fx_map: Dict[str, float] = {"USD": 1.0}
    fx_quotes: Dict[str, FxQuote] = {
        "USD": FxQuote(pair="USD", px=1.0, direction="direct", usd_per_ccy=1.0, source="BASE")
    }
    missing: List[str] = []

    cset = sorted({(c or "USD").upper().strip() for c in currencies})

    for ccy in cset:
        if ccy in {"", "USD"}:
            continue

        # --- IB attempt ---
        direct_pair = f"{ccy}USD"
        invert_pair = f"USD{ccy}"

        px_direct = try_fx_pair_ib(ib, direct_pair)
        if px_direct and px_direct > 0:
            fx_map[ccy] = px_direct
            fx_quotes[ccy] = FxQuote(pair=direct_pair, px=px_direct, direction="direct",
                                     usd_per_ccy=px_direct, source="IB")
            continue

        px_invert = try_fx_pair_ib(ib, invert_pair)
        if px_invert and px_invert > 0:
            usd_per_ccy = 1.0 / px_invert
            fx_map[ccy] = usd_per_ccy
            fx_quotes[ccy] = FxQuote(pair=invert_pair, px=px_invert, direction="invert",
                                     usd_per_ccy=usd_per_ccy, source="IB")
            continue

        # --- yfinance fallback ---
        yf_res = yfinance_usd_per_ccy(ccy)
        if yf_res is not None:
            yf_sym, px, direction, usd_per_ccy = yf_res
            fx_map[ccy] = usd_per_ccy
            fx_quotes[ccy] = FxQuote(pair=yf_sym, px=px, direction=direction,
                                     usd_per_ccy=usd_per_ccy, source="YF")
            continue

        missing.append(ccy)

    return fx_map, fx_quotes, missing


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


def add_fx_box(fig, fx_text: str):
    fig.text(
        0.01, 0.01,
        fx_text,
        ha="left", va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85)
    )


def main():
    ib = IB()
    ib.connect(HOST, PORT, clientId=CLIENT_ID)

    try:
        if USE_DELAYED_DATA:
            ib.reqMarketDataType(4)  # delayed-frozen

        items = ib.portfolio()
        if not items:
            print("No portfolio items returned. Check TWS login + API enabled + correct port.")
            return

        # ---- FILTER: STOCKS ONLY, LONG ONLY ----
        stk = []
        for p in items:
            c = p.contract
            if c.secType != "STK":
                continue
            pos = safe_float(p.position) or 0.0
            if pos <= 0:
                continue
            stk.append(p)

        if not stk:
            print("No LONG stock positions found (secType=STK, position>0).")
            return

        currencies = [(p.contract.currency or "USD").upper() for p in stk]
        fx_map, fx_quotes, missing_fx = fetch_fx_snapshot_to_usd(ib, currencies)

        if missing_fx:
            print("\nWARNING: Missing FX for:", ", ".join(missing_fx))
            print("Stock holdings in those currencies will be skipped.\n")

        rows = []
        for p in stk:
            c = p.contract
            ccy = (c.currency or "USD").upper()
            if ccy in missing_fx:
                continue

            symbol = c.symbol or c.localSymbol
            position = safe_float(p.position) or 0.0
            market_price = safe_float(p.marketPrice)
            mult = safe_float(getattr(c, "multiplier", 1) or 1) or 1.0

            if market_price is None:
                continue

            native_value = market_price * position * mult
            usd_value = native_value * fx_map.get(ccy, 1.0)

            if usd_value <= 0:
                continue

            rows.append({"symbol": symbol, "currency": ccy, "usdValue": usd_value})

        df = pd.DataFrame(rows)
        if df.empty:
            print("No USD-valued long stock holdings available (FX missing or prices missing).")
            return

        # ---- AGGREGATE: symbol + currency (USD valuation) ----
        agg = (
            df.groupby(["symbol", "currency"], as_index=False)["usdValue"]
              .sum()
              .sort_values("usdValue", ascending=False)
              .reset_index(drop=True)
        )

        total = agg["usdValue"].sum()
        agg["pct"] = agg["usdValue"] / total * 100.0

        # Print table
        printable = agg.copy()
        printable["usdValue"] = printable["usdValue"].map(fmt_money)
        printable["pct"] = printable["pct"].map(lambda x: f"{x:.2f}%")
        print("\nLONG STOCK portfolio allocation (values converted to USD):\n")
        print(printable.rename(columns={"usdValue": "valueUSD"}).to_string(index=False))
        print(f"\nTOTAL (USD): {fmt_money(total)}\n")

        # Split
        big = agg[agg["pct"] > THRESHOLD_PCT].copy()
        small = agg[agg["pct"] <= THRESHOLD_PCT].copy()

        # Timestamped output
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fx_text = fx_snapshot_text(fx_quotes, max_lines=16)

        # -------- MAIN PIE --------
        main_pie = big.copy()
        if not small.empty:
            main_pie = pd.concat(
                [main_pie, pd.DataFrame([{
                    "symbol": f"Other (≤{THRESHOLD_PCT:.0f}%)",
                    "currency": "",
                    "usdValue": small["usdValue"].sum(),
                    "pct": small["pct"].sum(),
                }])],
                ignore_index=True
            )

        main_labels = []
        for r in main_pie.itertuples(index=False):
            if str(r.symbol).startswith("Other"):
                main_labels.append(f"{r.symbol} ({r.pct:.1f}%)")
            else:
                main_labels.append(f"{r.symbol} ({r.currency}) ({r.pct:.1f}%)")

        fig = plt.figure(figsize=(9, 9))
        plt.pie(
            main_pie["usdValue"],
            labels=main_labels,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 2 else ""
        )
        plt.title("IBKR LONG STOCK Allocation (USD) (>1% + Other)")
        plt.tight_layout()
        add_fx_box(fig, fx_text)

        main_filename = f"portfolio_main_usd_{ts}.png"
        plt.savefig(main_filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved main chart → {os.path.abspath(main_filename)}")

        # -------- OTHER BREAKDOWN PIE --------
        if not small.empty:
            other_total = small["usdValue"].sum()
            small = small.sort_values("usdValue", ascending=False).reset_index(drop=True)

            other_labels = [
                f"{r.symbol} ({r.currency}) ({(r.usdValue/other_total*100):.1f}% of Other)"
                for r in small.itertuples(index=False)
            ]

            fig2 = plt.figure(figsize=(9, 9))
            plt.pie(
                small["usdValue"],
                labels=other_labels,
                autopct=lambda pct: f"{pct:.1f}%" if pct >= 2 else ""
            )
            plt.title(f"Breakdown of Other (USD) (Holdings ≤{THRESHOLD_PCT:.0f}%)")
            plt.tight_layout()
            add_fx_box(fig2, fx_text)

            other_filename = f"portfolio_other_breakdown_usd_{ts}.png"
            plt.savefig(other_filename, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved Other breakdown chart → {os.path.abspath(other_filename)}")
        else:
            print(f"No holdings ≤ {THRESHOLD_PCT:.0f}% — no Other breakdown chart created.")

    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
