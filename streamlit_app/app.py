# app.py
import uuid
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from snaptrade_client import SnapTrade
from portfolio_analyser_refactor import analyze_portfolio


# =========================
# Config
# =========================
st.set_page_config(page_title="SnapTrade Portfolio Analyzer", layout="wide")

EARNINGS_CSV = Path("sec_quarterly_net_income.csv")
MARKETCAP_CSV = Path("capital_flows/tickers_2026-02-23.csv")


# =========================
# Helpers
# =========================
def init_snaptrade() -> SnapTrade:
    client_id = st.secrets.get("SNAPTRADE_CLIENT_ID", "")
    consumer_key = st.secrets.get("SNAPTRADE_CONSUMER_KEY", "")
    if not client_id or not consumer_key:
        raise RuntimeError("Missing SNAPTRADE_CLIENT_ID / SNAPTRADE_CONSUMER_KEY in .streamlit/secrets.toml")
    return SnapTrade(consumer_key=consumer_key, client_id=client_id)


def normalize_weights(df: pd.DataFrame, ticker_col="ticker", value_col="value") -> pd.DataFrame:
    df = df.copy()
    df[ticker_col] = df[ticker_col].astype(str).str.upper().str.strip()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    total = float(df[value_col].sum())
    if total <= 0:
        raise ValueError("Total portfolio value is 0; cannot compute weights.")
    df["weight"] = df[value_col] / total
    return df[[ticker_col, "weight"]]


def extract_ticker_from_position(p: Any) -> Optional[str]:
    """
    Based on your Robinhood payload:
      p['symbol']['symbol'] -> 'KHC'
    We keep small fallbacks for other broker shapes.
    """
    if not isinstance(p, dict):
        # SnapTrade SDK sometimes returns model objects; try to convert
        if hasattr(p, "to_dict") and callable(getattr(p, "to_dict")):
            try:
                p = p.to_dict()
            except Exception:
                return None
        elif hasattr(p, "__dict__"):
            try:
                p = dict(p.__dict__)
            except Exception:
                return None
        else:
            return None

    sym = p.get("symbol") or {}
    if not isinstance(sym, dict):
        if hasattr(sym, "to_dict") and callable(getattr(sym, "to_dict")):
            try:
                sym = sym.to_dict()
            except Exception:
                sym = {}
        elif hasattr(sym, "__dict__"):
            try:
                sym = dict(sym.__dict__)
            except Exception:
                sym = {}
        else:
            sym = {}

    # ✅ Primary path you confirmed
    ticker = sym.get("symbol")

    # Fallbacks (other broker variations)
    if not ticker:
        ticker = sym.get("SYMBOL") or sym.get("rawSymbol") or sym.get("ticker") or sym.get("TICKER")

    if not ticker:
        return None

    ticker = str(ticker).upper().strip()

    # yfinance-sane filter (keeps AAPL, BRK.B, RDS-A, etc.)
    if not re.match(r"^[A-Z0-9.\-]+$", ticker):
        return None

    return ticker


def extract_market_value(p: Any) -> float:
    if not isinstance(p, dict):
        if hasattr(p, "to_dict") and callable(getattr(p, "to_dict")):
            try:
                p = p.to_dict()
            except Exception:
                return 0.0
        elif hasattr(p, "__dict__"):
            try:
                p = dict(p.__dict__)
            except Exception:
                return 0.0
        else:
            return 0.0

    mv = p.get("marketValue")
    if mv is not None:
        try:
            return float(mv)
        except Exception:
            pass

    # fallback: units * price
    units = p.get("units", 0) or 0
    price = p.get("price", 0) or 0
    try:
        return float(units) * float(price)
    except Exception:
        return 0.0


# =========================
# UI
# =========================
st.title("SnapTrade → Portfolio Analyzer")

with st.sidebar:
    st.header("Input mode")
    mode = st.radio("Choose input", ["SnapTrade Connect", "Upload CSV"], index=0)

    st.divider()
    st.header("Analysis settings")
    bench = st.text_input("Benchmark ticker", value="SPY")
    years = st.number_input("History (years)", min_value=1, max_value=10, value=3)

    st.caption("CSV format: columns `ticker,value` (value in account currency).")


# =========================
# Mode A: Upload CSV
# =========================
if mode == "Upload CSV":
    up = st.file_uploader("Upload holdings CSV", type=["csv"])
    if up is not None:
        raw = pd.read_csv(up)
        if not {"ticker", "value"}.issubset(raw.columns):
            st.error("CSV must contain columns: ticker, value")
            st.stop()

        weights_df = normalize_weights(raw, "ticker", "value")
        portfolio = weights_df.to_dict("records")

        st.success(f"Loaded {len(portfolio)} tickers from CSV.")
        st.dataframe(weights_df, use_container_width=True)

        if st.button("Analyze CSV"):
            out = analyze_portfolio(
                portfolio=portfolio,
                earnings_csv=EARNINGS_CSV if EARNINGS_CSV.exists() else None,
                marketcap_csv=MARKETCAP_CSV if MARKETCAP_CSV.exists() else None,
                bench_ticker=bench.strip().upper(),
                history_years=int(years),
            )
            st.session_state["analysis_out"] = out
            st.session_state["analysis_label"] = "CSV Upload"


# =========================
# Mode B: SnapTrade Connect (Accounts-first UX)
# =========================
if mode == "SnapTrade Connect":
    st.subheader("Connect brokerage")

    # Per-session user (MVP). Map to real app users later.
    if "snaptrade_user_id" not in st.session_state:
        st.session_state.snaptrade_user_id = f"st-{uuid.uuid4()}"
        st.session_state.snaptrade_user_secret = None

    user_id = st.session_state.snaptrade_user_id

    try:
        snaptrade = init_snaptrade()
    except Exception as e:
        st.error(str(e))
        st.stop()

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Create/Reset SnapTrade User"):
            resp = snaptrade.authentication.register_snap_trade_user(body={"userId": user_id})
            st.session_state.snaptrade_user_secret = resp.body["userSecret"]
            st.success("User created (or reset).")

    with c2:
        if st.button("Generate Connection Portal URL"):
            if not st.session_state.snaptrade_user_secret:
                st.warning("Click Create/Reset SnapTrade User first.")
            else:
                login = snaptrade.authentication.login_snap_trade_user(
                    query_params={
                        "userId": user_id,
                        "userSecret": st.session_state.snaptrade_user_secret,
                    }
                )
                st.session_state.portal_url = login.body["redirectURI"]
                st.info("Open the portal URL, connect your brokerage, then come back and click “Load Accounts”.")

    if "portal_url" in st.session_state:
        st.link_button("Open SnapTrade Connection Portal", st.session_state.portal_url)

    st.divider()

    # ---- Step 1: Load accounts (no holdings yet) ----
    st.subheader("Step 1 — Load accounts")

    if st.button("Load Accounts"):
        if not st.session_state.snaptrade_user_secret:
            st.warning("Create/Reset SnapTrade User first.")
            st.stop()

        accounts = snaptrade.account_information.list_user_accounts(
            query_params={
                "userId": user_id,
                "userSecret": st.session_state.snaptrade_user_secret,
            }
        ).body

        if not accounts:
            st.warning("No accounts found yet. If you just connected, try again in a moment.")
            st.stop()

        # Store minimal fields + keep raw for future debugging
        cleaned = []
        for a in accounts:
            cleaned.append(
                {
                    "id": a["id"],
                    "name": a.get("name") or a.get("number") or a["id"],
                    "institution": a.get("institution_name") or a.get("institution") or "",
                    "currency": a.get("currency") or "",
                    "raw": a,
                }
            )
        st.session_state["accounts"] = cleaned
        st.session_state.pop("selected_holdings_df", None)
        st.session_state.pop("selected_skipped_df", None)

    accounts = st.session_state.get("accounts", [])
    if accounts:
        st.success(f"Found {len(accounts)} accounts.")

        labels = []
        id_by_label = {}
        for a in accounts:
            inst = f"{a['institution']}" if a["institution"] else "Broker"
            label = f"{a['name']} ({inst}) — {a['id']}"
            labels.append(label)
            id_by_label[label] = a["id"]

        pick = st.selectbox("Choose an account to deep dive", labels, index=0)
        st.session_state["picked_account_id"] = id_by_label[pick]
        st.session_state["picked_account_label"] = pick

        st.divider()

        # ---- Step 2: Fetch holdings for selected account ----
        st.subheader("Step 2 — Fetch holdings for selected account")

        if st.button("Fetch Holdings (Selected Account)"):
            if not st.session_state.snaptrade_user_secret:
                st.warning("Create/Reset SnapTrade User first.")
                st.stop()

            account_id = st.session_state["picked_account_id"]

            h = snaptrade.account_information.get_user_holdings(
                path_params={"accountId": account_id},
                query_params={
                    "userId": user_id,
                    "userSecret": st.session_state.snaptrade_user_secret,
                },
            ).body

            positions = h.get("positions", []) or []
            print(f"DEBUG: Raw positions for account {account_id}: {positions}")
            rows: List[Dict[str, Any]] = []
            print(f"DEBUG: rows = {rows}")
            skipped: List[Dict[str, Any]] = []
            print(f"DEBUG: skipped = {skipped}")
            for p in positions:
                ticker = extract_ticker_from_position(p)
                if not ticker:
                    sym = p.get("symbol") if isinstance(p, dict) else getattr(p, "symbol", None)
                    skipped.append({"reason": "unparsed_ticker", "raw_symbol": str(sym)[:250]})
                    continue

                mv = extract_market_value(p)
                if mv <= 0:
                    skipped.append({"reason": "nonpositive_value", "raw_symbol": ticker})
                    continue

                rows.append({"ticker": ticker, "value": float(mv)})

            if not rows:
                st.error("Holdings returned, but no valid equity/ETF positions could be parsed.")
                if skipped:
                    with st.expander("Skipped positions (debug)"):
                        st.dataframe(pd.DataFrame(skipped), use_container_width=True)
                st.stop()

            holdings_df = pd.DataFrame(rows).groupby("ticker", as_index=False)["value"].sum()
            st.session_state["selected_holdings_df"] = holdings_df
            st.session_state["selected_skipped_df"] = pd.DataFrame(skipped)

            st.success(f"Parsed {len(holdings_df)} tickers for this account.")
            st.dataframe(holdings_df, use_container_width=True)

            if not st.session_state["selected_skipped_df"].empty:
                with st.expander("Skipped positions (debug)"):
                    st.dataframe(st.session_state["selected_skipped_df"], use_container_width=True)

        # ---- Step 3: Analyze selected account ----
        if "selected_holdings_df" in st.session_state:
            st.subheader("Step 3 — Analyze this account")

            holdings_df = st.session_state["selected_holdings_df"]
            weights_df = normalize_weights(holdings_df, "ticker", "value")
            portfolio = weights_df.to_dict("records")

            st.caption("Weights used for analysis")
            st.dataframe(weights_df, use_container_width=True)

            if st.button("Run analysis on this account"):
                out = analyze_portfolio(
                    portfolio=portfolio,
                    earnings_csv=EARNINGS_CSV if EARNINGS_CSV.exists() else None,
                    marketcap_csv=MARKETCAP_CSV if MARKETCAP_CSV.exists() else None,
                    bench_ticker=bench.strip().upper(),
                    history_years=int(years),
                )
                st.session_state["analysis_out"] = out
                st.session_state["analysis_label"] = st.session_state.get("picked_account_label", "Selected Account")


# =========================
# Render analysis output
# =========================
if "analysis_out" in st.session_state:
    out = st.session_state["analysis_out"]
    label = st.session_state.get("analysis_label", "")

    st.header(f"Portfolio Summary — {label}")

    colA, colB, colC = st.columns(3)
    colA.metric("Weighted Dividend Yield", f"{out['portfolio_dividend_yield']*100:.2f}%")

    mix = out["bucket_mix"]
    colB.metric("Defensive", f"{mix.get('Defensive', 0.0)*100:.1f}%")
    colC.metric("Degen", f"{mix.get('Degen', 0.0)*100:.1f}%")

    st.subheader("Bucket mix")
    mix_df = pd.DataFrame([mix]).T.rename(columns={0: "weight"})
    mix_df["pct"] = mix_df["weight"] * 100
    st.dataframe(mix_df, use_container_width=True)

    st.subheader("Leaders (biggest contributors = trait * portfolio weight)")
    leaders = out.get("leaders", {})
    if leaders:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "bucket": k,
                        "ticker": v["ticker"],
                        "portfolio_pct_in_bucket": v["portfolio_pct_in_bucket"] * 100,
                    }
                    for k, v in leaders.items()
                ]
            ),
            use_container_width=True,
        )

    st.subheader("Per-stock table")
    st.dataframe(out["per_stock"], use_container_width=True)

    st.subheader("Industry summary")
    ind = out["industry_summary"].copy()
    ind_fmt = ind.copy()

    if "industry_weight" in ind_fmt.columns:
        ind_fmt["industry_weight"] *= 100
    if "industry_dividend_yield" in ind_fmt.columns:
        ind_fmt["industry_dividend_yield"] *= 100
    for col in ["Defensive", "Quality", "Speculative", "Degen"]:
        if col in ind_fmt.columns:
            ind_fmt[col] *= 100

    st.dataframe(ind_fmt, use_container_width=True)