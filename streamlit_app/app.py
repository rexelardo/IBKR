import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from portfolio_analyser_refactor import analyze_portfolio

# SnapTrade SDK (maintained python sdk is referenced in their docs) :contentReference[oaicite:4]{index=4}
from snaptrade_client import SnapTrade

st.set_page_config(page_title="SnapTrade Portfolio Analyzer", layout="wide")

EARNINGS_CSV = Path("sec_quarterly_net_income.csv")
MARKETCAP_CSV = Path("capital_flows/tickers_2026-02-23.csv")

def normalize_weights(df: pd.DataFrame, ticker_col="ticker", value_col="value"):
    df = df.copy()
    df[ticker_col] = df[ticker_col].astype(str).str.upper().str.strip()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    total = df[value_col].sum()
    if total <= 0:
        raise ValueError("Total portfolio value is 0; cannot compute weights.")
    df["weight"] = df[value_col] / total
    return df[[ticker_col, "weight"]]

def init_snaptrade() -> SnapTrade:
    client_id = st.secrets.get("SNAPTRADE_CLIENT_ID", "")
    consumer_key = st.secrets.get("SNAPTRADE_CONSUMER_KEY", "")
    if not client_id or not consumer_key:
        raise RuntimeError("Missing SNAPTRADE_CLIENT_ID / SNAPTRADE_CONSUMER_KEY in .streamlit/secrets.toml")

    # Basic init follows SnapTrade “getting started / requests” docs :contentReference[oaicite:5]{index=5}
    return SnapTrade(
        consumer_key=consumer_key,
        client_id=client_id,
    )

st.title("SnapTrade → Portfolio Analyzer")

with st.sidebar:
    st.header("1) Get portfolio data")
    mode = st.radio("Choose input", ["SnapTrade Connect", "Upload CSV"], index=0)

    st.caption("CSV format: columns `ticker,value` (value in account currency).")

# -----------------------------
# Mode A: Upload CSV
# -----------------------------
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

        if st.button("Analyze"):
            out = analyze_portfolio(
                portfolio=portfolio,
                earnings_csv=EARNINGS_CSV if EARNINGS_CSV.exists() else None,
                marketcap_csv=MARKETCAP_CSV if MARKETCAP_CSV.exists() else None,
            )
            st.session_state["analysis_out"] = out

# -----------------------------
# Mode B: SnapTrade Connect
# -----------------------------
if mode == "SnapTrade Connect":
    st.subheader("Connect brokerage")

    # Create a per-session user in SnapTrade (MVP approach).
    # In production, you should map this to your real app user IDs.
    if "snaptrade_user_id" not in st.session_state:
        st.session_state.snaptrade_user_id = f"st-{uuid.uuid4()}"
        st.session_state.snaptrade_user_secret = None

    user_id = st.session_state.snaptrade_user_id

    try:
        snaptrade = init_snaptrade()
    except Exception as e:
        st.error(str(e))
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Create/Reset SnapTrade User"):
            resp = snaptrade.authentication.register_snap_trade_user(
                body={"userId": user_id}
            )
            st.session_state.snaptrade_user_secret = resp.body["userSecret"]
            st.success("User created (or reset).")

    with col2:
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
                st.info("Open the portal URL, connect your brokerage, then come back and click “Fetch Holdings”.")

    if "portal_url" in st.session_state:
        st.link_button("Open SnapTrade Connection Portal", st.session_state.portal_url)
        
    st.divider()
    st.subheader("Fetch holdings")

    if st.button("Fetch Holdings from SnapTrade"):
        if not st.session_state.snaptrade_user_secret:
            st.warning("Create/Reset SnapTrade User first.")
            st.stop()

        # Best practice per docs: list accounts then use account-specific holdings endpoint :contentReference[oaicite:8]{index=8}
        accounts = snaptrade.account_information.list_user_accounts(
            user_id=user_id,
            user_secret=st.session_state.snaptrade_user_secret,
        ).body

        if not accounts:
            st.warning("No accounts found yet. If you just connected, try again in a moment.")
            st.stop()

        # Pull holdings per account and aggregate by ticker
        rows = []
        for acct in accounts:
            account_id = acct["id"]
            h = snaptrade.account_information.get_user_holdings(
                user_id=user_id,
                user_secret=st.session_state.snaptrade_user_secret,
                account_id=account_id,
            ).body

            # This shape can vary by instrument type; we handle common equity ticker path.
            positions = h.get("positions", []) or []
            for p in positions:
                sym = p.get("symbol", {}) or {}
                ticker = sym.get("symbol", None) or sym.get("ticker", None)
                if not ticker:
                    continue

                # Prefer a market value field if present, else fall back to price * units.
                mv = p.get("marketValue", None)
                if mv is None:
                    units = p.get("units", 0) or 0
                    price = p.get("price", 0) or 0
                    mv = float(units) * float(price)

                rows.append({"ticker": str(ticker).upper(), "value": float(mv)})

        if not rows:
            st.error("Holdings returned, but no ticker positions could be parsed.")
            st.stop()

        holdings = pd.DataFrame(rows).groupby("ticker", as_index=False)["value"].sum()
        weights_df = normalize_weights(holdings, "ticker", "value")
        portfolio = weights_df.to_dict("records")

        st.success(f"Fetched and parsed {len(portfolio)} tickers.")
        st.dataframe(weights_df, use_container_width=True)

        out = analyze_portfolio(
            portfolio=portfolio,
            earnings_csv=EARNINGS_CSV if EARNINGS_CSV.exists() else None,
            marketcap_csv=MARKETCAP_CSV if MARKETCAP_CSV.exists() else None,
        )
        st.session_state["analysis_out"] = out

# -----------------------------
# Render analysis
# -----------------------------
if "analysis_out" in st.session_state:
    out = st.session_state["analysis_out"]

    st.header("Portfolio Summary")

    a, b, c = st.columns(3)
    a.metric("Weighted Dividend Yield", f"{out['portfolio_dividend_yield']*100:.2f}%")

    mix = out["bucket_mix"]
    b.metric("Defensive", f"{mix['Defensive']*100:.1f}%")
    c.metric("Degen", f"{mix['Degen']*100:.1f}%")

    st.subheader("Bucket mix")
    st.dataframe(
        pd.DataFrame([mix]).T.rename(columns={0: "weight"}).assign(pct=lambda x: x["weight"]*100),
        use_container_width=True,
    )

    st.subheader("Leaders")
    leaders = out["leaders"]
    st.dataframe(
        pd.DataFrame([
            {"bucket": k, "ticker": v["ticker"], "portfolio_pct_in_bucket": v["portfolio_pct_in_bucket"]*100}
            for k, v in leaders.items()
        ]),
        use_container_width=True,
    )

    st.subheader("Per-stock table")
    st.dataframe(out["per_stock"], use_container_width=True)

    st.subheader("Industry summary")
    ind = out["industry_summary"].copy()
    ind_fmt = ind.copy()
    ind_fmt["industry_weight"] *= 100
    ind_fmt["industry_dividend_yield"] *= 100
    for col in ["Defensive", "Quality", "Speculative", "Degen"]:
        if col in ind_fmt.columns:
            ind_fmt[col] *= 100
    st.dataframe(ind_fmt, use_container_width=True)