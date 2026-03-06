import uuid
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.express as px

from snaptrade_client import SnapTrade
from portfolio_analyser_refactor import analyze_portfolio

# =========================
# Config
# =========================
st.set_page_config(page_title="SnapTrade Portfolio Analyzer", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "snaptrade_users.sqlite3"

EARNINGS_CSV = BASE_DIR / "sec_quarterly_net_income.csv"
MARKETCAP_CSV = BASE_DIR / "tickers_2026-02-23.csv"


# =========================
# Helpers: SnapTrade init
# =========================
def init_snaptrade() -> SnapTrade:
    client_id = st.secrets.get("SNAPTRADE_CLIENT_ID", "")
    consumer_key = st.secrets.get("SNAPTRADE_CONSUMER_KEY", "")
    if not client_id or not consumer_key:
        raise RuntimeError("Missing SNAPTRADE_CLIENT_ID / SNAPTRADE_CONSUMER_KEY in .streamlit/secrets.toml")
    return SnapTrade(consumer_key=consumer_key, client_id=client_id)


# =========================
# Helpers: SQLite (keep old single-row-per-profile model)
# =========================
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_init() -> None:
    conn = db_connect()
    try:
        # Old schema (no id column). Keep it.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS snaptrade_users (
                profile TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                user_secret TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def db_get_profile(profile: str) -> Optional[Dict[str, str]]:
    conn = db_connect()
    try:
        row = conn.execute(
            "SELECT profile, user_id, user_secret, created_at, updated_at FROM snaptrade_users WHERE profile = ?",
            (profile,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def db_upsert_profile(profile: str, user_id: str, user_secret: str) -> None:
    conn = db_connect()
    try:
        now = _utc_now_iso()
        existing = conn.execute(
            "SELECT profile FROM snaptrade_users WHERE profile = ?",
            (profile,),
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE snaptrade_users
                SET user_id = ?, user_secret = ?, updated_at = ?
                WHERE profile = ?
                """,
                (user_id, user_secret, now, profile),
            )
        else:
            conn.execute(
                """
                INSERT INTO snaptrade_users (profile, user_id, user_secret, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (profile, user_id, user_secret, now, now),
            )
        conn.commit()
    finally:
        conn.close()


def mask_secret(s: str) -> str:
    if not s:
        return ""
    if len(s) <= 8:
        return "••••••••"
    return f"{s[:4]}••••••••{s[-4:]}"


# =========================
# Helpers: parsing positions
# =========================
def normalize_weights(df: pd.DataFrame, ticker_col="ticker", value_col="value") -> pd.DataFrame:
    df = df.copy()
    df[ticker_col] = df[ticker_col].astype(str).str.upper().str.strip()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    total = float(df[value_col].sum())
    if total <= 0:
        raise ValueError("Total portfolio value is 0; cannot compute weights.")
    df["weight"] = df[value_col] / total
    return df[[ticker_col, "weight"]]


def _to_dict_maybe(x: Any) -> Any:
    if isinstance(x, dict):
        return x
    if hasattr(x, "to_dict") and callable(getattr(x, "to_dict")):
        try:
            return x.to_dict()
        except Exception:
            return x
    if hasattr(x, "__dict__"):
        try:
            return dict(x.__dict__)
        except Exception:
            return x
    return x


def _deep_symbol_value(sym: Any) -> Optional[str]:
    """
    Handles nested shapes like:
      sym = {"symbol": "QQQ"}
      sym = {"symbol": {"symbol": "QQQ"}}
      sym = {"symbol": {"symbol": {"symbol": "QQQ"}}}
    and objects with .symbol.
    """
    cur = sym
    for _ in range(6):
        cur = _to_dict_maybe(cur)
        if isinstance(cur, str):
            return cur.strip().upper()
        if isinstance(cur, dict):
            if "symbol" not in cur:
                break
            cur = cur.get("symbol")
            continue
        if hasattr(cur, "symbol"):
            cur = getattr(cur, "symbol")
            continue
        break
    if isinstance(cur, str):
        return cur.strip().upper()
    return None


def extract_ticker_from_position(p: Any) -> Optional[str]:
    p = _to_dict_maybe(p)
    if not isinstance(p, dict):
        return None

    ticker = _deep_symbol_value(p.get("symbol"))
    if not ticker:
        return None

    if not re.match(r"^[A-Z0-9.\-]+$", ticker):
        return None

    return ticker


def extract_units_price_value(p: Any) -> Tuple[float, float, float]:
    p = _to_dict_maybe(p)
    if not isinstance(p, dict):
        return 0.0, 0.0, 0.0

    units = float(p.get("units") or 0.0)
    price = float(p.get("price") or 0.0)

    mv = p.get("marketValue")
    if mv is not None:
        try:
            return units, price, float(mv)
        except Exception:
            pass

    return units, price, units * price


# =========================
# Boot DB
# =========================
db_init()

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

    st.divider()
    st.header("SnapTrade storage")
    profile = st.text_input("Profile name", value="default").strip()
    st.caption(f"DB absolute path: {DB_PATH.resolve()}")


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
# Mode B: SnapTrade Connect (same buttons as before)
# =========================
if mode == "SnapTrade Connect":
    st.subheader("Connect brokerage")

    try:
        snaptrade = init_snaptrade()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Load persisted creds for this profile (if any)
    persisted = db_get_profile(profile)
    if persisted:
        st.success(f"Loaded persisted credentials for profile '{profile}'")
        st.caption(f"userId: {persisted['user_id']}")
        st.caption(f"userSecret: {mask_secret(persisted['user_secret'])}")
        user_id = persisted["user_id"]
        user_secret = persisted["user_secret"]
    else:
        st.warning(f"No persisted credentials for profile '{profile}' yet.")
        user_id = f"st-{uuid.uuid4()}"
        user_secret = ""

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Create/Reset SnapTrade User"):
            # IMPORTANT: This will create a NEW SnapTrade userId each time (and overwrite DB for this profile)
            user_id = f"st-{uuid.uuid4()}"
            resp = snaptrade.authentication.register_snap_trade_user(body={"userId": user_id})
            user_secret = resp.body["userSecret"]
            db_upsert_profile(profile, user_id, user_secret)
            st.success("User created/reset and persisted.")
            st.rerun()

    with c2:
        if st.button("Generate Connection Portal URL"):
            persisted2 = db_get_profile(profile)
            if not persisted2:
                st.warning("Click Create/Reset SnapTrade User first.")
            else:
                login = snaptrade.authentication.login_snap_trade_user(
                    query_params={"userId": persisted2["user_id"], "userSecret": persisted2["user_secret"]}
                )
                st.session_state.portal_url = login.body["redirectURI"]
                st.info("Open the portal URL, connect your brokerage, then come back and click “Load Accounts”.")

    if "portal_url" in st.session_state:
        st.link_button("Open SnapTrade Connection Portal", st.session_state.portal_url)

    st.divider()

    # ---- Step 1: Load accounts
    st.subheader("Step 1 — Load accounts")

    if st.button("Load Accounts"):
        persisted3 = db_get_profile(profile)
        if not persisted3:
            st.warning("Create/Reset SnapTrade User first.")
            st.stop()

        accounts = snaptrade.account_information.list_user_accounts(
            query_params={"userId": persisted3["user_id"], "userSecret": persisted3["user_secret"]}
        ).body

        if not accounts:
            st.warning("No accounts found yet. If you just connected, try again in a moment.")
            st.stop()

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
        st.session_state.pop("selected_skipped", None)

    accounts = st.session_state.get("accounts", [])
    if accounts:
        st.success(f"Found {len(accounts)} accounts.")

        labels = []
        id_by_label = {}
        for a in accounts:
            inst = a["institution"] or "Broker"
            label = f"{a['name']} ({inst}) — {a['id']}"
            labels.append(label)
            id_by_label[label] = a["id"]

        pick = st.selectbox("Choose an account to deep dive", labels, index=0)
        st.session_state["picked_account_id"] = id_by_label[pick]
        st.session_state["picked_account_label"] = pick

        st.divider()

        # ---- Step 2: Fetch holdings
        st.subheader("Step 2 — Fetch holdings for selected account")

        if st.button("Fetch Holdings (Selected Account)"):
            persisted4 = db_get_profile(profile)
            if not persisted4:
                st.warning("Create/Reset SnapTrade User first.")
                st.stop()

            account_id = st.session_state["picked_account_id"]
            h = snaptrade.account_information.get_user_holdings(
                path_params={"accountId": account_id},
                query_params={"userId": persisted4["user_id"], "userSecret": persisted4["user_secret"]},
            ).body

            positions = h.get("positions", []) or []

            rows: List[Dict[str, Any]] = []
            skipped: List[Dict[str, Any]] = []

            for p in positions:
                ticker = extract_ticker_from_position(p)
                if not ticker:
                    skipped.append({"reason": "unparsed_ticker", "raw": _to_dict_maybe(p)})
                    continue

                units, price, value = extract_units_price_value(p)
                if value <= 0:
                    skipped.append({"reason": "nonpositive_value", "ticker": ticker, "units": units, "price": price})
                    continue

                rows.append({"ticker": ticker, "value": value, "units": units, "price": price})

            if not rows:
                st.error("Holdings returned, but no valid positions could be parsed.")
                if skipped:
                    with st.expander("Skipped positions (debug)"):
                        st.json(skipped[:10])
                st.stop()

            holdings_df = (
                pd.DataFrame(rows)
                .groupby("ticker", as_index=False)[["value", "units"]]
                .sum()
                .sort_values("value", ascending=False)
            )

            st.session_state["selected_holdings_df"] = holdings_df
            st.session_state["selected_skipped"] = skipped

            st.success(f"Parsed {len(holdings_df)} tickers for this account.")
            st.dataframe(holdings_df, use_container_width=True)

            if skipped:
                with st.expander("Skipped positions (debug)"):
                    st.json(skipped[:10])

        # ---- Step 3: Analyze
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

    per_stock = out["per_stock"].copy()
    industry_summary = out["industry_summary"].copy()
    mix = out["bucket_mix"]
    leaders = out.get("leaders", {})
    most_x = out.get("most_x", {})

    st.header(f"Portfolio Summary — {label}")

    # -------------------------
    # Metrics
    # -------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Weighted Dividend Yield",
        f"{out['portfolio_dividend_yield'] * 100:.2f}%"
    )

    col2.metric(
        "Defensive",
        f"{mix.get('Defensive', 0.0) * 100:.1f}%"
    )

    col3.metric(
        "Quality",
        f"{mix.get('Quality', 0.0) * 100:.1f}%"
    )

    col4.metric(
        "Degen",
        f"{mix.get('Degen', 0.0) * 100:.1f}%"
    )

    # -------------------------
    # Prepare data
    # -------------------------

    per_stock_df = per_stock.reset_index().rename(columns={"index": "ticker"})

    per_stock_df["weight_pct"] = per_stock_df["weight"] * 100
    per_stock_df["ann_return_pct"] = per_stock_df["ann_return"] * 100
    per_stock_df["ann_vol_pct"] = per_stock_df["ann_vol"] * 100
    per_stock_df["max_dd_pct"] = per_stock_df["max_drawdown"] * 100
    per_stock_df["div_yield_pct"] = per_stock_df["dividend_yield"] * 100

    industry_df = industry_summary.reset_index().rename(columns={"index": "industry"})
    industry_df["industry_weight_pct"] = industry_df["industry_weight"] * 100

    # -------------------------
    # Charts row 1
    # -------------------------

    st.subheader("Portfolio Composition")

    c1, c2 = st.columns(2)

    with c1:

        mix_df = pd.DataFrame({
            "bucket": list(mix.keys()),
            "value": [v * 100 for v in mix.values()]
        })

        fig_mix = px.pie(
            mix_df,
            names="bucket",
            values="value",
            hole=0.45,
            title="Bucket Mix"
        )

        st.plotly_chart(fig_mix, use_container_width=True)

    with c2:

        top_holdings = per_stock_df.sort_values(
            "weight_pct",
            ascending=False
        ).head(12)

        fig_holdings = px.bar(
            top_holdings,
            x="ticker",
            y="weight_pct",
            color="industry",
            title="Top Holdings by Portfolio Weight",
            labels={"weight_pct": "Weight (%)"}
        )

        st.plotly_chart(fig_holdings, use_container_width=True)

    # -------------------------
    # Charts row 2
    # -------------------------

    c3, c4 = st.columns(2)

    with c3:

        ind = industry_df.sort_values(
            "industry_weight_pct",
            ascending=False
        ).head(12)

        fig_industry = px.bar(
            ind,
            x="industry_weight_pct",
            y="industry",
            orientation="h",
            title="Industry Allocation",
            labels={"industry_weight_pct": "Weight (%)"}
        )

        st.plotly_chart(fig_industry, use_container_width=True)

    with c4:

        fig_risk = px.scatter(
            per_stock_df,
            x="ann_vol_pct",
            y="ann_return_pct",
            size="weight_pct",
            color="industry",
            hover_name="ticker",
            title="Risk vs Return Map",
            labels={
                "ann_vol_pct": "Volatility (%)",
                "ann_return_pct": "Return (%)"
            }
        )

        st.plotly_chart(fig_risk, use_container_width=True)

    # -------------------------
    # Leaders
    # -------------------------

    st.subheader("Leaders (trait × portfolio weight)")

    if leaders:

        leaders_df = pd.DataFrame([
            {
                "bucket": k,
                "ticker": v["ticker"],
                "portfolio_pct_in_bucket": v["portfolio_pct_in_bucket"] * 100
            }
            for k, v in leaders.items()
        ])

        st.dataframe(leaders_df, use_container_width=True)

    # -------------------------
    # Tables
    # -------------------------

    st.subheader("Per-stock table")

    display_df = per_stock.copy()

    for col in [
        "weight",
        "ann_return",
        "ann_vol",
        "max_drawdown",
        "dividend_yield",
        "Defensive",
        "Quality",
        "Speculative",
        "Degen",
    ]:
        if col in display_df.columns:
            display_df[col] = display_df[col] * 100

    st.dataframe(display_df, use_container_width=True)

    st.subheader("Industry summary")

    ind_fmt = industry_summary.copy()

    if "industry_weight" in ind_fmt.columns:
        ind_fmt["industry_weight"] *= 100

    if "industry_dividend_yield" in ind_fmt.columns:
        ind_fmt["industry_dividend_yield"] *= 100

    for col in ["Defensive", "Quality", "Speculative", "Degen"]:
        if col in ind_fmt.columns:
            ind_fmt[col] *= 100

    st.dataframe(ind_fmt, use_container_width=True)