# treemap_drilldown_capweighted_returns_robust.py
#
# What this script does (end-to-end):
# 1) Load tickers + industries + marketCap from your CSV
# 2) Fix Yahoo tickers: "." -> "-" (e.g., BRK.B -> BRK-B)
# 3) Download Adj Close from yfinance robustly:
#    - chunked requests (avoids rate limits)
#    - retries + backoff
#    - threads=False (avoids sqlite/timezone cache contention)
#    - writable tz-cache location (avoids "unable to open database file")
# 4) Compute calendar-offset returns per ticker with fallback:
#    - if target date precedes first available price, use first available price
#    - if ticker has no data at all, returns = NaN
# 5) Build cap-weighted returns by industry (weights = ticker marketCap from CSV)
# 6) Treemap drilldown:
#    - at each level: bucket industries by THRESHOLD into "big" + "Other"
#    - plot 2x2 treemaps (YTD, 7D, 30D, 365D) for that universe
#    - then drill into "Other" universe (the small industries) and repeat until no small left
# 7) Find top industry (by TOP_HORIZON) in full universe and export a CSV of all its companies + returns
#
# Outputs:
# - treemap_level_{level}_{YYYY-MM-DD}.png (if SAVE_PNGS=True)
# - top_industry_{TOP_HORIZON}_{IndustryName}_{YYYYMMDD}.csv
# - optional: missing tickers report CSVs (see REPORT_MISSING_CSVS)

import os
import time
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import squarify
import yfinance as yf


# -----------------------
# Config
# -----------------------
CSV_PATH = "capital_flows/tickers_2026-02-16.csv"

THRESHOLD = 0.005          # 0.5% bucket threshold per level
TOP_HORIZON = "YTD"        # "YTD", "7D", "30D", "365D"
SAVE_PNGS = True
PNG_DIR = "."
REPORT_MISSING_CSVS = True # write csv reports of missing/no-data/failed batches
YF_CHUNK_SIZE = 150        # lower = safer for rate limits
YF_PAUSE_SECONDS = 1.2     # pause between chunks
YF_MAX_RETRIES = 4         # retries per chunk
LOOKBACK_DAYS = 420        # enough for 365D + buffer


# -----------------------
# yfinance cache fix (avoids sqlite write errors)
# -----------------------
os.makedirs("/tmp/yf_cache", exist_ok=True)
try:
    yf.set_tz_cache_location("/tmp/yf_cache")
except Exception:
    pass  # older yfinance versions may not have this


# -----------------------
# Helpers
# -----------------------
def format_money(x: float) -> str:
    ax = abs(x)
    if ax >= 1e12:
        return f"${x/1e12:.2f}T"
    elif ax >= 1e9:
        return f"${x/1e9:.2f}B"
    elif ax >= 1e6:
        return f"${x/1e6:.2f}M"
    elif ax >= 1e3:
        return f"${x/1e3:.2f}K"
    else:
        return f"${x:.0f}"

def safe_filename(s: str, max_len: int = 120) -> str:
    s = re.sub(r"[^\w\s\-.]", "", str(s)).strip()
    s = re.sub(r"\s+", "_", s)
    return s[:max_len] if len(s) > max_len else s

def cap_weighted_return(df_: pd.DataFrame, col: str) -> float:
    w = df_["marketCap"].to_numpy(dtype=float)
    r = df_[col].to_numpy(dtype=float)
    m = np.isfinite(w) & np.isfinite(r) & (w > 0)
    if m.sum() == 0:
        return np.nan
    return float(np.sum(w[m] * r[m]) / np.sum(w[m]))

def start_prices_with_fallback(px: pd.DataFrame, target_date: pd.Timestamp) -> pd.Series:
    """
    For each ticker column:
      - if there is a price on/before target_date -> last price <= target_date
      - else -> first available price (earliest)
      - if no data -> NaN
    """
    target_date = pd.Timestamp(target_date)
    out = {}
    for col in px.columns:
        s = px[col].dropna()
        if s.empty:
            out[col] = np.nan
            continue
        if s.index[0] > target_date:
            out[col] = float(s.iloc[0])
        else:
            out[col] = float(s.loc[:target_date].iloc[-1])
    return pd.Series(out)

def end_prices_latest(px: pd.DataFrame) -> pd.Series:
    """Latest available price for each ticker column (or NaN if none)."""
    out = {}
    for col in px.columns:
        s = px[col].dropna()
        out[col] = float(s.iloc[-1]) if not s.empty else np.nan
    return pd.Series(out)

def compute_returns_calendar(px: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """
    Returns per ticker using calendar offsets snapped with fallback:
      - YTD: Jan 1 of current year (fallback to first price if IPO after Jan 1)
      - 7D: asof - 7 calendar days (fallback to first price if newer)
      - 30D: asof - 30 calendar days
      - 365D: asof - 365 calendar days
    """
    asof = pd.Timestamp(asof).normalize()
    endp = end_prices_latest(px)

    year_start = pd.Timestamp(asof.year, 1, 1)
    targets = {
        "YTD": year_start,
        "7D": asof - pd.Timedelta(days=7),
        "30D": asof - pd.Timedelta(days=30),
        "365D": asof - pd.Timedelta(days=365),
    }

    out = {}
    for k, t in targets.items():
        startp = start_prices_with_fallback(px, t)
        out[k] = (endp / startp) - 1.0

    return pd.DataFrame(out)

def diverging_colors(vals: np.ndarray):
    """Color map based on returns with percentile clipping to avoid outlier domination."""
    v = vals.astype(float)
    finite = np.isfinite(v)
    if not finite.any():
        return [(0.9, 0.9, 0.9, 1.0)] * len(v)

    lo = np.nanpercentile(v[finite], 5)
    hi = np.nanpercentile(v[finite], 95)
    v_clip = np.clip(v, lo, hi)

    vmin = np.nanmin(v_clip[finite])
    vmax = np.nanmax(v_clip[finite])
    denom = (vmax - vmin) + 1e-12
    x = (v_clip - vmin) / denom

    out = []
    for xi, ok in zip(x, finite):
        out.append(plt.cm.RdYlGn(float(xi)) if ok else (0.9, 0.9, 0.9, 1.0))
    return out

def add_other_row(grouped: pd.DataFrame, threshold: float):
    """
    grouped must have: industry, marketCap, pct
    returns (plot_df, big_df, small_df)
    """
    big = grouped[grouped["pct"] > threshold].copy()
    small = grouped[grouped["pct"] <= threshold].copy()

    if not small.empty:
        total = float(grouped["marketCap"].sum())
        other_mcap = float(small["marketCap"].sum())
        other_row = pd.DataFrame({
            "industry": ["Other"],
            "marketCap": [other_mcap],
            "pct": [other_mcap / total if total else np.nan],
        })
        plot_df = pd.concat([big, other_row], ignore_index=True)
    else:
        plot_df = big.copy()

    plot_df = plot_df.sort_values("marketCap", ascending=False).reset_index(drop=True)
    return plot_df, big, small

def savefig_level(fig, level: int, asof: pd.Timestamp):
    if not SAVE_PNGS:
        return
    os.makedirs(PNG_DIR, exist_ok=True)
    path = os.path.join(PNG_DIR, f"capital_flows_treemap_level_{level}_{pd.Timestamp(asof):%Y-%m-%d}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[saved] {path}")

def yf_download_chunked_adjclose(
    tickers: list[str],
    start: str,
    end: str,
    chunk_size: int = 150,
    pause: float = 1.0,
    max_retries: int = 3,
):
    """
    Robust chunked yfinance download for Adj Close.

    Returns:
      px: DataFrame with Adj Close columns for tickers that returned data (may contain NaNs)
      failed_batches: list of tickers belonging to batches that failed completely
    """
    frames = []
    failed_batches = []

    for i in range(0, len(tickers), chunk_size):
        batch = tickers[i:i + chunk_size]

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                data = yf.download(
                    batch,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=False,
                    threads=False,   # important: reduces rate limit & sqlite issues
                    group_by="column"
                )
                if data is None or data.empty:
                    raise RuntimeError("Empty response from yfinance")
                # Expect MultiIndex columns; take Adj Close
                adj = data["Adj Close"] if "Adj Close" in data.columns.get_level_values(0) else data.get("Adj Close")
                if adj is None or (hasattr(adj, "empty") and adj.empty):
                    raise RuntimeError("No Adj Close in response")
                frames.append(adj)
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(pause * attempt)

        if last_err is not None:
            failed_batches.extend(batch)

        time.sleep(pause)

    if not frames:
        return pd.DataFrame(), failed_batches

    px = pd.concat(frames, axis=1)
    px = px.loc[:, ~px.columns.duplicated()]  # in case of overlap
    px = px.sort_index()
    return px, failed_batches


# -----------------------
# Load CSV
# -----------------------
df = pd.read_csv(CSV_PATH)

required = ["ticker", "name", "industry", "marketCap"]
missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    raise ValueError(f"CSV missing required columns: {missing_cols}")

df = df[required].dropna(subset=["ticker", "industry", "marketCap"]).copy()
df["ticker"] = df["ticker"].astype(str)

# yfinance symbol normalization (BRK.B -> BRK-B)
df["yf_ticker"] = df["ticker"].str.replace(".", "-", regex=False)

# Dedup tickers
df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

asof = pd.Timestamp.today().normalize()
start_needed = (asof - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
end_needed = (asof + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

tickers_yf = sorted(df["yf_ticker"].unique().tolist())
print(f"Tickers in CSV: {len(df)} | Unique yfinance tickers: {len(tickers_yf)}")


# -----------------------
# Download prices robustly
# -----------------------
px, failed_batches = yf_download_chunked_adjclose(
    tickers_yf,
    start=start_needed,
    end=end_needed,
    chunk_size=YF_CHUNK_SIZE,
    pause=YF_PAUSE_SECONDS,
    max_retries=YF_MAX_RETRIES,
)

if px.empty:
    raise RuntimeError("No price data downloaded at all. Try smaller chunk_size or wait (rate-limited).")

# Drop columns with absolutely no data
good_cols = [c for c in px.columns if px[c].notna().any()]
bad_cols_no_data = sorted(set(px.columns) - set(good_cols))
px = px[good_cols]

print(f"Downloaded columns with data: {len(good_cols)}")
if bad_cols_no_data:
    print(f"Tickers with NO data (examples): {bad_cols_no_data[:20]}")
if failed_batches:
    print(f"Failed batch tickers (examples): {failed_batches[:20]}")

# Optional reports
if REPORT_MISSING_CSVS:
    if bad_cols_no_data:
        pd.DataFrame({"yf_ticker": bad_cols_no_data}).to_csv(
            f"yfinance_no_price_data_{asof:%Y%m%d}.csv", index=False
        )
        print(f"[wrote] yfinance_no_price_data_{asof:%Y%m%d}.csv")
    if failed_batches:
        pd.DataFrame({"yf_ticker": failed_batches}).to_csv(
            f"yfinance_failed_batches_{asof:%Y%m%d}.csv", index=False
        )
        print(f"[wrote] yfinance_failed_batches_{asof:%Y%m%d}.csv")


# -----------------------
# Compute ticker returns (calendar offsets + fallback)
# -----------------------
ret_ticker = compute_returns_calendar(px, asof)
ret_ticker.index.name = "yf_ticker"

# Map metadata back onto returns
meta = df.set_index("yf_ticker")[["ticker", "name", "industry", "marketCap"]]
ret_ticker = ret_ticker.join(meta, how="left")

# Some tickers may have prices but be missing metadata join (rare); drop those
ret_ticker = ret_ticker.dropna(subset=["industry", "marketCap"])


# -----------------------
# Drilldown treemaps (cap-weighted returns)
# -----------------------
level = 0
current_universe = df.copy()

while True:
    # Industry market cap in current universe (for rectangle sizes)
    grouped = (
        current_universe.groupby("industry", as_index=False)["marketCap"]
        .sum()
        .sort_values("marketCap", ascending=False)
        .reset_index(drop=True)
    )
    total_mcap = float(grouped["marketCap"].sum())
    grouped["pct"] = grouped["marketCap"] / total_mcap if total_mcap else np.nan

    plot_df, big_df, small_df = add_other_row(grouped, THRESHOLD)

    # Restrict returns to tickers in current universe
    cur_yf = set(current_universe["yf_ticker"].tolist())
    cur_ret = ret_ticker[ret_ticker.index.isin(cur_yf)].copy()

    # Cap-weighted returns per industry in current universe
    ind_rows = []
    for ind, g in cur_ret.groupby("industry"):
        ind_rows.append({
            "industry": ind,
            "YTD": cap_weighted_return(g, "YTD"),
            "7D": cap_weighted_return(g, "7D"),
            "30D": cap_weighted_return(g, "30D"),
            "365D": cap_weighted_return(g, "365D"),
        })
    ind_rets = pd.DataFrame(ind_rows).set_index("industry") if ind_rows else pd.DataFrame(
        columns=["YTD", "7D", "30D", "365D"], index=pd.Index([], name="industry")
    )

    # Attach industry returns to plot df
    for col in ["YTD", "7D", "30D", "365D"]:
        plot_df[col] = plot_df["industry"].map(ind_rets[col]) if not ind_rets.empty else np.nan

    # Cap-weighted "Other" return across tickers in small industries of this level
    if not small_df.empty:
        small_set = set(small_df["industry"].tolist())
        other_mask = cur_ret["industry"].isin(small_set)
        if other_mask.any():
            for col in ["YTD", "7D", "30D", "365D"]:
                plot_df.loc[plot_df["industry"] == "Other", col] = cap_weighted_return(cur_ret.loc[other_mask], col)

    # Plot 2x2 treemaps for this level
    sizes = plot_df["marketCap"].to_numpy(dtype=float)
    inds = plot_df["industry"].to_numpy(dtype=str)
    horizons = [("YTD", "YTD"), ("7D", "7D"), ("30D", "30D"), ("365D", "365D")]

    fig, axes = plt.subplots(2, 2, figsize=(32, 16))
    axes = axes.ravel()

    for ax, (col, title_part) in zip(axes, horizons):
        ax.axis("off")
        r = plot_df[col].to_numpy(dtype=float)

        labels = []
        for ind, mcap, rv in zip(inds, sizes, r):
            share = (mcap / total_mcap) if total_mcap else np.nan
            rv_str = f"{rv:+.2%}" if np.isfinite(rv) else "n/a"
            labels.append(f"{ind}\n{format_money(mcap)} ({share:.1%})\n{title_part}: {rv_str}")

        colors = diverging_colors(r)

        squarify.plot(
            sizes=sizes,
            label=labels,
            color=colors,
            alpha=1.0,
            pad=True,
            ax=ax,
        )

        ax.set_title(
            f"Industry Treemap — Level {level} — {title_part} (as of {asof.date()})\nUniverse total: {format_money(total_mcap)}",
            fontsize=14
        )

    plt.tight_layout()
    savefig_level(fig, level, asof)
    plt.show()

    # Stop if nothing to drill
    if small_df.empty:
        break

    # Next universe = small industries only
    small_set = set(small_df["industry"].tolist())
    current_universe = current_universe[current_universe["industry"].isin(small_set)].copy()
    level += 1

    if current_universe.empty:
        break


# -----------------------
# Export: company breakdown for top industry (full universe, cap-weighted)
# -----------------------
full_ret = ret_ticker.copy()

# Cap-weighted industry returns on full universe
full_ind_rows = []
for ind, g in full_ret.groupby("industry"):
    full_ind_rows.append({
        "industry": ind,
        "YTD": cap_weighted_return(g, "YTD"),
        "7D": cap_weighted_return(g, "7D"),
        "30D": cap_weighted_return(g, "30D"),
        "365D": cap_weighted_return(g, "365D"),
        "industry_mcap": float(g["marketCap"].sum()),
        "n": int(len(g)),
    })

full_ind_df = pd.DataFrame(full_ind_rows)

if full_ind_df.empty or TOP_HORIZON not in full_ind_df.columns:
    print("No industries found to rank for export.")
else:
    full_ind_df = full_ind_df.sort_values(TOP_HORIZON, ascending=False)

    if not np.isfinite(full_ind_df.iloc[0][TOP_HORIZON]):
        print("Top industry return is NaN (insufficient price data). No export written.")
    else:
        top_industry = str(full_ind_df.iloc[0]["industry"])
        top_val = float(full_ind_df.iloc[0][TOP_HORIZON])
        print(f"Top industry by {TOP_HORIZON}: {top_industry} ({top_val:+.2%})")

        top_companies = full_ret[full_ret["industry"] == top_industry].copy()

        # Ensure yf_ticker exists as a COLUMN (in your pipeline it's often the index name)
        if "yf_ticker" not in top_companies.columns:
            top_companies = (
                top_companies
                .reset_index()
                .rename(columns={"index": "yf_ticker"})
            )
            # If reset_index created a column with the index name, prefer that
            if "yf_ticker" not in top_companies.columns and top_companies.columns[0] == "yf_ticker":
                pass

        # Compute within-industry weights + contribution proxy
        total_ind_mcap = float(top_companies["marketCap"].sum())
        top_companies["weight_in_industry"] = (
            top_companies["marketCap"] / total_ind_mcap if total_ind_mcap else np.nan
        )
        top_companies[f"cap_weighted_contrib_{TOP_HORIZON}"] = (
            top_companies["weight_in_industry"] * top_companies[TOP_HORIZON]
        )

        export_cols = [
            "ticker", "yf_ticker", "name", "industry", "marketCap",
            "YTD", "7D", "30D", "365D",
            "weight_in_industry", f"cap_weighted_contrib_{TOP_HORIZON}",
        ]

        # Safety: keep only columns that exist (prevents KeyError if something is missing)
        export_cols = [c for c in export_cols if c in top_companies.columns]

        out = top_companies[export_cols].sort_values(TOP_HORIZON, ascending=False)

        # Ensure output directory exists
        out_dir = os.path.dirname(f"capital_flows/top_industry_{TOP_HORIZON}_x.csv")
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        out_name = f"capital_flows/top_industry_{TOP_HORIZON}_{safe_filename(top_industry)}_{asof:%Y%m%d}.csv"
        out.to_csv(out_name, index=False)
        print(f"[wrote] {out_name}")
