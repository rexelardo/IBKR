import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks

import mplfinance as mpf
import matplotlib.pyplot as plt

# -----------------------------
# Helpers
# -----------------------------

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR using simple moving average of True Range."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

    return tr.rolling(period).mean()


# -----------------------------
# Core detector (SciPy, conservative)
# -----------------------------

def find_w_patterns_scipy_conservative(
    df: pd.DataFrame,
    atr_period: int = 14,

    # Swing detection (conservative)
    min_swing_distance: int = 8,     # bars between extrema
    trough_prom_atr: float = 1.0,    # prominence threshold in ATR for troughs (applied on -Low)
    peak_prom_atr: float = 1.0,      # prominence threshold in ATR for peaks (applied on High)

    # Geometry constraints
    min_sep: int = 12,
    max_sep: int = 70,
    low_atr_tol: float = 1.0,        # |L2-L1| <= 1.0 ATR
    min_bounce_atr: float = 2.0,     # neckline above avg bottom by >= 2 ATR

    # Context: prior downtrend into L1
    downtrend_lookback: int = 30,
    downtrend_min_drop: float = 0.08,  # >= 8% drop into L1 over lookback

    # Confirmation: breakout after L2
    breakout_lookahead: int = 20,
    breakout_buffer_atr: float = 0.25, # close above neckline by 0.25 ATR

    # Volume confirmation
    vol_ma_period: int = 20,
    vol_multiplier: float = 1.5,     # breakout vol >= 1.5x vol MA
    require_vol_weaken_on_L2: bool = True,  # vol(L2) <= vol(L1)
) -> pd.DataFrame:
    """
    df index must be DateTimeIndex and columns must include:
    High, Low, Close, Volume
    Returns confirmed patterns (i.e., breakout detected and volume-confirmed).
    """
    needed = {"High", "Low", "Close", "Volume"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    df["ATR"] = compute_atr(df, atr_period)
    df["VOL_MA"] = df["Volume"].rolling(vol_ma_period).mean()

    # Keep rows where ATR and VOL_MA exist
    dfv = df[df["ATR"].notna() & df["VOL_MA"].notna()].copy()
    if len(dfv) < 60:
        return pd.DataFrame()

    lows = dfv["Low"].to_numpy()
    highs = dfv["High"].to_numpy()
    closes = dfv["Close"].to_numpy()
    vols = dfv["Volume"].to_numpy()
    atrv = dfv["ATR"].to_numpy()
    vol_ma = dfv["VOL_MA"].to_numpy()
    idx = dfv.index

    # Use median ATR as a stable scale for prominence thresholds
    atr_scale = np.nanmedian(atrv)
    if not np.isfinite(atr_scale) or atr_scale <= 0:
        return pd.DataFrame()

    trough_prom = trough_prom_atr * atr_scale
    peak_prom = peak_prom_atr * atr_scale

    # Troughs as peaks on inverted lows
    troughs, _ = find_peaks(-lows, distance=min_swing_distance, prominence=trough_prom)
    # Peaks on highs
    peaks, _ = find_peaks(highs, distance=min_swing_distance, prominence=peak_prom)

    if len(troughs) < 2 or len(peaks) < 1:
        return pd.DataFrame()

    def best_neckline(t1: int, t2: int):
        between = peaks[(peaks > t1) & (peaks < t2)]
        if len(between) == 0:
            return None
        return int(between[np.argmax(highs[between])])

    patterns = []

    for i in range(len(troughs) - 1):
        t1 = int(troughs[i])

        # Prior downtrend into L1
        if t1 < downtrend_lookback:
            continue
        start = closes[t1 - downtrend_lookback]
        end = closes[t1]
        if not (np.isfinite(start) and np.isfinite(end)) or start <= 0:
            continue
        drop = (start - end) / start
        if drop < downtrend_min_drop:
            continue

        for j in range(i + 1, len(troughs)):
            t2 = int(troughs[j])
            sep = t2 - t1
            if sep < min_sep:
                continue
            if sep > max_sep:
                break

            h = best_neckline(t1, t2)
            if h is None:
                continue

            L1 = float(lows[t1])
            L2 = float(lows[t2])
            H = float(highs[h])
            avgL = 0.5 * (L1 + L2)

            # Similar bottoms (ATR-based)
            if abs(L2 - L1) > low_atr_tol * float(atrv[t2]):
                continue

            # Big bounce to neckline (ATR-based)
            if (H - avgL) < min_bounce_atr * float(atrv[h]):
                continue

            # Weakening selling pressure (optional)
            if require_vol_weaken_on_L2 and (vols[t2] > vols[t1]):
                continue

            # Breakout confirmation after L2
            end_i = min(len(closes) - 1, t2 + breakout_lookahead)
            if t2 + 1 > end_i:
                continue

            threshold = H + breakout_buffer_atr * float(atrv[t2])
            future = closes[t2 + 1:end_i + 1]
            above = np.where(future > threshold)[0]
            if len(above) == 0:
                continue
            bidx = int(t2 + 1 + above[0])

            # Volume confirmation: breakout volume "increasing" vs its MA
            if vol_ma[bidx] <= 0:
                continue
            if vols[bidx] < vol_multiplier * vol_ma[bidx]:
                continue

            patterns.append({
                "L1_date": idx[t1],
                "H_date": idx[h],
                "L2_date": idx[t2],
                "breakout_date": idx[bidx],
                "L1_low": L1,
                "L2_low": L2,
                "neckline_H": H,
                "sep_days": int(sep),
                "downtrend_drop_pct": float(drop),
                "bottom_diff_atr": float(abs(L2 - L1) / float(atrv[t2])),
                "bounce_atr": float((H - avgL) / float(atrv[h])),
                "breakout_close": float(closes[bidx]),
                "breakout_vol": float(vols[bidx]),
                "breakout_vol_vs_ma": float(vols[bidx] / float(vol_ma[bidx])),
            })

    if not patterns:
        return pd.DataFrame()

    return pd.DataFrame(patterns).sort_values("breakout_date").reset_index(drop=True)


# -----------------------------
# Scanner over your df_all (long format)
# -----------------------------

def scan_current_w_breakouts(
    df_all: pd.DataFrame,
    date_col: str = "Date",
    ticker_col: str = "ticker",
    require_broke_out_on_last_bar: bool = False,  # if True: only days_since_breakout==0
    **detector_kwargs
) -> pd.DataFrame:
    """
    df_all must contain columns:
      ticker, Date, High, Low, Close, Volume   (Open optional)
    Returns tickers currently above neckline for their most recent detected W,
    plus days_since_breakout (trading bars since breakout date).
    """

    # Ensure Date is datetime
    df_all = df_all.copy()
    df_all[date_col] = pd.to_datetime(df_all[date_col])

    rows = []

    for ticker, dfg in df_all.groupby(ticker_col):
        # Correct sort: sort_values on Date (NOT sort_index(by='Date'))
        dfg = dfg.sort_values(date_col)

        # Set Date as index for detector
        df = dfg.set_index(date_col)

        # Basic sanity
        needed = {"High", "Low", "Close", "Volume"}
        if not needed.issubset(df.columns):
            continue
        if len(df) < 120:
            continue

        sig = find_w_patterns_scipy_conservative(df, **detector_kwargs)
        if sig.empty:
            continue

        # Use the most recent breakout pattern for "current" evaluation
        last_pat = sig.sort_values("breakout_date").iloc[-1]

        last_date = df.index[-1]
        last_close = float(df["Close"].iloc[-1])
        neckline = float(last_pat["neckline_H"])
        breakout_date = last_pat["breakout_date"]

        # "Currently breaking out" = price is above neckline on last bar
        if last_close <= neckline:
            continue

        # Trading-bar distance since breakout
        try:
            bpos = df.index.get_loc(breakout_date)
            days_since = int((len(df.index) - 1) - bpos)
        except KeyError:
            continue

        if require_broke_out_on_last_bar and days_since != 0:
            continue

        rows.append({
            "ticker": ticker,
            "last_date": last_date,
            "last_close": last_close,
            "neckline": neckline,
            "breakout_date": breakout_date,
            "days_since_breakout": days_since,

            # pattern dates/levels for charting
            "L1_date": last_pat["L1_date"],
            "H_date": last_pat["H_date"],
            "L2_date": last_pat["L2_date"],
            "L1_low": float(last_pat["L1_low"]),
            "L2_low": float(last_pat["L2_low"]),

            # ranking / diagnostics
            "breakout_vol_vs_ma": float(last_pat["breakout_vol_vs_ma"]),
            "bounce_atr": float(last_pat["bounce_atr"]),
            "bottom_diff_atr": float(last_pat["bottom_diff_atr"]),
            "sep_days": int(last_pat["sep_days"]),
            "downtrend_drop_pct": float(last_pat["downtrend_drop_pct"]),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Freshest first, then strongest volume breakout, then bounce strength
    return out.sort_values(
        ["days_since_breakout", "breakout_vol_vs_ma", "bounce_atr"],
        ascending=[True, False, False]
    ).reset_index(drop=True)


# -----------------------------
# W chart: candlestick + where indicator flashed + now
# -----------------------------

def _date_to_xpos(plot_df: pd.DataFrame, d) -> int:
    """Map a date to mplfinance's integer x-axis position (0..len-1)."""
    ts = pd.Timestamp(d)
    try:
        return int(plot_df.index.get_loc(ts))
    except KeyError:
        return int(plot_df.index.get_indexer([ts], method="nearest")[0])


def plot_w_chart(
    ticker: str,
    df_ohlc: pd.DataFrame,
    row: pd.Series,
    out_path: Path,
    date_col: str = "Date",
    max_bars: int = 750,
    bars_before_l1: int = 30,
) -> None:
    """
    Plot OHLC candlestick with vertical lines and dots at L1, H, L2, breakout, and now.
    Window from before L1 to last bar (up to max_bars, e.g. ~3 years) so the full W is visible.
    Uses integer x-positions so dots/lines align with mplfinance's axis.
    """
    if df_ohlc.index.name != date_col and date_col in df_ohlc.columns:
        df_ohlc = df_ohlc.set_index(date_col)
    df_ohlc.index = pd.to_datetime(df_ohlc.index)
    df_ohlc = df_ohlc.sort_index()

    # Window: from before L1 to last bar, capped at max_bars
    l1_ts = pd.Timestamp(row["L1_date"])
    last_ts = pd.Timestamp(row["last_date"])
    start_ts = l1_ts - pd.Timedelta(days=max(1, bars_before_l1 * 2))
    mask = (df_ohlc.index >= start_ts) & (df_ohlc.index <= last_ts)
    slice_df = df_ohlc.loc[mask, ["Open", "High", "Low", "Close", "Volume"]]
    if len(slice_df) > max_bars:
        slice_df = slice_df.tail(max_bars)
    plot_df = slice_df.copy()
    if plot_df.empty or len(plot_df) < 10:
        return
    start_ts = plot_df.index[0]
    end_ts = plot_df.index[-1]

    vline_dates = [
        (row["L1_date"], "L1", "green"),
        (row["H_date"], "neck", "blue"),
        (row["L2_date"], "L2", "green"),
        (row["breakout_date"], "breakout", "orange"),
        (row["last_date"], "now", "red"),
    ]
    vlines = [(d, label, c) for d, label, c in vline_dates if start_ts <= pd.Timestamp(d) <= end_ts]

    fig, axes = mpf.plot(
        plot_df,
        type="candle",
        volume=True,
        style="charles",
        title=f"{ticker} — W breakout {pd.Timestamp(row['breakout_date']).strftime('%Y-%m-%d')} | neckline {row['neckline']:.2f} | now {pd.Timestamp(row['last_date']).strftime('%Y-%m-%d')}",
        ylabel="Price",
        ylabel_lower="Volume",
        returnfig=True,
        figsize=(14, 7),
        warn_too_much_data=10000,
    )
    ax_main = axes[0]

    # mplfinance x-axis is integer (0..len-1). Use position so lines/dots align.
    for d, label, color in vlines:
        pos = _date_to_xpos(plot_df, d)
        ax_main.axvline(x=pos, color=color, linestyle="--", alpha=0.8, linewidth=1)
        ytop = ax_main.get_ylim()[1]
        ax_main.annotate(label, xy=(pos, ytop), fontsize=8, color=color, ha="center")

    # Dots at key levels (L1, neckline H, L2) using integer x
    for date_key, price_key, dot_color in [
        ("L1_date", "L1_low", "green"),
        ("H_date", "neckline", "blue"),
        ("L2_date", "L2_low", "green"),
    ]:
        if date_key not in row or price_key not in row:
            continue
        ts = pd.Timestamp(row[date_key])
        if ts < start_ts or ts > end_ts:
            continue
        pos = _date_to_xpos(plot_df, ts)
        price = float(row[price_key])
        ax_main.scatter([pos], [price], color=dot_color, s=60, zorder=5, edgecolors="white")

    # Now dot at last bar (last index = len(plot_df)-1)
    ax_main.scatter([len(plot_df) - 1], [row["last_close"]], color="red", s=80, zorder=5, edgecolors="white", label="now")

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Example usage
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
STOCK_CSV = SCRIPT_DIR / "all_stock_info.csv"

df_all = pd.read_csv(STOCK_CSV)
if df_all.columns[0] == "Unnamed: 0" or df_all.columns[0] == "":
    df_all = df_all.drop(columns=[df_all.columns[0]], errors="ignore")

current = scan_current_w_breakouts(
    df_all,
    date_col="Date",
    ticker_col="ticker",
    require_broke_out_on_last_bar=False,  # True => only breakout on last bar

    # Conservative knobs (you can tweak later)
    min_swing_distance=8,
    trough_prom_atr=1.0,
    peak_prom_atr=1.0,
    min_sep=12,
    max_sep=70,
    low_atr_tol=1.0,
    min_bounce_atr=2.0,
    downtrend_lookback=30,
    downtrend_min_drop=0.08,
    breakout_lookahead=20,
    breakout_buffer_atr=0.25,
    vol_ma_period=20,
    vol_multiplier=1.5,
    require_vol_weaken_on_L2=True,
)

# Only breakouts in the last month
BREAKOUT_LAST_DAYS = 30
now = pd.Timestamp("now")
current["breakout_date"] = pd.to_datetime(current["breakout_date"])
current = current[current["breakout_date"] >= (now - pd.Timedelta(days=BREAKOUT_LAST_DAYS))].reset_index(drop=True)
print(f"Breakouts in last {BREAKOUT_LAST_DAYS} days: {len(current)}")
print(current.head(50))

# Save candlestick chart per ticker: where indicator flashed (L1, H, L2, breakout) and now
CHART_DIR = SCRIPT_DIR / "w_charts"
CHART_DIR.mkdir(exist_ok=True)
date_col = "Date"
ticker_col = "ticker"
df_all_indexed = df_all.copy()
df_all_indexed[date_col] = pd.to_datetime(df_all_indexed[date_col])

for _, row in current.iterrows():
    ticker = row["ticker"]
    dfg = df_all_indexed[df_all_indexed[ticker_col] == ticker].sort_values(date_col)
    if dfg.empty or len(dfg) < 20:
        continue
    df_ohlc = dfg.set_index(date_col)[["Open", "High", "Low", "Close", "Volume"]].copy()
    out_path = CHART_DIR / f"{ticker}_W.png"
    try:
        plot_w_chart(ticker, df_ohlc, row, out_path, date_col=date_col)
        print(f"  chart -> {out_path.name}")
    except Exception as e:
        print(f"  chart skip {ticker}: {e}")
