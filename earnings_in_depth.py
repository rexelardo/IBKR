import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("sec_quarterly_net_income.csv")
df["quarter_end"] = pd.to_datetime(df["quarter_end"])
df = df.sort_values(["ticker", "quarter_end"])

results = []

START_YEAR = 2020
MIN_Q = 8  # minimum quarters since 2020

for ticker, g in df.groupby("ticker"):
    g = g.sort_values("quarter_end")

    # Keep only data since 2020
    g = g[g["quarter_end"].dt.year >= START_YEAR]

    if len(g) < MIN_Q:
        continue

    y = g["net_income"].astype(float).values
    dates = g["quarter_end"].values

    # Must start and end positive for CAGR to make sense
    if y[0] <= 0 or y[-1] <= 0:
        continue

    # -------- Trend (Linear Regression Slope) --------
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]

    # -------- CAGR Calculation --------
    years = (dates[-1] - dates[0]).astype('timedelta64[D]').astype(int) / 365.25
    if years <= 0:
        continue

    cagr = (y[-1] / y[0]) ** (1 / years) - 1

    results.append({
        "ticker": ticker,
        "slope": slope,
        "cagr": cagr
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Keep only positive trend companies
results_df = results_df[results_df["slope"] > 0]

# Sort by CAGR descending
results_df = results_df.sort_values("cagr", ascending=False)

print("Companies with positive trend since 2020:", len(results_df))
print(results_df.head(20))