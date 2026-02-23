import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("sec_quarterly_net_income.csv")

# Prep
df["quarter_end"] = pd.to_datetime(df["quarter_end"])
df = df.sort_values(["ticker", "quarter_end"])

qualified = []

for ticker, g in df.groupby("ticker"):
    g = g.sort_values("quarter_end").reset_index(drop=True)
    
    earnings = g["net_income"].values
    dates = g["quarter_end"].values
    
    if len(g) < 10:
        continue
    
    # Check every rolling 10-quarter window
    for i in range(len(g) - 9):
        window_earnings = earnings[i:i+10]
        window_dates = dates[i:i+10]
        
        # Condition 1: all positive
        if not np.all(window_earnings > 0):
            continue
        
        # Condition 2: strictly increasing
        if not np.all(np.diff(window_earnings) > 0):
            continue
        
        # Condition 3: last quarter in 2025
        year = pd.to_datetime(window_dates[-1]).year
        if year not in [2025, 2026]:
            continue
        
        qualified.append(ticker)
        break  # stop once we find one valid streak

print("Companies meeting criteria:", len(qualified))
print(qualified)