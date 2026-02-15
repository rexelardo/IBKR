from ib_insync import *
import pandas as pd
import matplotlib.pyplot as plt
# ib = IB()
# ib.connect('127.0.0.1', 7496, clientId=1)


THRESHOLD_PCT = 1.0  # <= 1% goes into "Other" breakdown

def fmt_money(x: float) -> str:
    return f"{x:,.2f}"

def main():
    ib = IB()
    ib.connect('127.0.0.1', 7496, clientId=1)

    print("Connected:", ib.isConnected())
    print("Server time:", ib.reqCurrentTime())
    print("Accounts:", ib.managedAccounts())


    try:
        items = ib.portfolio()
        if not items:
            print("No portfolio items returned. Check TWS login + API enabled + correct port.")
            return

        rows = []
        for p in items:
            symbol = p.contract.symbol or p.contract.localSymbol
            rows.append({
                "symbol": symbol,
                "marketValue": float(p.marketValue),
            })

        df = pd.DataFrame(rows)
        df = df[df["marketValue"].abs() > 1e-9].copy()

        agg = (
            df.groupby("symbol", as_index=False)["marketValue"]
              .sum()
              .sort_values("marketValue", ascending=False)
              .reset_index(drop=True)
        )

        total = agg["marketValue"].sum()
        if total == 0:
            print("Total market value is 0. Nothing to plot.")
            return

        agg["pct"] = agg["marketValue"] / total * 100.0

        # Print table (all holdings)
        printable = agg.copy()
        printable["marketValue"] = printable["marketValue"].map(fmt_money)
        printable["pct"] = printable["pct"].map(lambda x: f"{x:.2f}%")
        print("\nPortfolio allocation (by market value):\n")
        print(printable.to_string(index=False))

        # Split into >1% and <=1%
        big = agg[agg["pct"] > THRESHOLD_PCT].copy()
        small = agg[agg["pct"] <= THRESHOLD_PCT].copy()

        # --- MAIN PIE ---
        main_pie = big.copy()

        if not small.empty:
            other_value = small["marketValue"].sum()
            other_pct = small["pct"].sum()
            main_pie = pd.concat(
                [main_pie, pd.DataFrame([{
                    "symbol": f"Other (≤{THRESHOLD_PCT:.0f}%)",
                    "marketValue": other_value,
                    "pct": other_pct
                }])],
                ignore_index=True
            )

        main_labels = [f"{s} ({p:.1f}%)" for s, p in zip(main_pie["symbol"], main_pie["pct"])]

        plt.figure(figsize=(9, 9))
        plt.pie(
            main_pie["marketValue"],
            labels=main_labels,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 2 else ""
        )
        plt.title("IBKR Portfolio Allocation (Holdings > 1% + Other bucket)")
        plt.tight_layout()
        plt.show()

        # --- SECOND PIE: BREAKDOWN OF "OTHER" ---
        if not small.empty:
            # To avoid unreadable labels, we can label only if slice >= 2% of the OTHER bucket
            other_total = small["marketValue"].sum()
            small = small.sort_values("marketValue", ascending=False).reset_index(drop=True)
            small_labels = [
                f"{row.symbol} ({(row.marketValue/other_total*100):.1f}% of Other)"
                for row in small.itertuples(index=False)
            ]

            plt.figure(figsize=(9, 9))
            plt.pie(
                small["marketValue"],
                labels=small_labels,
                autopct=lambda pct: f"{pct:.1f}%" if pct >= 2 else ""
            )
            plt.title(f"Breakdown of Other (Holdings ≤ {THRESHOLD_PCT:.0f}% of portfolio)")
            plt.tight_layout()
            plt.show()
        else:
            print(f"\nNo holdings at or below {THRESHOLD_PCT:.0f}% — no 'Other' breakdown needed.\n")

    finally:
        ib.disconnect()

if __name__ == "__main__":
    main()