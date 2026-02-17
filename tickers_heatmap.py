import pandas as pd
import matplotlib.pyplot as plt
import squarify

CSV_PATH = "capital_flows/tickers_2026-02-16.csv"
THRESHOLD = 0.005  # 0.5%

df = pd.read_csv(CSV_PATH)

# Base grouping (industry universe)
base = (
    df.groupby("industry", as_index=False)["marketCap"]
      .sum()
      .sort_values("marketCap", ascending=False)
      .reset_index(drop=True)
)

level = 0
current = base.copy()


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



while not current.empty:
    total = current["marketCap"].sum()
    current = current.copy()
    current["pct"] = current["marketCap"] / total

    # Split on threshold in the CURRENT universe
    big = current[current["pct"] > THRESHOLD].copy()
    small = current[current["pct"] <= THRESHOLD].copy()

    # Build what we plot for this level
    if not small.empty:
        other_row = pd.DataFrame({
            "industry": ["Other"],
            "marketCap": [small["marketCap"].sum()],
            "pct": [small["marketCap"].sum() / total],
        })
        plot_df = pd.concat([big, other_row], ignore_index=True)
    else:
        plot_df = big.copy()

    plot_df = plot_df.sort_values("marketCap", ascending=False)

    sizes = plot_df["marketCap"].values
    labels = plot_df["industry"].values

    pretty_labels = [
        f"{ind}\n{format_money(val)} ({val/total:.1%})"
        for ind, val in zip(labels, sizes)
    ]


    # Color scale per level
    norm = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-12)
    colors = [plt.cm.Blues(0.4 + 0.6*x) for x in norm]

    # Plot
    plt.figure(figsize=(32, 16))
    plt.axis("off")

    squarify.plot(
        sizes=sizes,
        label=pretty_labels,
        color=colors,
        alpha=1.0,
        pad=True
    )

    title_suffix = " (no small left)" if small.empty else f" (drill into Other: {len(small)} industries)"
    plt.title(f"capital_flows/Market Cap Treemap by Industry — Level {level} (Total {format_money(total)})",fontsize=18)
    plt.tight_layout()
    plt.savefig(f"treemap_level_{level}_{pd.Timestamp.today().date()}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Stop if no more "small" to drill into
    if small.empty:
        break

    # Next level: treat the previous small set as the new universe
    current = small[["industry", "marketCap"]].copy().sort_values("marketCap", ascending=False).reset_index(drop=True)
    level += 1
