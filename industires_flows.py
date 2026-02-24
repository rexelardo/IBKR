import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import squarify
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm


def plot_pct_change_treemap(file_old='capital_flows/tickers_2026-02-16.csv',
                            file_new='capital_flows/tickers_2026-02-23.csv',
                            top_n=None,
                            min_size=0.1,
                            save_to=None):
    """Plot a treemap where box size = |% change| and color = signed % change.

    - sizes: absolute percent change between snapshots (as percentages)
    - color: percent change (negative red -> positive green)

    Args:
        file_old: path to earlier CSV (must contain 'industry' and 'marketCap')
        file_new: path to later CSV
        top_n: if set, show only top N industries by absolute percent change
        min_size: minimum box size (percent) to ensure visibility
        save_to: optional path to save the figure
    Returns:
        DataFrame with industries, pct_change, and size used for plotting
    """
    df_old = pd.read_csv(file_old)
    df_new = pd.read_csv(file_new)

    old = df_old.groupby('industry', dropna=False)['marketCap'].sum()
    new = df_new.groupby('industry', dropna=False)['marketCap'].sum()

    industries = sorted(set(old.index).union(new.index))
    old = old.reindex(industries).fillna(0)
    new = new.reindex(industries).fillna(0)

    # percent change: handle old==0 by assigning 100% if new>0, 0 if both 0
    pct = (new - old) / old.replace(0, np.nan) * 100
    pct = pct.fillna(0)
    # where old==0 and new>0, set to 100% (interpretation choice)
    pct[(old == 0) & (new > 0)] = 100.0

    df_stats = pd.DataFrame({
        'industry': industries,
        'old_mc': old.values,
        'new_mc': new.values,
        'pct_change': pct.values,
    })

    # size = absolute percent change (ensure minimum size)
    df_stats['size'] = np.maximum(np.abs(df_stats['pct_change']), min_size)

    # optionally keep only top N by size
    df_stats = df_stats.sort_values('size', ascending=False)
    if top_n:
        df_stats = df_stats.head(top_n)

    labels = [f"{row['industry']}\n{row['pct_change']:+.1f}%" for _, row in df_stats.iterrows()]
    sizes = df_stats['size'].tolist()

    # color mapping (use pyplot.get_cmap to avoid deprecation)
    cmap = plt.get_cmap('RdYlGn')
    norm = TwoSlopeNorm(vmin=df_stats['pct_change'].min(), vcenter=0, vmax=df_stats['pct_change'].max())
    colors = [cmap(norm(v)) for v in df_stats['pct_change'].values]

    # create Figure and Axes so we can attach the colorbar cleanly
    fig, ax = plt.subplots(figsize=(12, 8))
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.9, pad=True, ax=ax)
    ax.axis('off')

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(df_stats['pct_change'].values)
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.6, pad=0.02)
    cbar.set_label('Percent change (new vs old)')

    plt.title('Industry percent-change treemap (size = |% change|, color = sign)')

    if save_to:
        plt.savefig(save_to, dpi=150, bbox_inches='tight')

    plt.show()
    return df_stats


if __name__ == '__main__':
    plot_pct_change_treemap()


# import pandas as pd

# df = pd.read_csv('capital_flows/tickers_2026-02-16.csv')
# df2 = pd.read_csv('capital_flows/tickers_2026-02-23.csv')

# first = df.groupby('industry').sum()['marketCap']
# second = df2.groupby('industry').sum()['marketCap']
# stats = ((first-second)/first)*100


# import squarify


