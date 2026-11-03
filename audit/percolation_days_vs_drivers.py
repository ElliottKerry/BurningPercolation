"""Is the 2010-2015 timing effect real? Timing effect against deforestation and the Amazon clustering ratio.

Timing effect (lag): join-up day of actual fire minus the median of 10 shuffled-date runs, scaled distance,
from percolation_days.csv. Its range = join day minus the latest / earliest shuffled run.
Deforestation: PRODES Legal Amazon yearly rate (Aug-Jul). Amazon clustering ratio: fire / random half-joined
distance for the Legal Amazon (percolation_legal_amazon.csv). Whole-Brazil ratio added for comparison.
Correlations on levels (Spearman, Pearson) and on year-to-year changes (removes shared slow trends).

    python audit/percolation_days_vs_drivers.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from audit.fire_percolation_pilot import GRID, INK, INK2, MUTED, OUT, SERIES
from audit.percolation_vs_deforestation import prodes


def corr(x, y):
    sp, pp = stats.spearmanr(x, y)
    pe, pq = stats.pearsonr(x, y)
    dx, dy = x.diff().dropna(), y.diff().dropna()
    dc, dp = stats.pearsonr(dx, dy)
    return {"spearman": round(float(sp), 3), "spearman_p": round(float(pp), 4), "pearson": round(float(pe), 3),
            "pearson_p": round(float(pq), 4), "changes_pearson": round(float(dc), 3), "changes_p": round(float(dp), 4)}


def main():
    import matplotlib.pyplot as plt
    d = pd.read_csv(OUT / "percolation_days.csv").set_index("year")
    la = pd.read_csv(OUT / "percolation_legal_amazon.csv").set_index("year")
    br = pd.read_csv(OUT / "percolation_by_year.csv").set_index("year")
    df = pd.DataFrame({"lag": d.scaled_lag, "lag_lo": d.scaled_join_day - d.scaled_null_hi,
                       "lag_hi": d.scaled_join_day - d.scaled_null_lo, "lag_fixed": d.fixed_lag,
                       "events": d.events, "deforestation_km2": prodes(), "amazon_ratio": la.ratio,
                       "brazil_ratio": br.ratio_events}).dropna(subset=["lag"])
    df.to_csv(OUT / "percolation_days_vs_drivers.csv")
    res = {"lag_vs_" + c: corr(df[c], df.lag) for c in ("deforestation_km2", "amazon_ratio", "brazil_ratio", "events")}
    res["lag_fixed_vs_deforestation_km2"] = corr(df.deforestation_km2, df.lag_fixed)
    res["lag_fixed_vs_amazon_ratio"] = corr(df.amazon_ratio, df.lag_fixed)
    w = df.index.isin(range(2010, 2016))
    res["lag_mean_2010_2015"] = float(df.lag[w].mean())
    res["lag_mean_other_years"] = float(df.lag[~w].mean())
    res["mannwhitney_2010_2015_vs_other_p"] = float(stats.mannwhitneyu(df.lag[w], df.lag[~w]).pvalue)
    res["years_whose_whole_range_is_above_zero"] = [int(y) for y in df.index[df.lag_lo > 0]]
    (OUT / "percolation_days_vs_drivers.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    a = axs[0]
    a.axvspan(2009.5, 2015.5, color=MUTED, alpha=0.12, lw=0)
    a.bar(df.index, df.lag, color=SERIES[0], width=0.7, label="Timing effect (median of shuffled runs)")
    a.errorbar(df.index, df.lag, yerr=[df.lag - df.lag_lo, df.lag_hi - df.lag], fmt="none", ecolor=INK2, lw=0.9,
               capsize=2, label="Range over 10 shuffled runs")
    a.axhline(0, color=INK2, lw=0.8)
    a.set_ylabel("Days later than shuffled dates", color=INK2)
    a.set_title("Timing effect by year (shaded 2010–2015)", loc="left", fontsize=10, color=INK)
    a.legend(frameon=False, fontsize=8.5, loc="upper left")
    for a, col, xlab, scale in ((axs[1], "deforestation_km2", "Deforestation, Legal Amazon (thousand km², PRODES)", 1000),
                                (axs[2], "amazon_ratio", "Amazon clustering ratio (fire / random)", 1)):
        x = df[col] / scale
        a.errorbar(x, df.lag, yerr=[df.lag - df.lag_lo, df.lag_hi - df.lag], fmt="none", ecolor=MUTED, lw=0.7)
        a.scatter(x[~w], df.lag[~w], s=34, color=SERIES[0], zorder=3, label="Other years")
        a.scatter(x[w], df.lag[w], s=34, color=SERIES[1], zorder=3, label="2010–2015")
        for y, xv, lv in zip(df.index, x, df.lag):
            a.annotate(str(y)[2:], (xv, lv), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=INK2)
        r = res[f"lag_vs_{col}"]
        a.set_title(f"Spearman {r['spearman']:+.2f} (p = {r['spearman_p']:.3f})\n"
                    f"year-to-year changes: r = {r['changes_pearson']:+.2f} (p = {r['changes_p']:.2f})",
                    loc="left", fontsize=9, color=INK)
        a.axhline(0, color=INK2, lw=0.8)
        a.set_xlabel(xlab, color=INK2)
        a.legend(frameon=False, fontsize=8.5)
    axs[1].set_ylabel("Timing effect (days)", color=INK2)
    for a in axs:
        a.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Is the timing effect linked to deforestation or Amazon clustering? 2003–2024", x=0.01, ha="left",
                 fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_days_vs_drivers.png", dpi=150)


if __name__ == "__main__":
    main()
