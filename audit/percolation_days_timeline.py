"""Timeline: join-up delay, Amazon critical point and deforestation on one time axis (stacked panels, no dual axis).

Panels: (1) delay (scaled distance, percolation_days.csv) with range over 10 shuffled runs; (2) Legal Amazon critical
distance d50 for fire and for random points (percolation_legal_amazon.csv); (3) their ratio; (4) PRODES deforestation.
Shading: 2010-2015.

    python audit/percolation_days_timeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from scipy import stats

from audit.fire_percolation_pilot import GRID, INK, INK2, MUTED, OUT, SERIES


def main():
    import matplotlib.pyplot as plt
    d = pd.read_csv(OUT / "percolation_days_vs_drivers.csv", index_col=0)
    la = pd.read_csv(OUT / "percolation_legal_amazon.csv").set_index("year")
    yrs = d.index
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    a = axs[0]
    a.bar(yrs, d.lag, color=SERIES[0], width=0.7)
    a.errorbar(yrs, d.lag, yerr=[d.lag - d.lag_lo, d.lag_hi - d.lag], fmt="none", ecolor=INK2, lw=0.8, capsize=2)
    a.axhline(0, color=INK2, lw=0.8)
    a.set_ylabel("Days", color=INK2)
    a.set_title("1. Join-up delay: days later than shuffled dates (bars = range of 10 shuffled runs)", loc="left",
                fontsize=10, color=INK)
    a = axs[1]
    a.plot(yrs, la.fire_d50_km.reindex(yrs), color=SERIES[0], lw=2, marker="o", ms=4, label="Actual fire")
    a.fill_between(yrs, la.random_d50_lo.reindex(yrs), la.random_d50_hi.reindex(yrs), color=MUTED, alpha=0.3, lw=0)
    a.plot(yrs, la.random_d50_mid.reindex(yrs), color=MUTED, lw=1.5, ls="--", label="Random points, same number")
    r, p = stats.spearmanr(yrs, la.fire_d50_km.reindex(yrs))
    a.set_ylabel("km", color=INK2)
    a.set_title(f"2. Legal Amazon critical point d50 (distance at which half the fires join). Rises over time "
                f"(Spearman with year {r:+.2f}, p {p:.2f})\n    because there are fewer fires, so random points rise too",
                loc="left", fontsize=10, color=INK)
    a.legend(frameon=False, fontsize=8.5, loc="upper left")
    a = axs[2]
    a.plot(yrs, d.amazon_ratio, color=SERIES[0], lw=2, marker="o", ms=4)
    a.axhline(1, color=INK2, lw=0.8)
    r1, p1 = stats.spearmanr(d.amazon_ratio, d.lag)
    a.set_ylabel("Fire / random", color=INK2)
    a.set_title(f"3. Amazon critical point adjusted for fire count (fire / random). 1 = no more clustered than random"
                f"\n    Dips to ~1 in 2007–2015. With the delay: Spearman {r1:+.2f} (p {p1:.3f})", loc="left",
                fontsize=10, color=INK)
    a = axs[3]
    a.bar(yrs, d.deforestation_km2 / 1000, color=SERIES[3], width=0.7)
    r2, p2 = stats.spearmanr(d.deforestation_km2, d.lag)
    a.set_ylabel("Thousand km²", color=INK2)
    a.set_title(f"4. Deforestation, Legal Amazon (PRODES). With the delay: Spearman {r2:+.2f} (p {p2:.3f})",
                loc="left", fontsize=10, color=INK)
    for a in axs:
        a.axvspan(2009.5, 2015.5, color=MUTED, alpha=0.12, lw=0)
        a.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Join-up delay, Amazon critical point and deforestation, 2003–2024 (shaded 2010–2015)", x=0.01,
                 ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_days_timeline.png", dpi=140)


if __name__ == "__main__":
    main()
