"""What was the climate doing in the years when fire joined up late (2010-2015), and does it track the delay?

Delay: timing effect from percolation_days.csv (scaled distance; actual join day minus median of 10 shuffled runs).
Climate: ERA5 monthly (era5_brazil.py) on the 0.25 deg cells, averaged with fixed long-run fire weights (each cell's
share of all fire events 2003-2024), dry season Jul-Oct. Three areas: all Brazil, Legal Amazon, rest of Brazil.
Tests per variable: Spearman with the delay (levels), Pearson of year-to-year changes, and 2010-2015 vs other years
(difference in means, Mann-Whitney). 15 variable-area pairs, so expect about one p < 0.05 by chance.

    python audit/percolation_days_climate.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from audit.fire_percolation_pilot import ALBERS, GRID, INK, INK2, MUTED, OUT, SERIES, YEARS
from audit.percolation_climate import CLIMATE, DRY
from audit.percolation_lattice_threshold import RES, brazil_cells, counts
from audit.percolation_legal_amazon import legal_amazon
from firms import config

AREAS = {"brazil": "All Brazil", "amazon": "Legal Amazon", "rest": "Rest of Brazil"}


def dry_season_by_area():
    import shapely
    from pyproj import Transformer
    mask, ix0, iy0 = brazil_cells()
    total = sum(counts(y, mask, ix0, iy0) for y in YEARS)
    yy, xx = np.nonzero(mask)
    lon, lat = (xx + ix0) * RES, (yy + iy0) * RES
    x, y = Transformer.from_crs("EPSG:4326", ALBERS, always_xy=True).transform(lon, lat)
    inside = shapely.contains_xy(legal_amazon(), x, y)
    cells = pd.DataFrame({"ix": xx + ix0, "iy": yy + iy0, "w": total[yy, xx].astype(float), "amazon": inside})
    c = pd.read_parquet(config.DATA_DIR / "cache" / "era5_brazil_monthly.parquet")
    c = c[c.month.dt.month.isin(DRY)].merge(cells)
    c["year"] = c.month.dt.year
    out = {}
    for area, sel in (("brazil", c), ("amazon", c[c.amazon]), ("rest", c[~c.amazon])):
        out[area] = sel.groupby("year").apply(
            lambda g: pd.Series({v: np.average(g[v], weights=g.w) for v in CLIMATE}), include_groups=False)
    return out


def main():
    import matplotlib.pyplot as plt
    lag = pd.read_csv(OUT / "percolation_days.csv").set_index("year").scaled_lag
    clim = dry_season_by_area()
    w = lag.index.isin(range(2010, 2016))
    res, rows = {}, []
    for area, df in clim.items():
        for v in CLIMATE:
            x = df[v].reindex(lag.index)
            sp, pp = stats.spearmanr(x, lag)
            dc, dp = stats.pearsonr(x.diff().dropna(), lag.diff().dropna())
            mw = stats.mannwhitneyu(x[w], x[~w]).pvalue
            r = {"area": area, "variable": v, "spearman_with_delay": round(float(sp), 2),
                 "spearman_p": round(float(pp), 3), "changes_r": round(float(dc), 2), "changes_p": round(float(dp), 3),
                 "mean_2010_2015": round(float(x[w].mean()), 3), "mean_other": round(float(x[~w].mean()), 3),
                 "sd_all_years": round(float(x.std()), 3), "mannwhitney_p": round(float(mw), 3)}
            rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "percolation_days_climate.csv", index=False)
    pd.concat(clim, names=["area"]).to_csv(OUT / "percolation_days_climate_dry_season.csv")
    (OUT / "percolation_days_climate.json").write_text(json.dumps(rows, indent=1))
    print(tab.to_string())

    fig, axs = plt.subplots(len(AREAS), len(CLIMATE), figsize=(18, 10), sharex=True)
    for i, (area, name) in enumerate(AREAS.items()):
        df = clim[area]
        for j, (v, vlab) in enumerate(CLIMATE.items()):
            a = axs[i, j]
            a.axvspan(2009.5, 2015.5, color=MUTED, alpha=0.12, lw=0)
            a.plot(df.index, df[v], color=MUTED, lw=1)
            a.scatter(df.index[~w], df[v][~w], s=22, color=SERIES[0], zorder=3, label="Other years")
            a.scatter(df.index[w], df[v][w], s=22, color=SERIES[1], zorder=3, label="2010–2015 (late join-up)")
            a.axhline(df[v][~w].mean(), color=SERIES[0], lw=0.8, ls=":")
            a.axhline(df[v][w].mean(), color=SERIES[1], lw=0.8, ls=":")
            r = tab[(tab.area == area) & (tab.variable == v)].iloc[0]
            strong = min(r.spearman_p, r.mannwhitney_p) < 0.05
            a.set_title(f"vs delay: Spearman {r.spearman_with_delay:+.2f} (p {r.spearman_p:.2f})\n"
                        f"2010–15 vs other: p {r.mannwhitney_p:.2f}", loc="left", fontsize=8.5,
                        color=INK if strong else INK2, weight="bold" if strong else "normal")
            if i == 0:
                a.text(0, 1.28, vlab + ", Jul–Oct", transform=a.transAxes, fontsize=10, color=INK, weight="bold")
            if j == 0:
                a.set_ylabel(name, color=INK, fontsize=10.5)
            a.grid(color=GRID, lw=0.6)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
    axs[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Dry-season climate where fire usually burns, 2003–2024. Shaded = 2010–2015, the years fire joined up "
                 "late. Dotted lines = mean of each group. Bold = p < 0.05 (15 tests per row of results, so about one "
                 "expected by chance)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "percolation_days_climate.png", dpi=130)


if __name__ == "__main__":
    main()
