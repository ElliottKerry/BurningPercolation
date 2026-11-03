"""The three connectivity periods on the map: one typical year per period, plus every year's curve by period.

Typical year = the year whose clustering ratio is closest to its period mean (percolation_change.json).
Maps: that year's fire events joined at the distance where the SAME NUMBER of random points would be half joined
(random d50 from the calibration in percolation_measure.py), so each map is at a comparable, count-adjusted
scale. If fire were spread like random, the largest cluster (purple) would hold about half the fires.
Bottom: every year's curve in units of typical spacing, coloured by period, with period means.

    python audit/percolation_periods_maps.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from audit.fire_percolation_pilot import ALBERS, GRID, INK, INK2, MUTED, OUT, labels_at, mst_edges, project
from audit.percolation_lattice_threshold import RANK_COLOURS
from audit.percolation_measure import REL, random_d50
from audit.percolation_stability import events_with_dates
from firms import config

PERIODS = {"2003–2006": range(2003, 2007), "2007–2015": range(2007, 2016), "2016–2024": range(2016, 2025)}
PCOL = {"2003–2006": "#2a78d6", "2007–2015": "#e8590c", "2016–2024": "#1b8a5a"}


def main():
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from pyproj import Transformer
    recs = {r["year"]: r for r in json.loads((OUT / "percolation_measure_years.json").read_text())}
    cal = json.loads((OUT / "percolation_measure_calibration.json").read_text())
    ch = json.loads((OUT / "percolation_change.json").read_text())["ratio_anywhere"]["period_means"]
    means = {p["years"].replace("-", "–"): p["mean"] for p in ch}
    biomes = gpd.read_file(config.RAW_DIR / "biomes" / "lm_bioma_250.shp").set_crs("EPSG:4674", allow_override=True)
    back = Transformer.from_crs(ALBERS, "EPSG:4326", always_xy=True)

    fig = plt.figure(figsize=(17, 10.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.12)
    for c, (name, yrs) in enumerate(PERIODS.items()):
        y = min(yrs, key=lambda q: abs(recs[q]["ratio_anywhere"] - means[name]))
        ev = events_with_dates(y)
        ev = ev[ev.start.dt.year == y]
        xy = np.unique(np.round(project(ev.lon.to_numpy(), ev.lat.to_numpy()), 4), axis=0)
        n = len(xy)
        d = random_d50(cal, "anywhere", n)
        lab = labels_at(n, *mst_edges(xy), d)
        size = np.bincount(lab)
        top = np.argsort(size)[::-1][:4]
        lon, lat = back.transform(xy[:, 0] * 1000, xy[:, 1] * 1000)
        a = fig.add_subplot(gs[0, c])
        biomes.boundary.plot(ax=a, color=INK2, lw=0.3)
        a.scatter(lon, lat, s=0.05, color="#cfcdc5", rasterized=True)
        for r in range(3, -1, -1):
            m = lab == top[r]
            a.scatter(lon[m], lat[m], s=0.05, color=RANK_COLOURS[r], rasterized=True)
        a.set_xlim(-74.5, -34)
        a.set_ylim(-34, 5.8)
        a.set_aspect("equal")
        a.set_axis_off()
        rec = recs[y]
        a.set_title(f"{name} (period mean ratio {means[name]:.2f})\nTypical year {y}: {rec['events']:,} fires, ratio "
                    f"{rec['ratio_anywhere']:.2f}\nJoined at {d:.1f} km: largest cluster {size.max() / n:.0%}, "
                    f"2nd {np.sort(size)[-2] / n:.0%}", loc="left", fontsize=10.5, color=PCOL[name])

    a = fig.add_subplot(gs[1, :2])
    rand = np.array(cal["anywhere"]["rel_curves"])
    a.fill_between(REL, rand.min(0), rand.max(0), color=MUTED, alpha=0.3, lw=0, label="Random points (anywhere)")
    for name, yrs in PERIODS.items():
        cs = np.array([recs[q]["curve_rel"] for q in yrs])
        for cv in cs:
            a.plot(REL, cv, color=PCOL[name], lw=0.7, alpha=0.35)
        a.plot(REL, cs.mean(0), color=PCOL[name], lw=2.6, label=f"{name}: mean of {len(cs)} years")
    a.axhline(0.5, color=MUTED, lw=0.6)
    a.set_xscale("log")
    a.set_xlim(0.25, 8)
    a.set_xticks([0.25, 0.5, 1, 2, 4, 8], ["¼", "½", "1", "2", "4", "8"])
    a.minorticks_off()
    a.set_xlabel("Joining distance ÷ typical spacing between fires (removes the fire count)", color=INK2)
    a.set_ylabel("Share of fires in the largest cluster", color=INK2)
    a.set_title("Every year's percolation curve, coloured by period. 2007–2015 joins up earlier (closer to random) "
                "but more gradually", loc="left", fontsize=10.5, color=INK)
    a.legend(frameon=False, fontsize=9.5, loc="upper left")
    a.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

    a = fig.add_subplot(gs[1, 2])
    yrs = sorted(recs)
    for name, rng_ in PERIODS.items():
        ys = [q for q in yrs if q in rng_]
        r = np.array([recs[q]["ratio_anywhere"] for q in ys])
        se = np.array([recs[q]["ratio_anywhere_se"] for q in ys])
        a.errorbar(ys, r, yerr=1.96 * se, fmt="o", color=PCOL[name], ms=4, lw=1, capsize=2)
        a.hlines(means[name], min(ys) - 0.4, max(ys) + 0.4, color=PCOL[name], lw=2.4)
    a.axhline(1, color=INK2, lw=0.8)
    a.set_ylabel("Clustering ratio (fire ÷ random)", color=INK2)
    a.set_title("Ratio by year with 95% intervals\nand period means", loc="left", fontsize=10.5, color=INK)
    a.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    fig.suptitle("The three periods of fire connectivity in Brazil, 2003–2024 (MODIS Aqua fire events). Maps: purple "
                 "largest cluster, green 2nd, pink 3rd, orange 4th, grey the rest", x=0.01, ha="left", fontsize=12,
                 color=INK)
    fig.savefig(OUT / "percolation_periods.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
