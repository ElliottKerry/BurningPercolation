"""Percolation threshold of frontier fire (near recent clearing) vs background fire, Amazon biome 2008-2023.

Groups as in percolation_clearing.py: near = within 1 km of a PRODES patch of PRODES year y or y+1.
Per group and year (distinct event positions, Albers km, Euclidean MST sweep):
  d50     distance where the largest cluster first holds half the group's events
  d_star  distance where the second-largest cluster is largest (classic percolation transition signal)
  random  d50 of the same number of random points inside the Amazon biome (calibrated); ratio = d50 / random
Also the typical spacing sqrt(biome area / n), so thresholds can be read in spacing units.

    python audit/percolation_frontier_threshold.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from audit.fire_percolation_pilot import GRID, INK, INK2, MUTED, OUT, mst_edges, project
from audit.percolation_stability import events_with_dates
from firms import config

YEARS = range(2008, 2024)
NEAR_M = 1000
GROUPS = {"frontier": ("Frontier fire (near recent clearing)", "#e8590c"),
          "background": ("Background fire (elsewhere)", "#2a78d6")}


def thresholds(xy):
    """d50 and the distance where the second-largest cluster peaks."""
    u = np.unique(np.round(xy, 4), axis=0)
    n = len(u)
    i, j, w = mst_edges(u)
    parent, size = np.arange(n), np.ones(n, np.int64)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    sizes = {}                                  # root -> size, for clusters of 2+
    big, d50, best2, d_star = 1, None, 0, None
    import heapq
    for k in range(len(w)):
        a, b = find(i[k]), find(j[k])
        if a == b:
            continue
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
        sizes.pop(b, None)
        sizes[a] = size[a]
        big = max(big, size[a])
        if d50 is None and big >= n / 2:
            d50 = float(w[k])
        if k % 50 == 0 or big >= n / 2:
            top2 = heapq.nlargest(2, sizes.values())
            second = top2[1] if len(top2) > 1 else 0
            if second > best2 and big < 0.9 * n:
                best2, d_star = second, float(w[k])
    return d50, d_star, best2 / n, n


def main():
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import shapely
    from audit.percolation_early_warning_regional import biome_shapes
    amazon = biome_shapes()["Amazon"]
    shapely.prepare(amazon)
    a, b = json.loads((OUT / "percolation_early_warning_regional_cal.json").read_text())["Amazon"]
    rd50 = lambda n: float(np.exp(a + b * np.log(n)))
    area = amazon.area / 1e6
    pr = gpd.read_parquet(config.DATA_DIR / "cache" / "prodes_amazon_2008_2024.parquet")
    pr = pr[pr.lon.notna() & ~pr.geometry.is_empty]
    rows = []
    for y in YEARS:
        ev = events_with_dates(y)
        ev = ev[ev.start.dt.year == y]
        xy = project(ev.lon.to_numpy(), ev.lat.to_numpy())
        xy = xy[shapely.contains_xy(amazon, xy[:, 0] * 1000, xy[:, 1] * 1000)]
        tree = shapely.STRtree(pr.geometry[pr.year.isin([y, y + 1])].to_numpy())
        near = np.zeros(len(xy), bool)
        near[np.unique(tree.query(shapely.points(xy * 1000), predicate="dwithin", distance=NEAR_M)[0])] = True
        for g, m in (("frontier", near), ("background", ~near), ("all", np.ones(len(xy), bool))):
            d50, ds, s2, n = thresholds(xy[m])
            rows.append({"year": y, "group": g, "events": n, "d50_km": d50, "d_star_km": ds, "second_peak_share": s2,
                         "random_d50_km": rd50(n), "ratio": d50 / rd50(n), "spacing_km": float(np.sqrt(area / n))})
        print(y, [(r["group"], round(r["d50_km"], 1), round(r["d_star_km"] or np.nan, 1), round(r["ratio"], 2))
                  for r in rows[-3:]], flush=True)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(OUT / "percolation_frontier_threshold.csv", index=False)
    summ = df.groupby("group")[["events", "d50_km", "d_star_km", "random_d50_km", "ratio", "spacing_km"]].median().round(2)
    rng = df.groupby("group")[["d50_km", "d_star_km"]].agg(["min", "max"]).round(1)
    print(summ.to_string())
    print(rng.to_string())
    (OUT / "percolation_frontier_threshold.json").write_text(json.dumps(
        {"median_by_group": summ.to_dict(orient="index"), "range_by_group": {g: {f"{c[0]}_{c[1]}": v for c, v in r.items()}
                                                                                for g, r in rng.iterrows()}}, indent=1))

    fig, axs = plt.subplots(1, 2, figsize=(15, 4.8))
    for g, (lab, col) in GROUPS.items():
        d = df[df.group == g]
        axs[0].plot(d.year, d.d50_km, marker="o", ms=4, lw=1.8, color=col, label=f"{lab}: d50")
        axs[0].plot(d.year, d.random_d50_km, lw=1.2, ls="--", color=col, alpha=0.6, label=f"Random points, same number")
        axs[1].plot(d.year, d.ratio, marker="o", ms=4, lw=1.8, color=col, label=lab)
    axs[0].set_ylabel("Half-joined distance d50 (km)", color=INK2)
    axs[0].set_title("Percolation threshold (d50) in km", loc="left", fontsize=10.5, color=INK)
    axs[1].axhline(1, color=INK2, lw=0.8)
    axs[1].set_ylabel("d50 ÷ random d50", color=INK2)
    axs[1].set_title("Threshold relative to random points with the same count", loc="left", fontsize=10.5, color=INK)
    for ax in axs:
        ax.legend(frameon=False, fontsize=8.5)
        ax.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Percolation threshold of frontier vs background fire, Amazon biome 2008–2023 (MODIS Aqua; PRODES)",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_frontier_threshold.png", dpi=140)


if __name__ == "__main__":
    main()
