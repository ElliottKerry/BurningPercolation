"""Lattice percolation by threshold: fire-event counts per 0.25° cell, as in the illiteracy-percolation figure.

Each 0.25° cell (about 28 km) whose centre is in Brazil gets the number of MODIS Aqua fire events (2 km / 2 day
rule, cached by fire_percolation_expected.py --events) whose centre falls in it that year. For a threshold k,
a cell is occupied when count >= k. Occupied cells sharing an edge (rook neighbours) form a cluster. Raising k
switches cells off, so clusters shrink and split. Cluster size = number of cells.

Critical threshold k*: where the second-largest cluster peaks (the largest cluster splits in two), as marked by
the stars in the paper's figure. The occupancy p* = share of Brazil cells still on at k* is also reported: for
random site percolation on a square lattice the theory value is 0.593.

Null (the paper's inset): the same counts shuffled at random among Brazil's cells, REPS times; keeps the count
distribution, destroys the geography.

    python audit/percolation_lattice_threshold.py [year]     # detail year, default 2019
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import ndimage

from audit.fire_percolation_pilot import CACHE, GRID, INK, INK2, MUTED, OUT, SERIES, YEARS
from firms import config

RES = 0.25
REPS_YEAR, REPS_ALL = 20, 5
RANK_COLOURS = ["#6a3d9a", "#33a02c", "#e7298a", "#ff9f1c"]      # largest, 2nd, 3rd, 4th (as in the paper)


def brazil_cells():
    """Boolean mask of 0.25° cells whose centre is inside Brazil, with the lon/lat index offsets."""
    import geopandas as gpd
    import shapely
    shape = gpd.read_parquet(config.DATA_DIR / "cache" / "brazil_outline_5880.parquet").to_crs("EPSG:4326").union_all()
    x0, y0, x1, y1 = shape.bounds
    ix = np.arange(int(np.floor(x0 / RES)), int(np.ceil(x1 / RES)) + 1)
    iy = np.arange(int(np.floor(y0 / RES)), int(np.ceil(y1 / RES)) + 1)
    lon, lat = np.meshgrid(ix * RES, iy * RES)
    mask = shapely.contains_xy(shape, lon, lat)
    return mask, ix[0], iy[0]


def counts(year, mask, ix0, iy0):
    f = pd.read_parquet(CACHE / f"events_{year}_2km_2d.parquet")
    cx = np.floor(f.lon.to_numpy() / RES + 0.5).astype(int) - ix0
    cy = np.floor(f.lat.to_numpy() / RES + 0.5).astype(int) - iy0
    ok = (cx >= 0) & (cy >= 0) & (cx < mask.shape[1]) & (cy < mask.shape[0])
    g = np.zeros(mask.shape, int)
    np.add.at(g, (cy[ok], cx[ok]), 1)
    return np.where(mask, g, 0)


def sweep(g, mask, ks):
    """Largest and second-largest cluster size (cells) and occupancy for each threshold k."""
    rows = []
    n = mask.sum()
    for k in ks:
        occ = mask & (g >= k)
        lab, m = ndimage.label(occ)
        s = np.sort(np.bincount(lab.ravel())[1:])[::-1] if m else np.array([0])
        rows.append({"k": k, "occupancy": occ.sum() / n, "largest": s[0] if len(s) else 0,
                     "second": s[1] if len(s) > 1 else 0, "clusters": m})
    return pd.DataFrame(rows)


def critical(c):
    i = c.second.idxmax()
    drop = (c.largest.shift(1) - c.largest).idxmax()
    return {"k_star": int(c.k[i]), "p_star": float(c.occupancy[i]), "second_peak": int(c.second[i]),
            "k_biggest_drop": int(c.k[drop]), "biggest_drop_cells": int(c.largest.shift(1)[drop] - c.largest[drop])}


def shuffled(g, mask, rng):
    v = g[mask].copy()
    rng.shuffle(v)
    s = np.zeros_like(g)
    s[mask] = v
    return s


def main(year):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    rng = np.random.default_rng(42)
    mask, ix0, iy0 = brazil_cells()
    ncell = int(mask.sum())
    print(f"{ncell:,} Brazil cells")

    # ---- every year: real vs shuffled critical threshold
    rows = []
    for y in YEARS:
        g = counts(y, mask, ix0, iy0)
        ks = np.arange(0, int(np.quantile(g[mask], 0.995)) + 2)
        cr = critical(sweep(g, mask, ks))
        nl = [critical(sweep(shuffled(g, mask, rng), mask, ks)) for _ in range(REPS_ALL)]
        r = {"year": y, "events": int(g.sum()), "cells_with_fire": float((g[mask] > 0).mean()), **cr,
             "null_k_star_mid": float(np.median([x["k_star"] for x in nl])),
             "null_p_star_mid": float(np.median([x["p_star"] for x in nl])),
             "null_p_star_lo": min(x["p_star"] for x in nl), "null_p_star_hi": max(x["p_star"] for x in nl)}
        rows.append(r)
        print(f"{y}: k* {cr['k_star']} (p* {cr['p_star']:.2f}) vs shuffled k* {r['null_k_star_mid']:.0f} "
              f"(p* {r['null_p_star_mid']:.2f}); biggest split at k={cr['k_biggest_drop']} "
              f"({cr['biggest_drop_cells']} cells)", flush=True)
    yrs = pd.DataFrame(rows).set_index("year")
    yrs.to_csv(OUT / "lattice_threshold_years.csv")

    # ---- the detail year
    g = counts(year, mask, ix0, iy0)
    ks = np.arange(0, int(np.quantile(g[mask], 0.995)) + 2)
    c = sweep(g, mask, ks)
    nulls = [sweep(shuffled(g, mask, rng), mask, ks) for _ in range(REPS_YEAR)]
    cr = critical(c)
    (OUT / f"lattice_threshold_{year}.json").write_text(json.dumps(
        {"year": year, "cells": ncell, "critical": cr, "null_k_star": [critical(n)["k_star"] for n in nulls],
         "years": yrs.round(4).reset_index().to_dict("records")}, indent=1, default=float))

    fig, ax = plt.subplots(figsize=(9, 5))
    for col, colour, lab in (("largest", RANK_COLOURS[0], "Largest"), ("second", RANK_COLOURS[1], "Second largest")):
        lo = np.min([n[col] for n in nulls], axis=0)
        hi = np.max([n[col] for n in nulls], axis=0)
        ax.fill_between(ks, lo, hi, color=MUTED, alpha=0.25, lw=0)
        ax.plot(ks, np.median([n[col] for n in nulls], axis=0), color=MUTED, lw=1.5, ls="--")
        ax.plot(ks, c[col], color=colour, lw=2, marker="o", ms=3, label=f"{lab}, actual fire")
    ax.plot([], [], color=MUTED, ls="--", label=f"Counts shuffled among cells ({REPS_YEAR} runs, band = range)")
    ax.plot(cr["k_star"], c.largest[cr["k_star"]], marker="*", ms=14, color=INK, ls="none")
    ax.annotate(f"k* = {cr['k_star']} events per cell\n({cr['p_star']:.0%} of cells still on)",
                (cr["k_star"], c.largest[cr["k_star"]]), xytext=(12, 8), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("Threshold k: a cell is on if it has at least k fire events", color=INK2)
    ax.set_ylabel("Cluster size (cells)", color=INK2)
    ax.set_title(f"Lattice percolation of fire, Brazil {year}: 0.25° cells ({ncell:,}), MODIS Aqua fire events",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"lattice_threshold_curve_{year}.png", dpi=150)

    # maps at: k=1, just before the split, at the split, and a high threshold
    show = sorted({1, max(cr["k_star"] - 1, 1), cr["k_star"], int(np.quantile(g[mask][g[mask] > 0], 0.9))})
    biomes = gpd.read_file(config.RAW_DIR / "biomes" / "lm_bioma_250.shp").set_crs("EPSG:4674", allow_override=True)
    ext = ((ix0 - 0.5) * RES, (ix0 + mask.shape[1] - 0.5) * RES, (iy0 - 0.5) * RES, (iy0 + mask.shape[0] - 0.5) * RES)
    cmap = ListedColormap(["#ffffff", "#d9d7d0", "#5f5f5f"] + RANK_COLOURS)
    fig, axs = plt.subplots(1, len(show), figsize=(4.3 * len(show), 4.8))
    for a, k in zip(np.atleast_1d(axs), show):
        occ = mask & (g >= k)
        lab, m = ndimage.label(occ)
        size = np.bincount(lab.ravel())
        size[0] = 0
        top = np.argsort(size)[::-1][:4]
        img = np.where(mask, 1, 0)                                  # 0 outside, 1 off
        img = np.where(occ, 2, img)                                 # 2 = on, other clusters
        for r_, cl in enumerate(top):
            if size[cl]:
                img = np.where(lab == cl, 3 + r_, img)
        a.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=6, interpolation="nearest")
        biomes.boundary.plot(ax=a, color=INK2, lw=0.3)
        a.set_title(f"k = {k}: {occ.sum() / mask.sum():.0%} of cells on,\nlargest {size.max():,} cells, "
                    f"2nd {np.sort(size)[-2]:,}", fontsize=9, loc="left", color=INK)
        a.set_axis_off()
        a.set_aspect("equal")
    fig.suptitle(f"Fire clusters by threshold, {year}: purple largest, green 2nd, pink 3rd, orange 4th; "
                 "dark grey other clusters; light grey cells below the threshold", x=0.01, ha="left", fontsize=10,
                 color=INK)
    fig.tight_layout()
    fig.savefig(OUT / f"lattice_threshold_maps_{year}.png", dpi=150)

    # years: critical occupancy, real vs shuffled
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.fill_between(yrs.index, yrs.null_p_star_lo, yrs.null_p_star_hi, color=MUTED, alpha=0.25, lw=0)
    ax.plot(yrs.index, yrs.null_p_star_mid, color=MUTED, lw=2, ls="--", marker="o", ms=3,
            label=f"Counts shuffled among cells ({REPS_ALL} runs, band = range)")
    ax.plot(yrs.index, yrs.p_star, color=SERIES[0], lw=2, marker="o", ms=4, label="Actual fire")
    ax.axhline(0.593, color=INK2, lw=0.8, ls=":")
    ax.annotate("random-lattice theory, 0.593", (yrs.index[0], 0.593), xytext=(0, 4), textcoords="offset points",
                fontsize=8, color=INK2)
    ax.set_ylabel("Share of cells on at the split", color=INK2)
    ax.set_title("Critical point by year: share of Brazil's cells still on when the largest fire cluster splits",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "lattice_threshold_years.png", dpi=150)
    print(json.dumps(cr, indent=1))


if __name__ == "__main__":
    main(next((int(a) for a in sys.argv[1:] if a.isdigit()), 2019))
