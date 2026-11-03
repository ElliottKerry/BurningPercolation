"""Q3: does spatial structure carry information beyond fire counts? Years with about the same number of fire
events but different percolation curves, side by side on identical thresholds and scales.

Years: 2014 (88,841 events) and 2019 (88,379), almost the same count, plus 2011 (81,855). Distance-method
fire / random ratio: 1.08, 1.35 and 0.99.

Figure 1 (maps), one row per year, same scales in every row:
  events per 0.25 deg cell (one log colour scale) | distance clusters at the same d | lattice clusters at the same k
  | one same-cell randomisation at the same d (each event moved to a random spot inside its own 28 km cell)
Figure 2 (curves): each year against its two spatially constrained randomisations (same 28 km cell; where fire
  usually is), REPS runs each, on one shared axis.

    python audit/percolation_similar_years.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if "--events" not in sys.argv:
    sys.argv.append("--events")

import numpy as np
from scipy import ndimage

from audit.fire_percolation_expected import Sampler, curve_of, observed
from audit.fire_percolation_pilot import ALBERS, GRID, INK, INK2, MUTED, OUT, SERIES, labels_at, mst_edges, summarise
from audit.percolation_lattice_threshold import RANK_COLOURS, RES, brazil_cells, counts
from firms import config

YEARS = [2014, 2019, 2011]
D_KM = 14            # between the d50 of the three years (12.2-16.0 km)
K = 5                # events per cell
REPS = 10
NULL_COLOUR = {"samecell": SERIES[2], "usual": SERIES[1]}
NULL_NAME = {"samecell": "Random: same 28 km cell", "usual": "Random: where fire usually is"}


def clusters_scatter(a, pts, back, biomes, title):
    pts = np.unique(np.round(pts, 4), axis=0)
    lab = labels_at(len(pts), *mst_edges(pts), D_KM)
    size = np.bincount(lab)
    top = np.argsort(size)[::-1][:4]
    lon, lat = back.transform(pts[:, 0] * 1000, pts[:, 1] * 1000)
    a.scatter(lon, lat, s=0.06, color="#cfcdc5", rasterized=True)
    for r, cl in enumerate(top):
        m = lab == cl
        a.scatter(lon[m], lat[m], s=0.06, color=RANK_COLOURS[r], rasterized=True)
    biomes.boundary.plot(ax=a, color=INK2, lw=0.3)
    a.set_title(f"{title}\nlargest {size.max() / len(pts):.0%}, 2nd {np.sort(size)[-2] / len(pts):.0%} of events",
                loc="left", fontsize=9, color=INK)
    return size.max() / len(pts)


def main():
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from pyproj import Transformer
    rng = np.random.default_rng(314)
    back = Transformer.from_crs(ALBERS, "EPSG:4326", always_xy=True)
    biomes = gpd.read_file(config.RAW_DIR / "biomes" / "lm_bioma_250.shp").set_crs("EPSG:4674", allow_override=True)
    mask, ix0, iy0 = brazil_cells()
    ext = ((ix0 - 0.5) * RES, (ix0 + mask.shape[1] - 0.5) * RES, (iy0 - 0.5) * RES, (iy0 + mask.shape[0] - 0.5) * RES)
    S = Sampler()
    data = {y: observed(y) for y in YEARS}

    # ---- figure 1: maps
    fig, axs = plt.subplots(len(YEARS), 4, figsize=(18, 4.7 * len(YEARS)))
    lattice_cmap = ListedColormap(["#ffffff", "#e4e2db", "#7a7a7a"] + RANK_COLOURS)
    summary = {}
    for r, y in enumerate(YEARS):
        lon, lat, xy = data[y]
        g = counts(y, mask, ix0, iy0)
        a = axs[r, 0]
        im = a.imshow(np.log10(np.where(g > 0, g, np.nan)), origin="lower", extent=ext, cmap="magma_r", vmin=0,
                      vmax=2.5, interpolation="nearest")
        biomes.boundary.plot(ax=a, color=INK2, lw=0.3)
        a.set_title(f"Events per cell ({len(xy):,} events)\n{(g[mask] > 0).mean():.0%} of cells have fire",
                    loc="left", fontsize=9, color=INK)
        a.text(-0.06, 0.5, str(y), transform=a.transAxes, fontsize=16, weight="bold", color=INK, rotation=90,
               va="center", ha="right")
        big = clusters_scatter(axs[r, 1], xy, back, biomes, f"Actual fire, clusters at d = {D_KM} km")
        occ = mask & (g >= K)
        lab, _ = ndimage.label(occ)
        size = np.bincount(lab.ravel())
        size[0] = 0
        img = np.where(occ, 2, np.where(mask, 1, 0))
        for k, cl in enumerate(np.argsort(size)[::-1][:4]):
            img = np.where(lab == cl, 3 + k, img)
        a = axs[r, 2]
        a.imshow(img, origin="lower", extent=ext, cmap=lattice_cmap, vmin=0, vmax=6, interpolation="nearest")
        biomes.boundary.plot(ax=a, color=INK2, lw=0.3)
        a.set_title(f"Actual fire, cells with at least {K} events\n{occ.sum() / mask.sum():.0%} of cells on, largest "
                    f"{size.max():,} cells, 2nd {np.sort(size)[-2]:,}", loc="left", fontsize=9, color=INK)
        nbig = clusters_scatter(axs[r, 3], S.draw("samecell", lon, lat, rng), back, biomes,
                                f"Random: same 28 km cell, d = {D_KM} km")
        summary[y] = {"events": len(xy), "largest_share_at_d": round(float(big), 3),
                      "samecell_largest_share_at_d": round(float(nbig), 3),
                      "lattice_on_share_at_k": round(float(occ.sum() / mask.sum()), 3),
                      "lattice_largest_cells_at_k": int(size.max())}
        for a in axs[r]:
            a.set_axis_off()
            a.set_aspect("equal")
    cb = fig.colorbar(im, ax=axs[-1, 0], shrink=0.8, pad=0.02, location="bottom")
    cb.set_ticks([0, 1, 2], labels=["1", "10", "100"])
    cb.set_label("Events per 0.25° cell", color=INK2)
    fig.suptitle(f"Similar fire totals, different structure: {', '.join(map(str, YEARS))}. Same scales in every row. "
                 "Cluster colours: purple largest, green 2nd, pink 3rd, orange 4th, grey the rest",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.savefig(OUT / "similar_years_maps.png", dpi=130, bbox_inches="tight")

    # ---- figure 2: curves against the spatially constrained randomisations
    fig, axs = plt.subplots(1, len(YEARS), figsize=(16, 4.6), sharey=True)
    for a, y in zip(axs, YEARS):
        lon, lat, xy = data[y]
        c = curve_of(xy)
        obs = summarise(c)["d50_km"]
        summary[y]["d50_km"] = obs
        for kind in ("samecell", "usual"):
            runs = [curve_of(S.draw(kind, lon, lat, rng)) for _ in range(REPS)]
            ys = np.array([x.largest for x in runs])
            a.fill_between(c.d_km, ys.min(0), ys.max(0), color=NULL_COLOUR[kind], alpha=0.2, lw=0)
            a.plot(c.d_km, np.median(ys, 0), color=NULL_COLOUR[kind], lw=1.4, ls="--", label=NULL_NAME[kind])
            summary[y][f"{kind}_d50_km"] = float(np.median([summarise(x)["d50_km"] for x in runs]))
        a.plot(c.d_km, c.largest, color=SERIES[0], lw=2.2, label="Actual fire")
        a.axhline(0.5, color=MUTED, lw=0.6)
        a.axvline(D_KM, color=MUTED, lw=0.6, ls=":")
        a.set_xscale("log")
        a.set_xlim(2, 100)
        a.set_xticks([2, 5, 10, 20, 50, 100], ["2", "5", "10", "20", "50", "100"])
        a.minorticks_off()
        a.set_title(f"{y}: {len(xy):,} events, d50 {obs:.1f} km vs same-cell "
                    f"{summary[y]['samecell_d50_km']:.1f} km", loc="left", fontsize=10, color=INK)
        a.set_xlabel("Joining distance d (km, log scale)", color=INK2)
        a.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    axs[0].set_ylabel("Share of events in the largest cluster", color=INK2)
    axs[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(f"Same axes for every year; bands = range of {REPS} random runs; dotted line = d of the maps",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "similar_years_curves.png", dpi=150)
    (OUT / "similar_years.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
