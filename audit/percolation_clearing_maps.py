"""Maps: fire near recent clearing vs other fire in the Amazon biome, and how each one clusters.

Rows: 2012 (low clearing) and 2019 (high clearing). MODIS Aqua fire events (2 km / 2 days) in the Amazon biome;
near = within 1 km of a PRODES deforestation patch of PRODES year y or y+1 (as percolation_clearing.py).
Column 1: both groups, cleared patches (y, y+1) as grey dots.
Columns 2-3: each group's clusters when joined at the distance where the same number of random points inside the
biome would be half joined (so a group spread like random points would have a largest cluster near 50%).

    python audit/percolation_clearing_maps.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from audit.fire_percolation_pilot import ALBERS, INK, INK2, OUT, labels_at, mst_edges, project
from audit.percolation_lattice_threshold import RANK_COLOURS
from audit.percolation_stability import events_with_dates
from firms import config

YEARS = [2012, 2019]
NEAR_M = 1000
NEAR_C, FAR_C = "#e8590c", "#2a78d6"


def main():
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import shapely
    from pyproj import Transformer
    from audit.percolation_early_warning_regional import biome_shapes
    amazon = biome_shapes()["Amazon"]
    shapely.prepare(amazon)
    a, b = json.loads((OUT / "percolation_early_warning_regional_cal.json").read_text())["Amazon"]
    rd50 = lambda n: float(np.exp(a + b * np.log(n)))
    pr = gpd.read_parquet(config.DATA_DIR / "cache" / "prodes_amazon_2008_2024.parquet")
    pr = pr[pr.lon.notna() & ~pr.geometry.is_empty]
    back = Transformer.from_crs(ALBERS, "EPSG:4326", always_xy=True)
    outline = gpd.GeoSeries([amazon], crs=ALBERS).to_crs("EPSG:4326")
    states = gpd.read_file(config.RAW_DIR / "boundaries" / "BR_Municipios_2025.shp")[["SIGLA_UF", "geometry"]]
    states = states.dissolve("SIGLA_UF").to_crs("EPSG:4326").simplify(0.03)
    fig, axs = plt.subplots(len(YEARS), 3, figsize=(19, 6.6 * len(YEARS)))
    for r, y in enumerate(YEARS):
        ev = events_with_dates(y)
        ev = ev[ev.start.dt.year == y]
        xy = project(ev.lon.to_numpy(), ev.lat.to_numpy())
        xy = xy[shapely.contains_xy(amazon, xy[:, 0] * 1000, xy[:, 1] * 1000)]
        patches = pr[pr.year.isin([y, y + 1])]
        tree = shapely.STRtree(patches.geometry.to_numpy())
        near = np.zeros(len(xy), bool)
        near[np.unique(tree.query(shapely.points(xy * 1000), predicate="dwithin", distance=NEAR_M)[0])] = True
        lon, lat = back.transform(xy[:, 0] * 1000, xy[:, 1] * 1000)

        ax = axs[r, 0]
        ax.scatter(patches.lon, patches.lat, s=0.3, color="#b9b7ae", rasterized=True, label="Cleared patches (PRODES)")
        ax.scatter(lon[~near], lat[~near], s=0.25, color=FAR_C, rasterized=True)
        ax.scatter(lon[near], lat[near], s=0.25, color=NEAR_C, rasterized=True)
        ax.set_title(f"{y}: {len(xy):,} Amazon fire events\n{near.mean():.0%} within 1 km of recent clearing (orange), "
                     f"{(~near).mean():.0%} elsewhere (blue)", loc="left", fontsize=10.5, color=INK)
        ax.text(-0.04, 0.5, str(y), transform=ax.transAxes, fontsize=18, weight="bold", color=INK, rotation=90,
                va="center", ha="right")
        for c, (m, name, col) in enumerate(((near, "Fires near recent clearing", NEAR_C),
                                            (~near, "Other fires", FAR_C)), start=1):
            g = xy[m]
            u, inv = np.unique(np.round(g, 4), axis=0, return_inverse=True)
            d = rd50(len(u))
            lab = labels_at(len(u), *mst_edges(u), d)[inv.ravel()]
            size = np.bincount(lab)
            top = np.argsort(size)[::-1][:4]
            ax = axs[r, c]
            ax.scatter(lon[m], lat[m], s=0.25, color="#cfcdc5", rasterized=True)
            for k in range(3, -1, -1):
                mm = lab == top[k]
                ax.scatter(lon[m][mm], lat[m][mm], s=0.25, color=RANK_COLOURS[k], rasterized=True)
            ax.set_title(f"{name}: clusters at {d:.0f} km\nlargest {size.max() / len(g):.0%}, 2nd "
                         f"{np.sort(size)[-2] / len(g):.0%} (random points would give about 50%)", loc="left",
                         fontsize=10.5, color=col)
        for ax in axs[r]:
            states.boundary.plot(ax=ax, color="#d6d4cc", lw=0.4)
            outline.boundary.plot(ax=ax, color=INK2, lw=0.6)
            ax.set_xlim(-74.2, -43.5)
            ax.set_ylim(-17.5, 5.5)
            ax.set_aspect("equal")
            ax.set_axis_off()
    fig.suptitle("Amazon fire on recently cleared land vs other fire (MODIS Aqua fire events, PRODES deforestation). "
                 "Cluster colours: purple largest, green 2nd, pink 3rd, orange 4th, grey the rest",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    fig.savefig(OUT / "percolation_clearing_maps.png", dpi=120)


if __name__ == "__main__":
    main()
