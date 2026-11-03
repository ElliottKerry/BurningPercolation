"""Map grid: actual fire events against the three random versions, at several joining distances.

Rows: actual fire, then one random run of each baseline (same 28 km cell, where fire usually is, anywhere in
Brazil). Columns: joining distances d. In every panel the biggest cluster is blue, the next 7 have their own
colours and everything else is grey. It shows on a map what the percolation curve shows as lines.

    python audit/percolation_map_grid.py [year]      # default 2019; always uses fire events (2 km / 2 days)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if "--events" not in sys.argv:
    sys.argv.append("--events")                      # the imported module reads this flag at import time

import numpy as np

from audit.fire_percolation_expected import NULLS, Sampler, observed
from audit.fire_percolation_pilot import ALBERS, INK, INK2, OUT, SERIES, labels_at, mst_edges
from firms import config

DISTANCES = [8, 12, 16, 20, 65]                      # km, the points discussed in the curve write-up


def main(year):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from pyproj import Transformer
    rng = np.random.default_rng(2019)
    back = Transformer.from_crs(ALBERS, "EPSG:4326", always_xy=True)
    biomes = gpd.read_file(config.RAW_DIR / "biomes" / "lm_bioma_250.shp").set_crs("EPSG:4674", allow_override=True)
    lon, lat, xy = observed(year)
    S = Sampler()
    order = {k: lab for k, lab, _ in NULLS}
    rows = [("Actual fire", xy)] + [("Random: " + order[k][0].lower() + order[k][1:], S.draw(k, lon, lat, rng))
                                    for k in ("samecell", "usual", "anywhere")]

    fig, axs = plt.subplots(len(rows), len(DISTANCES), figsize=(3.3 * len(DISTANCES), 3.5 * len(rows)))
    for r, (name, pts) in enumerate(rows):
        pts = np.unique(np.round(pts, 4), axis=0)
        i, j, w = mst_edges(pts)
        plon, plat = back.transform(pts[:, 0] * 1000, pts[:, 1] * 1000)
        for c, d in enumerate(DISTANCES):
            a = axs[r, c]
            lab = labels_at(len(pts), i, j, w, d)
            sizes = np.bincount(lab)
            top = np.argsort(sizes)[::-1][:8]
            a.scatter(plon, plat, s=0.08, color="#d6d4cc", rasterized=True)
            for k, cl in enumerate(top):
                m = lab == cl
                a.scatter(plon[m], plat[m], s=0.08, color=SERIES[k], rasterized=True)
            biomes.boundary.plot(ax=a, color=INK2, lw=0.3)
            a.set_title(f"biggest cluster {sizes.max() / len(pts):.0%}", fontsize=9, loc="left", color=INK)
            a.set_axis_off()
            a.set_aspect("equal")
            if r == 0:
                a.text(0, 1.18, f"d = {d} km", transform=a.transAxes, fontsize=12, weight="bold", color=INK)
            if c == 0:
                a.text(-0.08, 0.5, name, transform=a.transAxes, fontsize=11, weight="bold", color=INK,
                       rotation=90, va="center", ha="right")
    fig.suptitle(f"Fire events {year} ({len(xy):,}): two events are in the same cluster if a chain of links shorter "
                 f"than d joins them.\nBiggest cluster blue, next 7 coloured, the rest grey. Each random row is one run "
                 f"with the same number of points.", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0.02, 0, 1, 0.95))
    fig.savefig(OUT / f"percolation_map_grid_{year}_events.png", dpi=130)
    print("saved", OUT / f"percolation_map_grid_{year}_events.png")


if __name__ == "__main__":
    main(next((int(a) for a in sys.argv[1:] if a.isdigit()), 2019))
