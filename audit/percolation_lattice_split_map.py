"""Large map of the critical point in lattice_threshold_curve_2019.png: just before (k*-1) and at (k*) the split.

    python audit/percolation_lattice_split_map.py [year]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import ndimage

from audit.fire_percolation_pilot import INK, INK2, OUT
from audit.percolation_lattice_threshold import RANK_COLOURS, RES, brazil_cells, counts, critical, sweep
from firms import config

NAMES = ["Largest", "2nd", "3rd", "4th"]


def main(year):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    mask, ix0, iy0 = brazil_cells()
    g = counts(year, mask, ix0, iy0)
    ks = np.arange(0, int(np.quantile(g[mask], 0.995)) + 2)
    k_star = critical(sweep(g, mask, ks))["k_star"]
    biomes = gpd.read_file(config.RAW_DIR / "biomes" / "lm_bioma_250.shp").set_crs("EPSG:4674", allow_override=True)
    ext = ((ix0 - 0.5) * RES, (ix0 + mask.shape[1] - 0.5) * RES, (iy0 - 0.5) * RES, (iy0 + mask.shape[0] - 0.5) * RES)
    cmap = ListedColormap(["#ffffff", "#e4e2db", "#7a7a7a"] + RANK_COLOURS)
    fig, axs = plt.subplots(1, 2, figsize=(15, 7.6))
    for a, k, head in zip(axs, (k_star - 1, k_star), ("Just before the split", "At the split (the star)")):
        occ = mask & (g >= k)
        lab, _ = ndimage.label(occ)
        size = np.bincount(lab.ravel())
        size[0] = 0
        top = np.argsort(size)[::-1][:4]
        img = np.where(occ, 2, np.where(mask, 1, 0))
        for r, cl in enumerate(top):
            img = np.where(lab == cl, 3 + r, img)
        a.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=6, interpolation="nearest")
        biomes.boundary.plot(ax=a, color=INK2, lw=0.4)
        handles = [Patch(color=RANK_COLOURS[r], label=f"{NAMES[r]} cluster: {size[cl]:,} cells") for r, cl in enumerate(top)]
        handles += [Patch(color="#7a7a7a", label="Other small clusters"),
                    Patch(color="#e4e2db", label=f"Off: fewer than {k} events")]
        a.legend(handles=handles, loc="lower left", frameon=False, fontsize=9)
        a.set_title(f"{head}: k = {k} events per cell, {occ.sum() / mask.sum():.0%} of cells on",
                    loc="left", fontsize=11, color=INK)
        a.set_axis_off()
        a.set_aspect("equal")
    fig.suptitle(f"Where the largest fire cluster splits, {year} (0.25° cells, MODIS Aqua fire events; lines = biomes)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / f"lattice_threshold_split_map_{year}.png", dpi=150)


if __name__ == "__main__":
    main(next((int(a) for a in sys.argv[1:] if a.isdigit()), 2019))
