"""Forest share of each 0.005 degree fire-grid cell, from MapBiomas land cover (30 m).

    python audit/mapbiomas_forest_grid.py 2019 para     # box: Para (default)
    python audit/mapbiomas_forest_grid.py 2019 amazon

Cells match the fire grid in fire_types_2019.py: ix = floor(lon / 0.005), iy = floor(lat / 0.005).
Forest = MapBiomas class 3 (forest formation) or 6 (floodable forest); this includes old secondary forest that
MapBiomas maps as forest. Each 30 m pixel is assigned to the cell holding its centre; the forest share is
forest pixels / pixels with any class (class 0 = no data is ignored).
The raster is read in strips of rows and summed with np.add.reduceat, first across columns, then down rows.
Output: data/cache/mapbiomas_forest_<box>_<year>.parquet (ix, iy, forest_share, pixels).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds

from firms import config

CELL = 0.005
FOREST = (3, 6)
BOXES = {"para": (-59.0, -10.0, -45.8, 2.8), "amazon": (-74.0, -18.1, -43.9, 5.3)}
STRIP = 1024


def groups(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start positions of runs of equal values in a sorted index array, and the value of each run."""
    starts = np.r_[0, np.flatnonzero(np.diff(idx)) + 1]
    return starts, idx[starts]


def main():
    year = sys.argv[1] if len(sys.argv) > 1 else "2019"
    box_name = sys.argv[2] if len(sys.argv) > 2 else "para"
    x0, y0, x1, y1 = BOXES[box_name]
    src = config.RAW_DIR / "mapbiomas" / f"coverage_{year}.tif"
    out = config.DATA_DIR / "cache" / f"mapbiomas_forest_{box_name}_{year}.parquet"
    parts = []
    with rasterio.open(src) as r:
        w = from_bounds(x0, y0, x1, y1, r.transform).round_offsets().round_lengths()
        a, e = r.transform.a, r.transform.e
        cols = np.arange(w.col_off, w.col_off + w.width)
        lon = r.transform.c + (cols + 0.5) * a
        cstart, ixs = groups(np.floor(lon / CELL).astype(np.int64))
        for row0 in range(w.row_off, w.row_off + w.height, STRIP):
            n = min(STRIP, w.row_off + w.height - row0)
            block = r.read(1, window=Window(w.col_off, row0, w.width, n))
            lat = r.transform.f + (np.arange(row0, row0 + n) + 0.5) * e
            rstart, iys = groups(np.floor(lat / CELL).astype(np.int64))
            forest = np.isin(block, FOREST).astype(np.uint32)
            valid = (block > 0).astype(np.uint32)
            f = np.add.reduceat(np.add.reduceat(forest, cstart, axis=1), rstart, axis=0)
            v = np.add.reduceat(np.add.reduceat(valid, cstart, axis=1), rstart, axis=0)
            iy, ix = np.meshgrid(iys, ixs, indexing="ij")
            parts.append(pd.DataFrame({"ix": ix.ravel(), "iy": iy.ravel(), "forest": f.ravel(), "pixels": v.ravel()}))
            print(f"rows {row0 - w.row_off + n:,} of {w.height:,}", end="\r")
    g = pd.concat(parts).groupby(["ix", "iy"], as_index=False).sum()  # strips can split a cell row
    g = g[g.pixels > 0]
    g["forest_share"] = g.forest / g.pixels
    g[["ix", "iy", "forest_share", "pixels"]].to_parquet(out)
    print(f"\n{out.name}: {len(g):,} cells; mean forest share {g.forest_share.mean():.2f}; "
          f"cells >= 80% forest: {100 * (g.forest_share >= 0.8).mean():.0f}%")


if __name__ == "__main__":
    sys.exit(main())
