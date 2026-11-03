"""Data for the interactive 'clusters growing' page (audit/results/artifact_growth/).

Per year, MODIS Aqua fire events (2 km / 2 days) started that year, shown on 0.25 deg cells.
  season mode   events added on their start day, joined within the scaled distance 2 x sqrt(Brazil area / n)
                (same rule as percolation_days.py); snapshots on the 1st and 15th of Jun-Dec and on 31 Dec.
  distance mode all the year's events, joined within d (DISTANCES km), via the Euclidean MST.
Cell code per snapshot: '0'-'3' = the cell holds an event of the largest .. 4th largest cluster (best rank wins),
'4' = other events, '.' = no event yet (season mode only). Cluster ranks are by size at that snapshot.

    python audit/percolation_growth_artifact.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from audit.fire_percolation_pilot import ALBERS, OUT, YEARS, labels_at, mst_edges, project
from audit.percolation_days import SCALE
from audit.percolation_lattice_threshold import RES, brazil_cells
from audit.percolation_stability import events_with_dates
from firms import config

DEST = OUT / "artifact_growth"
DISTANCES = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60]
SNAP_DATES = [f"{m:02d}-{d:02d}" for m in range(6, 13) for d in (1, 15)] + ["12-31"]


def rank_of(lab, alive):
    """Rank 0-3 for the four largest clusters among alive points, 4 for the rest, -1 for not alive."""
    size = np.bincount(lab[alive], minlength=lab.max() + 1)
    top = np.argsort(size)[::-1][:4]
    r = np.full(len(lab), 4)
    for k, c in enumerate(top):
        r[lab == c] = k
    r[~alive] = -1
    return r, size.max() if alive.any() else 0


def cell_codes(cell, r):
    """One character per cell: best (lowest) rank of its alive events, '.' if none alive."""
    best = np.full(cell.max() + 1, 9)
    ok = r >= 0
    np.minimum.at(best, cell[ok], r[ok])
    return "".join("." if b == 9 else str(b) for b in best)


def main():
    import geopandas as gpd
    DEST.mkdir(parents=True, exist_ok=True)
    area = gpd.read_parquet(config.DATA_DIR / "cache" / "brazil_outline_5880.parquet").to_crs(ALBERS).area.sum() / 1e6
    days = pd.read_csv(OUT / "percolation_days.csv").set_index("year")
    summary = {"distances": DISTANCES, "snap_dates": SNAP_DATES, "years": {}}
    for y in YEARS:
        ev = events_with_dates(y)
        ev = ev[ev.start.dt.year == y]
        lon, lat = ev.lon.to_numpy(), ev.lat.to_numpy()
        xy = project(lon, lat)
        n = len(xy)
        doy = (ev.start.dt.dayofyear - 1).to_numpy()
        ix = np.floor(lon / RES + 0.5).astype(int)
        iy = np.floor(lat / RES + 0.5).astype(int)
        keys, cell = np.unique(np.column_stack([ix, iy]), axis=0, return_inverse=True)
        cell = cell.ravel()
        counts = np.bincount(cell)

        # season mode
        d = SCALE * np.sqrt(area / n)
        pr = cKDTree(xy).query_pairs(d, output_type="ndarray")
        season, s_share, s_sofar = [], [], []
        for md in SNAP_DATES:
            day = pd.Timestamp(f"{y}-{md}").dayofyear - 1
            alive = doy <= day
            k = alive[pr[:, 0]] & alive[pr[:, 1]]
            _, lab = connected_components(coo_matrix((np.ones(k.sum()), (pr[k, 0], pr[k, 1])), shape=(n, n)),
                                          directed=False)
            r, big = rank_of(lab, alive)
            season.append(cell_codes(cell, r))
            s_share.append(round(big / n, 4))
            s_sofar.append(round(float(alive.mean()), 4))

        # distance mode (MST on unique positions, mapped back to events)
        u, inv = np.unique(np.round(xy, 4), axis=0, return_inverse=True)
        inv = inv.ravel()
        i, j, w = mst_edges(u)
        dist, d_share = [], []
        alive = np.ones(n, bool)
        for dk in DISTANCES:
            lab = labels_at(len(u), i, j, w, dk)[inv]
            r, big = rank_of(lab, alive)
            dist.append(cell_codes(cell, r))
            d_share.append(round(big / n, 4))

        rec = {"year": y, "events": int(n), "d_km": round(float(d), 1),
               "ix": keys[:, 0].tolist(), "iy": keys[:, 1].tolist(), "count": counts.tolist(),
               "season": season, "distance": dist}
        (DEST / f"{y}.json").write_text(json.dumps(rec, separators=(",", ":")))
        row = days.loc[y]
        summary["years"][str(y)] = {"events": int(n), "d_km": round(float(d), 1), "season_share": s_share,
                                    "season_sofar": s_sofar, "distance_share": d_share,
                                    "join_day": round(float(row.scaled_join_day), 1),
                                    "shuffled_join_day": round(float(row.scaled_null_day), 1),
                                    "delay": round(float(row.scaled_lag), 1)}
        print(y, n, len(keys), flush=True)

    # Brazil cells for the background
    mask, ix0, iy0 = brazil_cells()
    yy, xx = np.nonzero(mask)
    summary["brazil"] = {"ix": (xx + ix0).tolist(), "iy": (yy + iy0).tolist()}
    s = gpd.read_file(config.RAW_DIR / "boundaries" / "BR_Municipios_2025.shp")[["SIGLA_UF", "geometry"]]
    st = s.dissolve("SIGLA_UF").to_crs("EPSG:4326").simplify(0.03)
    rings = []
    for g in st.geometry:
        polys = g.geoms if g.geom_type == "MultiPolygon" else [g]
        for p in polys:
            if p.area < 0.05:
                continue
            rings.append([[round(x, 2), round(yv, 2)] for x, yv in p.exterior.coords])
    summary["states"] = rings
    (DEST / "summary.json").write_text(json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
