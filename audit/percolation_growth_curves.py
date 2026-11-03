"""Add fine largest-cluster curves to the growth page data (artifact_growth/summary.json).

Per year: share of fire events in the largest cluster on FINE_D (60 log-spaced distances, 1-100 km), from the
Euclidean MST of the year's events (duplicate positions counted with their multiplicity), and the typical
spacing sqrt(Brazil area / events) used for the 'times typical spacing' axis. d50 in km and relative.

    python audit/percolation_growth_curves.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from audit.fire_percolation_pilot import ALBERS, OUT, YEARS, mst_edges, project
from audit.percolation_stability import events_with_dates
from firms import config

DEST = OUT / "artifact_growth" / "summary.json"
FINE_D = np.round(np.logspace(0, 2, 60), 3)


def largest_curve(xy):
    u, cnt = np.unique(np.round(xy, 4), axis=0, return_counts=True)
    i, j, w = mst_edges(u)
    parent = np.arange(len(u))
    size = cnt.astype(np.int64)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    big, best = np.empty(len(w)), size.max()
    for k in range(len(w)):
        a, b = find(i[k]), find(j[k])
        if a != b:
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]
            best = max(best, size[a])
        big[k] = best
    pos = np.searchsorted(w, FINE_D, side="right") - 1
    return np.where(pos >= 0, big[np.maximum(pos, 0)], cnt.max()) / cnt.sum()


def main():
    import geopandas as gpd
    area = gpd.read_parquet(config.DATA_DIR / "cache" / "brazil_outline_5880.parquet").to_crs(ALBERS).area.sum() / 1e6
    s = json.loads(DEST.read_text())
    s["fine_d"] = FINE_D.tolist()
    for y in YEARS:
        ev = events_with_dates(y)
        ev = ev[ev.start.dt.year == y]
        c = largest_curve(project(ev.lon.to_numpy(), ev.lat.to_numpy()))
        sp = float(np.sqrt(area / len(ev)))
        d50 = float(np.exp(np.interp(0.5, c, np.log(FINE_D))))
        s["years"][str(y)].update({"fine_share": np.round(c, 4).tolist(), "spacing_km": round(sp, 2),
                                   "d50_km": round(d50, 2), "d50_rel": round(d50 / sp, 3)})
        print(y, round(sp, 2), round(d50, 2), round(d50 / sp, 3), flush=True)
    y = s["years"].values()
    km, rel = np.array([v["d50_km"] for v in y]), np.array([v["d50_rel"] for v in y])
    s["d50_spread"] = {"km": [float(km.min()), float(km.max())], "rel": [float(rel.min()), float(rel.max())],
                       "cv_km": round(float(km.std() / km.mean()), 3), "cv_rel": round(float(rel.std() / rel.mean()), 3)}
    print(s["d50_spread"])
    DEST.write_text(json.dumps(s, separators=(",", ":")))


if __name__ == "__main__":
    main()
