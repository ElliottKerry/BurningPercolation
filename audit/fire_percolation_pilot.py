"""Pilot: hierarchical percolation of fire detections, after Arcaute et al. 2016 (R. Soc. Open Sci. 3:150691).

The paper treats street intersections as points and joins any two points closer than a distance d. Sweeping d
from small to large, clusters grow and merge; the size of the largest cluster against d is the "percolation
curve", and the distances where big clusters suddenly merge are the critical points. Here the points are
MODIS Aqua daytime fire detections in Brazil (the trusted series, 2003-2024).

    python audit/fire_percolation_pilot.py extract          # DuckDB -> data/cache/percolation/aqua_<year>.parquet
    python audit/fire_percolation_pilot.py run [year]       # curves, maps, critical points (default 2019)
    python audit/fire_percolation_pilot.py run 2019 --events  # same, on fire events (run fire_percolation_expected.py --events first)

How it works:
- Points are projected to the IBGE Albers equal-area projection (metres). Repeat detections at exactly the same
  coordinates count once, as the paper counts each intersection once.
- Clusters at every d come from the minimum spanning tree (MST) of the points: two points are in the same
  cluster at distance d exactly when the MST path between them has no edge longer than d. The MST is built
  from the Delaunay triangulation, so every threshold is exact.
- Critical points, two ways: (a) "big merges", where two clusters that each hold at least 1% of all points
  join (the paper reads these off its curve by eye); (b) the distance where the second-largest cluster peaks,
  the standard percolation estimate of the threshold.
- Null model (not in the paper): the same number of points scattered at random over Brazil. A curve that
  differs from the null shows structure beyond "there are this many fires in a country this size".
- MODIS pixels are about 1 km, so distances below about 1-2 km mostly reflect the pixel grid.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from firms import config

CACHE = config.DATA_DIR / "cache" / "percolation"
OUT = Path(__file__).resolve().parent / "results"
ALBERS = "+proj=aea +lat_0=-12 +lon_0=-54 +lat_1=-2 +lat_2=-22 +ellps=GRS80 +units=m +no_defs"
YEARS = range(2003, 2025)
D_GRID = np.round(np.geomspace(1, 400, 600), 3)        # km
BIG = 0.01                                              # a "big merge": both clusters >= 1% of points
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
EVENTS = "--events" in sys.argv                         # use fire events (2 km / 2 days) from fire_percolation_expected.py
TAG = "_events" if EVENTS else ""
WHAT = "Fire events" if EVENTS else "Fire detections"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def extract():
    import duckdb
    CACHE.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    for y in YEARS:
        con.execute(f"""COPY (SELECT longitude AS lon, latitude AS lat, acq_date AS d FROM modis_aqua_day_clean
                             WHERE year(acq_date) = {y}) TO '{(CACHE / f'aqua_{y}.parquet').as_posix()}' (FORMAT parquet)""")
    print("extracted", list(YEARS))


# ------------------------------------------------------------------------------------------- core method
def project(lon, lat):
    from pyproj import Transformer
    x, y = Transformer.from_crs("EPSG:4326", ALBERS, always_xy=True).transform(lon, lat)
    return np.column_stack([x, y]) / 1000.0              # km


def mst_edges(xy):
    """Edges (i, j, length km) of the Euclidean minimum spanning tree, sorted by length."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial import Delaunay
    tri = Delaunay(xy)
    s = tri.simplices
    e = np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    w = np.linalg.norm(xy[e[:, 0]] - xy[e[:, 1]], axis=1)
    n = len(xy)
    m = minimum_spanning_tree(coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n))).tocoo()
    o = np.argsort(m.data)
    return m.row[o], m.col[o], m.data[o]


def sweep(n, i, j, w):
    """Union-find along the sorted MST edges. Returns the curve on D_GRID (largest and second-largest cluster
    share, number of clusters with >= 2 points) and the list of big merges."""
    parent = np.arange(n)
    size = np.ones(n, dtype=np.int64)

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    largest = 1
    sizes_count = {1: n}                     # size -> number of clusters with that size (for 2nd largest)
    curve, merges, g = [], [], 0
    big = BIG * n

    def second():
        ks = sorted((k for k, c in sizes_count.items() if c > 0), reverse=True)
        if not ks:
            return 0
        if sizes_count[ks[0]] > 1:
            return ks[0]
        return ks[1] if len(ks) > 1 else 0

    for a, b, d in zip(i, j, w):
        while g < len(D_GRID) and D_GRID[g] < d:
            curve.append((D_GRID[g], largest / n, second() / n, n - sum(sizes_count.values()) + 0))
            g += 1
        ra, rb = find(a), find(b)
        sa, sb = size[ra], size[rb]
        if min(sa, sb) >= big:
            merges.append({"d_km": round(float(d), 2), "a_share": sa / n, "b_share": sb / n, "ra": int(ra), "rb": int(rb),
                           "a_pt": int(a), "b_pt": int(b)})
        if sa < sb:
            ra, rb, sa, sb = rb, ra, sb, sa
        parent[rb] = ra
        size[ra] = sa + sb
        for s_ in (sa, sb):
            sizes_count[s_] -= 1
        sizes_count[sa + sb] = sizes_count.get(sa + sb, 0) + 1
        largest = max(largest, sa + sb)
    while g < len(D_GRID):
        curve.append((D_GRID[g], largest / n, second() / n, 1))
        g += 1
    c = pd.DataFrame(curve, columns=["d_km", "largest", "second", "_"]).drop(columns="_")
    return c, merges


def labels_at(n, i, j, w, d):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    k = w <= d
    _, lab = connected_components(coo_matrix((np.ones(k.sum()), (i[k], j[k])), shape=(n, n)), directed=False)
    return lab


def random_points(n, rng):
    """n points uniformly at random inside Brazil (km, Albers)."""
    import geopandas as gpd
    import shapely
    outline = gpd.read_parquet(config.DATA_DIR / "cache" / "brazil_outline_5880.parquet").to_crs(ALBERS)
    shape = outline.union_all()
    shapely.prepare(shape)
    x0, y0, x1, y1 = shape.bounds
    pts = []
    while sum(len(p) for p in pts) < n:
        cand = rng.uniform([x0, y0], [x1, y1], size=(n, 2))
        pts.append(cand[shapely.contains_xy(shape, cand[:, 0], cand[:, 1])])
    return np.vstack(pts)[:n] / 1000.0


def load_points(year):
    f = pd.read_parquet(CACHE / (f"events_{year}_2km_2d.parquet" if EVENTS else f"aqua_{year}.parquet"))
    xy = project(f.lon.to_numpy(), f.lat.to_numpy())
    u = np.unique(np.round(xy, 4), axis=0)
    return f, u


def summarise(curve):
    k = curve.second.idxmax()
    half = curve[curve.largest >= 0.5]
    return {"d_c_km": float(curve.d_km[k]), "second_peak": float(curve.second[k]),
            "d50_km": float(half.d_km.iloc[0]) if len(half) else None,
            "largest_at_10km": float(np.interp(10, curve.d_km, curve.largest)),
            "largest_at_25km": float(np.interp(25, curve.d_km, curve.largest))}


# -------------------------------------------------------------------------------------------------- run
def place(lon, lat, ufs):
    """State of a point (nearest municipality polygon)."""
    import geopandas as gpd
    p = gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon], [lat]), crs="EPSG:4326").to_crs(ufs.crs)
    j = gpd.sjoin_nearest(p, ufs, how="left")
    return j.SIGLA_UF.iloc[0]


def run(year: int):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from pyproj import Transformer
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(2016)
    back = Transformer.from_crs(ALBERS, "EPSG:4326", always_xy=True)

    # 1. Evolution across years: real vs random, so the critical distances can be compared fairly.
    evo = []
    for y in YEARS:
        raw, xy = load_points(y)
        n = len(xy)
        c, _m = sweep(n, *mst_edges(xy))
        r, _ = sweep(n, *mst_edges(random_points(n, rng) ))
        sr, sn = summarise(c), summarise(r)
        evo.append({"year": y, "detections": len(raw), "points": n, "big_merges": len(_m), **{f"real_{k}": v for k, v in sr.items()},
                    **{f"null_{k}": v for k, v in sn.items()}})
        if y == year:
            curve, null = c, r
        print(f"{y}: {n:,} points, d50 real {sr['d50_km']:.1f} vs random {sn['d50_km']:.1f} km; big merges {len(_m)}", flush=True)
    evo = pd.DataFrame(evo)

    # 2. The chosen year in detail.
    raw, xy = load_points(year)
    n = len(xy)
    i, j, w = mst_edges(xy)
    curve, merges = sweep(n, i, j, w)
    ufs = gpd.read_file(config.RAW_DIR / "boundaries" / "BR_Municipios_2025.shp")[["SIGLA_UF", "geometry"]]
    for m in merges:
        for k in ("a", "b"):
            lon, lat = back.transform(xy[m[f"{k}_pt"], 0] * 1000, xy[m[f"{k}_pt"], 1] * 1000)
            m[f"{k}_uf"] = place(lon, lat, ufs)
    crit = sorted({round(m["d_km"], 1) for m in merges})
    print(f"\n{year}: {len(raw):,} detections, {n:,} distinct points; big merges at (km): {crit}")

    # Pick map distances: the first big merge, the second-largest-cluster peak, and two later merges.
    dc = summarise(curve)["d_c_km"]
    show = sorted(set([crit[0] if crit else 5.0, dc] + crit[len(crit) // 2:len(crit) // 2 + 1] + crit[-1:]))[:4]

    # ---- Figure 1: percolation curve, real vs random
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for a, col, title in ((ax[0], "largest", "Largest cluster"), (ax[1], "second", "Second-largest cluster")):
        a.plot(null.d_km, null[col], color=MUTED, lw=2, ls="--", label="Same number of points, random")
        a.plot(curve.d_km, curve[col], color=SERIES[0], lw=2, label=f"{WHAT} {year}")
        a.set_xscale("log")
        a.set_title(f"{title} (share of all points)", loc="left", fontsize=10, color=INK)
        a.grid(color=GRID, lw=0.6)
        a.set_xlabel("Joining distance d (km)", color=INK2)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    for d in crit:
        ax[0].axvline(d, color=SERIES[1], lw=0.8, alpha=0.6)
    ax[0].legend(frameon=False, fontsize=9, loc="upper left")
    ax[1].axvline(dc, color=SERIES[1], lw=1)
    ax[1].annotate(f"peak {dc:.1f} km", (dc, curve.second.max()), xytext=(6, -2), textcoords="offset points",
                   fontsize=9, color=INK2)
    fig.suptitle(f"Fire percolation curve, Brazil {year} (MODIS Aqua{", fire events" if EVENTS else ""}). Orange lines: two clusters of ≥1% of points merge",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / f"percolation_curve_{year}{TAG}.png", dpi=160)

    # ---- Figure 2: cluster maps
    lon, lat = back.transform(xy[:, 0] * 1000, xy[:, 1] * 1000)
    biomes = gpd.read_file(config.RAW_DIR / "biomes" / "lm_bioma_250.shp").set_crs("EPSG:4674", allow_override=True)
    fig, ax = plt.subplots(1, len(show), figsize=(4.2 * len(show), 4.6))
    for a, d in zip(np.atleast_1d(ax), show):
        lab = labels_at(n, i, j, w, d)
        sizes = np.bincount(lab)
        top = np.argsort(sizes)[::-1][:8]
        a.scatter(lon, lat, s=0.2, color=GRID, rasterized=True)
        for k, c in enumerate(top):
            m = lab == c
            a.scatter(lon[m], lat[m], s=0.2, color=SERIES[k], rasterized=True)
        biomes.boundary.plot(ax=a, color=INK2, lw=0.4)
        a.set_title(f"d = {d:.1f} km: largest {sizes.max() / n:.0%},\n{(sizes >= 0.001 * n).sum()} clusters ≥0.1%",
                    fontsize=9, loc="left", color=INK)
        a.set_axis_off()
        a.set_aspect("equal")
    fig.suptitle(f"{WHAT}: clusters at critical distances, {year}: the 8 largest in colour, smaller in grey; lines = biome borders",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / f"percolation_maps_{year}{TAG}.png", dpi=160)

    # ---- Figure 3: evolution of the critical distance, 2003-2024
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(evo.year, evo.real_d50_km, color=SERIES[0], lw=2, marker="o", ms=4, label=WHAT)
    ax[0].plot(evo.year, evo.null_d50_km, color=MUTED, lw=2, ls="--", marker="o", ms=4, label="Random, same number")
    ax[0].set_title("Distance at which half of all points form one cluster (km)", loc="left", fontsize=10, color=INK)
    ax[0].legend(frameon=False, fontsize=9)
    ax[1].plot(evo.year, evo.real_d50_km / evo.null_d50_km, color=SERIES[0], lw=2, marker="o", ms=4)
    ax[1].set_title("Ratio fire / random (above 1 = fire needs longer links to join up)", loc="left", fontsize=10, color=INK)
    for a in ax:
        a.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle(f"Evolution of the critical point, MODIS Aqua 2003–2024, {WHAT.lower()} (2021+ affected by Aqua orbit drift)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / f"percolation_evolution{TAG}.png", dpi=160)

    res = {"paper": "Arcaute et al. 2016 R. Soc. Open Sci. 3:150691", "year": year, "detections": int(len(raw)),
           "points": int(n), "summary": summarise(curve), "null_summary": summarise(null),
           "big_merges": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
                           if k not in ("ra", "rb", "a_pt", "b_pt")} for m in merges],
           "map_distances_km": show, "evolution": evo.round(3).to_dict("records")}
    (OUT / f"percolation_pilot_{year}{TAG}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("summary", "null_summary", "big_merges")}, indent=1, default=float))


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract()
    else:
        run(next((int(a) for a in sys.argv[2:] if a.isdigit()), 2019))
