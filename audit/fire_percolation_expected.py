"""Expected vs observed: fire percolation compared with three "no structure" baselines (null models).

Each baseline keeps the same number of points as the real year but scrambles part of the pattern:
  1. Anywhere in Brazil       - points uniform over Brazil: fire with no geography at all.
  2. Where fire usually is    - points drawn in proportion to the 2003-2024 fire map (0.25 deg cells), placed
                                at random inside the cell: the usual fire geography, nothing specific to the year.
  3. Same 28 km cell          - every real point moved to a random spot in its own 0.25 deg cell (about 28 km):
                                keeps this year's regional pattern exactly, removes clumping within a cell.
The gaps between observed and each baseline say where the structure comes from:
  observed vs 1 -> fire has a geography at all;  observed vs 2 -> this year differs from the usual map;
  observed vs 3 -> fire clumps at scales below about 28 km (local spread or clustered burning).

    python audit/fire_percolation_expected.py [year]            # detections; writes audit/results/expected_*.png/.json
    python audit/fire_percolation_expected.py [year] --events   # fire events instead of detections (*_events.*)

Fire events (--events): detections within EVENT_KM of each other and at most EVENT_DAYS apart are the same fire,
chained through neighbours (connected components), so a fire burning or spreading for days is one event. Each
event becomes one point at the mean position of its detections. This removes clumping that only comes from
the same fire being seen on several days or in neighbouring pixels. The "usual fire map" baseline is then built
from events too.
Method of the curves themselves: audit/fire_percolation_pilot.py (after Arcaute et al. 2016).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from audit.fire_percolation_pilot import (ALBERS, CACHE, D_GRID, GRID, INK, INK2, MUTED, OUT, SERIES, YEARS,
                                          labels_at, mst_edges, project, random_points, summarise, sweep)
from firms import config

RES = 0.25
REPS_YEAR, REPS_ALL_YEARS = 20, 5
EVENTS = "--events" in sys.argv
EVENT_KM, EVENT_DAYS = 2.0, 2
TAG = "_events" if EVENTS else ""
NULLS = [("anywhere", "Anywhere in Brazil", "#898781"),
         ("usual", "Where fire usually is", SERIES[1]),
         ("samecell", "Same 28 km cell", SERIES[2])]
OBS = SERIES[0]


def load_year(year):
    """lon, lat of the points for one year: detections, or event centres with --events (cached)."""
    f = pd.read_parquet(CACHE / f"aqua_{year}.parquet")
    if not EVENTS:
        return f
    cache = CACHE / f"events_{year}_{EVENT_KM:g}km_{EVENT_DAYS}d.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    xy = project(f.lon.to_numpy(), f.lat.to_numpy())
    t = (pd.to_datetime(f.d) - pd.Timestamp(f"{year}-01-01")).dt.days.to_numpy()
    pr = cKDTree(xy).query_pairs(EVENT_KM, output_type="ndarray")
    pr = pr[np.abs(t[pr[:, 0]] - t[pr[:, 1]]) <= EVENT_DAYS]
    n = len(f)
    _, lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n)), directed=False)
    ev = pd.DataFrame({"e": lab, "lon": f.lon, "lat": f.lat}).groupby("e").agg(lon=("lon", "mean"), lat=("lat", "mean"),
                                                                               detections=("lon", "size"))
    ev.to_parquet(cache, index=False)
    print(f"{year}: {n:,} detections -> {len(ev):,} events (median {ev.detections.median():.0f}, "
          f"largest {ev.detections.max():,} detections)", flush=True)
    return ev


class Sampler:
    def __init__(self):
        import geopandas as gpd
        import shapely
        from pyproj import Transformer
        shape = gpd.read_parquet(config.DATA_DIR / "cache" / "brazil_outline_5880.parquet").to_crs(ALBERS).union_all()
        shapely.prepare(shape)
        self.shape, self.shapely = shape, shapely
        self.fwd = Transformer.from_crs("EPSG:4326", ALBERS, always_xy=True)
        parts = [load_year(y)[["lon", "lat"]] for y in YEARS]
        allp = pd.concat(parts)
        ix, iy = np.floor(allp.lon / RES + 0.5).astype(int), np.floor(allp.lat / RES + 0.5).astype(int)
        c = pd.DataFrame({"ix": ix, "iy": iy}).value_counts()
        self.cells = np.array(c.index.tolist())
        self.p = c.to_numpy() / c.sum()

    def in_cells(self, ix, iy, rng):
        """One uniform point inside each given cell, redrawn (up to 30 times) while it falls outside Brazil."""
        n = len(ix)
        xy = np.full((n, 2), np.nan)
        todo = np.arange(n)
        for _ in range(30):
            lon = (ix[todo] + rng.uniform(-0.5, 0.5, len(todo))) * RES
            lat = (iy[todo] + rng.uniform(-0.5, 0.5, len(todo))) * RES
            x, y = self.fwd.transform(lon, lat)
            ok = self.shapely.contains_xy(self.shape, x, y)
            xy[todo[ok]] = np.column_stack([x, y])[ok]
            todo = todo[~ok]
            if not len(todo):
                break
        if len(todo):                                    # tiny coastal slivers: keep the cell centre
            x, y = self.fwd.transform(ix[todo] * RES, iy[todo] * RES)
            xy[todo] = np.column_stack([x, y])
        return xy / 1000.0

    def draw(self, kind, lon, lat, rng):
        n = len(lon)
        if kind == "anywhere":
            return random_points(n, rng)
        if kind == "usual":
            k = rng.choice(len(self.cells), size=n, p=self.p)
            return self.in_cells(self.cells[k, 0], self.cells[k, 1], rng)
        ix, iy = np.floor(lon / RES + 0.5).astype(int), np.floor(lat / RES + 0.5).astype(int)
        return self.in_cells(ix, iy, rng)


def observed(year):
    f = load_year(year)
    xy = project(f.lon.to_numpy(), f.lat.to_numpy())
    _, keep = np.unique(np.round(xy, 4), axis=0, return_index=True)
    keep = np.sort(keep)
    return f.lon.to_numpy()[keep], f.lat.to_numpy()[keep], xy[keep]


def curve_of(xy):
    xy = np.unique(np.round(xy, 4), axis=0)
    c, _ = sweep(len(xy), *mst_edges(xy))
    return c


def style(a):
    a.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.tick_params(colors=INK2)


def main(year):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from pyproj import Transformer
    rng = np.random.default_rng(7)
    S = Sampler()
    lon, lat, xy = observed(year)
    n = len(xy)
    obs = curve_of(xy)

    # ---- 1. The chosen year: observed curve against the range of each baseline (REPS_YEAR runs each)
    env = {}
    for key, _, _ in NULLS:
        runs = [curve_of(S.draw(key, lon, lat, rng)) for _ in range(REPS_YEAR)]
        L = np.vstack([r.largest for r in runs])
        env[key] = {"lo": L.min(0), "hi": L.max(0), "mid": np.median(L, 0),
                    "d50": [summarise(r)["d50_km"] for r in runs]}
        print(f"{year} {key}: d50 {np.median(env[key]['d50']):.1f} km "
              f"(range {min(env[key]['d50']):.1f}-{max(env[key]['d50']):.1f})", flush=True)
    d50_obs = summarise(obs)["d50_km"]
    print(f"{year} observed: d50 {d50_obs:.1f} km", flush=True)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    for key, label, col in NULLS:
        e = env[key]
        ax.fill_between(D_GRID, e["lo"], e["hi"], color=col, alpha=0.25, lw=0)
        ax.plot(D_GRID, e["mid"], color=col, lw=2, ls="--", label=f"Expected: {label.lower()}")
    ax.plot(obs.d_km, obs.largest, color=OBS, lw=2.5, label=f"Actual fire{' events' if EVENTS else ''}, {year}")
    ax.annotate(f"Actual {year}", (d50_obs, 0.5), xytext=(8, -14), textcoords="offset points", fontsize=10,
                color=INK, weight="bold")
    ax.axhline(0.5, color=MUTED, lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(1, 200)
    ax.set_xlabel("Joining distance d (km): fires closer than d are linked", color=INK2)
    ax.set_ylabel("Share of all fire points in the biggest cluster", color=INK2)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax)
    ax.set_title(f"Expected vs actual percolation, Brazil {year} (MODIS Aqua, {n:,} fire points). "
                 f"Bands: range of {REPS_YEAR} random runs", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / f"expected_vs_actual_curve_{year}{TAG}.png", dpi=160)

    # ---- 2. Maps at the observed half-joined distance: actual next to one run of each baseline
    back = Transformer.from_crs(ALBERS, "EPSG:4326", always_xy=True)
    biomes = gpd.read_file(config.RAW_DIR / "biomes" / "lm_bioma_250.shp").set_crs("EPSG:4674", allow_override=True)
    panels = [("Actual fire", xy)] + [(f"Expected: {lab.lower()}", S.draw(k, lon, lat, rng)) for k, lab, _ in NULLS[::-1]]
    fig, axs = plt.subplots(1, 4, figsize=(17, 5))
    for a, (title, pts) in zip(axs, panels):
        pts = np.unique(np.round(pts, 4), axis=0)
        i, j, w = mst_edges(pts)
        lab = labels_at(len(pts), i, j, w, d50_obs)
        sizes = np.bincount(lab)
        top = np.argsort(sizes)[::-1][:8]
        plon, plat = back.transform(pts[:, 0] * 1000, pts[:, 1] * 1000)
        a.scatter(plon, plat, s=0.15, color="#d6d4cc", rasterized=True)
        for k, c in enumerate(top):
            m = lab == c
            a.scatter(plon[m], plat[m], s=0.15, color=SERIES[k], rasterized=True)
        biomes.boundary.plot(ax=a, color=INK2, lw=0.35)
        a.set_title(f"{title}\nbiggest cluster {sizes.max() / len(pts):.0%} of points", loc="left", fontsize=10, color=INK)
        a.set_axis_off()
        a.set_aspect("equal")
    fig.suptitle(f"Clusters at d = {d50_obs:.1f} km (where half of the actual {year} fire points have joined up). "
                 f"8 biggest clusters in colour", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / f"expected_vs_actual_maps_{year}{TAG}.png", dpi=150)

    # ---- 3. Every year: observed half-joined distance against the baselines' range (REPS_ALL_YEARS runs)
    rows = []
    for y in YEARS:
        ylon, ylat, yxy = observed(y)
        r = {"year": y, "points": len(yxy), "observed": summarise(curve_of(yxy))["d50_km"]}
        for key, _, _ in NULLS:
            d = [summarise(curve_of(S.draw(key, ylon, ylat, rng)))["d50_km"] for _ in range(REPS_ALL_YEARS)]
            r.update({f"{key}_lo": min(d), f"{key}_mid": float(np.median(d)), f"{key}_hi": max(d)})
        rows.append(r)
        print(f"{y}: observed {r['observed']:.1f} | usual {r['usual_mid']:.1f} | same cell {r['samecell_mid']:.1f} | "
              f"anywhere {r['anywhere_mid']:.1f} km", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / f"expected_vs_actual_years{TAG}.csv", index=False)   # saved before plotting

    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label, col in NULLS:
        ax.fill_between(t.year, t[f"{key}_lo"], t[f"{key}_hi"], color=col, alpha=0.25, lw=0)
        ax.plot(t.year, t[f"{key}_mid"], color=col, lw=2, ls="--", marker="o", ms=3, label=f"Expected: {label.lower()}")
    ax.plot(t.year, t.observed, color=OBS, lw=2.5, marker="o", ms=5, label="Actual fire")
    for yy in (2005, 2010, 2019, 2024):
        v = t.loc[t.year == yy, "observed"].iloc[0]
        ax.annotate(str(yy), (yy, v), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, color=INK2)
    ax.axvspan(2021, 2024.5, color=MUTED, alpha=0.08, lw=0)
    ax.text(2021.1, ax.get_ylim()[1] * 0.97, "Aqua orbit drift", fontsize=8, color=INK2, va="top")
    ax.set_ylabel("Distance where half of fire points join (km)", color=INK2)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    style(ax)
    ax.set_title(f"Expected vs actual, every year 2003–2024 (bands: range of {REPS_ALL_YEARS} random runs). "
                 + ("Fire events. " if EVENTS else "") + "Higher = fire more broken into separate regions", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / f"expected_vs_actual_years{TAG}.png", dpi=160)

    (OUT / f"expected_vs_actual_{year}{TAG}.json").write_text(json.dumps({
        "year": year, "points": int(n), "observed_d50_km": d50_obs,
        "baselines_d50_km": {k: {"median": float(np.median(env[k]["d50"])), "min": min(env[k]["d50"]),
                                 "max": max(env[k]["d50"])} for k, _, _ in NULLS},
        "years": t.round(2).to_dict("records")}, indent=1))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(int(args[0]) if args else 2019)
