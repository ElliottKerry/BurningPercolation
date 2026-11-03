"""Separate recurring (persistent) burning from spreading, wildfire-like fire, and redo the percolation for each.

FIRMS fixed heat sources (type 2/3, plus anything within ~1 km of one) were already removed in firms/clean.py.
What remains is vegetation fire, which mixes two kinds:

Split 1, by place: does the same 1 km spot burn year after year?
  For every 0.01 deg cell (~1 km), count the years 2003-2024 with at least one detection. An event is
  "recurring spot" when its detections sit, on average, in cells that burn in >= REC_YEARS of the 22 years
  (typical of pasture and crop burning repeated each dry season). Otherwise "new spot".
Split 2, by behaviour: does the fire spread?
  An event is "spreading" when it lasts >= SPREAD_DAYS days or has >= SPREAD_DET detections (a fire that
  grows across pixels and days, closer to an uncontrolled wildfire). Otherwise "small" (a one-off burn).

Events: detections within 2 km and 2 days, chained (as in fire_percolation_expected.py --events).
Measure for each group and year: ratio = d50(group) / median d50 of REPS sets of the same number of points
spread uniformly over Brazil (1 = joins like random points; > 1 = split into separate regional hotspots).

    python audit/percolation_fire_types.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from audit.fire_percolation_pilot import (CACHE, GRID, INK, INK2, MUTED, OUT, SERIES, YEARS, mst_edges, project,
                                          random_points, summarise, sweep)

REC_YEARS = 5
SPREAD_DAYS, SPREAD_DET = 3, 10
REPS = 3
DETAIL = 2019
GROUPS = {"Recurring spot": ("rec", True), "New spot": ("rec", False),
          "Spreading fire": ("spread", True), "Small burn": ("spread", False)}
PAIRS = (["Recurring spot", "New spot"], ["Spreading fire", "Small burn"])
COLOUR = {"All events": INK, "Recurring spot": SERIES[1], "New spot": SERIES[0],
          "Spreading fire": SERIES[4], "Small burn": SERIES[2]}


def burn_years():
    """Years (of 22) with any detection, per 0.01 deg cell."""
    a = pd.concat([pd.read_parquet(CACHE / f"aqua_{y}.parquet", columns=["lon", "lat"]).assign(y=y) for y in YEARS])
    return a.groupby([np.floor(a.lon / 0.01).astype(int), np.floor(a.lat / 0.01).astype(int)]).y.nunique()


def typed_events(year, yrs):
    cache = CACHE / f"events_typed_{year}_2km_2d.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    f = pd.read_parquet(CACHE / f"aqua_{year}.parquet")
    xy = project(f.lon.to_numpy(), f.lat.to_numpy())
    t = pd.to_datetime(f.d).to_numpy().astype("datetime64[D]").astype(np.int64)
    pr = cKDTree(xy).query_pairs(2.0, output_type="ndarray")
    pr = pr[np.abs(t[pr[:, 0]] - t[pr[:, 1]]) <= 2]
    n = len(f)
    _, f["e"] = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n)),
                                     directed=False)
    f["t"] = t
    f["yrs"] = yrs.reindex(pd.MultiIndex.from_arrays([np.floor(f.lon / 0.01).astype(int),
                                                      np.floor(f.lat / 0.01).astype(int)])).to_numpy()
    ev = f.groupby("e").agg(lon=("lon", "mean"), lat=("lat", "mean"), det=("t", "size"), t0=("t", "min"),
                            t1=("t", "max"), burn_years=("yrs", "mean"))
    ev["days"] = ev.t1 - ev.t0 + 1
    ev["rec"] = ev.burn_years >= REC_YEARS
    ev["spread"] = (ev.days >= SPREAD_DAYS) | (ev.det >= SPREAD_DET)
    ev = ev.drop(columns=["t0", "t1"]).reset_index(drop=True)
    ev.to_parquet(cache, index=False)
    return ev


def curve(xy):
    xy = np.unique(np.round(xy, 4), axis=0)
    c, _ = sweep(len(xy), *mst_edges(xy))
    return c, len(xy)


def ratio(ev, rng):
    c, n = curve(project(ev.lon.to_numpy(), ev.lat.to_numpy()))
    nulls = [curve(random_points(n, rng))[0] for _ in range(REPS)]
    d0 = float(np.median([summarise(x)["d50_km"] for x in nulls]))
    return summarise(c)["d50_km"], d0, c, nulls


def style(a):
    a.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


def main():
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from firms import config
    rng = np.random.default_rng(11)
    yrs = burn_years()
    rows, detail = [], {}
    for y in YEARS:
        ev = typed_events(y, yrs)
        r = {"year": y, "events": len(ev), "detections": int(ev.det.sum())}
        for name, (col, val) in {"All events": (None, None), **GROUPS}.items():
            sub = ev if col is None else ev[ev[col] == val]
            o, d0, c, nulls = ratio(sub, rng)
            r[f"{name}: events"] = len(sub)
            r[f"{name}: share of detections"] = sub.det.sum() / ev.det.sum()
            r[f"{name}: ratio"] = o / d0
            if y == DETAIL:
                detail[name] = (sub, c, nulls, o, d0)
        rows.append(r)
        print(y, {k.split(":")[0]: round(v, 2) for k, v in r.items() if k.endswith("ratio")}, flush=True)
    df = pd.DataFrame(rows).set_index("year")
    df.to_csv(OUT / "fire_types_years.csv")
    names = ["All events", *GROUPS]
    summary = {"rules": {"recurring_spot_years": REC_YEARS, "spreading_days": SPREAD_DAYS,
                         "spreading_detections": SPREAD_DET},
               "mean_share_of_events": {g: float((df[f"{g}: events"] / df.events).mean()) for g in GROUPS},
               "mean_share_of_detections": {g: float(df[f"{g}: share of detections"].mean()) for g in GROUPS},
               "mean_ratio": {g: float(df[f"{g}: ratio"].mean()) for g in names},
               "min_ratio": {g: float(df[f"{g}: ratio"].min()) for g in names},
               "max_ratio": {g: float(df[f"{g}: ratio"].max()) for g in names},
               "corr_with_all_events": {g: float(df[f"{g}: ratio"].corr(df["All events: ratio"])) for g in GROUPS}}
    (OUT / "fire_types.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    # ---- 2019 maps: events per 0.25 deg cell for each group
    biomes = gpd.read_file(config.RAW_DIR / "biomes" / "lm_bioma_250.shp").set_crs("EPSG:4674", allow_override=True)
    fig, axs = plt.subplots(1, 4, figsize=(18, 5.2))
    all_ev = detail["All events"][0]
    for a, g in zip(axs, GROUPS):
        sub = detail[g][0]
        h, xe, ye = np.histogram2d(sub.lon, sub.lat, bins=[np.arange(-74, -28.9, 0.25), np.arange(-34, 5.6, 0.25)])
        a.imshow(np.log10(np.where(h > 0, h, np.nan)).T, origin="lower", extent=(xe[0], xe[-1], ye[0], ye[-1]),
                 cmap="magma_r", vmin=0, vmax=2.5, interpolation="nearest")
        biomes.boundary.plot(ax=a, color=INK2, lw=0.3)
        a.set_title(f"{g}: {len(sub):,} events\n({len(sub) / len(all_ev):.0%} of events, "
                    f"{sub.det.sum() / all_ev.det.sum():.0%} of detections)", loc="left", fontsize=9.5, color=INK)
        a.set_axis_off()
        a.set_aspect("equal")
    sm = plt.cm.ScalarMappable(cmap="magma_r", norm=plt.Normalize(0, 2.5))
    cb = fig.colorbar(sm, ax=axs, shrink=0.6, pad=0.01)
    cb.set_ticks([0, 1, 2], labels=["1", "10", "100"])
    cb.set_label("Events per 0.25° cell", color=INK2)
    fig.suptitle(f"Kinds of fire, {DETAIL}. Recurring spot: the same 1 km spot burns in at least {REC_YEARS} of 22 "
                 f"years. Spreading: lasts at least {SPREAD_DAYS} days or has at least {SPREAD_DET} detections",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(OUT / f"fire_types_maps_{DETAIL}.png", dpi=150, bbox_inches="tight")

    # ---- 2019 percolation curves; d in units of each group's own random d50 so groups of different size compare
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for a, pair, head in zip(axs, PAIRS, ("Split by place", "Split by behaviour")):
        for g in ["All events", *pair]:
            sub, c, nulls, o, d0 = detail[g]
            main_line = g != "All events"
            a.plot(c.d_km / d0, c.largest, color=COLOUR[g], lw=2 if main_line else 1.2, ls="-" if main_line else ":",
                   label=f"{g} (fire / random {o / d0:.2f})")
        sub, c, nulls, o, d0 = detail[pair[0]]
        a.plot(nulls[0].d_km / d0, nulls[0].largest, color=MUTED, lw=1.2, ls="--", label="Random points")
        a.axhline(0.5, color=MUTED, lw=0.6)
        a.set_xscale("log")
        a.set_xlim(0.2, 6)
        a.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
        a.minorticks_off()
        a.set_xlabel("Joining distance, in units of the random half-joined distance", color=INK2)
        a.set_title(head, loc="left", fontsize=10, color=INK)
        a.legend(frameon=False, fontsize=9, loc="upper left")
        style(a)
    axs[0].set_ylabel("Share in the largest cluster", color=INK2)
    fig.suptitle(f"Percolation curves by kind of fire, {DETAIL}: curves further right join up later than random",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / f"fire_types_curves_{DETAIL}.png", dpi=150)

    # ---- every year
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
    for a, pair, head in zip(axs, PAIRS, ("Split by place", "Split by behaviour")):
        a.plot(df.index, df["All events: ratio"], color=INK, lw=1.2, ls=":", label="All events")
        for g in pair:
            a.plot(df.index, df[f"{g}: ratio"], color=COLOUR[g], lw=2, marker="o", ms=4, label=g)
        a.axhline(1, color=MUTED, lw=0.8, ls="--")
        a.set_title(f"{head}: fire / random half-joined distance", loc="left", fontsize=10, color=INK)
        a.legend(frameon=False, fontsize=9)
        style(a)
    axs[0].set_ylabel("Fire / random (1 = like random)", color=INK2)
    fig.suptitle("Critical point by kind of fire, 2003–2024 (MODIS Aqua fire events, all Brazil)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fire_types_years.png", dpi=150)


if __name__ == "__main__":
    main()
