"""Is the critical point stable? How much it moves year to year and month to month, 2003-2024 (all Brazil).

Fire events use the same rule as fire_percolation_expected.py --events (detections within 2 km and 2 days,
chained), but each event also keeps its start date, so events can be grouped by month. An event belongs to
the month of its first detection.

The raw half-joined distance d50 depends strongly on how many fires there are (fewer points = bigger gaps),
and the number of fires swings tenfold over the season. So the measure used is the ratio to random points:
rho = d50(fire) / median d50 of REPS sets of the same number of points scattered uniformly over Brazil.
rho = 1: fire joins like random points; rho > 1: fire is split into separate regional hotspots.
Months with fewer than MIN_EVENTS events are left out (too few points for a stable d50).

    python audit/percolation_stability.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from audit.fire_percolation_pilot import (CACHE, GRID, INK, INK2, MUTED, OUT, SERIES, YEARS, mst_edges, project,
                                          random_points, summarise, sweep)

EVENT_KM, EVENT_DAYS = 2.0, 2
REPS = 3
MIN_EVENTS = 500


def events_with_dates(year):
    cache = CACHE / f"events_dated_{year}_2km_2d.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    f = pd.read_parquet(CACHE / f"aqua_{year}.parquet")
    xy = project(f.lon.to_numpy(), f.lat.to_numpy())
    t = pd.to_datetime(f.d).to_numpy().astype("datetime64[D]").astype(np.int64)
    pr = cKDTree(xy).query_pairs(EVENT_KM, output_type="ndarray")
    pr = pr[np.abs(t[pr[:, 0]] - t[pr[:, 1]]) <= EVENT_DAYS]
    n = len(f)
    _, lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n)), directed=False)
    ev = (pd.DataFrame({"e": lab, "lon": f.lon, "lat": f.lat, "t": t}).groupby("e")
          .agg(lon=("lon", "mean"), lat=("lat", "mean"), start=("t", "min")))
    ev["start"] = ev.start.astype("datetime64[D]")
    ev.to_parquet(cache, index=False)
    return ev


def d50(xy):
    xy = np.unique(np.round(xy, 4), axis=0)
    c, _ = sweep(len(xy), *mst_edges(xy))
    return summarise(c)["d50_km"], len(xy)


def ratio(lon, lat, rng):
    obs, n = d50(project(lon, lat))
    null = float(np.median([d50(random_points(n, rng))[0] for _ in range(REPS)]))
    return obs, null, n


def acf1(x):
    x = np.asarray(x, float)
    x = x - x.mean()
    return float((x[1:] * x[:-1]).sum() / (x * x).sum())


def main():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(7)
    months, years = [], []
    for y in YEARS:
        ev = events_with_dates(y)
        ev = ev[ev.start.dt.year == y]
        o, r, n = ratio(ev.lon.to_numpy(), ev.lat.to_numpy(), rng)
        years.append({"year": y, "events": n, "fire_d50": o, "random_d50": r, "rho": o / r})
        for m, g in ev.groupby(ev.start.dt.month):
            if len(g) < MIN_EVENTS:
                months.append({"date": pd.Timestamp(y, m, 1), "events": len(g), "rho": np.nan})
                continue
            o, r, n = ratio(g.lon.to_numpy(), g.lat.to_numpy(), rng)
            months.append({"date": pd.Timestamp(y, m, 1), "events": n, "fire_d50": o, "random_d50": r, "rho": o / r})
        print(f"{y}: year rho {years[-1]['rho']:.2f}", flush=True)
    yr = pd.DataFrame(years).set_index("year")
    mo = pd.DataFrame(months).set_index("date").asfreq("MS")
    mo["month"] = mo.index.month
    mo["season_mean"] = mo.groupby("month").rho.transform("mean")
    mo["anomaly"] = mo.rho - mo.season_mean
    yr.to_csv(OUT / "percolation_stability_years.csv")
    mo.to_csv(OUT / "percolation_stability_months.csv")

    ok = mo.dropna(subset=["anomaly"])
    contiguous = mo.anomaly.dropna()
    a = mo.anomaly.to_numpy()
    pairs = ~np.isnan(a[1:]) & ~np.isnan(a[:-1])
    acf_m = float(np.corrcoef(a[1:][pairs], a[:-1][pairs])[0, 1])
    dry = mo[mo.month.isin([8, 9, 10])].dropna(subset=["rho"])
    res = {
        "years": {"mean": yr.rho.mean(), "sd": yr.rho.std(), "cv": yr.rho.std() / yr.rho.mean(),
                  "min": yr.rho.min(), "max": yr.rho.max(), "mean_abs_change": yr.rho.diff().abs().mean(),
                  "lag1_autocorrelation": acf1(yr.rho)},
        "months": {"months_used": int(len(ok)), "months_total": int(len(mo)),
                   "seasonal_mean_by_month": mo.groupby("month").rho.mean().round(3).to_dict(),
                   "seasonal_sd_by_month": mo.groupby("month").rho.std().round(3).to_dict(),
                   "months_used_by_month": mo.groupby("month").rho.count().to_dict(),
                   "anomaly_sd": float(contiguous.std()), "anomaly_lag1_autocorrelation": acf_m,
                   "fire_season_aug_oct_mean": dry.rho.mean(), "fire_season_aug_oct_sd": dry.rho.std()},
    }
    (OUT / "percolation_stability.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))

    fig, ax = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [1, 1.2, 1]})
    a0 = ax[0]
    a0.plot(yr.index, yr.rho, color=SERIES[0], lw=2, marker="o", ms=4)
    a0.axhline(yr.rho.mean(), color=MUTED, lw=0.8, ls="--")
    a0.axhline(1, color=MUTED, lw=0.8, ls=":")
    a0.set_title(f"Year to year: fire / random half-joined distance (mean {yr.rho.mean():.2f}, "
                 f"SD {yr.rho.std():.2f}, year-to-year correlation {acf1(yr.rho):.2f})", loc="left", fontsize=10, color=INK)
    a1 = ax[1]
    a1.plot(mo.index, mo.rho, color=SERIES[0], lw=1, marker="o", ms=2)
    a1.axhline(1, color=MUTED, lw=0.8, ls=":")
    a1.set_title(f"Month to month (months with ≥{MIN_EVENTS} fire events; gaps = too few fires)", loc="left",
                 fontsize=10, color=INK)
    a2 = ax[2]
    g = mo.groupby("month").rho
    q = pd.DataFrame({"mean": g.mean(), "lo": g.quantile(0.1), "hi": g.quantile(0.9)})
    a2.fill_between(q.index, q.lo, q.hi, color=SERIES[0], alpha=0.18, lw=0, label="10th–90th percentile of years")
    a2.plot(q.index, q["mean"], color=SERIES[0], lw=2, marker="o", ms=4, label="Mean over years")
    a2.axhline(1, color=MUTED, lw=0.8, ls=":")
    a2.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    a2.set_title("The seasonal cycle: typical value for each calendar month, 2003–2024", loc="left", fontsize=10, color=INK)
    a2.legend(frameon=False, fontsize=9)
    for a in ax:
        a.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.set_ylabel("Fire / random", color=INK2)
    fig.suptitle("How stable is the critical point? MODIS Aqua fire events, all Brazil. 1 = joins up like random points",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_stability.png", dpi=150)


if __name__ == "__main__":
    main()
