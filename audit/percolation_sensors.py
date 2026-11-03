"""Robustness: repeat the measurement (step 1) and the periods (step 2) with other fire satellites.

Sensors (daytime passes, fire_detections_clean, same cleaning rules as Aqua):
  aqua   MODIS Aqua   2003-2024 (the main series; also minus the 19 incomplete days found against INPE)
  terra  MODIS Terra  2003-2024 (morning pass, ~10:30)
  snpp   VIIRS S-NPP  2012-2024 (375 m pixels, ~13:30)
Events: detections chained when within 2 km and 2 days, as for Aqua. Built a day at a time (pairs between a day
and the next two days), keeping only a spanning forest of each window, so VIIRS fits in memory.
Measurement: as percolation_measure.py (MST curve, d50, ratio to random points anywhere in Brazil at the same
count, d80/d20 width, B half-samples of start weeks for the SE). The 'anywhere' random d50 is c n^-0.5
(calibrated on MODIS-sized samples; for random points this law is exact up to small edge effects).
Change: the same fixed periods 2003-06, 2007-15, 2016-24; BIC break search per sensor; agreement between sensors.

    python audit/percolation_sensors.py events     # build event files (VIIRS takes a while)
    python audit/percolation_sensors.py measure    # half-samples, parallel
    python audit/percolation_sensors.py report     # tests and figures
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from audit.fire_percolation_pilot import CACHE, GRID, INK, INK2, MUTED, OUT, SERIES, project
from firms import config

SENSORS = {"aqua": ("modis", "Aqua", range(2003, 2025), "MODIS Aqua (main)"),
           "terra": ("modis", "Terra", range(2003, 2025), "MODIS Terra (morning)"),
           "snpp": ("viirs_snpp", "N", range(2012, 2025), "VIIRS S-NPP (375 m)")}
COLOURS = {"aqua": SERIES[0], "terra": SERIES[1], "snpp": SERIES[2]}
PERIODS = {"2003–2006": range(2003, 2007), "2007–2015": range(2007, 2016), "2016–2024": range(2016, 2025)}
EVENT_KM, EVENT_DAYS = 2.0, 2


def build_events(lon, lat, day):
    """Fire events: detections chained when within EVENT_KM and EVENT_DAYS. Returns event label per detection."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    xy = project(lon, lat)
    n = len(xy)
    order = np.argsort(day, kind="stable")
    days = day[order]
    ei, ej = [], []
    uniq = np.unique(days)
    starts = np.searchsorted(days, uniq)
    ends = np.searchsorted(days, uniq + EVENT_DAYS, side="right")
    for d, a0, b1 in zip(uniq, starts, ends):
        a1 = np.searchsorted(days, d, side="right")
        A, Bw = order[a0:a1], order[a0:b1]
        m = cKDTree(xy[A]).sparse_distance_matrix(cKDTree(xy[Bw]), EVENT_KM, output_type="coo_matrix")
        if m.nnz == 0:
            continue
        ia, ib = A[m.row], Bw[m.col]
        loc, inv = np.unique(np.r_[ia, ib], return_inverse=True)
        k = len(ia)
        g = coo_matrix((np.ones(k), (inv[:k], inv[k:])), shape=(len(loc), len(loc)))
        _, lab = connected_components(g, directed=False)
        first = np.full(lab.max() + 1, -1)
        for idx in range(len(loc)):                      # root of each component = its first member
            if first[lab[idx]] < 0:
                first[lab[idx]] = idx
        ei.append(loc)
        ej.append(loc[first[lab]])
    ei, ej = (np.concatenate(ei), np.concatenate(ej)) if ei else (np.array([], int), np.array([], int))
    _, lab = connected_components(coo_matrix((np.ones(len(ei)), (ei, ej)), shape=(n, n)), directed=False)
    return lab


def events_file(tag, y):
    return CACHE / f"events_{tag}_{y}_2km_2d.parquet"


def events():
    import duckdb
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    for tag, (sensor, sat, years, _) in SENSORS.items():
        for y in years:
            f = events_file(tag, y)
            if f.exists():
                continue
            q = f"""SELECT longitude AS lon, latitude AS lat, acq_date AS d FROM fire_detections_clean
                    WHERE sensor = '{sensor}' AND satellite = '{sat}' AND daynight = 'D' AND year(acq_date) = {y}"""
            if tag == "aqua":
                q += " AND NOT (acq_date IN (SELECT day FROM incomplete_days))"
            det = con.execute(q).df()
            day = pd.to_datetime(det.d).to_numpy().astype("datetime64[D]").astype(np.int64)
            lab = build_events(det.lon.to_numpy(), det.lat.to_numpy(), day)
            ev = (pd.DataFrame({"e": lab, "lon": det.lon, "lat": det.lat, "t": day}).groupby("e")
                  .agg(lon=("lon", "mean"), lat=("lat", "mean"), start=("t", "min"), det=("t", "size")))
            ev["start"] = ev.start.astype("datetime64[D]")
            ev.to_parquet(f, index=False)
            print(tag, y, len(det), "detections ->", len(ev), "events", flush=True)


def one(args):
    tag, y, area, cal, seed = args
    from audit.percolation_measure import B, REL, curve, random_d50
    ev = pd.read_parquet(events_file(tag, y))
    ev = ev[ev.start.dt.year == y]
    xy = project(ev.lon.to_numpy(), ev.lat.to_numpy())
    week = ((ev.start.dt.dayofyear - 1) // 7).to_numpy()
    _, d50, n = curve(xy, REL)
    width = curve.levels[0.8] / curve.levels[0.2]
    rng = np.random.default_rng(seed)
    weeks = np.unique(week)
    H = []
    for _ in range(B):
        m = np.isin(week, rng.choice(weeks, size=len(weeks) // 2, replace=False))
        _, dh, nh = curve(xy[m], REL)
        H.append((dh / random_d50(cal, "anywhere", nh), curve.levels[0.8] / curve.levels[0.2]))
    H = np.array(H)
    rec = {"sensor": tag, "year": y, "events": int(n), "d50_km": d50,
           "ratio": d50 / random_d50(cal, "anywhere", n), "ratio_se": float(H[:, 0].std(ddof=1)),
           "width": width, "width_se": float(H[:, 1].std(ddof=1))}
    print(tag, y, n, round(rec["ratio"], 3), round(rec["ratio_se"], 3), round(width, 2), flush=True)
    return rec


def measure():
    from multiprocessing import Pool
    from audit.percolation_measure import brazil_area, calibrate
    area = brazil_area()
    cal = calibrate(area)
    jobs = [(tag, y, area, cal, 5000 + y + 100 * k) for k, (tag, v) in enumerate(SENSORS.items()) for y in v[2]]
    jobs.sort(key=lambda j: j[0] != "snpp")              # big VIIRS jobs first
    with Pool(15) as pool:
        recs = pool.map(one, jobs, chunksize=1)
    (OUT / "percolation_sensors_years.json").write_text(json.dumps(recs))


def breaks(y, years):
    from audit.percolation_change import MAX_BREAKS, best_breaks
    n = len(y)
    bic = {}
    for k in range(MAX_BREAKS + 1):
        br, rss = best_breaks(y, k)
        bic[k] = (n * np.log(rss / n) + (2 * k + 1) * np.log(n), [int(years[b]) for b in br])
    k = min(bic, key=lambda q: bic[q][0])
    return {"bic": {q: round(v[0], 2) for q, v in bic.items()}, "chosen": k, "new_periods_start": bic[k][1]}


def report():
    import matplotlib.pyplot as plt
    from scipy import stats
    df = pd.DataFrame(json.loads((OUT / "percolation_sensors_years.json").read_text()))
    df.round(4).to_csv(OUT / "percolation_sensors_years.csv", index=False)
    res = {}
    for col in ("ratio", "width"):
        wide = df.pivot(index="year", columns="sensor", values=col)
        r = {"agreement_spearman": {}, "period_means": {}, "breaks": {}, "tests": {}}
        for a, b in (("aqua", "terra"), ("aqua", "snpp"), ("terra", "snpp")):
            d = wide[[a, b]].dropna()
            sp = stats.spearmanr(d[a], d[b])
            r["agreement_spearman"][f"{a}_vs_{b}"] = [round(sp[0], 2), round(sp[1], 4), len(d)]
        for s in SENSORS:
            x = wide[s].dropna()
            r["period_means"][s] = {p: round(float(x[x.index.isin(yrs)].mean()), 3) for p, yrs in PERIODS.items()
                                    if x.index.isin(yrs).any()}
            r["breaks"][s] = breaks(x.to_numpy(), x.index.to_numpy())
            mid, late = x[x.index.isin(range(2007, 2016))], x[x.index >= 2016]
            if s != "snpp":
                other = x[~x.index.isin(range(2007, 2016))]
                r["tests"][f"{s}: 2007-15 vs other"] = round(float(stats.ttest_ind(mid, other, equal_var=False).pvalue), 5)
            m12 = x[x.index.isin(range(2012, 2016))]
            r["tests"][f"{s}: 2012-15 vs 2016-24"] = [round(float(m12.mean()), 3), round(float(late.mean()), 3),
                                                     round(float(stats.ttest_ind(m12, late, equal_var=False).pvalue), 4)]
        res[col] = r
    (OUT / "percolation_sensors.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    ev = df.pivot(index="year", columns="sensor", values="events")
    print(ev.to_string())

    fig, axs = plt.subplots(1, 3, figsize=(19, 5.2), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    for a, col, lab in ((axs[0], "ratio", "Clustering ratio (fire ÷ random, same count)"),
                        (axs[1], "width", "Transition width (d80 ÷ d20)")):
        for k, (s, v) in enumerate(SENSORS.items()):
            d = df[df.sensor == s].sort_values("year")
            off = (k - 1) * 0.2
            a.errorbar(d.year + off, d[col], yerr=1.96 * d[f"{col}_se"], fmt="o", ms=4, color=COLOURS[s],
                       capsize=2, lw=1, label=v[3])
            for p, yrs in PERIODS.items():
                x = d[d.year.isin(yrs)][col]
                if len(x):
                    lo = max(min(yrs), d.year.min())
                    a.hlines(x.mean(), lo - 0.4, max(yrs) + 0.4, color=COLOURS[s], lw=2, alpha=0.8)
        if col == "ratio":
            a.axhline(1, color=INK2, lw=0.8)
        a.axvspan(2006.5, 2015.5, color=MUTED, alpha=0.1, lw=0)
        a.set_title(f"{lab}\n95% intervals; lines = period means (2003–06, 2007–15, 2016–24)", loc="left",
                    fontsize=10, color=INK)
        a.legend(frameon=False, fontsize=8.5, loc="upper left")
    a = axs[2]
    wide = df.pivot(index="year", columns="sensor", values="ratio")
    for s, mk in (("terra", "o"), ("snpp", "s")):
        d = wide[["aqua", s]].dropna()
        a.scatter(d.aqua, d[s], s=30, color=COLOURS[s], marker=mk,
                  label=f"{SENSORS[s][3]}: ρ {res['ratio']['agreement_spearman'][f'aqua_vs_{s}'][0]:+.2f}")
        for yv, xa, xb in zip(d.index, d.aqua, d[s]):
            a.annotate(str(yv)[2:], (xa, xb), xytext=(3, 2), textcoords="offset points", fontsize=7, color=INK2)
    lim = [min(wide.min()) - 0.05, max(wide.max()) + 0.05]
    a.plot(lim, lim, color=MUTED, lw=0.8)
    a.set_xlabel("Ratio, MODIS Aqua", color=INK2)
    a.set_ylabel("Ratio, other satellite", color=INK2)
    a.set_title("Do the satellites agree year by year?", loc="left", fontsize=10, color=INK)
    a.legend(frameon=False, fontsize=8.5, loc="upper left")
    for a in axs:
        a.grid(color=GRID, lw=0.6)
        for sp_ in ("top", "right"):
            a.spines[sp_].set_visible(False)
    fig.suptitle("Is the three-period pattern the same on other fire satellites? (daytime fire events, Brazil)",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_sensors.png", dpi=140)


if __name__ == "__main__":
    {"events": events, "measure": measure, "report": report}[sys.argv[1]]()
