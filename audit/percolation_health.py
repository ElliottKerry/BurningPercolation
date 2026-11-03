"""Smoke and health: with the same amount of fire nearby, do municipalities in well-connected (clumped) fire
landscapes have more respiratory admissions?

Units: municipalities of residence in the 9 Legal Amazon states (MA whole state, as SIH codes are by state),
years 2012-2019 (SIH coverage; 2020 dropped for COVID).
Outcome: SIH admissions by municipality of residence, Aug-Nov (smoke season), log(1 + count):
  resp (J00-J99, all ages), resp_u5, resp_65, and injuries (S00-T98) as a negative control.
Fire (MODIS Aqua fire events, 2 km / 2 days, starting Jul-Oct):
  local       log(1 + events inside the municipality)
  nearby      log(1 + events within RADIUS_KM of the municipality's representative point)
  connected   share of the nearby events that belong to a large cluster of that year (>= BIG_SHARE of the year's
              events) when all the year's events are joined at the random-point d50 for the same count -- how
              connected the fire landscape around the municipality is, for a given amount of fire
Model: OLS with municipality and state-by-year fixed effects (absorbed), SE clustered by municipality,
  outcome ~ local + nearby + connected + log(1 + off-season admissions of the same group, Jan-Jun)
Coefficients are per one standard deviation of each exposure (within-municipality SD reported too).
The negative control should show no effect; if it does, the design is picking up something other than smoke.

    python audit/percolation_health.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from audit.fire_percolation_pilot import ALBERS, GRID, INK, INK2, MUTED, OUT, SERIES, labels_at, mst_edges, project
from audit.percolation_measure import random_d50
from audit.percolation_stability import events_with_dates
from audit.priority_municipalities import fe_ols
from firms import config

YEARS = range(2012, 2020)
UF_CODES = {"11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO", "21": "MA", "51": "MT"}
RADIUS_KM = 100
BIG_SHARE = 0.01
OUTCOMES = {"resp": "Respiratory, all ages", "resp_u5": "Respiratory, under 5", "resp_65": "Respiratory, 65+",
            "injury": "Injuries (negative control)"}


def admissions():
    import duckdb
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    q = f"""SELECT munic_res, year(dt_inter) AS year,
                   month(dt_inter) BETWEEN 8 AND 11 AS season,
                   sum(n) FILTER (WHERE grp = 'resp') AS resp,
                   sum(n) FILTER (WHERE grp = 'resp' AND age_grp = 'u5') AS resp_u5,
                   sum(n) FILTER (WHERE grp = 'resp' AND age_grp = '65+') AS resp_65,
                   sum(n) FILTER (WHERE grp = 'injury') AS injury
            FROM sih_daily
            WHERE year(dt_inter) BETWEEN {min(YEARS)} AND {max(YEARS)}
              AND (month(dt_inter) BETWEEN 8 AND 11 OR month(dt_inter) <= 6)
              AND left(munic_res, 2) IN ({','.join(repr(k) for k in UF_CODES)})
            GROUP BY ALL"""
    a = con.execute(q).df().fillna(0)
    s = a[a.season].drop(columns="season").set_index(["munic_res", "year"])
    o = a[~a.season].drop(columns="season").set_index(["munic_res", "year"]).add_prefix("off_")
    return s.join(o, how="outer").fillna(0).reset_index()


def main():
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import shapely
    from scipy.spatial import cKDTree
    cal = json.loads((OUT / "percolation_measure_calibration.json").read_text())
    m = gpd.read_file(config.RAW_DIR / "boundaries" / "BR_Municipios_2025.shp")
    m = m[m.CD_MUN.str[:2].isin(UF_CODES)].to_crs(ALBERS)
    m["munic_res"] = m.CD_MUN.str[:6]
    pts = m.geometry.representative_point()
    cen = np.column_stack([pts.x, pts.y]) / 1000
    tree_c = cKDTree(cen)
    rows = []
    for y in YEARS:
        ev = events_with_dates(y)
        ev = ev[ev.start.dt.year == y]
        xy = project(ev.lon.to_numpy(), ev.lat.to_numpy())
        u, inv = np.unique(np.round(xy, 4), axis=0, return_inverse=True)
        inv = inv.ravel()
        lab = labels_at(len(u), *mst_edges(u), random_d50(cal, "anywhere", len(u)))[inv]
        size = np.bincount(lab)
        big = size[lab] >= BIG_SHARE * len(xy)
        season = ev.start.dt.month.between(7, 10).to_numpy()
        xs, bs = xy[season], big[season]
        g = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs[:, 0] * 1000, xs[:, 1] * 1000), crs=ALBERS)
        j = gpd.sjoin(g, m[["munic_res", "geometry"]], predicate="within")
        local = j.groupby("munic_res").size()
        tree_e = cKDTree(xs)
        nb = tree_e.query_ball_point(cen, RADIUS_KM)
        n_near = np.array([len(k) for k in nb])
        n_big = np.array([bs[k].sum() if k else 0 for k in nb])
        d = pd.DataFrame({"munic_res": m.munic_res.to_numpy(), "year": y, "uf": m.munic_res.str[:2].map(UF_CODES).to_numpy(),
                          "nearby_n": n_near, "connected": np.where(n_near > 0, n_big / np.maximum(n_near, 1), 0.0),
                          "big_share_of_year": float(big.mean())})
        d["local_n"] = d.munic_res.map(local).fillna(0).to_numpy()
        rows.append(d)
        print(y, len(xy), "events,", round(float(big.mean()), 3), "in large clusters", flush=True)
    fire = pd.concat(rows)
    adm = admissions()
    df = fire.merge(adm, on=["munic_res", "year"], how="inner")
    df["local"] = np.log1p(df.local_n)
    df["nearby"] = np.log1p(df.nearby_n)
    df["st_yr"] = df.uf + df.year.astype(str)
    df["CD_MUN"] = df.munic_res
    for c in OUTCOMES:
        df[f"y_{c}"] = np.log1p(df[c])
        df[f"off_{c}"] = np.log1p(df[f"off_{c}"])
    df.to_csv(OUT / "percolation_health_panel.csv", index=False)

    expo = ["local", "nearby", "connected"]
    sd = df[expo].std()
    within_sd = df.groupby("munic_res")[expo].transform(lambda s: s - s.mean()).std()
    res = {"n_municipalities": int(df.munic_res.nunique()), "n_obs": int(len(df)), "radius_km": RADIUS_KM,
           "exposure_sd": sd.round(3).to_dict(), "exposure_within_municipality_sd": within_sd.round(3).to_dict(),
           "connected_mean": round(float(df.connected.mean()), 3), "models": {}}
    from scipy import stats
    for c in OUTCOMES:
        for name, xs in (("counts only", ["local", "nearby"]), ("with connectivity", expo)):
            d = df.copy()
            for e in xs:
                d[e] = d[e] / sd[e]
            b, V = fe_ols(d, f"y_{c}", [*xs, f"off_{c}"])
            se = np.sqrt(np.diag(V))
            res["models"][f"{c} | {name}"] = {e: {"pct_per_sd": round(float(100 * (np.exp(b[e]) - 1)), 2),
                                                  "lo": round(float(100 * (np.exp(b[e] - 1.96 * se[i]) - 1)), 2),
                                                  "hi": round(float(100 * (np.exp(b[e] + 1.96 * se[i]) - 1)), 2),
                                                  "p": round(float(2 * stats.norm.sf(abs(b[e] / se[i]))), 4)}
                                              for i, e in enumerate(xs)}
    (OUT / "percolation_health.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "models"}, indent=1))
    for k, v in res["models"].items():
        print(k, v)

    fig, axs = plt.subplots(1, 3, figsize=(17, 5), sharex=True)
    labels = {"local": "Fires in the municipality", "nearby": f"Fires within {RADIUS_KM} km",
              "connected": "Connectivity (share of nearby fires\nin large clusters)"}
    for a, e in zip(axs, expo):
        ys = np.arange(len(OUTCOMES))[::-1]
        for i, c in enumerate(OUTCOMES):
            r = res["models"][f"{c} | with connectivity"][e]
            col = MUTED if c == "injury" else SERIES[0]
            a.errorbar(r["pct_per_sd"], ys[i], xerr=[[r["pct_per_sd"] - r["lo"]], [r["hi"] - r["pct_per_sd"]]], fmt="o",
                       color=col, capsize=3, ms=6)
        a.axvline(0, color=INK2, lw=0.8)
        a.set_yticks(ys, list(OUTCOMES.values()) if e == "local" else [""] * len(ys))
        a.set_title(labels[e], loc="left", fontsize=10.5, color=INK)
        a.set_xlabel("% change in Aug–Nov admissions per 1 SD (95% CI)", color=INK2)
        a.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle(f"Fire and hospital admissions, Legal Amazon municipalities 2012–2019 ({res['n_municipalities']} "
                 "municipalities). Each municipality compared with itself in other years; state-wide yearly changes "
                 "removed. Grey = negative control", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_health.png", dpi=140)


if __name__ == "__main__":
    main()
