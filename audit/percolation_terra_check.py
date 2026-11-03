"""Why is the high clustering of 2003-2006 seen by MODIS Aqua (13:30) but not by MODIS Terra (10:30)?

Checks:
  1 thinning   randomly keep Aqua events down to Terra's count (REPS runs): does Aqua's ratio drop to Terra's?
  2 coverage   days with Terra / Aqua daytime data per year (sensor_days)
  3 where      Terra events as a share of Aqua events, by region and year (Legal Amazon vs rest of Brazil)
  4 which      ratio of each satellite inside the Legal Amazon and outside it, 2003-06 vs later
  5 time       Aqua events that Terra also saw (a Terra detection within 2 km on the same or previous day):
               clustering of the 'seen by both' and 'Aqua only' events

    python audit/percolation_terra_check.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from audit.fire_percolation_pilot import ALBERS, GRID, INK, INK2, MUTED, OUT, SERIES, project
from audit.percolation_measure import REL, brazil_area, curve, random_d50
from audit.percolation_sensors import events_file
from firms import config

YEARS = range(2003, 2025)
REPS = 5


def ratio(xy, cal, area_scale=1.0):
    _, d50, n = curve(xy, REL)
    return d50 / (random_d50(cal, "anywhere", n) * area_scale)


def main():
    import duckdb
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import shapely
    from scipy.spatial import cKDTree
    from audit.percolation_legal_amazon import legal_amazon
    cal = json.loads((OUT / "percolation_measure_calibration.json").read_text())
    area = brazil_area()
    la = legal_amazon()
    shapely.prepare(la)
    la_scale = np.sqrt(la.area / 1e6 / area)        # random d50 inside a smaller region scales with sqrt(area)
    rng = np.random.default_rng(4)
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    cov = con.execute("""SELECT satellite, year("day") y, count(*) FILTER (WHERE observed) obs FROM sensor_days
                         WHERE sensor = 'modis' AND daynight = 'D' GROUP BY ALL""").df()
    cov = cov.pivot(index="y", columns="satellite", values="obs")
    rows = []
    for y in YEARS:
        ev = {}
        for s in ("aqua", "terra"):
            e = pd.read_parquet(events_file(s, y))
            e = e[e.start.dt.year == y]
            xy = project(e.lon.to_numpy(), e.lat.to_numpy())
            ev[s] = (e, xy, shapely.contains_xy(la, xy[:, 0] * 1000, xy[:, 1] * 1000))
        (ea, xa, ina), (et, xt, int_) = ev["aqua"], ev["terra"]
        r = {"year": y, "aqua_events": len(ea), "terra_events": len(et),
             "aqua_days": int(cov.loc[y, "Aqua"]) if y in cov.index else None,
             "terra_days": int(cov.loc[y, "Terra"]) if y in cov.index else None,
             "ratio_aqua": ratio(xa, cal), "ratio_terra": ratio(xt, cal),
             "terra_per_aqua_amazon": int_.sum() / ina.sum(), "terra_per_aqua_rest": (~int_).sum() / (~ina).sum(),
             "amazon_share_aqua": ina.mean(), "amazon_share_terra": int_.mean()}
        r["ratio_aqua_thinned"] = float(np.median([ratio(xa[rng.choice(len(xa), len(xt), replace=False)], cal)
                                                   for _ in range(REPS)]))
        r["ratio_aqua_amazon"] = ratio(xa[ina], cal, la_scale)
        r["ratio_terra_amazon"] = ratio(xt[int_], cal, la_scale)
        # Aqua events also seen by Terra: a Terra event within 2 km starting the same day or the day before
        ta = ea.start.to_numpy().astype("datetime64[D]").astype(np.int64)
        tt = et.start.to_numpy().astype("datetime64[D]").astype(np.int64)
        seen = np.zeros(len(xa), bool)
        tree = cKDTree(xt)
        for i, nb in enumerate(tree.query_ball_point(xa, 2.0)):
            if nb and np.any((ta[i] - tt[nb] >= 0) & (ta[i] - tt[nb] <= 1)):
                seen[i] = True
        r["aqua_seen_by_terra_share"] = seen.mean()
        r["ratio_aqua_seen_by_terra"] = ratio(xa[seen], cal)
        r["ratio_aqua_only"] = ratio(xa[~seen], cal)
        r["seen_share_amazon"] = seen[ina].mean()
        r["seen_share_rest"] = seen[~ina].mean()
        rows.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    df = pd.DataFrame(rows).set_index("year")
    df.round(4).to_csv(OUT / "percolation_terra_check.csv")
    early, mid, late = df.index <= 2006, df.index.isin(range(2007, 2016)), df.index >= 2016
    summ = {c: {"2003-06": round(float(df[c][early].mean()), 3), "2007-15": round(float(df[c][mid].mean()), 3),
                "2016-24": round(float(df[c][late].mean()), 3)} for c in df.columns if c != "year"}
    (OUT / "percolation_terra_check.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    a = axs[0]
    for c, col, lab in (("ratio_aqua", SERIES[0], "Aqua, all events"), ("ratio_terra", SERIES[1], "Terra, all events"),
                        ("ratio_aqua_thinned", MUTED, "Aqua thinned at random to Terra's count")):
        a.plot(df.index, df[c], marker="o", ms=3.5, lw=1.6 if c != "ratio_aqua_thinned" else 1.2, color=col,
               ls="-" if c != "ratio_aqua_thinned" else "--", label=lab)
    a.set_title("1. Is it just fewer fires? (thinning)", loc="left", fontsize=10, color=INK)
    a = axs[1]
    for c, col, lab in (("terra_per_aqua_amazon", SERIES[2], "Legal Amazon"), ("terra_per_aqua_rest", SERIES[3],
                                                                                "Rest of Brazil")):
        a.plot(df.index, df[c], marker="o", ms=3.5, lw=1.6, color=col, label=lab)
    a.set_title("2. Terra events per Aqua event, by region", loc="left", fontsize=10, color=INK)
    a = axs[2]
    for c, col, lab in (("ratio_aqua_seen_by_terra", SERIES[1], "Aqua events Terra also saw"),
                        ("ratio_aqua_only", SERIES[0], "Aqua events Terra did not see")):
        a.plot(df.index, df[c], marker="o", ms=3.5, lw=1.6, color=col, label=lab)
    a.set_title("3. Clustering of fires seen by both vs afternoon-only", loc="left", fontsize=10, color=INK)
    for a in axs:
        a.axvspan(2002.5, 2006.5, color=MUTED, alpha=0.12, lw=0)
        if a is not axs[1]:
            a.axhline(1, color=INK2, lw=0.8)
            a.set_ylabel("Clustering ratio (fire ÷ random)", color=INK2)
        a.legend(frameon=False, fontsize=8.5)
        a.grid(color=GRID, lw=0.6)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("Why does Terra not show the high clustering of 2003–2006 (shaded)?", x=0.01, ha="left", fontsize=11.5,
                 color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_terra_check.png", dpi=140)


if __name__ == "__main__":
    main()
