"""Early warning by region: inside one biome, do early-season fire patterns predict the rest of the season?

Same test as percolation_early_warning.py, but inside the Amazon biome and inside the Cerrado biome (IBGE
1:250,000 biome map), so regions whose fire seasons peak at different times are not mixed.
Region quantities (MODIS Aqua fire events, 2 km / 2 days, centre inside the biome):
  ratio        fire d50 / random d50 of the same number of points inside the biome (random d50 calibrated per
               biome as c n^b from CAL_REPS runs at CAL_SIZES)
  join_day     events added on their start day, joined within 2 x sqrt(biome area / n); first days the largest
               cluster holds 40/50/60% of the year's biome events, averaged
  share_sofar  share of the year's biome events in the largest cluster at the cut-off, same distance
Cut-offs, predictors, targets and leave-one-year-out skill as in percolation_early_warning.py.

    python audit/percolation_early_warning_regional.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from audit.fire_percolation_pilot import ALBERS, GRID, INK, INK2, OUT, SERIES, YEARS, project
from audit.percolation_days import SCALE, join_day, largest_by_day
from audit.percolation_early_warning import CUTS, loo
from audit.percolation_legal_amazon import random_in
from audit.percolation_measure import REL, curve
from audit.percolation_stability import events_with_dates
from firms import config

BIOMES = {"Amazon": 1, "Cerrado": 3}
CAL_SIZES = np.geomspace(2000, 120000, 7).astype(int)
CAL_REPS = 2
PREDICTORS = {"ratio_sofar": "clustering ratio so far", "log_n_sofar": "number of fires so far",
              "share_sofar": "share in the largest cluster so far"}
TARGETS = {"ratio_rest": "Rest-of-season clustering", "ratio_year": "Full-year clustering", "join_day": "Join-up day"}


def biome_shapes():
    import geopandas as gpd
    b = gpd.read_file(config.RAW_DIR / "biomes" / "lm_bioma_250.shp").set_crs("EPSG:4674", allow_override=True)
    b = b.to_crs(ALBERS)
    b["geometry"] = b.geometry.make_valid()
    return {name: b[b.CD_Bioma == code].geometry.make_valid().union_all().buffer(0) for name, code in BIOMES.items()}


def calibrate(shape, rng):
    rows = []
    for n in CAL_SIZES:
        for _ in range(CAL_REPS):
            _, d50, nd = curve(random_in(shape, n, rng), REL)
            rows.append((nd, d50))
    r = np.array(rows, float)
    b, a = np.polyfit(np.log(r[:, 0]), np.log(r[:, 1]), 1)
    return a, b


def main():
    import matplotlib.pyplot as plt
    import shapely
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(77)
    shapes = biome_shapes()
    cal_file = OUT / "percolation_early_warning_regional_cal.json"
    if cal_file.exists():
        cal = json.loads(cal_file.read_text())
    else:
        cal = {k: calibrate(s, rng) for k, s in shapes.items()}
        cal_file.write_text(json.dumps(cal))
    rd50 = lambda k, n: float(np.exp(cal[k][0] + cal[k][1] * np.log(n)))
    rows = []
    for y in YEARS:
        ev = events_with_dates(y)
        ev = ev[ev.start.dt.year == y]
        xy_all = project(ev.lon.to_numpy(), ev.lat.to_numpy())
        doy_all = ev.start.dt.dayofyear.to_numpy()
        for k, shape in shapes.items():
            shapely.prepare(shape)
            m = shapely.contains_xy(shape, xy_all[:, 0] * 1000, xy_all[:, 1] * 1000)
            xy, doy = xy_all[m], doy_all[m]
            n = len(xy)
            _, d50y, ny = curve(xy, REL)
            d = SCALE * np.sqrt(shape.area / 1e6 / n)
            pr = cKDTree(xy).query_pairs(d, output_type="ndarray")
            big = largest_by_day(n, pr[:, 0], pr[:, 1], doy - 1)
            jd = join_day(big, n)
            for cut, md in CUTS.items():
                c = pd.Timestamp(f"{y}-{md}").dayofyear
                before = doy < c
                _, d50b, nb = curve(xy[before], REL)
                _, d50a, na = curve(xy[~before], REL)
                rows.append({"region": k, "year": y, "cut": cut, "events": n, "n_sofar": nb,
                             "share_of_year_sofar": nb / n, "ratio_sofar": d50b / rd50(k, nb),
                             "ratio_rest": d50a / rd50(k, na), "ratio_year": d50y / rd50(k, ny),
                             "share_sofar": big[c - 2] / n, "join_day": jd})
        print(y, flush=True)
    df = pd.DataFrame(rows)
    df["log_n_sofar"] = np.log(df.n_sofar)
    df.round(4).to_csv(OUT / "percolation_early_warning_regional.csv", index=False)

    res = {}
    for k in BIOMES:
        res[k] = {}
        for cut in CUTS:
            d = df[(df.region == k) & (df.cut == cut)].sort_values("year")
            r = {"share_of_year_fires_so_far": round(float(d.share_of_year_sofar.mean()), 3)}
            for t in TARGETS:
                dd = d.dropna(subset=[t])
                r[t] = {p: loo(dd[p], dd[t]) for p in PREDICTORS}
                r[t]["n_years"] = int(len(dd))
            res[k][cut] = r
    (OUT / "percolation_early_warning_regional.json").write_text(json.dumps(res, indent=1, default=float))
    for k in BIOMES:
        for cut in CUTS:
            print(k, cut, res[k][cut]["share_of_year_fires_so_far"])
            for t in TARGETS:
                print("   ", t, {p: (v["skill"], v["spearman"][0], v["spearman"][1]) for p, v in res[k][cut][t].items()
                                 if isinstance(v, dict)}, "n", res[k][cut][t]["n_years"])

    fig, axs = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    xs = np.arange(len(CUTS))
    for a, k in zip(axs, BIOMES):
        for (t, p), col, ls in ((("ratio_rest", "ratio_sofar"), SERIES[1], "-"),
                                (("ratio_year", "ratio_sofar"), SERIES[0], "-"),
                                (("join_day", "share_sofar"), SERIES[2], "-"),
                                (("join_day", "log_n_sofar"), SERIES[2], "--")):
            a.plot(xs, [res[k][c][t][p]["skill"] for c in CUTS], marker="o", color=col, ls=ls, lw=2,
                   label=f"{TARGETS[t]} from {PREDICTORS[p]}")
        a.axhline(0, color=INK2, lw=0.8)
        a.set_xticks(xs, [f"{c}\n({res[k][c]['share_of_year_fires_so_far']:.0%} of fires)" for c in CUTS])
        a.set_title(f"{k} biome", loc="left", fontsize=11, color=INK)
        a.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    axs[0].set_ylabel("Skill vs guessing the average (leave-one-year-out)", color=INK2)
    axs[1].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("Early warning inside one biome: does what is known by each date beat the long-run average? "
                 "(MODIS Aqua, 2003–2024)", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_early_warning_regional.png", dpi=140)


if __name__ == "__main__":
    main()
