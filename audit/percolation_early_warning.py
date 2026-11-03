"""Early warning: do early-season fire patterns predict the rest of the season?

MODIS Aqua fire events (2 km / 2 days), Brazil, 2003-2024. Cut-offs: fires that started before 1 Jul, 1 Aug,
1 Sep, 1 Oct.
Known at the cut-off:
  ratio_sofar   clustering ratio of the fires so far (fire d50 / random d50 at the same count)
  log_n_sofar   number of fires so far (log)
  share_sofar   share of the YEAR's fires already in the largest cluster at the scaled distance (season mode of
                the growth page) -- uses the year's total only to scale, so it is reported but flagged
Targets:
  ratio_year    full-year clustering ratio (percolation_measure_years.json); partly contains the known fires
  ratio_rest    clustering ratio of the fires that start after the cut-off only (the honest forecast)
  join_day      join-up day (scaled distance; percolation_days.csv)
Scoring, leave-one-year-out (fit a straight line on 21 years, predict the 22nd):
  skill = 1 - MAE(model) / MAE(climatology), climatology = mean of the other 21 years. > 0 beats guessing the
  average. Also Spearman, and a persistence baseline (last year's value) for the ratios.

    python audit/percolation_early_warning.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from audit.fire_percolation_pilot import GRID, INK, INK2, MUTED, OUT, SERIES, YEARS, project
from audit.percolation_measure import REL, curve, random_d50
from audit.percolation_stability import events_with_dates

CUTS = {"1 Jul": "07-01", "1 Aug": "08-01", "1 Sep": "09-01", "1 Oct": "10-01"}
PREDICTORS = {"ratio_sofar": "Clustering ratio so far", "log_n_sofar": "Number of fires so far (log)"}
TARGETS = {"ratio_year": "Full-year clustering ratio", "ratio_rest": "Rest-of-season clustering ratio",
           "join_day": "Join-up day"}


def ratio_of(xy, cal):
    _, d50, n = curve(xy, REL)
    return d50 / random_d50(cal, "anywhere", n)


def loo(x, y):
    """Leave-one-out predictions from a straight line, and from the mean of the other years."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    pm, pc = np.empty(len(y)), np.empty(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        b, a = np.polyfit(x[m], y[m], 1)
        pm[i], pc[i] = a + b * x[i], y[m].mean()
    mae_m, mae_c = np.abs(pm - y).mean(), np.abs(pc - y).mean()
    return {"skill": round(float(1 - mae_m / mae_c), 3), "mae_model": round(float(mae_m), 4),
            "mae_climatology": round(float(mae_c), 4), "spearman": [round(v, 3) for v in stats.spearmanr(x, y)]}


def main():
    import matplotlib.pyplot as plt
    cal = json.loads((OUT / "percolation_measure_calibration.json").read_text())
    full = {r["year"]: r for r in json.loads((OUT / "percolation_measure_years.json").read_text())}
    days = pd.read_csv(OUT / "percolation_days.csv").set_index("year")
    summ = json.loads((OUT / "artifact_growth" / "summary.json").read_text())
    rows = []
    for y in YEARS:
        ev = events_with_dates(y)
        ev = ev[ev.start.dt.year == y]
        xy = project(ev.lon.to_numpy(), ev.lat.to_numpy())
        doy = ev.start.dt.dayofyear.to_numpy()
        for cut, md in CUTS.items():
            c = pd.Timestamp(f"{y}-{md}").dayofyear
            before = doy < c
            k = summ["snap_dates"].index(md)
            rows.append({"year": y, "cut": cut, "n_sofar": int(before.sum()), "share_of_year_sofar": before.mean(),
                         "ratio_sofar": ratio_of(xy[before], cal), "ratio_rest": ratio_of(xy[~before], cal),
                         "share_sofar": summ["years"][str(y)]["season_share"][k],
                         "ratio_year": full[y]["ratio_anywhere"], "join_day": float(days.loc[y, "scaled_join_day"])})
        print(y, flush=True)
    df = pd.DataFrame(rows)
    df["log_n_sofar"] = np.log(df.n_sofar)
    df.round(4).to_csv(OUT / "percolation_early_warning.csv", index=False)

    res = {}
    for cut in CUTS:
        d = df[df.cut == cut].sort_values("year")
        r = {"share_of_year_fires_so_far_mean": round(float(d.share_of_year_sofar.mean()), 3)}
        for t in TARGETS:
            r[t] = {p: loo(d[p], d[t]) for p in (*PREDICTORS, "share_sofar")}
            if t != "join_day":
                prev = d[t].shift(1).to_numpy()[1:]
                act = d[t].to_numpy()[1:]
                clim = np.array([d[t].to_numpy()[np.arange(len(d)) != i].mean() for i in range(1, len(d))])
                r[t]["persistence_last_year_skill"] = round(float(1 - np.abs(prev - act).mean() / np.abs(clim - act).mean()), 3)
        res[cut] = r
    (OUT / "percolation_early_warning.json").write_text(json.dumps(res, indent=1, default=float))
    for cut in CUTS:
        print(cut, "fires so far", res[cut]["share_of_year_fires_so_far_mean"])
        for t in TARGETS:
            print("  ", t, {p: (v["skill"], v["spearman"][0], v["spearman"][1]) for p, v in res[cut][t].items()
                            if isinstance(v, dict)}, "persistence", res[cut][t].get("persistence_last_year_skill"))

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    a = axs[0]
    xs = np.arange(len(CUTS))
    for (t, p), col, ls in ((("ratio_year", "ratio_sofar"), SERIES[0], "-"),
                            (("ratio_rest", "ratio_sofar"), SERIES[1], "-"),
                            (("join_day", "share_sofar"), SERIES[2], "-"),
                            (("join_day", "log_n_sofar"), SERIES[2], "--")):
        a.plot(xs, [res[c][t][p]["skill"] for c in CUTS], marker="o", color=col, ls=ls, lw=2,
               label=f"{TARGETS[t]} from {'share in largest cluster' if p == 'share_sofar' else PREDICTORS[p].lower()}")
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xticks(xs, [f"{c}\n({res[c]['share_of_year_fires_so_far_mean']:.0%} of fires)" for c in CUTS])
    a.set_ylabel("Skill vs guessing the average (leave-one-year-out)", color=INK2)
    a.set_title("How early can we tell? Above 0 = better than the long-run average", loc="left", fontsize=10,
                color=INK)
    a.legend(frameon=False, fontsize=8, loc="upper left")
    for a, (t, p, cut) in zip(axs[1:], (("ratio_rest", "ratio_sofar", "1 Aug"), ("join_day", "share_sofar", "1 Sep"))):
        d = df[df.cut == cut]
        a.scatter(d[p], d[t], s=30, color=SERIES[0], zorder=3)
        for yv, xv, tv in zip(d.year, d[p], d[t]):
            a.annotate(str(yv)[2:], (xv, tv), xytext=(3, 2), textcoords="offset points", fontsize=7, color=INK2)
        r = res[cut][t][p]
        a.set_xlabel(f"{'Share of the year’s fires in the largest cluster' if p == 'share_sofar' else PREDICTORS[p]}"
                     f", by {cut}", color=INK2)
        a.set_ylabel(TARGETS[t], color=INK2)
        a.set_title(f"{TARGETS[t]} vs what was known by {cut}\nSpearman {r['spearman'][0]:+.2f} (p {r['spearman'][1]:.3f}), "
                    f"skill {r['skill']:+.2f}", loc="left", fontsize=10, color=INK)
    for a in axs:
        a.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Early warning test: do early-season fire patterns predict the rest of the season? (MODIS Aqua, "
                 "Brazil 2003–2024)", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_early_warning.png", dpi=140)


if __name__ == "__main__":
    main()
