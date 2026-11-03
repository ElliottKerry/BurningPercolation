"""Model-based comparison: does a simple model of WHERE fire happens also reproduce HOW CLUSTERED it is?

Step 1, count model (Poisson GLM), unit = 0.25 deg cell x year, Brazil, 2003-2024:
  events_cy ~ year effects                                             M0  (fire spread evenly over cells)
            + log(1 + cell's mean events in the OTHER years)           M1  (where fire usually is)
            + dry-season climate anomalies of the cell (Jul-Oct mean    M2  (that year's weather, per place)
              minus the cell's own 2003-2024 mean, standardised): rain, temperature, humidity, wind, soil moisture
  Events: MODIS Aqua fire events (2 km / 2 days) starting in the year, by the cell of their position.
  Climate: ERA5 monthly (era5_brazil.py). Deforestation / land cover are not in yet (MapBiomas downloads suspended).
Step 2, simulate: for each year and model, place exactly that year's number of events in cells with
  probabilities proportional to the model's fitted means (multinomial), uniformly inside each cell (inside Brazil).
Step 3, compare: clustering ratio (d50 / random-anywhere d50 at the same count) of the simulated fires (REPS runs,
  median) against the observed ratio; and the residual clustering = observed d50 / model d50.
Benchmark 'oracle': the ACTUAL number of events in each cell, scattered at random inside the cell -- the best any
0.25 deg cell model could do. Clustering finer than a cell (d50 is ~10-15 km, cells ~28 km) is out of reach of
any cell model; the oracle shows how much of the pattern lives between cells (which cells burn).
If a model reproduces the observed ratio and its three periods, its ingredients explain the clustering pattern.

    python audit/percolation_model.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if "--events" not in sys.argv:
    sys.argv.append("--events")

import numpy as np
import pandas as pd

from audit.fire_percolation_pilot import GRID, INK, INK2, MUTED, OUT, SERIES, YEARS, project
from audit.percolation_lattice_threshold import RES, brazil_cells
from audit.percolation_measure import REL, curve, random_d50
from audit.percolation_stability import events_with_dates
from firms import config

CLIMATE = ["rain_mm", "temp_c", "rh_pct", "wind_ms", "soil_0_7cm"]
DRY = [7, 8, 9, 10]
REPS = 3
PERIODS = {"2003–06": range(2003, 2007), "2007–15": range(2007, 2016), "2016–24": range(2016, 2025)}
MODELS = {"M0": "Year only (fire spread evenly)", "M1": "+ where fire usually is",
          "M2": "+ that year's local climate"}


def panel():
    mask, ix0, iy0 = brazil_cells()
    yy, xx = np.nonzero(mask)
    cells = pd.DataFrame({"ix": xx + ix0, "iy": iy0 + yy})
    rows, pts = [], {}
    for y in YEARS:
        ev = events_with_dates(y)
        ev = ev[ev.start.dt.year == y]
        ix = np.floor(ev.lon.to_numpy() / RES + 0.5).astype(int)
        iy = np.floor(ev.lat.to_numpy() / RES + 0.5).astype(int)
        c = pd.DataFrame({"ix": ix, "iy": iy}).value_counts().rename("events").reset_index()
        d = cells.merge(c, on=["ix", "iy"], how="left").fillna({"events": 0})
        d["year"] = y
        rows.append(d)
        pts[y] = project(ev.lon.to_numpy(), ev.lat.to_numpy())
    p = pd.concat(rows, ignore_index=True)
    tot = p.groupby(["ix", "iy"]).events.transform("sum")
    p["baseline"] = np.log1p((tot - p.events) / (len(YEARS) - 1))
    cl = pd.read_parquet(config.DATA_DIR / "cache" / "era5_brazil_monthly.parquet")
    cl = cl[cl.month.dt.month.isin(DRY)]
    cl = cl.assign(year=cl.month.dt.year).groupby(["ix", "iy", "year"])[CLIMATE].mean().reset_index()
    for v in CLIMATE:
        a = cl[v] - cl.groupby(["ix", "iy"])[v].transform("mean")
        cl[v + "_anom"] = a / a.std()
    p = p.merge(cl[["ix", "iy", "year", *[v + "_anom" for v in CLIMATE]]], on=["ix", "iy", "year"], how="left")
    p = p.fillna({v + "_anom": 0.0 for v in CLIMATE})
    return p, pts


def fit(p):
    import statsmodels.api as sm
    yd = pd.get_dummies(p.year, prefix="y", dtype=float)
    anom = [v + "_anom" for v in CLIMATE]
    specs = {"M0": [], "M1": ["baseline"], "M2": ["baseline", *anom]}
    out, info = {}, {}
    for k, xs in specs.items():
        X = pd.concat([yd, p[xs]], axis=1)
        f = sm.GLM(p.events, X, family=sm.families.Poisson()).fit(cov_type="HC0")
        out[k] = f.fittedvalues.to_numpy()
        info[k] = {"deviance_explained": round(float(1 - f.deviance / f.null_deviance), 4),
                   "pearson_dispersion": round(float(f.pearson_chi2 / f.df_resid), 2)}
        if xs:
            info[k]["rate_ratio_per_unit"] = {x: [round(float(np.exp(f.params[x])), 3),
                                                  round(float(np.exp(f.params[x] - 1.96 * f.bse[x])), 3),
                                                  round(float(np.exp(f.params[x] + 1.96 * f.bse[x])), 3)] for x in xs}
    return out, info


def main():
    import matplotlib.pyplot as plt
    from audit.fire_percolation_expected import Sampler
    cal = json.loads((OUT / "percolation_measure_calibration.json").read_text())
    p, pts = panel()
    mu, info = fit(p)
    print(json.dumps(info, indent=1))
    S = Sampler()
    rng = np.random.default_rng(21)
    obs = {r["year"]: r for r in json.loads((OUT / "percolation_measure_years.json").read_text())}
    rows = []
    for y in YEARS:
        m = (p.year == y).to_numpy()
        n = len(pts[y])
        cix, ciy = p.ix.to_numpy()[m], p.iy.to_numpy()[m]
        r = {"year": y, "events": n, "observed_ratio": obs[y]["ratio_anywhere"],
             "observed_ratio_se": obs[y]["ratio_anywhere_se"], "observed_d50": obs[y]["d50_km"]}
        cnt = p.events.to_numpy()[m].astype(int)
        k_idx = np.repeat(np.arange(len(cnt)), cnt)
        d = np.median([curve(S.in_cells(cix[k_idx], ciy[k_idx], rng), REL)[1] for _ in range(REPS)])
        r["oracle_ratio"] = float(d) / random_d50(cal, "anywhere", n)
        for k in MODELS:
            w = mu[k][m] / mu[k][m].sum()
            d50s = []
            for _ in range(REPS):
                cnt = rng.multinomial(n, w)
                k_idx = np.repeat(np.arange(len(w)), cnt)
                xy = S.in_cells(cix[k_idx], ciy[k_idx], rng)
                d50s.append(curve(xy, REL)[1])
            d50 = float(np.median(d50s))
            r[f"{k}_ratio"] = d50 / random_d50(cal, "anywhere", n)
            r[f"{k}_residual"] = obs[y]["d50_km"] / d50
        rows.append(r)
        print(y, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    df = pd.DataFrame(rows).set_index("year")
    df.round(4).to_csv(OUT / "percolation_model_years.csv")

    from scipy import stats
    res = {"glm": info, "periods": {}, "agreement": {}}
    for c in ["observed_ratio", "oracle_ratio", *[f"{k}_ratio" for k in MODELS], *[f"{k}_residual" for k in MODELS]]:
        res["periods"][c] = {p_: round(float(df[c][df.index.isin(yrs)].mean()), 3) for p_, yrs in PERIODS.items()}
    mid = df.index.isin(range(2007, 2016))
    res["agreement"]["oracle"] = {"spearman_with_observed": [round(v, 3) for v in stats.spearmanr(df.observed_ratio,
                                                                                                   df.oracle_ratio)]}
    for k in MODELS:
        sp = stats.spearmanr(df.observed_ratio, df[f"{k}_ratio"])
        res["agreement"][k] = {"spearman_with_observed": [round(sp[0], 3), round(sp[1], 4)],
                               "spearman_with_oracle": [round(v, 3) for v in stats.spearmanr(df.oracle_ratio, df[f"{k}_ratio"])],
                               "residual_2007_15_vs_other_p": round(float(stats.ttest_ind(
                                   df[f"{k}_residual"][mid], df[f"{k}_residual"][~mid], equal_var=False).pvalue), 5)}
    (OUT / "percolation_model.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "glm"}, indent=1))

    fig, axs = plt.subplots(1, 2, figsize=(17, 6.2))
    a = axs[0]
    a.errorbar(df.index, df.observed_ratio, yerr=1.96 * df.observed_ratio_se, fmt="o-", color=INK, lw=1.6, ms=4,
               capsize=2, label="Observed fire")
    for k, col in zip(MODELS, (MUTED, SERIES[1], SERIES[0])):
        a.plot(df.index, df[f"{k}_ratio"], marker="s", ms=3.5, lw=1.4, color=col,
               label=f"{k}: {MODELS[k]} (ρ {res['agreement'][k]['spearman_with_observed'][0]:+.2f})")
    a.plot(df.index, df.oracle_ratio, marker="^", ms=4, lw=1.6, ls="--", color=SERIES[2],
           label=f"Best possible cell model: actual counts per cell (ρ {res['agreement']['oracle']['spearman_with_observed'][0]:+.2f})")
    a.axhline(1, color=INK2, lw=0.8)
    a.axvspan(2006.5, 2015.5, color=MUTED, alpha=0.1, lw=0)
    a.set_ylabel("Clustering ratio (fire ÷ random anywhere)", color=INK2)
    a.set_title("Does the model's simulated fire cluster like real fire?", loc="left", fontsize=10.5, color=INK)
    a.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    a = axs[1]
    for k, col in zip(MODELS, (MUTED, SERIES[1], SERIES[0])):
        a.plot(df.index, df[f"{k}_residual"], marker="o", ms=3.5, lw=1.4, color=col, label=f"{k}: {MODELS[k]}")
        for p_, yrs in PERIODS.items():
            v = df[f"{k}_residual"][df.index.isin(yrs)]
            a.hlines(v.mean(), min(yrs) - 0.4, max(yrs) + 0.4, color=col, lw=2.2, alpha=0.7)
    a.axhline(1, color=INK2, lw=0.8)
    a.axvspan(2006.5, 2015.5, color=MUTED, alpha=0.1, lw=0)
    a.set_ylabel("Observed d50 ÷ model d50", color=INK2)
    a.set_title("Clustering the model does NOT explain (1 = fully explained); lines = period means", loc="left",
                fontsize=10.5, color=INK)
    a.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    for a in axs:
        a.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("A model of where fire happens (Poisson GLM on 0.25° cells, MODIS Aqua 2003–2024), used to simulate "
                 "fire and test whether it reproduces the clustering", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_model.png", dpi=140)


if __name__ == "__main__":
    main()
