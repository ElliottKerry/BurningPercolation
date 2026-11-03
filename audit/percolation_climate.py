"""Q2: does climate explain the year-to-year (and month-to-month) differences in fire connectivity?

Climate per cell-month: ERA5 monthly means on the same 0.25 deg cells (audit/era5_brazil.py): rain (mm),
2 m temperature (C), relative humidity (%), 10 m wind speed (m/s), soil moisture 0-7 cm (m3/m3).

Brazil-wide climate is summarised where fire happens: a weighted mean over cells, weights = each cell's share of
all fire events 2003-2024 (fixed over time, so a year's own fire does not set its own weights).

Annual (22 years). Connectivity: d50 of fire events (distance method), the fire / random ratio, lattice k* and
p*. Climate: dry-season (Jul-Oct) mean of the fire-weighted series. Steps:
  1. scatterplots + Spearman correlations;
  2. regression log(d50) ~ log(events) + one climate variable (standardised), one variable at a time,
     then all five together. log(events) is the control for fire totals.
Monthly (264 months). Connectivity: d50 of that month's events (percolation_stability_months.csv).
  log(d50) ~ log(events) + calendar-month effects (seasonality) + climate anomaly (standardised),
  Newey-West standard errors (3 lags). Study area is all of Brazil every year, so area is constant here.

    python audit/percolation_climate.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from audit.fire_percolation_pilot import GRID, INK, INK2, OUT, SERIES, YEARS
from audit.percolation_lattice_threshold import brazil_cells, counts
from firms import config

CLIMATE = {"rain_mm": "Rain (mm/month)", "temp_c": "Temperature (°C)", "rh_pct": "Relative humidity (%)",
           "wind_ms": "Wind speed (m/s)", "soil_0_7cm": "Soil moisture (m³/m³)"}
DRY = [7, 8, 9, 10]


def fire_weighted_climate():
    mask, ix0, iy0 = brazil_cells()
    total = sum(counts(y, mask, ix0, iy0) for y in YEARS)
    yy, xx = np.nonzero(mask)
    w = pd.DataFrame({"ix": xx + ix0, "iy": yy + iy0, "w": total[yy, xx] / total[mask].sum()})
    c = pd.read_parquet(config.DATA_DIR / "cache" / "era5_brazil_monthly.parquet").merge(w)
    out = c.groupby("month").apply(lambda g: pd.Series({v: np.average(g[v], weights=g.w) for v in CLIMATE}),
                                   include_groups=False)
    return out


def ols(y, X, hac=None):
    import statsmodels.api as sm
    m = sm.OLS(y, sm.add_constant(X), missing="drop")
    return m.fit(cov_type="HAC", cov_kwds={"maxlags": hac}) if hac else m.fit()


def main():
    import matplotlib.pyplot as plt
    clim = fire_weighted_climate()
    clim.to_csv(OUT / "climate_fire_weighted_monthly.csv")

    # ---------------- annual
    yr = pd.read_csv(OUT / "percolation_by_year.csv").set_index("year")
    dry = clim[clim.index.month.isin(DRY)].groupby(clim[clim.index.month.isin(DRY)].index.year).mean()
    a = yr.join(dry)
    a.to_csv(OUT / "percolation_climate_years.csv")
    metrics = {"d50_events_km": "d50 (km)", "ratio_events": "Fire / random", "k_star": "k*", "p_star": "p*"}
    spear = {m: {v: dict(zip(("rho", "p"), map(float, stats.spearmanr(a[v], a[m])))) for v in CLIMATE}
             for m in [*metrics, "events"]}
    z = (a[list(CLIMATE)] - a[list(CLIMATE)].mean()) / a[list(CLIMATE)].std()
    reg = {}
    for v in CLIMATE:
        f = ols(np.log(a.d50_events_km), pd.DataFrame({"log_events": np.log(a.events), v: z[v]}))
        reg[v] = {"coef": float(f.params[v]), "p": float(f.pvalues[v]), "log_events_coef": float(f.params.log_events),
                  "r2": float(f.rsquared)}
    base = ols(np.log(a.d50_events_km), pd.DataFrame({"log_events": np.log(a.events)}))
    full = ols(np.log(a.d50_events_km), pd.concat([np.log(a.events).rename("log_events"), z], axis=1))
    reg["_events_only_r2"] = float(base.rsquared)
    reg["_all_five"] = {"r2": float(full.rsquared), "adj_r2": float(full.rsquared_adj),
                        "coef": {k: float(v) for k, v in full.params.items()},
                        "p": {k: float(v) for k, v in full.pvalues.items()},
                        "F_test_climate_p": float(full.compare_f_test(base)[1])}

    # ---------------- monthly
    mo = pd.read_csv(OUT / "percolation_stability_months.csv", parse_dates=["date"]).set_index("date")
    mo = mo.join(clim)
    mo["moy"] = mo.index.month
    anom = mo[list(CLIMATE)] - mo.groupby("moy")[list(CLIMATE)].transform("mean")
    anom = anom / anom.std()
    dummies = pd.get_dummies(mo.moy, prefix="m", drop_first=True).astype(float)
    y = np.log(mo.fire_d50)
    mbase = ols(y, pd.concat([np.log(mo.events).rename("log_events"), dummies], axis=1), hac=3)
    monthly = {"_events_and_season_r2": float(mbase.rsquared)}
    for v in CLIMATE:
        f = ols(y, pd.concat([np.log(mo.events).rename("log_events"), dummies, anom[v]], axis=1), hac=3)
        monthly[v] = {"coef": float(f.params[v]), "p": float(f.pvalues[v]), "r2": float(f.rsquared)}
    mfull = ols(y, pd.concat([np.log(mo.events).rename("log_events"), dummies, anom], axis=1), hac=3)
    monthly["_all_five"] = {"r2": float(mfull.rsquared), "coef": {v: float(mfull.params[v]) for v in CLIMATE},
                            "p": {v: float(mfull.pvalues[v]) for v in CLIMATE}}
    mo_rho = {v: dict(zip(("rho", "p"), map(float, stats.spearmanr(anom[v], mo.rho, nan_policy="omit"))))
              for v in CLIMATE}

    res = {"annual_spearman": spear, "annual_regression_log_d50": reg, "monthly_regression_log_d50": monthly,
           "monthly_spearman_ratio_vs_climate_anomaly": mo_rho}
    (OUT / "percolation_climate.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

    # ---------------- figure: annual scatterplots
    fig, axs = plt.subplots(len(metrics), len(CLIMATE), figsize=(17, 13))
    early = a.index <= 2008
    for r, (m, mlab) in enumerate(metrics.items()):
        for c, (v, vlab) in enumerate(CLIMATE.items()):
            ax = axs[r, c]
            ax.scatter(a[v][~early], a[m][~early], s=26, color=SERIES[0], zorder=3, label="2009–2024")
            ax.scatter(a[v][early], a[m][early], s=26, color=SERIES[1], zorder=3, label="2003–2008")
            for yv, row in a.iterrows():
                ax.annotate(str(yv)[2:], (row[v], row[m]), xytext=(3, 2), textcoords="offset points", fontsize=7,
                            color=INK2)
            s = spear[m][v]
            ax.set_title(f"Spearman {s['rho']:+.2f} (p = {s['p']:.3f})", loc="left", fontsize=9,
                         color=INK if s["p"] < 0.05 else INK2, weight="bold" if s["p"] < 0.05 else "normal")
            if r == len(metrics) - 1:
                ax.set_xlabel(vlab + ", Jul–Oct", color=INK2)
            if c == 0:
                ax.set_ylabel(mlab, color=INK2)
            ax.grid(color=GRID, lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    axs[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Fire connectivity against dry-season climate, one dot per year 2003–2024 (climate averaged over "
                 "where fire usually is). Bold = p < 0.05", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_climate_scatter.png", dpi=130)


if __name__ == "__main__":
    main()
