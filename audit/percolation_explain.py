"""Explain: do the changes in fire connectivity follow the fire count, climate or deforestation?

Outcomes (percolation_measure_years.json): clustering ratio (fire d50 / random d50 at the same count, with SE) and
transition width (d80 / d20). Periods from percolation_change.py: 2003-2006, 2007-2015, 2016-2024.
Candidates, one value per year:
  events         number of fire events (log)
  climate        Jul-Oct means where fire usually burns (fixed long-run fire weights): rain, temperature,
                 relative humidity, wind, soil moisture (percolation_climate_years.csv)
  deforestation  PRODES Legal Amazon rate, Aug of the previous year to Jul (log)
  amazon_share   share of Brazil's fire events inside the Legal Amazon (where fire happens)
Tests per candidate and outcome:
  levels   Spearman across the 22 years
  changes  Pearson of year-to-year differences (removes slow shared drifts; a link that is only a match between
           eras disappears here)
  periods  2007-2015 vs other years (Welch t), i.e. does the candidate shift with the connectivity periods
  within   OLS outcome ~ candidate (standardised) + period dummies, HAC SE (2 lags): does it explain anything
           inside the periods, beyond the period shifts
Plus the main model: ratio ~ log deforestation + log events + climate summary (first principal component of the
five climate variables), HAC SE, and the same with period dummies.

    python audit/percolation_explain.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from audit.fire_percolation_pilot import GRID, INK, INK2, MUTED, OUT, SERIES
from audit.percolation_vs_deforestation import prodes

PERIODS = {"2003–2006": range(2003, 2007), "2007–2015": range(2007, 2016), "2016–2024": range(2016, 2025)}
CLIMATE = {"rain_mm": "Dry-season rain (mm/month)", "temp_c": "Dry-season temperature (°C)",
           "rh_pct": "Dry-season humidity (%)", "wind_ms": "Dry-season wind (m/s)",
           "soil_0_7cm": "Dry-season soil moisture"}
CANDS = {"log_events": "Fire count (log events)", **CLIMATE, "log_defor": "Deforestation (log km², PRODES)",
         "amazon_share": "Share of fires in the Legal Amazon"}
GROUP = {"log_events": "fire count", **{k: "climate" for k in CLIMATE}, "log_defor": "deforestation",
         "amazon_share": "where fire happens"}


def ols(y, X, hac=2):
    import statsmodels.api as sm
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": hac})


def main():
    import matplotlib.pyplot as plt
    recs = json.loads((OUT / "percolation_measure_years.json").read_text())
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("curve")} for r in recs]).set_index("year")
    cl = pd.read_csv(OUT / "percolation_climate_years.csv").set_index("year")[list(CLIMATE)]
    la = pd.read_csv(OUT / "percolation_legal_amazon.csv").set_index("year")
    df = df.join(cl)
    df["log_events"] = np.log(df.events)
    df["log_defor"] = np.log(prodes().reindex(df.index))
    df["amazon_share"] = la.share_of_brazil.reindex(df.index)
    df["period"] = ""
    for name, yrs in PERIODS.items():
        df.loc[df.index.isin(yrs), "period"] = name
    mid = (df.period == "2007–2015").to_numpy()
    dummies = pd.get_dummies(df.period, drop_first=True, dtype=float)
    z = (df[list(CLIMATE)] - df[list(CLIMATE)].mean()) / df[list(CLIMATE)].std()
    u, s, vt = np.linalg.svd(z.to_numpy(), full_matrices=False)
    df["climate_pc1"] = u[:, 0] * s[0]
    pc1_loadings = dict(zip(CLIMATE, np.round(vt[0], 3)))
    pc1_share = float(s[0] ** 2 / (s ** 2).sum())

    rows = []
    for out in ("ratio_anywhere", "width"):
        y = df[out]
        for c in CANDS:
            x = df[c]
            sp = stats.spearmanr(x, y)
            ch = stats.pearsonr(x.diff().dropna(), y.diff().dropna())
            t = stats.ttest_ind(x[mid], x[~mid], equal_var=False)
            xs = (x - x.mean()) / x.std()
            w = ols(y, pd.concat([xs.rename("x"), dummies], axis=1))
            rows.append({"outcome": out, "candidate": c, "group": GROUP[c],
                         "levels_spearman": round(sp[0], 2), "levels_p": round(sp[1], 3),
                         "changes_r": round(ch[0], 2), "changes_p": round(ch[1], 3),
                         "mid_period_shift_sd": round(float((x[mid].mean() - x[~mid].mean()) / x.std()), 2),
                         "mid_period_p": round(float(t.pvalue), 4),
                         "within_periods_coef": round(float(w.params.x), 3), "within_periods_p": round(float(w.pvalues.x), 3)})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "percolation_explain.csv", index=False)

    y = df.ratio_anywhere
    X = df[["log_defor", "log_events", "climate_pc1"]]
    Xs = (X - X.mean()) / X.std()
    m1 = ols(y, Xs)
    m2 = ols(y, pd.concat([Xs, dummies], axis=1))
    m0 = ols(y, dummies)
    res = {"climate_pc1": {"loadings": pc1_loadings, "share_of_climate_variance": round(pc1_share, 3)},
           "model_drivers": {"r2": round(m1.rsquared, 3), "coef_per_sd": m1.params.round(3).to_dict(),
                             "p": m1.pvalues.round(4).to_dict()},
           "model_periods_only": {"r2": round(m0.rsquared, 3)},
           "model_drivers_plus_periods": {"r2": round(m2.rsquared, 3), "coef_per_sd": m2.params.round(3).to_dict(),
                                          "p": m2.pvalues.round(4).to_dict()},
           "table": rows}
    (OUT / "percolation_explain.json").write_text(json.dumps(res, indent=1, default=float))
    print(tab[tab.outcome == "ratio_anywhere"].drop(columns="outcome").to_string(index=False))
    print(tab[tab.outcome == "width"].drop(columns="outcome").to_string(index=False))
    print(json.dumps({k: v for k, v in res.items() if k != "table"}, indent=1, default=float))

    # figure: each candidate over time (period means) and against the ratio
    show = ["log_events", "rain_mm", "temp_c", "rh_pct", "log_defor", "amazon_share"]
    fig, axs = plt.subplots(2, len(show), figsize=(20, 7.6))
    pcol = dict(zip(PERIODS, [SERIES[0], SERIES[1], SERIES[0]]))
    for j, c in enumerate(show):
        x = df[c]
        a = axs[0, j]
        a.plot(df.index, x, color=MUTED, lw=1)
        a.scatter(df.index, x, s=16, color=[pcol[p] for p in df.period], zorder=3)
        for name, yrs in PERIODS.items():
            v = x[df.index.isin(yrs)]
            a.hlines(v.mean(), min(yrs) - 0.4, max(yrs) + 0.4, color=pcol[name], lw=2)
        a.axvspan(2006.5, 2015.5, color=MUTED, alpha=0.1, lw=0)
        a.set_title(CANDS[c], loc="left", fontsize=9.5, color=INK)
        r = tab[(tab.outcome == "ratio_anywhere") & (tab.candidate == c)].iloc[0]
        a = axs[1, j]
        a.errorbar(x, y, yerr=1.96 * df.ratio_anywhere_se, fmt="none", ecolor=MUTED, lw=0.7)
        a.scatter(x, y, s=22, color=[pcol[p] for p in df.period], zorder=3)
        for yr, xv, yv in zip(df.index, x, y):
            a.annotate(str(yr)[2:], (xv, yv), xytext=(3, 2), textcoords="offset points", fontsize=6.5, color=INK2)
        strong = min(r.levels_p, r.changes_p) < 0.05
        a.set_title(f"levels ρ {r.levels_spearman:+.2f} (p {r.levels_p:.2f})\nchanges r {r.changes_r:+.2f} "
                    f"(p {r.changes_p:.2f})\nwithin periods p {r.within_periods_p:.2f}", loc="left", fontsize=8.5,
                    color=INK if strong else INK2, weight="bold" if strong else "normal")
        a.set_xlabel(CANDS[c], color=INK2, fontsize=8.5)
    axs[1, 0].set_ylabel("Clustering ratio (fire ÷ random)", color=INK2)
    for a in axs.ravel():
        a.grid(color=GRID, lw=0.5)
        for sp_ in ("top", "right"):
            a.spines[sp_].set_visible(False)
    from matplotlib.lines import Line2D
    axs[0, 0].legend(handles=[Line2D([], [], marker="o", ls="", color=SERIES[1], label="2007–2015"),
                              Line2D([], [], marker="o", ls="", color=SERIES[0], label="2003–06 and 2016–24"),
                              Line2D([], [], color=MUTED, lw=2, label="Period mean")],
                     frameon=False, fontsize=8, loc="best")
    fig.suptitle("What lines up with the connectivity periods? Top: each candidate by year (shaded 2007–2015, lines = "
                 "period means). Bottom: clustering ratio against it. Bold = p < 0.05", x=0.01, ha="left",
                 fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_explain.png", dpi=130)


if __name__ == "__main__":
    main()
