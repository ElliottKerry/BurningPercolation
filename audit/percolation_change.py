"""Describe change: how fire connectivity changed 2003-2024 after adjusting for the fire count.

Inputs: percolation_measure_years.json (per year: d50, ratio to random with SE, transition width with SE).
Count adjustment:
  ratio  = fire d50 / random d50 at the same number of points (random d50 ~ c n^-1/2), so it is count-free by
           construction; checked by its correlation with the fire count.
  width  = d80 / d20 (distance where the largest cluster holds 80% over where it holds 20%); a ratio of two
           distances, so the density scale cancels. Random points have a width near 1 (a sudden jump).
  raw d50: regressed on log(events) to show how much of the raw change is the count.
Change, for the ratio and the width:
  1. trend   Kendall's tau with year, Sen's slope.
  2. periods piecewise-constant mean with 0-3 breaks (segments >= MIN_SEG years), chosen by BIC; the data pick
             the break years. Break-year uncertainty: residual bootstrap (BOOT runs) refitting the chosen model.
  3. means   per period with 95% CI (years as units), plus the 2007-2015 window chosen earlier by eye.

    python audit/percolation_change.py
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from audit.fire_percolation_pilot import GRID, INK, INK2, MUTED, OUT, SERIES
from audit.percolation_measure import NULL_NAME, REL

MIN_SEG = 3
MAX_BREAKS = 3
BOOT = 2000


def fit_segments(y, breaks):
    """Piecewise-constant fit for given break positions (index where a new segment starts)."""
    edges = [0, *breaks, len(y)]
    fitted = np.concatenate([np.full(b - a, y[a:b].mean()) for a, b in zip(edges[:-1], edges[1:])])
    return fitted, float(((y - fitted) ** 2).sum())


def best_breaks(y, k):
    n = len(y)
    best = (None, np.inf)
    for br in itertools.combinations(range(MIN_SEG, n - MIN_SEG + 1), k):
        if any(b - a < MIN_SEG for a, b in zip((0, *br), (*br, n))):
            continue
        _, rss = fit_segments(y, br)
        if rss < best[1]:
            best = (br, rss)
    return best


def periods(y, years, rng):
    n = len(y)
    out = {}
    for k in range(MAX_BREAKS + 1):
        br, rss = best_breaks(y, k)
        out[k] = {"breaks": list(br), "rss": rss, "bic": n * np.log(rss / n) + (2 * k + 1) * np.log(n)}
    k = min(out, key=lambda q: out[q]["bic"])
    br = out[k]["breaks"]
    fitted, _ = fit_segments(y, br)
    resid = y - fitted
    boot = []
    for _ in range(BOOT if k else 0):
        ys = fitted + rng.choice(resid, n, replace=True)
        boot.append(best_breaks(ys, k)[0])
    boot = np.array(boot)
    res = {"bic_by_breaks": {q: round(v["bic"], 2) for q, v in out.items()}, "chosen_breaks": k,
           "first_years_of_new_periods": [int(years[b]) for b in br], "fitted": fitted.tolist()}
    if k:
        res["break_year_bootstrap"] = []
        for j, b in enumerate(br):
            yrs = years[boot[:, j]]
            res["break_year_bootstrap"].append({"estimate": int(years[b]),
                                                "within_1_year": float(np.mean(np.abs(boot[:, j] - b) <= 1)),
                                                "90pct_range": [int(np.percentile(yrs, 5)), int(np.percentile(yrs, 95))]})
    edges = [0, *br, n]
    res["period_means"] = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = y[a:b]
        half = stats.t.ppf(0.975, len(seg) - 1) * seg.std(ddof=1) / np.sqrt(len(seg))
        res["period_means"].append({"years": f"{years[a]}-{years[b - 1]}", "mean": round(float(seg.mean()), 3),
                                    "ci95": [round(float(seg.mean() - half), 3), round(float(seg.mean() + half), 3)]})
    return res


def sen_slope(y, x):
    s = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x))]
    return float(np.median(s))


def random_width(kind):
    cal = json.loads((OUT / "percolation_measure_calibration.json").read_text())
    w = []
    for c in cal[kind]["rel_curves"]:
        c = np.maximum.accumulate(np.array(c))
        d20, d80 = np.interp(0.2, c, REL), np.interp(0.8, c, REL)
        w.append(d80 / d20)
    return float(np.median(w))


def main():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(11)
    recs = json.loads((OUT / "percolation_measure_years.json").read_text())
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("curve")} for r in recs]).set_index("year")
    years = df.index.to_numpy()
    res = {}

    # raw d50 and the fire count
    lr = stats.linregress(np.log(df.events), np.log(df.d50_km))
    resid = np.log(df.d50_km) - (lr.intercept + lr.slope * np.log(df.events))
    res["raw_d50"] = {"slope_log_d50_on_log_events": round(lr.slope, 3), "r2": round(lr.rvalue ** 2, 3),
                      "expected_slope_if_only_count": -0.5,
                      "kendall_raw_d50_vs_year": [round(v, 3) for v in stats.kendalltau(years, df.d50_km)],
                      "kendall_events_vs_year": [round(v, 3) for v in stats.kendalltau(years, df.events)],
                      "kendall_residual_vs_year": [round(v, 3) for v in stats.kendalltau(years, resid)]}

    res["random_width"] = {k: round(random_width(k), 2) for k in NULL_NAME}
    for col in ("ratio_anywhere", "width"):
        y = df[col].to_numpy()
        tau, p = stats.kendalltau(years, y)
        r = {"count_check_spearman_with_events": [round(v, 3) for v in stats.spearmanr(df.events, y)],
             "trend_kendall_tau": round(tau, 3), "trend_p": round(p, 4),
             "sen_slope_per_decade": round(10 * sen_slope(y, years), 3)}
        r.update(periods(y, years, rng))
        era = df.index.isin(range(2007, 2016))
        t = stats.ttest_ind(y[era], y[~era], equal_var=False)
        r["by_eye_2007_2015"] = {"mean": round(float(y[era].mean()), 3), "other": round(float(y[~era].mean()), 3),
                                 "welch_p": float(t.pvalue)}
        res[col] = r
    corr = stats.spearmanr(df.ratio_anywhere, df.width)
    res["ratio_vs_width_spearman"] = [round(v, 3) for v in corr]
    (OUT / "percolation_change.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    a = axs[0]
    a.errorbar(years, df.d50_km, yerr=1.96 * df.d50_km_se, fmt="o-", color=SERIES[0], ms=4, lw=1.2, capsize=2,
               label="Actual fire d50")
    a.plot(years, np.exp(lr.intercept + lr.slope * np.log(df.events)), color=MUTED, lw=1.6, ls="--",
           label=f"Predicted from fire count alone (R² {lr.rvalue ** 2:.2f})")
    a.set_ylabel("km", color=INK2)
    a.set_title("1. Raw half-joined distance: rises mostly because\nthere are fewer fires", loc="left", fontsize=10,
                color=INK)
    a.legend(frameon=False, fontsize=8.5, loc="upper left")
    for a, col, lab, ref in ((axs[1], "ratio_anywhere", "2. Clustering ratio (fire ÷ random, same count)", 1.0),
                             (axs[2], "width", "3. Transition width (d80 ÷ d20). Random ≈ "
                              f"{res['random_width']['anywhere']:.1f}", None)):
        r = res[col]
        a.errorbar(years, df[col], yerr=1.96 * df[f"{col}_se"], fmt="o", color=SERIES[0], ms=4, lw=1.1, capsize=2,
                   label="Each year, 95% interval")
        a.step(np.r_[years - 0.5, years[-1] + 0.5], np.r_[r["fitted"], r["fitted"][-1]], where="post",
               color=SERIES[1], lw=2, label="Periods chosen by the data (BIC)")
        if ref:
            a.axhline(ref, color=INK2, lw=0.8)
        brk = ", ".join(str(b) for b in r["first_years_of_new_periods"]) or "none"
        a.set_title(f"{lab}\nnew periods start: {brk}; trend τ {r['trend_kendall_tau']:+.2f} (p {r['trend_p']:.2f})",
                    loc="left", fontsize=10, color=INK)
        a.legend(frameon=False, fontsize=8.5, loc="upper right")
    for a in axs:
        a.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("How fire connectivity changed 2003–2024, with the fire count taken out (MODIS Aqua fire events, "
                 "Brazil)", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_change.png", dpi=140)


if __name__ == "__main__":
    main()
