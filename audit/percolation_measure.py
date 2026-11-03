"""Measure: percolation curve of fire events for each year 2003-2024, compared with random points, with uncertainty.

Fire: MODIS Aqua events (2 km / 2 days) that start in the year, distinct positions, Albers km.
Curve: share of events in the largest cluster when events closer than d are joined (Euclidean MST sweep), on
FINE_D (1-100 km). d50 = distance where the largest cluster first holds half the events (log-interpolated).
Typical spacing s = sqrt(Brazil area / n); relative distance = d / s.

Random comparisons (same number of points):
  anywhere  uniform inside Brazil
  usual     inside 0.25 deg cells in proportion to their share of all fire events 2003-2024
For both, d50 scales as n^(-1/2). It is calibrated once from CAL_REPS runs at CAL_SIZES points (log-log fit), so
any sample size gets its random d50 without new runs. Ratio = fire d50 / random d50 at the same n.

Uncertainty: random half-samples of start weeks (B per year). Each draws half of the year's weeks with fire,
without replacement, and recomputes d50 and the ratios on those events. For a statistic close to a mean, the
spread of half-sample estimates around the full estimate equals the sampling variance of the full estimate
(half-sampling; McCarthy 1969, Efron 1982), so SE = SD of the half-sample estimates, 95% CI = estimate +- 1.96 SE.
d50 in km is put on the full-sample density first (x sqrt(n_half / n)). Whole weeks are kept together so fires
burning near each other over a few days are not split.

Tests: heterogeneity across years (Cochran's Q on the ratios with their SEs); 2007-2015 vs other years.

    python audit/percolation_measure.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from audit.fire_percolation_pilot import ALBERS, GRID, INK, INK2, MUTED, OUT, SERIES, YEARS, mst_edges, project

FINE_D = np.logspace(0, 2, 80)
REL = np.logspace(np.log10(0.2), np.log10(8), 80)
B = 200
CAL_SIZES = np.geomspace(15000, 180000, 9).astype(int)
CAL_REPS = 3
ERA = range(2007, 2016)
NULL_NAME = {"anywhere": "Random: anywhere in Brazil", "usual": "Random: where fire usually is"}


def curve(xy, grid):
    """Largest-cluster share on grid (km) for distinct points xy (km), and d50 (km)."""
    u = np.unique(np.round(xy, 4), axis=0)
    n = len(u)
    i, j, w = mst_edges(u)
    parent = np.arange(n)
    size = np.ones(n, np.int64)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    big, best = np.empty(len(w)), 1
    for k in range(len(w)):
        a, b = find(i[k]), find(j[k])
        if a != b:
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]
            best = max(best, size[a])
        big[k] = best
    share = big / n
    curve.levels = {lv: float(w[np.searchsorted(share, lv)]) for lv in (0.2, 0.5, 0.8)}
    d50 = curve.levels[0.5]
    pos = np.searchsorted(w, grid, side="right") - 1
    return np.where(pos >= 0, share[np.maximum(pos, 0)], 1 / n), d50, n


def brazil_area():
    import geopandas as gpd
    from firms import config
    return gpd.read_parquet(config.DATA_DIR / "cache" / "brazil_outline_5880.parquet").to_crs(ALBERS).area.sum() / 1e6


def calibrate(area):
    """Random d50 against n for both nulls; also their curves in relative units."""
    cache = OUT / "percolation_measure_calibration.json"
    if cache.exists():
        return json.loads(cache.read_text())
    sys.argv.append("--events")
    from audit.fire_percolation_expected import Sampler
    S = Sampler()
    rng = np.random.default_rng(2003)
    out = {}
    for kind in NULL_NAME:
        rows, rel_curves = [], []
        for n in CAL_SIZES:
            for _ in range(CAL_REPS):
                xy = S.draw(kind, np.zeros(n), np.zeros(n), rng) if kind == "anywhere" else \
                    S.draw("usual", np.zeros(n), np.zeros(n), rng)
                s = np.sqrt(area / n)
                c, d50, nd = curve(xy, REL * s)
                rows.append((int(nd), d50))
                rel_curves.append(c.tolist())
                print(kind, n, round(d50, 2), flush=True)
        r = np.array(rows, float)
        b, a = np.polyfit(np.log(r[:, 0]), np.log(r[:, 1]), 1)
        resid = np.log(r[:, 1]) - (a + b * np.log(r[:, 0]))
        out[kind] = {"a": float(a), "b": float(b), "resid_sd": float(resid.std(ddof=2)), "runs": rows,
                     "rel_curves": rel_curves}
    cache.write_text(json.dumps(out))
    return out


def random_d50(cal, kind, n):
    return float(np.exp(cal[kind]["a"] + cal[kind]["b"] * np.log(n)))


def one_year(args):
    y, area, cal, seed = args
    from audit.percolation_stability import events_with_dates
    ev = events_with_dates(y)
    ev = ev[ev.start.dt.year == y]
    xy = project(ev.lon.to_numpy(), ev.lat.to_numpy())
    week = ((ev.start.dt.dayofyear - 1) // 7).to_numpy()
    c_km, d50, n = curve(xy, FINE_D)
    width = curve.levels[0.8] / curve.levels[0.2]
    s = np.sqrt(area / n)
    c_rel = np.interp(np.log(REL * s), np.log(FINE_D), c_km)
    rng = np.random.default_rng(seed)
    weeks = np.unique(week)
    half = []
    for _ in range(B):
        pick = rng.choice(weeks, size=len(weeks) // 2, replace=False)
        m = np.isin(week, pick)
        ch, dh, nh = curve(xy[m], REL * np.sqrt(area / m.sum()))
        half.append({"n": nh, "d50": dh, "d50_fullscale": dh * np.sqrt(nh / n),
                     "width": curve.levels[0.8] / curve.levels[0.2],
                     "rel": dh / np.sqrt(area / nh),
                     "ratio_anywhere": dh / random_d50(cal, "anywhere", nh),
                     "ratio_usual": dh / random_d50(cal, "usual", nh), "curve_rel": ch})
    H = pd.DataFrame(half)
    rec = {"year": y, "events": int(n), "spacing_km": float(s), "d50_km": d50,
           "d50_km_se": float(H.d50_fullscale.std()), "d50_rel": d50 / s, "d50_rel_se": float(H.rel.std()),
           "ratio_anywhere": d50 / random_d50(cal, "anywhere", n), "ratio_anywhere_se": float(H.ratio_anywhere.std()),
           "ratio_usual": d50 / random_d50(cal, "usual", n), "ratio_usual_se": float(H.ratio_usual.std()),
           "width": width, "width_se": float(H.width.std()),
           "half_n_mean": float(H.n.mean())}
    hc = np.vstack(H.curve_rel.to_numpy())
    rec["curve_rel"] = c_rel.tolist()
    rec["curve_rel_sd"] = hc.std(0).tolist()
    print(y, {k: round(v, 3) for k, v in rec.items() if isinstance(v, float)}, flush=True)
    return rec


def main():
    import matplotlib.pyplot as plt
    from multiprocessing import Pool
    from scipy import stats
    area = brazil_area()
    cal = calibrate(area)
    cache = OUT / "percolation_measure_years.json"
    if cache.exists():
        recs = json.loads(cache.read_text())
    else:
        with Pool(min(15, len(YEARS))) as pool:
            recs = pool.map(one_year, [(y, area, cal, 1000 + y) for y in YEARS])
        cache.write_text(json.dumps(recs))
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("curve")} for r in recs]).set_index("year")
    for k in ("d50_km", "ratio_anywhere", "ratio_usual"):
        df[f"{k}_lo"] = df[k] - 1.96 * df[f"{k}_se"]
        df[f"{k}_hi"] = df[k] + 1.96 * df[f"{k}_se"]
    df.round(4).to_csv(OUT / "percolation_measure_years.csv")

    res = {"calibration": {k: {"d50_km = exp(a) * n^b": [round(np.exp(v["a"]), 2), round(v["b"], 3)],
                               "resid_sd_log": round(v["resid_sd"], 4)} for k, v in cal.items()},
           "half_samples_per_year": B}
    for k in ("ratio_anywhere", "ratio_usual", "d50_rel"):
        r, se = df[k], df[f"{k}_se"]
        w = 1 / se ** 2
        mean_w = float((w * r).sum() / w.sum())
        Q = float((w * (r - mean_w) ** 2).sum())
        era = df.index.isin(ERA)
        diff = float(r[era].mean() - r[~era].mean())
        diff_se = float(np.sqrt((se[era] ** 2).sum() / era.sum() ** 2 + (se[~era] ** 2).sum() / (~era).sum() ** 2))
        res[k] = {"range": [round(float(r.min()), 3), round(float(r.max()), 3)], "median_se": round(float(se.median()), 3),
                  "cochran_Q": round(Q, 1), "df": len(r) - 1, "Q_p": float(stats.chi2.sf(Q, len(r) - 1)),
                  "share_of_variation_beyond_noise_I2": round(max(0.0, (Q - (len(r) - 1)) / Q), 3),
                  "years_above_1_with_CI": int(((r - 1.96 * se) > 1).sum()) if k != "d50_rel" else None,
                  "years_CI_includes_1": int((((r - 1.96 * se) <= 1) & ((r + 1.96 * se) >= 1)).sum()) if k != "d50_rel" else None,
                  "era_2007_2015_minus_other": round(diff, 3), "era_diff_se": round(diff_se, 3),
                  "era_p": float(2 * stats.norm.sf(abs(diff) / diff_se))}
    res["d50_km"] = {"range": [round(float(df.d50_km.min()), 2), round(float(df.d50_km.max()), 2)],
                     "median_se_km": round(float(df.d50_km_se.median()), 2)}
    (OUT / "percolation_measure.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))

    # ---- figure 1: curves in relative units, one panel per year
    rand = {k: np.array(cal[k]["rel_curves"]) for k in NULL_NAME}
    fig, axs = plt.subplots(4, 6, figsize=(18, 11.5), sharex=True, sharey=True)
    for a, r in zip(axs.ravel(), recs):
        for k, col in (("anywhere", MUTED), ("usual", SERIES[1])):
            a.fill_between(REL, rand[k].min(0), rand[k].max(0), color=col, alpha=0.25, lw=0)
        c, sd = np.array(r["curve_rel"]), np.array(r["curve_rel_sd"])
        a.fill_between(REL, np.clip(c - 1.96 * sd, 0, 1), np.clip(c + 1.96 * sd, 0, 1), color=SERIES[0], alpha=0.3, lw=0)
        a.plot(REL, c, color=SERIES[0], lw=1.6)
        a.axhline(0.5, color=MUTED, lw=0.5)
        a.set_xscale("log")
        a.set_xticks([0.25, 0.5, 1, 2, 4, 8], ["¼", "½", "1", "2", "4", "8"])
        a.minorticks_off()
        a.set_title(f"{r['year']}: ratio {r['ratio_anywhere']:.2f} ± {1.96 * r['ratio_anywhere_se']:.2f}",
                    loc="left", fontsize=9, color=INK)
        a.grid(color=GRID, lw=0.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    for a in axs.ravel()[len(recs):]:
        a.set_axis_off()
    from matplotlib.patches import Patch
    axs.ravel()[len(recs)].legend(handles=[Patch(color=SERIES[0], alpha=0.5, label="Actual fire (95% band)"),
                                           Patch(color=MUTED, alpha=0.5, label=NULL_NAME["anywhere"]),
                                           Patch(color=SERIES[1], alpha=0.5, label=NULL_NAME["usual"])],
                                  frameon=False, fontsize=10, loc="center")
    for a in axs[:, 0]:
        a.set_ylabel("Largest cluster share", color=INK2, fontsize=9)
    for a in axs[-1, :]:
        a.set_xlabel("Distance ÷ typical spacing", color=INK2, fontsize=9)
    fig.suptitle("Percolation curves by year, fire vs random points, distance in units of each year's typical spacing "
                 "√(area / n). Title: fire / random-anywhere half-joined distance, ± 95% (half-samples of weeks)",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_measure_curves.png", dpi=130)

    # ---- figure 2: d50 and ratios with 95% CIs
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    a = axs[0]
    a.errorbar(df.index, df.d50_km, yerr=1.96 * df.d50_km_se, fmt="o-", color=SERIES[0], ms=4, capsize=2, lw=1.2,
               label="Actual fire")
    for k, col in (("anywhere", MUTED), ("usual", SERIES[1])):
        a.plot(df.index, [random_d50(cal, k, n) for n in df.events], color=col, lw=1.5, ls="--",
               label=NULL_NAME[k] + ", same number")
    a.set_ylabel("Half-joined distance d50 (km)", color=INK2)
    a.set_title(f"d50 by year with 95% intervals (median ± {1.96 * res['d50_km']['median_se_km']:.1f} km)",
                loc="left", fontsize=10, color=INK)
    a.legend(frameon=False, fontsize=9)
    a = axs[1]
    for k, col, off in (("anywhere", SERIES[0], -0.15), ("usual", SERIES[1], 0.15)):
        a.errorbar(df.index + off, df[f"ratio_{k}"], yerr=1.96 * df[f"ratio_{k}_se"], fmt="o", color=col, ms=4,
                   capsize=2, lw=1.2, label=f"vs {NULL_NAME[k].split(': ')[1]} (Q p = {res['ratio_' + k]['Q_p']:.1g})")
    a.axhline(1, color=INK2, lw=0.8)
    a.set_ylabel("Fire d50 ÷ random d50", color=INK2)
    a.set_title("Clustering ratio with 95% intervals. Above 1 = more clustered than random", loc="left", fontsize=10,
                color=INK)
    a.legend(frameon=False, fontsize=9)
    for a in axs:
        a.grid(color=GRID, lw=0.6)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle(f"Measuring fire connectivity 2003–2024 with uncertainty ({B} half-samples of weeks per year)",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_measure_years.png", dpi=140)


if __name__ == "__main__":
    main()
