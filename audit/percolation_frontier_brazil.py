"""Frontier fire across all of Brazil, 2003-2023: does fire next to new clearing carry the clustering everywhere?

Brazil-wide version of percolation_clearing.py / percolation_frontier_threshold.py (PRODES, Amazon, 2008-2023).
Fire: MODIS Aqua fire events (2 km / 2 days) starting in year y. Clearing: MapBiomas Collection 10, 30 m pixels that
change from natural land (forest, savanna, grassland) to farmland (pasture, crops, mosaic), in connected patches of
at least ~6.3 ha (the PRODES minimum), between the maps of y-1 and y or of y and y+1 (mapbiomas_ee.py near).
Frontier fire = within 1 km of such a patch (2 km as a check). 2024 is left out (no 2025 map).
Clustering ratio = d50 / d50 of the same number of random points: anywhere in Brazil for Brazil, inside the biome
for the Amazon and Cerrado (regional calibration).

Tests
  1 frontier share per year and its link to the three periods and to Brazil's clustering ratio
  2 clustering of frontier and other fire, each year (Brazil, Amazon, Cerrado); is each one stable over time?
  3 fixed mix: subsample each year so the frontier share equals its 2003-2023 average. If the periods disappear,
    the frontier share explains them
  4 check against PRODES (Amazon 2008-2023): MapBiomas frontier share vs PRODES near-clearing share

    python audit/percolation_frontier_brazil.py
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from audit.fire_percolation_pilot import GRID, INK, INK2, MUTED, OUT, project
from audit.mapbiomas_ee import DIR
from audit.percolation_measure import REL, curve, random_d50

YEARS = range(2003, 2024)
NEAR_M = 1000
PERIODS = {"2003–06": range(2003, 2007), "2007–15": range(2007, 2016), "2016–23": range(2016, 2024)}
NEAR_C, FAR_C = "#c2410c", "#2a78d6"
REPS = 4
MIN_N = 500


def load(y):
    f = pd.read_parquet(DIR / f"fires_{y}.parquet")
    nd = pd.read_parquet(DIR / f"near_{y}.parquet")
    f["dist"] = nd[["conv_a", "conv_b"]].min(axis=1).to_numpy()
    f["dist_a"] = nd.conv_a.to_numpy()
    return f


def one_year(args):
    y, cal, reg = args
    import shapely
    from audit.percolation_early_warning_regional import biome_shapes
    f = load(y)
    xy = project(f.lon.to_numpy(), f.lat.to_numpy())
    areas = {"Brazil": np.ones(len(f), bool)}
    for name, shape in biome_shapes().items():
        if name in reg:
            shapely.prepare(shape)
            areas[name] = shapely.contains_xy(shape, xy[:, 0] * 1000, xy[:, 1] * 1000)
    rd = {"Brazil": lambda n: random_d50(cal, "anywhere", n)}
    for name, (a, b) in reg.items():
        rd[name] = (lambda a, b: lambda n: float(np.exp(a + b * np.log(n))))(a, b)
    rows = []
    for area, m in areas.items():
        sub, dist = xy[m], f.dist.to_numpy()[m]
        near = dist <= NEAR_M
        r = {"year": y, "area": area, "events": int(m.sum()), "share_near": float(near.mean()),
             "share_near_2km": float((dist <= 2 * NEAR_M).mean()),
             "share_near_prior_only": float((f.dist_a.to_numpy()[m] <= NEAR_M).mean())}
        for tag, mk in (("all", np.ones(len(sub), bool)), ("near", near), ("far", ~near)):
            if mk.sum() >= MIN_N:
                _, d, n = curve(sub[mk], REL)
                r[f"ratio_{tag}"], r[f"d50_{tag}"], r[f"random_d50_{tag}"] = d / rd[area](n), d, rd[area](n)
            else:
                r[f"ratio_{tag}"] = r[f"d50_{tag}"] = r[f"random_d50_{tag}"] = np.nan
        rows.append(r)
    return rows


def fixed_mix(args):
    """Subsample year y (Brazil) so the frontier share equals `target`; median ratio over REPS draws, plus the same
    subsample size drawn with the year's own share."""
    y, cal, target, seed = args
    rng = np.random.default_rng(seed)
    f = load(y)
    xy = project(f.lon.to_numpy(), f.lat.to_numpy())
    near = (f.dist <= NEAR_M).to_numpy()
    i_near, i_far = np.nonzero(near)[0], np.nonzero(~near)[0]
    N = int(min(len(i_near) / target, len(i_far) / (1 - target)))
    out = {"year": y, "subsample_n": N}
    for tag, share in (("fixed_mix", target), ("own_mix", near.mean())):
        vals = []
        for _ in range(REPS):
            k = int(round(share * N))
            idx = np.concatenate([rng.choice(i_near, k, replace=False), rng.choice(i_far, N - k, replace=False)])
            _, d, n = curve(xy[idx], REL)
            vals.append(d / random_d50(cal, "anywhere", n))
        out[tag] = float(np.median(vals))
    return out


def main():
    import matplotlib.pyplot as plt
    from scipy import stats
    cal = json.loads((OUT / "percolation_measure_calibration.json").read_text())
    reg = json.loads((OUT / "percolation_early_warning_regional_cal.json").read_text())
    with Pool(11) as pool:
        df = pd.DataFrame([r for part in pool.map(one_year, [(y, cal, reg) for y in YEARS]) for r in part])
    br = df[df.area == "Brazil"].set_index("year")
    target = float(br.share_near.mean())
    with Pool(11) as pool:
        fm = pd.DataFrame(pool.map(fixed_mix, [(y, cal, target, 500 + y) for y in YEARS])).set_index("year")
    br = br.join(fm)
    df.round(4).to_csv(OUT / "percolation_frontier_brazil_years.csv", index=False)
    fm.round(4).to_csv(OUT / "percolation_frontier_brazil_fixed_mix.csv")

    def summary(a, cols):
        mid = a.index.isin(PERIODS["2007–15"])
        out = {}
        for c in cols:
            v = a[c]
            out[c] = {**{p: round(float(v[v.index.isin(yrs)].mean()), 3) for p, yrs in PERIODS.items()},
                      "median": round(float(v.median()), 3),
                      "p_2007_15_vs_other": round(float(stats.ttest_ind(v[mid].dropna(), v[~mid].dropna(),
                                                                         equal_var=False).pvalue), 4)}
        return out

    sp = lambda a, b: [round(float(v), 4) for v in stats.spearmanr(a, b, nan_policy="omit")]
    cols = ["share_near", "share_near_2km", "ratio_all", "ratio_near", "ratio_far"]
    res = {"target_share": round(target, 3), "Brazil": summary(br, cols + ["fixed_mix", "own_mix"])}
    res["Brazil"]["spearman_ratio_all_vs_share_near"] = sp(br.ratio_all, br.share_near)
    res["Brazil"]["spearman_ratio_all_vs_share_near_2km"] = sp(br.ratio_all, br.share_near_2km)
    res["Brazil"]["spearman_own_vs_fixed"] = sp(br.own_mix, br.fixed_mix)
    for area in ("Amazon", "Cerrado"):
        a = df[df.area == area].set_index("year")
        res[area] = summary(a, cols)
        res[area]["spearman_ratio_all_vs_share_near"] = sp(a.ratio_all, a.share_near)
        res[area]["median_d50_km"] = {t: round(float(a[f"d50_{t}"].median()), 1) for t in ("near", "far")}
        res[area]["median_random_d50_km"] = {t: round(float(a[f"random_d50_{t}"].median()), 1) for t in ("near", "far")}
    pc = OUT / "percolation_clearing_years.csv"
    if pc.exists():
        c = pd.read_csv(pc).set_index("year")
        a = df[df.area == "Amazon"].set_index("year").join(c[["near_share_main", "ratio_near", "ratio_far"]],
                                                           how="inner", rsuffix="_prodes")
        res["prodes_check_amazon_2008_23"] = {
            "spearman_share_near": sp(a.share_near, a.near_share_main),
            "mean_share_near_mapbiomas": round(float(a.share_near.mean()), 3),
            "mean_share_near_prodes": round(float(a.near_share_main.mean()), 3),
            "median_ratio_near": [round(float(a.ratio_near.median()), 3), round(float(a.ratio_near_prodes.median()), 3)],
            "median_ratio_far": [round(float(a.ratio_far.median()), 3), round(float(a.ratio_far_prodes.median()), 3)]}
    (OUT / "percolation_frontier_brazil.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))

    fig, axs = plt.subplots(2, 2, figsize=(16, 10.5))
    ax = axs[0, 0]
    for area, ls in (("Brazil", "-"), ("Amazon", "--"), ("Cerrado", ":")):
        a = df[df.area == area].set_index("year")
        ax.plot(a.index, 100 * a.share_near, ls=ls, marker="o", ms=3.5, lw=1.9, color=NEAR_C, label=area)
    ax.set_ylabel("% of fire events within 1 km of new clearing", color=INK2)
    r = res["Brazil"]["spearman_ratio_all_vs_share_near"]
    ax.set_title(f"A. Share of fire on the clearing frontier\nBrazil: Spearman with the clustering ratio ρ {r[0]:+.2f} "
                 f"(p {r[1]:.3f})", loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=9)
    for ax, area in ((axs[0, 1], "Brazil"), (axs[1, 0], "Amazon"), (axs[1, 1], "Cerrado")):
        a = df[df.area == area].set_index("year")
        s = res[area]
        for tag, col, lab in (("near", NEAR_C, "Frontier fire (≤ 1 km of new clearing)"),
                              ("far", FAR_C, "Other fire"), ("all", INK, "All fire")):
            ax.plot(a.index, a[f"ratio_{tag}"], marker="o", ms=3.5, lw=2.2 if tag == "all" else 1.7, color=col,
                    label=f"{lab}: median {s[f'ratio_{tag}']['median']:.2f}")
        ax.axhline(1, color=INK2, lw=0.8)
        where = "anywhere in Brazil" if area == "Brazil" else f"inside the {area} biome"
        ax.set_ylabel(f"Clustering ratio (fire ÷ random {where})", color=INK2)
        ax.set_title(f"{'BCD'[['Brazil', 'Amazon', 'Cerrado'].index(area)]}. {area}: how clustered each kind of fire is",
                     loc="left", fontsize=10.5, color=INK)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for ax in axs.ravel():
        ax.grid(color=GRID, lw=0.6)
        ax.axvspan(2006.5, 2015.5, color=MUTED, alpha=0.08, lw=0)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.suptitle("Fire next to new clearing across Brazil, 2003–2023 (MODIS Aqua fire events; MapBiomas Collection 10 "
                 "conversion of natural land to farmland, patches ≥ 6.3 ha)", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_frontier_brazil.png", dpi=140)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(br.index, br.own_mix, marker="o", ms=4, lw=2, color=INK, label="Each year's own frontier share")
    ax.plot(br.index, br.fixed_mix, marker="s", ms=4, lw=2, color=NEAR_C,
            label=f"Frontier share held at its average ({100 * target:.0f}%)")
    for col, c in ((INK, "own_mix"), (NEAR_C, "fixed_mix")):
        for p, yrs in PERIODS.items():
            v = br[c][br.index.isin(yrs)]
            ax.hlines(v.mean(), min(yrs) - 0.4, max(yrs) + 0.4, color=col, lw=2.5, alpha=0.4)
    ax.axhline(1, color=INK2, lw=0.8)
    ax.axvspan(2006.5, 2015.5, color=MUTED, alpha=0.08, lw=0)
    fmt = lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
    ax.set_title(f"If the frontier share had stayed the same, would the three periods remain? (lines = period means)\n"
                 f"2007–15 vs other years: own share p {fmt(res['Brazil']['own_mix']['p_2007_15_vs_other'])}, "
                 f"fixed share p {fmt(res['Brazil']['fixed_mix']['p_2007_15_vs_other'])}", loc="left", fontsize=10.5,
                 color=INK)
    ax.set_ylabel("Clustering ratio (subsample, Brazil)", color=INK2)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color=GRID, lw=0.6)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_frontier_brazil_fixed_mix.png", dpi=140)


if __name__ == "__main__":
    main()
