"""What land is burning, and does the mix of land types explain how clustered fire is? Brazil 2003-2024.

Fire: MODIS Aqua fire events (2 km / 2 days) starting in year y. Land cover: MapBiomas Collection 10 class at each
event position (mapbiomas_ee.py fires), in the year before the fire (lc_prev) and the fire's year (lc_same).
Fire types:
  conversion   natural land (forest, savanna, grassland) the year before, farmland (pasture, crops, mosaic) in the
               fire's year: fire on land being converted. Brazil-wide version of 'frontier fire' (PRODES, Amazon only)
  forest       forest the year before, not converted
  open_natural savanna or natural grassland the year before, not converted
  farmland     pasture, crops or mosaic of uses the year before
  other        everything else (left out of the type results)
Clustering ratio = d50 / d50 of the same number of random points: anywhere in Brazil for the Brazil results
(percolation_measure calibration), inside the biome for the Amazon and Cerrado results (regional calibration).

Tests
  1 share of each type per year, and its clustering ratio (Brazil, Amazon biome, Cerrado biome)
  2 does the conversion share follow the three periods (2003-06, 2007-15, 2016-24) and the overall ratio?
  3 fixed mix: subsample each year so the type shares equal the 2003-2024 average; if the periods disappear,
    the change in mix explains them
  4 check against PRODES: Amazon conversion share vs share of fire within 1 km of recent PRODES clearing

    python audit/percolation_landcover.py
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from audit.fire_percolation_pilot import GRID, INK, INK2, MUTED, OUT, SERIES, YEARS, project
from audit.mapbiomas_ee import DIR, group_of
from audit.percolation_measure import REL, curve, random_d50

TYPES = {"conversion": ("Conversion (natural → farmland that year)", "#c2410c"),
         "forest": ("Forest", "#2f7d4f"),
         "open_natural": ("Savanna / natural grassland", "#c99a2e"),
         "farmland": ("Farmland (pasture, crops)", "#2a78d6")}
PERIODS = {"2003–06": range(2003, 2007), "2007–15": range(2007, 2016), "2016–24": range(2016, 2025)}
REPS = 4
MIN_N = 500


def fire_types(f):
    prev, same = group_of(f.lc_prev.fillna(0)), group_of(f.lc_same.fillna(0))
    nat = np.isin(prev, ["forest", "savanna", "grassland"])
    farm_now = np.isin(same, ["pasture", "agriculture", "mosaic"])
    t = np.full(len(f), "other", dtype=object)
    t[np.isin(prev, ["pasture", "agriculture", "mosaic"])] = "farmland"
    t[prev == "forest"] = "forest"
    t[np.isin(prev, ["savanna", "grassland"])] = "open_natural"
    t[nat & farm_now] = "conversion"
    return t


def one_year(args):
    y, cal, reg, seed = args
    import shapely
    from audit.percolation_early_warning_regional import biome_shapes
    rng = np.random.default_rng(seed)
    f = pd.read_parquet(DIR / f"fires_{y}.parquet")
    f["type"] = fire_types(f)
    xy = project(f.lon.to_numpy(), f.lat.to_numpy())
    rows = []
    areas = {"Brazil": np.ones(len(f), bool)}
    for name, shape in biome_shapes().items():
        if name in reg:
            shapely.prepare(shape)
            areas[name] = shapely.contains_xy(shape, xy[:, 0] * 1000, xy[:, 1] * 1000)
    rd = {"Brazil": lambda n: random_d50(cal, "anywhere", n)}
    for name, (a, b) in reg.items():
        rd[name] = (lambda a, b: lambda n: float(np.exp(a + b * np.log(n))))(a, b)
    for area, m in areas.items():
        sub, t = xy[m], f.type.to_numpy()[m]
        r = {"year": y, "area": area, "events": int(m.sum())}
        _, d, n = curve(sub, REL)
        r["ratio_all"] = d / rd[area](n)
        for k in TYPES:
            mk = t == k
            r[f"share_{k}"] = float(mk.mean())
            if mk.sum() >= MIN_N:
                _, d, n = curve(sub[mk], REL)
                r[f"ratio_{k}"] = d / rd[area](n)
                r[f"d50_{k}"] = d
            else:
                r[f"ratio_{k}"] = r[f"d50_{k}"] = np.nan
        rows.append(r)
    return rows


def fixed_mix(args):
    """Subsample year y so that the type shares equal `target`; median ratio over REPS draws."""
    y, cal, target, seed = args
    rng = np.random.default_rng(seed)
    f = pd.read_parquet(DIR / f"fires_{y}.parquet")
    t = fire_types(f)
    keep = t != "other"
    xy, t = project(f.lon.to_numpy(), f.lat.to_numpy())[keep], t[keep]
    avail = {k: int((t == k).sum()) for k in TYPES}
    N = int(min(avail[k] / target[k] for k in TYPES))
    out = {}
    for tag, w in (("fixed_mix", target), ("own_mix", {k: avail[k] / sum(avail.values()) for k in TYPES})):
        vals = []
        for _ in range(REPS):
            idx = np.concatenate([rng.choice(np.nonzero(t == k)[0], int(round(w[k] * N)), replace=False) for k in TYPES])
            _, d, n = curve(xy[idx], REL)
            vals.append(d / random_d50(cal, "anywhere", n))
        out[tag] = float(np.median(vals))
    out.update({"year": y, "subsample_n": N})
    return out


def main():
    import matplotlib.pyplot as plt
    from scipy import stats
    cal = json.loads((OUT / "percolation_measure_calibration.json").read_text())
    reg = json.loads((OUT / "percolation_early_warning_regional_cal.json").read_text())
    with Pool(11) as pool:
        rows = [r for part in pool.map(one_year, [(y, cal, reg, 100 + y) for y in YEARS]) for r in part]
    df = pd.DataFrame(rows)
    br = df[df.area == "Brazil"].set_index("year")
    target = {k: float(br[f"share_{k}"].mean()) for k in TYPES}
    s = sum(target.values())
    target = {k: v / s for k, v in target.items()}
    with Pool(11) as pool:
        fm = pd.DataFrame(pool.map(fixed_mix, [(y, cal, target, 300 + y) for y in YEARS])).set_index("year")
    br = br.join(fm)
    obs = {r["year"]: r for r in json.loads((OUT / "percolation_measure_years.json").read_text())}
    br["ratio_measure"] = [obs[y]["ratio_anywhere"] for y in br.index]
    df.round(4).to_csv(OUT / "percolation_landcover_years.csv", index=False)

    mid = br.index.isin(PERIODS["2007–15"])
    welch = lambda v: round(float(stats.ttest_ind(v[mid], v[~mid], equal_var=False).pvalue), 4)
    per = lambda v: {p: round(float(v[v.index.isin(yrs)].mean()), 3) for p, yrs in PERIODS.items()}
    res = {"target_mix": {k: round(v, 3) for k, v in target.items()}, "brazil": {}, "biomes": {}}
    for c in ["ratio_all", *[f"share_{k}" for k in TYPES], *[f"ratio_{k}" for k in TYPES], "fixed_mix", "own_mix"]:
        res["brazil"][c] = {"periods": per(br[c]), "p_2007_15_vs_other": welch(br[c].dropna())
                            if br[c].notna().all() else None}
    sp = lambda a, b: [round(float(v), 3) for v in stats.spearmanr(a, b, nan_policy="omit")]
    res["brazil"]["spearman_ratio_vs_conversion_share"] = sp(br.ratio_all, br.share_conversion)
    res["brazil"]["spearman_ratio_vs_share"] = {k: sp(br.ratio_all, br[f"share_{k}"]) for k in TYPES}
    res["brazil"]["spearman_ratio_all_vs_measure"] = sp(br.ratio_all, br.ratio_measure)
    res["brazil"]["spearman_fixed_mix_vs_ratio"] = sp(br.fixed_mix, br.own_mix)
    for area in ("Amazon", "Cerrado"):
        a = df[df.area == area].set_index("year")
        res["biomes"][area] = {c: per(a[c]) for c in ["ratio_all", *[f"share_{k}" for k in TYPES],
                                                     *[f"ratio_{k}" for k in TYPES]]}
        res["biomes"][area]["median_ratio_by_type"] = {k: round(float(a[f"ratio_{k}"].median()), 3) for k in TYPES}
        res["biomes"][area]["spearman_ratio_vs_conversion_share"] = sp(a.ratio_all, a.share_conversion)
    pc = OUT / "percolation_clearing_years.csv"
    if pc.exists():
        c = pd.read_csv(pc).set_index("year")
        a = df[df.area == "Amazon"].set_index("year").join(c[["near_share_main"]], how="inner")
        res["prodes_check_amazon"] = {"spearman_conversion_share_vs_near_prodes": sp(a.share_conversion, a.near_share_main),
                                      "mean_conversion_share": round(float(a.share_conversion.mean()), 3),
                                      "mean_near_prodes_share": round(float(a.near_share_main.mean()), 3)}
    (OUT / "percolation_landcover.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))

    fig, axs = plt.subplots(2, 2, figsize=(16, 10.5))
    ax = axs[0, 0]
    bottom = np.zeros(len(br))
    for k, (lab, col) in TYPES.items():
        v = 100 * br[f"share_{k}"].to_numpy()
        ax.bar(br.index, v, bottom=bottom, color=col, width=0.75, label=lab, edgecolor="white", lw=0.6)
        bottom += v
    ax.set_ylabel("% of Brazil's fire events", color=INK2)
    ax.set_title("A. What land is burning (land cover the year before the fire)", loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2)
    ax = axs[0, 1]
    for k, (lab, col) in TYPES.items():
        ax.plot(br.index, br[f"ratio_{k}"], marker="o", ms=3.5, lw=1.6, color=col,
                label=f"{lab}: median {br[f'ratio_{k}'].median():.2f}")
    ax.plot(br.index, br.ratio_all, marker="o", ms=3.5, lw=2.2, color=INK, label=f"All fire: median {br.ratio_all.median():.2f}")
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_ylabel("Clustering ratio (fire ÷ random anywhere in Brazil)", color=INK2)
    ax.set_title("B. How clustered each type of fire is", loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2)
    ax = axs[1, 0]
    ax.plot(br.index, 100 * br.share_conversion, marker="o", ms=4, lw=2, color=TYPES["conversion"][1])
    for p, yrs in PERIODS.items():
        v = 100 * br.share_conversion[br.index.isin(yrs)]
        ax.hlines(v.mean(), min(yrs) - 0.4, max(yrs) + 0.4, color=TYPES["conversion"][1], lw=2.5, alpha=0.45)
    ax.set_ylabel("% of fire on land being converted", color=INK2)
    r = res["brazil"]["spearman_ratio_vs_conversion_share"]
    ax.set_title(f"C. Share of fire on land being converted (lines = period means)\n"
                 f"Spearman with Brazil's clustering ratio: ρ {r[0]:+.2f} (p {r[1]:.3f})", loc="left", fontsize=10.5,
                 color=INK)
    ax = axs[1, 1]
    ax.plot(br.index, br.own_mix, marker="o", ms=4, lw=2, color=INK, label="Each year's own mix")
    ax.plot(br.index, br.fixed_mix, marker="s", ms=4, lw=2, color=SERIES[0], label="Mix held at the 2003–24 average")
    for col, c in ((INK, "own_mix"), (SERIES[0], "fixed_mix")):
        for p, yrs in PERIODS.items():
            v = br[c][br.index.isin(yrs)]
            ax.hlines(v.mean(), min(yrs) - 0.4, max(yrs) + 0.4, color=col, lw=2.5, alpha=0.4)
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_ylabel("Clustering ratio (subsample)", color=INK2)
    ax.set_title(f"D. If the mix of land types had stayed the same, would the periods remain?\n"
                 f"2007–15 vs other years: own mix p {res['brazil']['own_mix']['p_2007_15_vs_other']}, "
                 f"fixed mix p {res['brazil']['fixed_mix']['p_2007_15_vs_other']}", loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2)
    for ax in axs.ravel():
        ax.grid(color=GRID, lw=0.6)
        ax.axvspan(2006.5, 2015.5, color=MUTED, alpha=0.08, lw=0)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
    fig.suptitle("Land cover under Brazil's fires and their clustering, 2003–2024 (MODIS Aqua fire events; MapBiomas "
                 "Collection 10)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_landcover.png", dpi=140)


if __name__ == "__main__":
    main()
