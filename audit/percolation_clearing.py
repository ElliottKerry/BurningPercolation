"""Does recent clearing explain the clustering of Amazon fire and its change between periods?

Area: Amazon biome (PRODES maps clearing only there). Fire: MODIS Aqua fire events (2 km / 2 days) starting in
calendar year y. Clearing: PRODES Amazon yearly deforestation polygons (prodes_amazon_cache.py); PRODES year y runs
Aug y-1 to Jul y, so fires of Aug-Oct y sit between PRODES y and y+1. 'Recent clearing' for fire year y = patches
of PRODES years y and y+1. Fire years 2008-2023 (2024 has no PRODES 2025).
Clustering ratio inside the biome = fire d50 / d50 of the same number of random points in the biome (calibration
from percolation_early_warning_regional.py).

Test A, fine scale: split each year's fires into NEAR recent clearing (within NEAR_M of a patch) and FAR; measure
  the share near, and the clustering of all, near and far fires. If the near fires carry the clustering and their
  share moves with the periods, recent clearing explains the fine-scale part. Sensitivity: 2 km; years y-1 and y.
Test B, cell level: Poisson GLM on 0.25 deg Amazon cells x year (as percolation_model.py),
  M2 = year + usual fire level + climate anomalies; M3 = M2 + log(1 + cleared km2 in the cell, PRODES y and y+1);
  simulate fires from each and from the oracle (true cell counts); compare clustering with the observed.

    python audit/percolation_clearing.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if "--events" not in sys.argv:
    sys.argv.append("--events")

import numpy as np
import pandas as pd

from audit.fire_percolation_pilot import GRID, INK, INK2, MUTED, OUT, SERIES, project
from audit.percolation_lattice_threshold import RES
from audit.percolation_measure import REL, curve
from audit.percolation_stability import events_with_dates
from firms import config

YEARS = range(2008, 2024)
NEAR_M = 1000
REPS = 3
EARLY, LATE = range(2008, 2016), range(2016, 2024)


def main():
    import matplotlib.pyplot as plt
    import shapely
    import geopandas as gpd
    from scipy import stats
    from audit.percolation_early_warning_regional import biome_shapes
    from audit.percolation_model import CLIMATE, panel
    from audit.fire_percolation_expected import Sampler
    import statsmodels.api as sm
    amazon = biome_shapes()["Amazon"]
    shapely.prepare(amazon)
    a, b = json.loads((OUT / "percolation_early_warning_regional_cal.json").read_text())["Amazon"]
    rd50 = lambda n: float(np.exp(a + b * np.log(n)))
    ratio = lambda xy: (lambda c: c[1] / rd50(c[2]))(curve(xy, REL)) if len(xy) > 500 else np.nan
    pr = gpd.read_parquet(config.DATA_DIR / "cache" / "prodes_amazon_2008_2024.parquet")
    pr = pr[pr.lon.notna() & ~pr.geometry.is_empty].reset_index(drop=True)
    geoms = {y: pr.geometry[pr.year == y].to_numpy() for y in range(2008, 2025)}

    # ---------------- test A
    rows, ev_by_year = [], {}
    for y in YEARS:
        ev = events_with_dates(y)
        ev = ev[ev.start.dt.year == y]
        xy = project(ev.lon.to_numpy(), ev.lat.to_numpy())
        inb = shapely.contains_xy(amazon, xy[:, 0] * 1000, xy[:, 1] * 1000)
        xy = xy[inb]
        ev_by_year[y] = (ev[inb], xy)
        pts = shapely.points(xy * 1000)
        r = {"year": y, "events": len(xy)}
        for tag, yrs, dist in (("main", (y, y + 1), NEAR_M), ("2km", (y, y + 1), 2000), ("prior", (y - 1, y), NEAR_M)):
            gs = np.concatenate([geoms[k] for k in yrs if k in geoms])
            tree = shapely.STRtree(gs)
            hit = np.zeros(len(xy), bool)
            hit[np.unique(tree.query(pts, predicate="dwithin", distance=dist)[0])] = True
            r[f"near_share_{tag}"] = float(hit.mean())
            if tag == "main":
                near = hit
        r["ratio_all"] = ratio(xy)
        r["ratio_near"] = ratio(xy[near])
        r["ratio_far"] = ratio(xy[~near])
        r["cleared_km2"] = float(pr.area_km2[pr.year.isin([y, y + 1])].sum() / 2)
        rows.append(r)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    A = pd.DataFrame(rows).set_index("year")
    e, l = A.index.isin(EARLY), A.index.isin(LATE)
    resA = {c: {"2008-15": round(float(A[c][e].mean()), 3), "2016-23": round(float(A[c][l].mean()), 3),
                "p": round(float(stats.ttest_ind(A[c][e], A[c][l], equal_var=False).pvalue), 4)}
            for c in ["ratio_all", "ratio_near", "ratio_far", "near_share_main", "near_share_2km", "near_share_prior",
                      "cleared_km2"]}
    resA["spearman_ratio_all_vs_near_share"] = [round(v, 3) for v in stats.spearmanr(A.ratio_all, A.near_share_main)]
    resA["spearman_near_share_vs_cleared"] = [round(v, 3) for v in stats.spearmanr(A.near_share_main, A.cleared_km2)]

    # ---------------- test B
    p, _ = panel()
    cells = p[["ix", "iy"]].drop_duplicates()
    from pyproj import Transformer
    from audit.fire_percolation_pilot import ALBERS
    x, yv = Transformer.from_crs("EPSG:4326", ALBERS, always_xy=True).transform(cells.ix * RES, cells.iy * RES)
    cells = cells[shapely.contains_xy(amazon, x, yv)]
    p = p.merge(cells, on=["ix", "iy"])
    pr["ix"] = np.floor(pr.lon / RES + 0.5).astype(int)
    pr["iy"] = np.floor(pr.lat / RES + 0.5).astype(int)
    cl = pr.groupby(["ix", "iy", "year"]).area_km2.sum().rename("clr").reset_index()
    p = p.merge(cl.rename(columns={"clr": "clear_y"}), on=["ix", "iy", "year"], how="left")
    p = p.merge(cl.assign(year=cl.year - 1).rename(columns={"clr": "clear_next"}), on=["ix", "iy", "year"], how="left")
    p = p[p.year.isin(YEARS)].fillna({"clear_y": 0, "clear_next": 0}).reset_index(drop=True)
    p["log_clear_y"], p["log_clear_next"] = np.log1p(p.clear_y), np.log1p(p.clear_next)
    yd = pd.get_dummies(p.year, prefix="y", dtype=float)
    anom = [v + "_anom" for v in CLIMATE]
    mu, glm = {}, {}
    for k, xs in (("M2", ["baseline", *anom]), ("M3", ["baseline", *anom, "log_clear_y", "log_clear_next"])):
        f = sm.GLM(p.events, pd.concat([yd, p[xs]], axis=1), family=sm.families.Poisson()).fit()
        mu[k] = f.fittedvalues.to_numpy()
        glm[k] = {"deviance_explained": round(float(1 - f.deviance / f.null_deviance), 4)}
        if k == "M3":
            glm[k]["rate_ratio"] = {c: round(float(np.exp(f.params[c])), 3) for c in ("log_clear_y", "log_clear_next")}
    S = Sampler()
    rng = np.random.default_rng(8)
    rowsB = []
    for y in YEARS:
        m = (p.year == y).to_numpy()
        cix, ciy = p.ix.to_numpy()[m], p.iy.to_numpy()[m]
        n = int(p.events.to_numpy()[m].sum())
        r = {"year": y, "observed_ratio": A.loc[y, "ratio_all"]}
        sets = {"oracle": p.events.to_numpy()[m].astype(int)}
        for k in mu:
            sets[k] = None
        for k in ("oracle", "M2", "M3"):
            vals = []
            for _ in range(REPS):
                cnt = sets[k] if k == "oracle" else rng.multinomial(n, mu[k][m] / mu[k][m].sum())
                idx = np.repeat(np.arange(len(cnt)), cnt)
                vals.append(curve(S.in_cells(cix[idx], ciy[idx], rng), REL)[1] / rd50(len(idx)))
            r[f"{k}_ratio"] = float(np.median(vals))
        rowsB.append(r)
        print(y, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    B = pd.DataFrame(rowsB).set_index("year")
    resB = {"glm": glm}
    for c in ["observed_ratio", "oracle_ratio", "M2_ratio", "M3_ratio"]:
        resB[c] = {"2008-15": round(float(B[c][e].mean()), 3), "2016-23": round(float(B[c][l].mean()), 3),
                   "spearman_with_observed": [round(v, 3) for v in stats.spearmanr(B.observed_ratio, B[c])]}
    A.join(B.drop(columns="observed_ratio")).round(4).to_csv(OUT / "percolation_clearing_years.csv")
    res = {"test_A_fine_scale": resA, "test_B_cell_level": resB}
    (OUT / "percolation_clearing.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

    fig, axs = plt.subplots(1, 3, figsize=(19, 5.4))
    ax = axs[0]
    ax.bar(A.index, 100 * A.near_share_main, color=SERIES[3], width=0.7, label="Fires within 1 km of recent clearing")
    ax.set_ylabel("% of Amazon-biome fire events", color=INK2)
    ax2_vals = A.cleared_km2 / 1000
    ax.set_title(f"A1. Share of fires on or next to recent clearing\n2008–15 {100 * resA['near_share_main']['2008-15']:.0f}% "
                 f"vs 2016–23 {100 * resA['near_share_main']['2016-23']:.0f}% (p {resA['near_share_main']['p']:.3f})",
                 loc="left", fontsize=10, color=INK)
    ax = axs[1]
    for c, col, lab in (("ratio_all", INK, "All Amazon fires"), ("ratio_near", SERIES[3], "Near recent clearing"),
                        ("ratio_far", SERIES[0], "Away from recent clearing")):
        ax.plot(A.index, A[c], marker="o", ms=4, lw=1.8, color=col,
                label=f"{lab}: {resA[c]['2008-15']:.2f} → {resA[c]['2016-23']:.2f} (p {resA[c]['p']:.3f})")
    ax.axhline(1, color=INK2, lw=0.8)
    ax.axvline(2015.5, color=MUTED, lw=0.8, ls="--")
    ax.set_ylabel("Clustering ratio inside the Amazon biome", color=INK2)
    ax.set_title("A2. Which fires are clustered? (period means 2008–15 → 2016–23)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax = axs[2]
    for c, col, lab in (("observed_ratio", INK, "Observed"), ("oracle_ratio", SERIES[2], "Best possible cell model"),
                        ("M2_ratio", SERIES[1], "Usual level + climate"),
                        ("M3_ratio", SERIES[0], "+ recent clearing in the cell")):
        ax.plot(B.index, B[c], marker="o", ms=3.5, lw=1.6, color=col, ls="--" if c == "oracle_ratio" else "-",
                label=f"{lab}: {resB[c]['2008-15']:.2f} → {resB[c]['2016-23']:.2f}")
    ax.axhline(1, color=INK2, lw=0.8)
    ax.axvline(2015.5, color=MUTED, lw=0.8, ls="--")
    ax.set_title("B. Cell model: does adding clearing reproduce the change?", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for ax in axs:
        ax.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Recent clearing and the clustering of Amazon fire, 2008–2023 (MODIS Aqua fire events; PRODES "
                 "deforestation)", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_clearing.png", dpi=140)


if __name__ == "__main__":
    main()
