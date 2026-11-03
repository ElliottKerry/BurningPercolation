"""Critical point over time for the Legal Amazon only, against PRODES deforestation (same area), 2003-2024.

The whole-Brazil version compared Brazil-wide fire with Legal Amazon deforestation. Here both cover the same
area. Legal Amazon, by its legal definition (Lei Complementar 124/2007): the states of Acre, Amapá, Amazonas,
Mato Grosso, Pará, Rondônia, Roraima and Tocantins, plus Maranhão west of the 44°W meridian. Built from the
IBGE 2025 municipal map; Maranhão is clipped at 44°W.

Points: MODIS Aqua fire events (2 km / 2 days, cached by fire_percolation_expected.py --events) whose centre is
inside the Legal Amazon. Random baseline: the same number of points scattered uniformly over the Legal Amazon,
REPS runs per year (the band is their range; the ratio uses their median).

    python audit/percolation_legal_amazon.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from audit.fire_percolation_pilot import (ALBERS, CACHE, GRID, INK, INK2, MUTED, OUT, SERIES, YEARS, mst_edges,
                                          project, summarise, sweep)
from audit.percolation_vs_deforestation import prodes
from firms import config

STATES = ["AC", "AP", "AM", "MT", "PA", "RO", "RR", "TO"]
REPS = 10
SHAPE_CACHE = config.DATA_DIR / "cache" / "legal_amazon_albers.parquet"


def legal_amazon():
    """Legal Amazon polygon in Albers metres (cached)."""
    import geopandas as gpd
    from shapely.geometry import box
    if SHAPE_CACHE.exists():
        return gpd.read_parquet(SHAPE_CACHE).union_all()
    m = gpd.read_file(config.RAW_DIR / "boundaries" / "BR_Municipios_2025.shp")[["SIGLA_UF", "geometry"]]
    whole = m[m.SIGLA_UF.isin(STATES)]
    ma = m[m.SIGLA_UF == "MA"].to_crs("EPSG:4674").clip(box(-80, -40, -44, 10)).to_crs(m.crs)
    shape = pd.concat([whole, ma]).to_crs(ALBERS).union_all().buffer(0)
    gpd.GeoDataFrame(geometry=[shape], crs=ALBERS).to_parquet(SHAPE_CACHE)
    return shape


def random_in(shape, n, rng):
    import shapely
    x0, y0, x1, y1 = shape.bounds
    pts = []
    while sum(len(p) for p in pts) < n:
        cand = rng.uniform([x0, y0], [x1, y1], size=(n, 2))
        pts.append(cand[shapely.contains_xy(shape, cand[:, 0], cand[:, 1])])
    return np.vstack(pts)[:n] / 1000.0


def d50(xy):
    xy = np.unique(np.round(xy, 4), axis=0)
    c, merges = sweep(len(xy), *mst_edges(xy))
    return summarise(c)["d50_km"], len(merges), len(xy)


def main():
    import matplotlib.pyplot as plt
    import shapely
    shape = legal_amazon()
    shapely.prepare(shape)
    print(f"Legal Amazon area: {shape.area / 1e6:,.0f} km²")
    rng = np.random.default_rng(2003)
    rows = []
    for y in YEARS:
        f = pd.read_parquet(CACHE / f"events_{y}_2km_2d.parquet")
        xy = project(f.lon.to_numpy(), f.lat.to_numpy())
        xy = xy[shapely.contains_xy(shape, xy[:, 0] * 1000, xy[:, 1] * 1000)]
        obs, merges, n = d50(xy)
        null = [d50(random_in(shape, n, rng))[0] for _ in range(REPS)]
        rows.append({"year": y, "events": n, "share_of_brazil": n / len(f), "fire_d50_km": obs, "big_merges": merges,
                     "random_d50_lo": min(null), "random_d50_mid": float(np.median(null)), "random_d50_hi": max(null)})
        print(f"{y}: {n:,} events ({n / len(f):.0%} of Brazil), d50 fire {obs:.1f} vs random {np.median(null):.1f} km",
              flush=True)
    df = pd.DataFrame(rows).set_index("year").join(prodes(), how="left")
    df["ratio"] = df.fire_d50_km / df.random_d50_mid
    df["ratio_lo"] = df.fire_d50_km / df.random_d50_hi
    df["ratio_hi"] = df.fire_d50_km / df.random_d50_lo
    df.to_csv(OUT / "percolation_legal_amazon.csv")

    res = {}
    for col in ("ratio", "fire_d50_km"):
        x, v = df.deforestation_km2, df[col]
        res[col] = {"pearson": stats.pearsonr(x, v), "spearman": stats.spearmanr(x, v),
                    "pearson_year_to_year_change": stats.pearsonr(x.diff().dropna(), v.diff().dropna()),
                    "pearson_2009_2024": stats.pearsonr(x[x.index >= 2009], v[v.index >= 2009])}
    out = {c: {k: {"r": round(float(t[0]), 3), "p": round(float(t[1]), 4)} for k, t in r.items()} for c, r in res.items()}
    (OUT / "percolation_legal_amazon.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    a = ax[0]
    a.bar(df.index, df.deforestation_km2 / 1000, color=SERIES[3], width=0.75)
    a.set_title("Deforestation, Legal Amazon (PRODES, thousand km² per year)", loc="left", fontsize=10, color=INK)
    a = ax[1]
    a.fill_between(df.index, df.ratio_lo, df.ratio_hi, color=SERIES[0], alpha=0.18, lw=0,
                   label=f"Range over {REPS} random runs")
    a.plot(df.index, df.ratio, color=SERIES[0], lw=2, marker="o", ms=4, label="Fire / random (median)")
    a.axhline(1, color=MUTED, lw=0.8, ls=":")
    a.legend(frameon=False, fontsize=9, loc="lower right")
    a.set_title("Critical point, Legal Amazon: fire / random half-joined distance", loc="left", fontsize=10, color=INK)
    a = ax[2]
    early = df.index <= 2008
    a.scatter(df.deforestation_km2[~early] / 1000, df.ratio[~early], color=SERIES[0], s=40, zorder=3, label="2009–2024")
    a.scatter(df.deforestation_km2[early] / 1000, df.ratio[early], color=SERIES[1], s=40, zorder=3, label="2003–2008")
    for y, row in df.iterrows():
        a.annotate(str(y), (row.deforestation_km2 / 1000, row.ratio), xytext=(4, 3), textcoords="offset points",
                   fontsize=7.5, color=INK2)
    rp, pp = res["ratio"]["pearson"]
    a.set_title(f"Each dot is a year (r = {rp:.2f}, p = {pp:.3f})", loc="left", fontsize=10, color=INK)
    a.set_xlabel("Deforestation (thousand km²)", color=INK2)
    a.set_ylabel("Fire / random", color=INK2)
    a.legend(frameon=False, fontsize=9)
    for a in ax:
        a.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Legal Amazon only: fire and deforestation over the same area. MODIS Aqua fire events 2003–2024; "
                 "PRODES year = Aug–Jul", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_legal_amazon.png", dpi=150)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
