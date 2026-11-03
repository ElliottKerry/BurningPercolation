"""Cattle herd per municipality and year (IBGE PPM, SIDRA table 3939, cattle = 'Bovino'), 2003-2024, put onto the
0.25 deg lattice.

Download: SIDRA caps a request at 50,000 values, so years are fetched in blocks.
  -> data/raw/ibge/ppm_cattle_{first}_{last}.json, data/cache/cattle_municipal.parquet (cd_mun, year, cattle)
Lattice: each municipality's herd is spread evenly over its area (cattle per km2), and each 0.25 deg cell (same cells
as the lattice / ERA5: index floor(lon / 0.25 + 0.5)) gets the sum over the pieces of municipalities inside it.
Area weighting is a first pass; weighting by pasture (MapBiomas cell shares) is better once those are in.
  -> data/cache/cattle_cells.parquet (ix, iy, year, cattle, cattle_per_km2)
Quick look: audit/results/cattle_ppm.png

    python audit/cattle_ppm.py
"""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from firms import config

SIDRA = "https://apisidra.ibge.gov.br/values/t/3939/n6/all/v/105/p/{years}/c79/2670"
BLOCKS = [(2003, 2010), (2011, 2018), (2019, 2024)]
RAW = config.RAW_DIR / "ibge"
CACHE = config.DATA_DIR / "cache"
ALBERS = "+proj=aea +lat_0=-12 +lon_0=-54 +lat_1=-2 +lat_2=-22 +ellps=GRS80 +units=m +no_defs"
RES = 0.25
LEGAL_AMAZON = ("AC", "AM", "AP", "MA", "MT", "PA", "RO", "RR", "TO")


def download():
    parts = []
    for a, b in BLOCKS:
        path = RAW / f"ppm_cattle_{a}_{b}.json"
        if not path.exists():
            url = SIDRA.format(years=",".join(str(y) for y in range(a, b + 1)))
            with urllib.request.urlopen(url, timeout=600) as r:
                path.write_bytes(r.read())
            print("downloaded", path.name, f"{path.stat().st_size:,} bytes", flush=True)
        rows = json.loads(path.read_text(encoding="utf-8"))[1:]
        parts.append(pd.DataFrame(rows)[["D1C", "D3C", "V"]])
    df = pd.concat(parts, ignore_index=True).rename(columns={"D1C": "cd_mun", "D3C": "year", "V": "cattle"})
    df["year"] = df.year.astype(int)
    df["cattle"] = pd.to_numeric(df.cattle, errors="coerce")             # '-' = none, '...' / 'X' = not available
    df.to_parquet(CACHE / "cattle_municipal.parquet", index=False)
    return df


def to_cells(df):
    import geopandas as gpd
    from shapely.geometry import box
    mun = gpd.read_file(config.RAW_DIR / "boundaries" / "BR_Municipios_2025.shp")[["CD_MUN", "SIGLA_UF", "geometry"]]
    mun = mun.to_crs("EPSG:4326")
    x0, y0, x1, y1 = mun.total_bounds
    ixs = np.arange(int(np.floor(x0 / RES + 0.5)), int(np.floor(x1 / RES + 0.5)) + 1)
    iys = np.arange(int(np.floor(y0 / RES + 0.5)), int(np.floor(y1 / RES + 0.5)) + 1)
    ix, iy = np.meshgrid(ixs, iys)
    ix, iy = ix.ravel(), iy.ravel()
    cells = gpd.GeoDataFrame({"ix": ix, "iy": iy},
                             geometry=[box(a * RES - RES / 2, b * RES - RES / 2, a * RES + RES / 2, b * RES + RES / 2)
                                       for a, b in zip(ix, iy)], crs="EPSG:4326")
    mun_a, cells_a = mun.to_crs(ALBERS), cells.to_crs(ALBERS)
    mun_a["mun_km2"] = mun_a.area / 1e6
    pieces = gpd.overlay(cells_a, mun_a, how="intersection", keep_geom_type=True)
    pieces["frac"] = pieces.area / 1e6 / pieces.mun_km2
    cell_km2 = (cells_a.set_index(["ix", "iy"]).area / 1e6).rename("cell_km2")
    w = pieces[["ix", "iy", "CD_MUN", "frac"]].rename(columns={"CD_MUN": "cd_mun"})
    out = w.merge(df, on="cd_mun").assign(cattle=lambda d: d.cattle.fillna(0) * d.frac)
    out = out.groupby(["ix", "iy", "year"], as_index=False).cattle.sum()
    out = out.merge(cell_km2.reset_index(), on=["ix", "iy"])
    out["cattle_per_km2"] = out.cattle / out.cell_km2
    out.drop(columns="cell_km2").to_parquet(CACHE / "cattle_cells.parquet", index=False)
    return out, mun


def main():
    import matplotlib.pyplot as plt
    from audit.fire_percolation_pilot import GRID, INK, INK2, MUTED, OUT
    df = download()
    cells, mun = to_cells(df)
    uf = mun.set_index("CD_MUN").SIGLA_UF
    df["legal_amazon"] = df.cd_mun.map(uf).isin(LEGAL_AMAZON)
    tot = df.groupby(["year", "legal_amazon"]).cattle.sum().unstack() / 1e6
    print("cattle (millions):\n", tot.round(1).to_string())
    print("municipal total vs cell total, 2023:", round(df[df.year == 2023].cattle.sum() / 1e6, 1),
          round(cells[cells.year == 2023].cattle.sum() / 1e6, 1))

    fig, axs = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    ax = axs[0]
    ax.plot(tot.index, tot[True], marker="o", ms=3.5, lw=2, color="#c2410c", label="Legal Amazon states")
    ax.plot(tot.index, tot[False], marker="o", ms=3.5, lw=2, color="#2a78d6", label="Rest of Brazil")
    ax.axvspan(2006.5, 2015.5, color=MUTED, alpha=0.08, lw=0)
    ax.set_ylabel("Cattle (millions of head)", color=INK2)
    ax.set_title("Cattle herd per year (IBGE PPM)", loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    vmax = np.nanpercentile(cells.cattle_per_km2, 99)
    for ax, y in zip(axs[1:], (2003, 2023)):
        c = cells[cells.year == y]
        ax.scatter(c.ix * RES, c.iy * RES, c=c.cattle_per_km2, s=2.2, marker="s", cmap="YlOrBr", vmin=0, vmax=vmax)
        mun.dissolve("SIGLA_UF").boundary.plot(ax=ax, color="#9a9890", lw=0.3)
        ax.set_title(f"{y}: cattle per km² in each 0.25° cell", loc="left", fontsize=10.5, color=INK)
        ax.set_xlim(-74.2, -34.5)
        ax.set_ylim(-34, 5.5)
        ax.set_aspect("equal")
        ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap="YlOrBr", norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=axs[1:], shrink=0.7, label="Cattle per km²")
    fig.suptitle("Cattle herd by municipality, 2003–2024, on the 0.25° lattice (municipal totals spread evenly over "
                 "each municipality's area)", x=0.01, ha="left", fontsize=12,
                 color=INK)
    fig.savefig(OUT / "cattle_ppm.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
