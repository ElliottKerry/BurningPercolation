"""Fire-driver lattice: a 0.25 degree grid over the Amazon and Cerrado, one row per cell per month, 2016-2024.

Each step is its own command and writes a parquet file to data/cache/lattice/; `assemble` joins them.
Steps that use geopandas and steps that use DuckDB run as separate commands, because loading both in one
process has crashed before.

    python audit/lattice_build.py download    # ERA5 + ERA5-Land monthly means from the Copernicus CDS
    python audit/lattice_build.py grid        # cells, their area, and the share in Brazil / Amazon / Cerrado
    python audit/lattice_build.py fires       # VIIRS fire counts per cell-month (DuckDB)
    python audit/lattice_build.py weather     # rain, temperature, humidity, wind, soil moisture per cell-month
    python audit/lattice_build.py landcover   # MapBiomas land-cover shares per cell-year (slow: about 1 h)
    python audit/lattice_build.py clearing    # DETER and PRODES cleared area per cell-month
    python audit/lattice_build.py assemble    # join everything -> data/lattice/ + DuckDB tables

Grid: cell centres sit on ERA5's own 0.25 degree points (lon = ix * 0.25, lat = iy * 0.25), each cell spans
+/- 0.125 degrees, so ERA5 values need no regridding. Cells are kept if any part lies in the Amazon or
Cerrado biome (IBGE 1:250,000). Column meanings and caveats: audit/lattice_notes.md.
"""
import calendar
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from firms import config

RES = 0.25
IX0, IX1 = -296, -165          # lon centres -74.00 .. -41.25 (Amazon + Cerrado extent)
IY0, IY1 = -99, 21             # lat centres -24.75 ..   5.25
YEARS = range(2016, 2025)
AREA = [5.5, -74.25, -25.0, -41.0]  # CDS request box N, W, S, E (covers every cell edge)
ALBERS = "+proj=aea +lat_0=-12 +lon_0=-54 +lat_1=-2 +lat_2=-22 +ellps=GRS80 +units=m +no_defs"  # IBGE equal-area
CACHE = config.DATA_DIR / "cache" / "lattice"
RAW = config.RAW_DIR / "era5_lattice"
OUT = config.DATA_DIR / "lattice"

# MapBiomas Collection 9 classes grouped for fire work. Savanna formation (4) is kept apart from forest,
# although MapBiomas level 1 puts it under "forest".
LC_GROUPS = {
    "forest": [3, 5, 6, 49],
    "savanna": [4],
    "grass_other_natural": [11, 12, 13, 29, 32, 50],
    "pasture": [15],
    "agriculture": [9, 18, 19, 20, 35, 36, 39, 40, 41, 46, 47, 48, 62],
    "mosaic": [21],
    "nonvegetated": [22, 23, 24, 25, 30],
    "water": [26, 31, 33],
}
DETER_CLEARING = ["DESMATAMENTO_CR", "DESMATAMENTO_VEG", "MINERACAO"]  # same classes as the placebo screen
DETER_CERRADO_START = "2018-05-01"


def cell_index(lon, lat):
    """Cell ix, iy holding a point (cell edges at centre +/- 0.125 degrees)."""
    return (np.floor(np.asarray(lon) / RES + 0.5).astype(np.int32),
            np.floor(np.asarray(lat) / RES + 0.5).astype(np.int32))


# ---------------------------------------------------------------------------------------------- download
def download():
    import cdsapi
    RAW.mkdir(parents=True, exist_ok=True)
    years = [str(y) for y in YEARS]
    months = [f"{m:02d}" for m in range(1, 13)]
    jobs = {
        # Monthly mean over all hours of every day.
        "era5_mean": ("reanalysis-era5-single-levels-monthly-means", "monthly_averaged_reanalysis", "00:00",
                      ["2m_temperature", "2m_dewpoint_temperature", "10m_u_component_of_wind",
                       "10m_v_component_of_wind", "10m_wind_speed", "total_precipitation"]),
        # Monthly mean of the 18:00 UTC hour only = 15:00 in Brasilia time, mid-afternoon, the fire-weather peak.
        "era5_afternoon": ("reanalysis-era5-single-levels-monthly-means", "monthly_averaged_reanalysis_by_hour_of_day",
                           "18:00", ["2m_temperature", "2m_dewpoint_temperature", "10m_u_component_of_wind",
                                     "10m_v_component_of_wind", "10m_wind_speed"]),
        "era5land_soil": ("reanalysis-era5-land-monthly-means", "monthly_averaged_reanalysis", "00:00",
                          ["volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2"]),
    }
    client = cdsapi.Client(quiet=True, progress=False)
    for name, (dataset, product, hour, variables) in jobs.items():
        dest = RAW / name
        if dest.exists() and any(dest.glob("*.nc")):
            print(f"{name}: already downloaded")
            continue
        zpath = RAW / f"{name}.zip"
        print(f"{name}: requesting from the CDS (may queue)...", flush=True)
        client.retrieve(dataset, {"product_type": [product], "variable": variables, "year": years,
                                  "month": months, "time": [hour], "area": AREA,
                                  "data_format": "netcdf", "download_format": "zip"}, str(zpath))
        dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest)
        zpath.unlink()
        print(f"{name}: {[p.name for p in dest.glob('*.nc')]}", flush=True)


# -------------------------------------------------------------------------------------------------- grid
def grid():
    import geopandas as gpd
    from shapely.geometry import box
    ix, iy = np.meshgrid(np.arange(IX0, IX1 + 1), np.arange(IY0, IY1 + 1))
    ix, iy = ix.ravel(), iy.ravel()
    lon, lat = ix * RES, iy * RES
    h = RES / 2
    cells = gpd.GeoDataFrame({"ix": ix, "iy": iy, "lon": lon, "lat": lat},
                             geometry=[box(x - h, y - h, x + h, y + h) for x, y in zip(lon, lat)], crs="EPSG:4674")
    # Area of a lat-lon box on the sphere (authalic radius): R^2 * dlon * (sin lat_top - sin lat_bottom).
    r = 6371.007
    cells["area_km2"] = r ** 2 * np.radians(RES) * (np.sin(np.radians(lat + h)) - np.sin(np.radians(lat - h)))
    biomes = gpd.read_file(config.RAW_DIR / "biomes" / "lm_bioma_250.shp")[["CD_Bioma", "geometry"]]
    biomes = biomes.set_crs("EPSG:4674", allow_override=True)
    eq = cells.to_crs(ALBERS)
    full = eq.geometry.area
    pieces = gpd.overlay(eq[["ix", "iy", "geometry"]], biomes.to_crs(ALBERS), how="intersection", keep_geom_type=True)
    pieces["a"] = pieces.geometry.area
    share = pieces.pivot_table(index=["ix", "iy"], columns="CD_Bioma", values="a", aggfunc="sum", fill_value=0)
    out = cells.drop(columns="geometry").merge(share.reset_index(), on=["ix", "iy"], how="left").fillna(0)
    denom = full.to_numpy()
    out["share_brazil"] = out[[c for c in share.columns]].sum(axis=1) / denom
    out["share_amazon"] = out.get(1, 0) / denom
    out["share_cerrado"] = out.get(3, 0) / denom
    out = out[(out.share_amazon + out.share_cerrado) > 0].copy()
    names = {1: "Amazon", 2: "Caatinga", 3: "Cerrado", 4: "Atlantic Forest", 5: "Pampa", 6: "Pantanal"}
    bcols = [c for c in share.columns]
    out["main_biome"] = out[bcols].idxmax(axis=1).map(names)
    out["cell_id"] = (out.iy - IY0) * (IX1 - IX0 + 1) + (out.ix - IX0)
    out = out[["cell_id", "ix", "iy", "lon", "lat", "area_km2", "share_brazil", "share_amazon", "share_cerrado",
               "main_biome"]].round({"area_km2": 2, "share_brazil": 4, "share_amazon": 4, "share_cerrado": 4})
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE / "cells.parquet", index=False)
    print(f"cells: {len(out):,} (main biome: {out.main_biome.value_counts().to_dict()}); "
          f"area {out.area_km2.min():.0f}-{out.area_km2.max():.0f} km2")


# ------------------------------------------------------------------------------------------------- fires
def fires():
    """VIIRS S-NPP (low-confidence already removed), local date = UTC-3; NOAA-20 on S-NPP outage days,
    exactly as in the clearing-to-burning work."""
    import duckdb
    gaps = (config.DATA_DIR / "cache" / "snpp_gap_days.csv").as_posix()
    local = "(acq_ts - INTERVAL 3 HOUR)::DATE"
    w, e = (IX0 - 0.5) * RES, (IX1 + 0.5) * RES
    s, n = (IY0 - 0.5) * RES, (IY1 + 0.5) * RES
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    f = con.sql(f"""
        WITH gaps AS (SELECT d::DATE AS d FROM read_csv_auto('{gaps}')),
        pts AS (
          SELECT CAST(floor(longitude / {RES} + 0.5) AS INTEGER) AS ix, CAST(floor(latitude / {RES} + 0.5) AS INTEGER) AS iy,
                 date_trunc('month', {local})::DATE AS month, frp
            FROM fire_detections_clean
           WHERE longitude >= {w} AND longitude < {e} AND latitude >= {s} AND latitude < {n}
             AND {local} BETWEEN DATE '{YEARS[0]}-01-01' AND DATE '{YEARS[-1]}-12-31'
             AND ((satellite = 'N' AND {local} NOT IN (SELECT d FROM gaps))
               OR (satellite = 'N20' AND {local} IN (SELECT d FROM gaps))))
        SELECT ix, iy, month, count(*) AS fires, sum(frp) AS frp_sum FROM pts GROUP BY ALL""").df()
    # Days per month with no VIIRS data at all: an S-NPP outage day with no NOAA-20 detections in the whole box.
    days = con.sql(f"""
        WITH gaps AS (SELECT d::DATE AS d FROM read_csv_auto('{gaps}')),
        n20 AS (SELECT DISTINCT {local} AS d FROM fire_detections_clean
                 WHERE satellite = 'N20' AND {local} IN (SELECT d FROM gaps)
                   AND longitude >= {w} AND longitude < {e} AND latitude >= {s} AND latitude < {n})
        SELECT date_trunc('month', g.d)::DATE AS month, count(*) AS days_gapfilled,
               count(*) FILTER (WHERE n20.d IS NULL) AS days_no_viirs
          FROM gaps g LEFT JOIN n20 ON g.d = n20.d
         WHERE g.d BETWEEN DATE '{YEARS[0]}-01-01' AND DATE '{YEARS[-1]}-12-31' GROUP BY ALL""").df()
    CACHE.mkdir(parents=True, exist_ok=True)
    f.to_parquet(CACHE / "fires.parquet", index=False)
    days.to_parquet(CACHE / "fire_days.parquet", index=False)
    print(f"fires: {f.fires.sum():,} detections in {len(f):,} cell-months; outage months: {len(days)}, "
          f"days with no VIIRS at all: {int(days.days_no_viirs.sum())}")


# ----------------------------------------------------------------------------------------------- weather
def _open(folder: str):
    import xarray as xr
    parts = []
    for p in sorted((RAW / folder).glob("*.nc")):
        d = xr.open_dataset(p)
        d = d.rename({"valid_time": "time"}) if "valid_time" in d.dims else d
        # Accumulated variables (rain) are stamped 06:00, the rest 00:00 or 18:00: align all to the month start.
        parts.append(d.assign_coords(time=d.time.dt.floor("D") - (d.time.dt.day - 1) * np.timedelta64(1, "D")))
    ds = xr.merge(parts, compat="override", join="exact")
    return ds.drop_vars([v for v in ("number", "expver") if v in ds.variables], errors="ignore")


def _to_cells(ds, cols: dict) -> pd.DataFrame:
    d = ds[list(cols)].to_dataframe().reset_index().rename(columns=cols)
    d["month"] = pd.to_datetime(d.time).dt.to_period("M").dt.to_timestamp()
    d["ix"], d["iy"] = cell_index(d.longitude, d.latitude)
    return d.drop(columns=["time", "longitude", "latitude"])


def rh(t_c, td_c):
    """Relative humidity (%) from temperature and dewpoint in deg C (Magnus formula, Alduchov & Eskridge 1996)."""
    return 100 * np.exp(17.625 * td_c / (243.04 + td_c)) / np.exp(17.625 * t_c / (243.04 + t_c))


def wind_from(u, v):
    """Direction the wind blows FROM, degrees clockwise from north (meteorological convention)."""
    return np.mod(270 - np.degrees(np.arctan2(v, u)), 360)


def weather():
    cells = pd.read_parquet(CACHE / "cells.parquet")[["ix", "iy"]]
    m = _to_cells(_open("era5_mean"), {"t2m": "t", "d2m": "td", "u10": "u", "v10": "v", "si10": "ws", "tp": "tp"})
    a = _to_cells(_open("era5_afternoon"), {"t2m": "t", "d2m": "td", "u10": "u", "v10": "v", "si10": "ws"})
    out = pd.DataFrame({"ix": m.ix, "iy": m.iy, "month": m.month})
    ndays = m.month.dt.days_in_month
    out["rain_mm"] = m.tp * 1000 * ndays            # ERA5 monthly mean is metres per day
    out["temp_c"] = m.t - 273.15
    out["rh_pct"] = rh(m.t - 273.15, m.td - 273.15)
    out["wind_speed_ms"] = m.ws
    out["wind_u_ms"], out["wind_v_ms"] = m.u, m.v
    out["wind_from_deg"] = wind_from(m.u, m.v)
    out["wind_steadiness"] = np.hypot(m.u, m.v) / m.ws   # 1 = always the same direction, 0 = no prevailing direction
    aft = pd.DataFrame({"ix": a.ix, "iy": a.iy, "month": a.month, "temp_pm_c": a.t - 273.15,
                        "rh_pm_pct": rh(a.t - 273.15, a.td - 273.15), "wind_speed_pm_ms": a.ws,
                        "wind_from_pm_deg": wind_from(a.u, a.v)})
    out = out.merge(aft, on=["ix", "iy", "month"], how="left")
    s = _to_cells(_open("era5land_soil"), {"swvl1": "soil_0_7cm", "swvl2": "soil_7_28cm"})
    s = s.groupby(["ix", "iy", "month"], as_index=False)[["soil_0_7cm", "soil_7_28cm"]].mean()  # 0.1 -> 0.25 deg
    out = out.merge(s, on=["ix", "iy", "month"], how="left")
    out = cells.merge(out, on=["ix", "iy"], how="left")
    out.to_parquet(CACHE / "weather.parquet", index=False)
    print(f"weather: {len(out):,} cell-months; missing soil moisture {out.soil_0_7cm.isna().mean():.1%} "
          f"(ERA5-Land has no value over open water)")
    print(out.drop(columns=["ix", "iy", "month"]).describe().T[["mean", "min", "max"]].round(2).to_string())


# --------------------------------------------------------------------------------------------- landcover
def _landcover_year(year: int) -> pd.DataFrame:
    """Pixel counts per cell and land-cover group for one year (cached per year, so a killed run resumes)."""
    cache = CACHE / f"landcover_{year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    d = _count_landcover(year)
    d.to_parquet(cache, index=False)
    return d


def _count_landcover(year: int) -> pd.DataFrame:
    import rasterio
    from rasterio.windows import Window, from_bounds
    lut = np.full(256, 255, dtype=np.uint8)               # 255 = no data / unlisted class
    for k, classes in enumerate(LC_GROUPS.values()):
        lut[classes] = k
    ng = len(LC_GROUPS)
    ncx = IX1 - IX0 + 1
    counts = np.zeros(((IY1 - IY0 + 1) * ncx, ng + 1), dtype=np.int64)   # last column: unlisted classes
    with rasterio.open(config.RAW_DIR / "mapbiomas" / f"coverage_{year}.tif") as r:
        w = from_bounds((IX0 - 0.5) * RES, (IY0 - 0.5) * RES, (IX1 + 0.5) * RES, (IY1 + 0.5) * RES,
                        r.transform).round_offsets().round_lengths()
        a, e = r.transform.a, r.transform.e
        lon = r.transform.c + (np.arange(w.col_off, w.col_off + w.width) + 0.5) * a
        cx = np.clip(np.floor(lon / RES + 0.5).astype(np.int64) - IX0, 0, ncx - 1)
        for row0 in range(w.row_off, w.row_off + w.height, 512):
            n = min(512, w.row_off + w.height - row0)
            block = r.read(1, window=Window(w.col_off, row0, w.width, n))
            lat = r.transform.f + (np.arange(row0, row0 + n) + 0.5) * e
            cy = np.clip(np.floor(lat / RES + 0.5).astype(np.int64) - IY0, 0, IY1 - IY0)
            g = lut[block]
            keep = block > 0
            g = np.where(g == 255, ng, g)
            idx = ((cy[:, None] * ncx + cx[None, :]) * (ng + 1) + g)[keep]
            counts += np.bincount(idx, minlength=counts.size).reshape(counts.shape)
    rows = np.flatnonzero(counts.sum(axis=1))
    d = pd.DataFrame(counts[rows], columns=[*LC_GROUPS, "unlisted"])
    d["iy"], d["ix"] = rows // ncx + IY0, rows % ncx + IX0
    d["year"] = year
    return d


def landcover():
    from concurrent.futures import ProcessPoolExecutor
    cells = pd.read_parquet(CACHE / "cells.parquet")[["ix", "iy"]]
    years = [y for y in YEARS if (config.RAW_DIR / "mapbiomas" / f"coverage_{y}.tif").exists()]
    with ProcessPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(_landcover_year, years))
    d = pd.concat(parts, ignore_index=True)
    total = d[[*LC_GROUPS, "unlisted"]].sum(axis=1)
    out = d[["ix", "iy", "year"]].copy()
    for g in LC_GROUPS:
        out[f"lc_{g}"] = (d[g] / total).round(4)
    out["lc_unlisted"] = (d.unlisted / total).round(4)
    out = cells.merge(out, on=["ix", "iy"], how="inner")
    out.to_parquet(CACHE / "landcover.parquet", index=False)
    print(f"landcover: years {years}; {len(out):,} cell-years; max unlisted share {out.lc_unlisted.max():.3f}")


# ---------------------------------------------------------------------------------------------- clearing
def clearing():
    """Cleared area per cell-month. Each polygon goes to the cell holding its centroid (patches are tiny next
    to a 28 km cell); area in km2 from the IBGE Albers equal-area projection."""
    import geopandas as gpd
    cells = pd.read_parquet(CACHE / "cells.parquet")[["ix", "iy"]]
    parts = []
    for src, prefix, field, keep in [("deter", "amz", "view_date", "classname"),
                                     ("deter", "cerrado", "view_date", "classname"),
                                     ("prodes", "amz", "image_date", "main_class"),
                                     ("prodes", "cerrado", "image_date", "main_class")]:
        wanted = DETER_CLEARING if src == "deter" else ["DESMATAMENTO"]
        for p in sorted((config.RAW_DIR / src).glob(f"{prefix}_*.geojson")):
            g = gpd.read_file(p, columns=[keep, field])
            g = g[g[keep].isin(wanted)]
            g = g.set_crs("EPSG:4674", allow_override=True) if g.crs is None else g
            a = g.to_crs(ALBERS)
            c = a.geometry.centroid.to_crs("EPSG:4674")
            ix, iy = cell_index(c.x, c.y)
            parts.append(pd.DataFrame({"col": f"{src}_{prefix}_km2", "ix": ix, "iy": iy,
                                       "date": pd.to_datetime(g[field], errors="coerce"),
                                       "km2": a.geometry.area.to_numpy() / 1e6}))
        print(f"{src} {prefix}: read", flush=True)
    d = pd.concat(parts, ignore_index=True)
    bad = d.date.isna().sum()
    d = d[(d.date >= f"{YEARS[0]}-01-01") & (d.date <= f"{YEARS[-1]}-12-31")]
    d["month"] = d.date.dt.to_period("M").dt.to_timestamp()
    out = d.pivot_table(index=["ix", "iy", "month"], columns="col", values="km2", aggfunc="sum").reset_index()
    out.columns.name = None
    out = cells.merge(out, on=["ix", "iy"], how="inner")
    out.to_parquet(CACHE / "clearing.parquet", index=False)
    print(f"clearing: {len(out):,} cell-months with clearing; polygons without a date dropped: {bad:,}")
    print(d.groupby(["col", d.date.dt.year]).km2.sum().unstack(0).round(0).to_string())


# ---------------------------------------------------------------------------------------------- assemble
def assemble():
    import duckdb
    cells = pd.read_parquet(CACHE / "cells.parquet")
    months = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-01", freq="MS")
    p = cells[["cell_id", "ix", "iy"]].merge(pd.DataFrame({"month": months}), how="cross")
    f = pd.read_parquet(CACHE / "fires.parquet")
    f["month"] = pd.to_datetime(f.month)
    p = p.merge(f, on=["ix", "iy", "month"], how="left")
    p[["fires", "frp_sum"]] = p[["fires", "frp_sum"]].fillna(0)
    p["fires"] = p.fires.astype("int32")
    days = pd.read_parquet(CACHE / "fire_days.parquet")
    days["month"] = pd.to_datetime(days.month)
    p = p.merge(days, on="month", how="left")
    p[["days_gapfilled", "days_no_viirs"]] = p[["days_gapfilled", "days_no_viirs"]].fillna(0).astype("int8")
    w = pd.read_parquet(CACHE / "weather.parquet")
    w["month"] = pd.to_datetime(w.month)
    p = p.merge(w, on=["ix", "iy", "month"], how="left")
    c = pd.read_parquet(CACHE / "clearing.parquet")
    c["month"] = pd.to_datetime(c.month)
    p = p.merge(c, on=["ix", "iy", "month"], how="left")
    kcols = ["deter_amz_km2", "deter_cerrado_km2", "prodes_amz_km2", "prodes_cerrado_km2"]
    for k in kcols:
        p[k] = p[k].fillna(0) if k in p else 0.0
    p.loc[p.month < DETER_CERRADO_START, "deter_cerrado_km2"] = np.nan   # DETER-Cerrado did not exist yet
    lc = pd.read_parquet(CACHE / "landcover.parquet")
    p["year"] = p.month.dt.year
    p = p.merge(lc, on=["ix", "iy", "year"], how="left").drop(columns=["year", "ix", "iy"])
    p = p.sort_values(["cell_id", "month"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(OUT / "lattice_025_cells.parquet", index=False)
    p.to_parquet(OUT / "lattice_025_monthly.parquet", index=False)
    con = duckdb.connect(str(config.DB_PATH))
    for t in ("cells", "monthly"):
        con.execute(f"CREATE OR REPLACE TABLE lattice_025_{t} AS SELECT * FROM read_parquet("
                    f"'{(OUT / f'lattice_025_{t}.parquet').as_posix()}')")
    con.close()
    print(f"lattice: {len(cells):,} cells x {len(months)} months = {len(p):,} rows, {p.shape[1]} columns")
    print(f"cell-months with any fire: {(p.fires > 0).mean():.1%}; total detections {p.fires.sum():,}")
    print("missing share by column:\n" + p.isna().mean().loc[lambda s: s > 0].round(3).to_string())


if __name__ == "__main__":
    steps = {"download": download, "grid": grid, "fires": fires, "weather": weather,
             "landcover": landcover, "clearing": clearing, "assemble": assemble}
    steps[sys.argv[1]]()
