"""ERA5 monthly weather for every 0.25 deg cell in Brazil, 2003-2024 (for the percolation climate test).

Same cells as the lattice and the percolation work: centres on ERA5's own 0.25 deg points, cell index
floor(lon / 0.25 + 0.5), so no regridding. One CDS request (monthly means of all hours):
  rain (total_precipitation), 2 m temperature and dewpoint (-> relative humidity), 10 m wind speed,
  volumetric soil water layer 1 (0-7 cm; ERA5 itself at 0.25 deg, not ERA5-Land).
Rain in ERA5 monthly means is metres per day, so mm in month = tp * 1000 * days in month.

    python audit/era5_brazil.py download
    python audit/era5_brazil.py cells        # -> data/cache/era5_brazil_monthly.parquet (ix, iy, month, ...)
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from firms import config

RAW = config.RAW_DIR / "era5_brazil"
OUT = config.DATA_DIR / "cache" / "era5_brazil_monthly.parquet"
YEARS = range(2003, 2025)
AREA = [5.5, -74.25, -34.0, -28.5]          # N, W, S, E: every Brazil cell edge
VARS = ["total_precipitation", "2m_temperature", "2m_dewpoint_temperature", "10m_wind_speed",
        "volumetric_soil_water_layer_1"]


def download():
    import cdsapi
    RAW.mkdir(parents=True, exist_ok=True)
    if any(RAW.glob("*.nc")):
        print("already downloaded")
        return
    z = RAW / "era5_brazil.zip"
    cdsapi.Client(quiet=True, progress=False).retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {"product_type": ["monthly_averaged_reanalysis"], "variable": VARS, "year": [str(y) for y in YEARS],
         "month": [f"{m:02d}" for m in range(1, 13)], "time": ["00:00"], "area": AREA,
         "data_format": "netcdf", "download_format": "zip"}, str(z))
    with zipfile.ZipFile(z) as f:
        f.extractall(RAW)
    z.unlink()
    print([p.name for p in RAW.glob("*.nc")])


def cells():
    import xarray as xr
    ds = xr.merge([xr.open_dataset(p).drop_vars(["expver", "number"], errors="ignore") for p in sorted(RAW.glob("*.nc"))],
                  compat="override")
    t = "valid_time" if "valid_time" in ds.dims else "time"
    df = ds.to_dataframe().reset_index()
    df["month"] = pd.to_datetime(df[t]).dt.to_period("M").dt.to_timestamp()
    df["ix"] = np.floor(df.longitude / 0.25 + 0.5).astype(int)
    df["iy"] = np.floor(df.latitude / 0.25 + 0.5).astype(int)
    df = df.groupby(["ix", "iy", "month"], as_index=False)[["tp", "t2m", "d2m", "si10", "swvl1"]].mean()
    days = df.month.dt.days_in_month
    t, td = df.t2m - 273.15, df.d2m - 273.15
    rh = 100 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t))
    out = pd.DataFrame({"ix": df.ix, "iy": df.iy, "month": df.month, "rain_mm": df.tp * 1000 * days, "temp_c": t,
                        "rh_pct": rh.clip(0, 100), "wind_ms": df.si10, "soil_0_7cm": df.swvl1})
    out.to_parquet(OUT, index=False)
    print(f"{len(out):,} cell-months, {out.month.min():%Y-%m} to {out.month.max():%Y-%m}")


if __name__ == "__main__":
    {"download": download, "cells": cells}[sys.argv[1]]()
