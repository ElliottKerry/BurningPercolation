# Fire-driver lattice (0.25°, monthly, Amazon + Cerrado, 2016–2024)

A separate analysis from clearing to burning, started 2026-10-22. It puts fire counts and possible fire drivers on one regular grid, so they can be modelled as lattice (areal) data: a fixed set of cells, each with neighbours, each with a value per month.

Build: `audit/lattice_build.py` (steps listed at the top of the script). Output:

- `data/lattice/lattice_025_cells.parquet` and DuckDB table `lattice_025_cells`: one row per cell.
- `data/lattice/lattice_025_monthly.parquet` and DuckDB table `lattice_025_monthly`: one row per cell per month.

## The grid

- Cell centres sit on ERA5's own 0.25° points. Each cell runs ±0.125° around its centre, about 28 km across.
- Cells are kept if any part lies in the Amazon or Cerrado biome (IBGE 1:250,000).
- Cell area is between about 700 and 775 km², because degree cells shrink away from the equator. Use `area_km2 × share_brazil` as the exposure (offset) in count models.
- Border and coast cells are only partly in Brazil. FIRMS fire and MapBiomas data stop at the border, but ERA5 weather covers the whole cell.
- Neighbours: cells `(ix±1, iy)` and `(ix, iy±1)` are rook neighbours; add the diagonals for queen neighbours.

## Columns

| Column | Source | Meaning |
|---|---|---|
| `fires` | VIIRS S-NPP 375 m (FIRMS), NOAA-20 on S-NPP outage days | Detections in the cell that month, with the date taken in local time (UTC−3) |
| `frp_sum` | same | Total fire radiative power (MW), a rough intensity measure |
| `days_gapfilled`, `days_no_viirs` | `data/cache/snpp_gap_days.csv` | S-NPP outage days in the month, and days with no VIIRS data at all |
| `rain_mm` | ERA5 monthly means | Total rainfall in the month (mm) |
| `temp_c`, `rh_pct` | ERA5 | Mean over all hours of 2 m air temperature and relative humidity |
| `temp_pm_c`, `rh_pm_pct`, `wind_speed_pm_ms`, `wind_from_pm_deg` | ERA5, 18:00 UTC hour | The same at 15:00 Brasília time, the afternoon fire-weather peak |
| `wind_speed_ms` | ERA5 | Mean 10 m wind speed |
| `wind_u_ms`, `wind_v_ms` | ERA5 | Mean east-west and north-south wind |
| `wind_from_deg` | from u, v | Direction the wind comes from (0 = north, 90 = east) |
| `wind_steadiness` | from u, v and speed | 1 = always from the same direction; near 0 = no prevailing direction |
| `soil_0_7cm`, `soil_7_28cm` | ERA5-Land (0.1°, averaged to the cell) | Volumetric soil water (m³/m³) |
| `lc_*` | MapBiomas Collection 9, 30 m | Share of the cell's mapped land in each group, for that calendar year |
| `deter_amz_km2`, `deter_cerrado_km2` | INPE DETER | Area of clear-cut, vegetation-clearing and mining alerts, dated by the image date |
| `prodes_amz_km2`, `prodes_cerrado_km2` | INPE PRODES | Yearly mapped clearing, dated by the image date |

Land-cover groups: forest (MapBiomas classes 3, 5, 6, 49), savanna (4), grassland and other natural (11, 12, 13, 29, 32, 50), pasture (15), agriculture (9, 18–20, 35, 36, 39–41, 46–48, 62), mosaic of uses (21), non-vegetated (22–25, 30), water (26, 31, 33).

## Things to know before modelling

1. **Lag the covariates.** Fire lowers soil moisture, changes land cover and follows clearing. Use earlier values, such as last month's or last year's land cover, so a covariate is not an effect of the fire it explains.
2. **Wind direction is a circular variable.** Use `wind_u_ms`, `wind_v_ms` (or the sine and cosine of the direction) in a model, not the degrees.
3. **Humidity is approximate.** It is computed from the monthly mean temperature and dewpoint, not as the average of hourly humidity, which reads slightly high in the driest months.
4. **DETER and PRODES overlap.** They often map the same clearing, so treat them as two measures and don't add them. DETER is near real time with monthly dates; PRODES is the more complete yearly map.
5. **Gaps by design:**
   - `deter_cerrado_km2` is empty before May 2018, because DETER-Cerrado did not exist yet.
   - Land cover is empty for 2024, because MapBiomas Collection 9 ends in 2023. Collection 10 covers 2024 but uses different methods.
   - Soil moisture is missing where ERA5-Land has no land.
6. **Clearing is assigned by centroid.** Each DETER or PRODES polygon goes to the cell holding its centre. Patches are small next to a 28 km cell, so this barely matters, but a large polygon on a cell edge counts entirely in one cell.
7. **Resolution varies.** Inputs range from 30 m (MapBiomas) through 375 m (VIIRS) to about 28 km (ERA5), so fine-scale detail is averaged away in the weather.
8. **Cloud.** Satellites miss fires under cloud, and cloud is patchy across space, so some apparent spatial clustering of fire may come from cloud.
9. **Not included yet:** vegetation condition (MODIS NDVI/EVI). It needs a free NASA Earthdata login.
