# Burning Percolation

**How connected is fire across Brazil, and has that changed since 2003?**

This project treats fire in Brazil as a percolation problem. Each MODIS Aqua fire (detections within 2 km and 2 days are merged into one fire event) is a point on the map. Points are joined when they are closer than a distance *d*. As *d* grows, small clusters merge into one large cluster. The distance at which the largest cluster holds half of all fires (**d50**) measures how clustered fire is. Dividing it by the d50 of the same number of random points gives a **clustering ratio** that can be compared from year to year.

The idea follows hierarchical percolation of road networks (Arcaute et al. 2016), applied here to fire.

> Status: exploratory PhD research (University of Derby, 2026). Results are pilot findings, not peer reviewed.

## Findings so far

- **Three periods.** Fire clustering was high in 2003–06, lower in 2007–15 and high again in 2016–24.
- **Human, not weather.** The clustering ratio tracks the yearly deforestation rate (PRODES; Spearman −0.53, p 0.011 for the days-until-joined measure). It shows no clear link to rainfall, temperature, humidity or soil moisture.
- **Frontier fires.** In the Amazon, fire on newly cleared land is about twice as clustered as other fire (percolation threshold about 37 km vs about 17 km). Overall clustering depends on the share of fire on the frontier, and that share follows deforestation.
- **Land class alone does not explain it.** Holding the mix of land-cover types (MapBiomas) fixed leaves the three periods in place. What matters is *closeness to recent clearing*, not the class of the pixel that burns.
- **Brazil-wide frontier share** (fires within 1 km of a MapBiomas natural-to-farm conversion of at least 6.3 ha) runs high, low, high: about 37% in 2003–06, 26% in 2007–15 and 31% in 2016–23.

Full notes: [`audit/lattice_notes.md`](audit/lattice_notes.md). The maths of the expected-vs-actual comparison is in `audit/results/Fire percolation - the maths behind expected vs actual.docx`.

## Layout

| Path | What it is |
|---|---|
| `firms/`, `backfill.py`, `assign_areas.py`, `quality.py` | Download NASA FIRMS fire detections for Brazil into a local DuckDB database |
| `audit/fire_percolation_pilot.py` | First pilot: extract MODIS Aqua fires, merge into events, cluster-size curves |
| `audit/percolation_measure.py` | The core measure: d50 and the random-points calibration |
| `audit/percolation_*.py` | Each later analysis (by year, stability, climate, deforestation, frontier, land cover, sensors, early warning, health, maps) |
| `audit/lattice_build.py`, `audit/era5_brazil.py` | The 0.25° monthly lattice of fire and climate (ERA5) |
| `audit/mapbiomas_ee.py` | MapBiomas land cover through Google Earth Engine |
| `audit/cattle_ppm.py` | Cattle herd per municipality (IBGE PPM), put on the 0.25° lattice |
| `audit/prodes_amazon_cache.py`, `audit/priority_municipalities.py` | PRODES clearing and the priority-municipalities test |
| `audit/results/` | Figures (`.png`), tables (`.csv`) and summary numbers (`.json`) |

Each script lists its inputs, outputs and how to run it at the top.

## Data

The data are **not** in this repository (about 32 GB). Scripts expect them under `data/`:

- NASA FIRMS active fires (MODIS, VIIRS), from a free FIRMS map key in `.env` as `FIRMS_MAP_KEY=...`
- INPE PRODES deforestation (yearly rates and polygons)
- MapBiomas Collection 10 land cover (via Google Earth Engine)
- ERA5 climate (Copernicus CDS)
- IBGE boundaries, census and livestock (SIDRA)

```
pip install -r requirements.txt
python backfill.py          # fire detections -> data/firms_brazil.duckdb
python audit/fire_percolation_pilot.py extract
```
