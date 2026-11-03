"""Cache PRODES Amazon yearly deforestation polygons 2008-2024 in one parquet (Albers equal-area).

Sources already on disk: data/raw/prodes_lc/yearly0813_*.geojson (PRODES years 2008-2013) and
data/raw/prodes/amz_*.geojson (2014-2024). A PRODES year y runs from August y-1 to July y.
Output: data/cache/prodes_amazon_2008_2024.parquet with year, area_km2, centroid lon/lat and geometry.

    python audit/prodes_amazon_cache.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from firms import config

ALBERS = "+proj=aea +lat_0=-12 +lon_0=-54 +lat_1=-2 +lat_2=-22 +ellps=GRS80 +units=m +no_defs"
OUT = config.DATA_DIR / "cache" / "prodes_amazon_2008_2024.parquet"


def main():
    import geopandas as gpd
    files = sorted((config.RAW_DIR / "prodes_lc").glob("yearly0813_*.geojson")) + \
        sorted((config.RAW_DIR / "prodes").glob("amz_*.geojson"))
    parts = []
    for f in files:
        g = gpd.read_file(f, columns=["year", "main_class"], engine="pyogrio")
        g = g[g.main_class == "DESMATAMENTO"][["year", "geometry"]]
        g = g.set_crs("EPSG:4674", allow_override=True) if g.crs is None else g
        parts.append(g.to_crs(ALBERS))
        print(f.name, len(g), flush=True)
    g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=ALBERS)
    g["year"] = g.year.astype(int)
    bad = ~g.is_valid
    g.loc[bad, "geometry"] = g.loc[bad, "geometry"].buffer(0)
    print("repaired invalid polygons:", int(bad.sum()))
    g["area_km2"] = g.area / 1e6
    c = g.geometry.representative_point().to_crs("EPSG:4326")
    g["lon"], g["lat"] = c.x, c.y
    g.to_parquet(OUT)
    print(g.groupby("year").agg(patches=("area_km2", "size"), km2=("area_km2", "sum")).round(0).to_string())


if __name__ == "__main__":
    main()
