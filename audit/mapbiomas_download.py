"""Download MapBiomas Brazil land cover (Collection 9, 30 m, annual) GeoTIFFs.

    python audit/mapbiomas_download.py 2019 [2020 ...]

Source: public Google Cloud bucket (mapbiomas-public, initiatives/brasil/collection_9/lclu/coverage), one
tiled LZW GeoTIFF per year for all Brazil (~1 GB), EPSG:4326, uint8 class codes (3 = forest formation,
6 = floodable forest, 15 = pasture, ...; see the MapBiomas legend). Saved as data/raw/mapbiomas/coverage_<year>.tif;
files already on disk at the right size are skipped. Download goes to a .part file first.
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from firms import config

URL = "https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/collection_9/lclu/coverage/brasil_coverage_{year}.tif"
OUT = config.RAW_DIR / "mapbiomas"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for year in sys.argv[1:] or ["2019"]:
        url, dest = URL.format(year=year), OUT / f"coverage_{year}.tif"
        size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])
        if dest.exists() and dest.stat().st_size == size:
            print(f"{dest.name}: already downloaded")
            continue
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
        assert tmp.stat().st_size == size, f"{dest.name}: size mismatch"
        tmp.replace(dest)
        print(f"{dest.name}: {size / 1e9:.2f} GB")


if __name__ == "__main__":
    sys.exit(main())
