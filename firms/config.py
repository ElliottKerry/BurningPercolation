"""Static configuration for the Brazil FIRMS backfill."""
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LOG_DIR = ROOT / "logs"
DB_PATH = DATA_DIR / "firms_brazil.duckdb"

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov"
ARCHIVE_URL = FIRMS_BASE + "/data/country/{slug}/{year}/{slug}_{year}_Brazil.csv"
COUNTRY = "BRA"
# Generous Brazil bounding box (W, S, E, N) used for the area-endpoint fallback and validation.
BRAZIL_BBOX = (-74.5, -34.5, -28.0, 5.8)

API_MAX_DAYS = 10            # FIRMS API maximum day_range per request
QUOTA_SAFETY_MARGIN = 50     # stop short of the transaction limit by this much
QUOTA_WINDOW_SECONDS = 600   # FIRMS transaction window (10 minutes)
HTTP_TIMEOUT = (30, 600)     # (connect, read) seconds
MAX_RETRIES = 6


@dataclass(frozen=True)
class Sensor:
    name: str               # our canonical sensor id
    archive_slug: str | None  # folder slug on the yearly country archive (None = not published)
    api_sp: str             # standard-processing API source
    api_nrt: str            # near-real-time API source
    first_year: int


SENSORS = {
    "modis": Sensor("modis", "modis", "MODIS_SP", "MODIS_NRT", 2000),
    "viirs_snpp": Sensor("viirs_snpp", "viirs-snpp", "VIIRS_SNPP_SP", "VIIRS_SNPP_NRT", 2012),
    "viirs_noaa20": Sensor("viirs_noaa20", "viirs-jpss1", "VIIRS_NOAA20_SP", "VIIRS_NOAA20_NRT", 2018),
    "viirs_noaa21": Sensor("viirs_noaa21", None, "VIIRS_NOAA21_SP", "VIIRS_NOAA21_NRT", 2024),
}


def today() -> date:
    return date.today()


def map_key() -> str | None:
    """FIRMS MAP_KEY from the environment or a local .env file."""
    key = os.environ.get("FIRMS_MAP_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "FIRMS_MAP_KEY":
                key = v.strip().strip('"').strip("'")
    return key or None
