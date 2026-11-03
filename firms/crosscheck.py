"""Cross-check against INPE BDQueimadas' reference-satellite series (Aqua MODIS afternoon pass).

INPE's reference series is its long-term consistency baseline: one satellite, one overpass.
The FIRMS equivalent is MODIS, satellite 'Aqua', daynight 'D'. Both derive from the same
instrument but through independent processing chains, so they should track closely;
state-years where they diverge point at a data problem on one side.
"""
import logging
import zipfile

from . import config, db
from .client import Client, sha256_of
from .log import RUN_ID, log_event

INPE_URL = "https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/anual/Brasil_sat_ref/focos_br_ref_{year}.zip"
INPE_DIR = config.RAW_DIR / "inpe"
FIRST_YEAR = 2003
RATIO_BAND = (0.8, 1.25)
DAILY_MIN_INPE = 200       # daily check: only days where INPE has at least this many foci
DAILY_MAX_RATIO = 0.5      # ...and FIRMS has less than this share of them -> incomplete day   # FIRMS/INPE outside this band is flagged
MIN_FOR_FLAG = 500         # ...when INPE has at least this many foci in the state-year

SCHEMA = """
CREATE TABLE IF NOT EXISTS inpe_focos_ref (
    id_bdq BIGINT, foco_id VARCHAR, latitude DOUBLE, longitude DOUBLE, data_pas TIMESTAMP,
    pais VARCHAR, estado VARCHAR, municipio VARCHAR, bioma VARCHAR, year INTEGER
);
"""


def _norm(col):
    return f"upper(strip_accents(trim({col})))"


def load_inpe(con, client: Client, to_year: int):
    con.execute(SCHEMA)
    for year in range(FIRST_YEAR, to_year + 1):
        unit, url = f"inpe:ref:{year}", INPE_URL.format(year=year)
        zpath = INPE_DIR / f"focos_br_ref_{year}.zip"
        # the current year's file is still growing: always refresh it
        is_final = year < config.today().year
        prev = db.unit_status(con, unit)
        if is_final and prev and prev[0] == "done" and zpath.exists() and sha256_of(zpath) == prev[1]:
            log_event("skip", f"{unit} already loaded ({prev[2]:,} foci)", unit=unit, rows=prev[2])
            continue
        status, _ = client.head(url, unit)
        if status != 200:
            log_event("discover", f"{unit} not published ({status})", unit=unit, http_status=status)
            continue
        started = db.utcnow()
        try:
            n, digest = client.download(url, zpath, unit)
            with zipfile.ZipFile(zpath) as z:
                name = next(m for m in z.namelist() if m.lower().endswith(".csv"))
                z.extract(name, INPE_DIR)
            csv = (INPE_DIR / name).as_posix()
            con.execute("BEGIN")
            con.execute("DELETE FROM inpe_focos_ref WHERE year = ?", [year])
            rows = con.execute(f"""
                INSERT INTO inpe_focos_ref
                SELECT id_bdq, foco_id, lat, lon, data_pas, pais, estado, municipio, bioma, {year}
                FROM read_csv('{csv}', header=true, types={{'id_bdq': 'BIGINT', 'lat': 'DOUBLE', 'lon': 'DOUBLE',
                                                            'data_pas': 'TIMESTAMP'}})""").fetchone()[0]
            con.execute("COMMIT")
            db.record_unit(con, unit, run_id=RUN_ID, sensor="inpe_aqua_ref", kind="inpe", url=url, status="done",
                           rows=rows, bytes=n, sha256=digest, raw_path=str(zpath), started_at=started, error=None)
            log_event("loaded", f"{unit}: {rows:,} foci", unit=unit, rows=rows, bytes=n)
        except Exception as e:
            db.safe_rollback(con)
            db.record_unit(con, unit, run_id=RUN_ID, sensor="inpe_aqua_ref", kind="inpe", url=url,
                           status="failed", started_at=started, error=repr(e))
            log_event("unit_failed", f"{unit}: {e!r}", level=logging.ERROR, exc_info=True, unit=unit)


def compare(con):
    """Build crosscheck_inpe (state x year) and log national and flagged state-level results."""
    con.execute(f"""
        CREATE OR REPLACE TABLE crosscheck_inpe AS
        WITH inpe AS (
            SELECT year, {_norm('estado')} AS uf_name, count(*) AS inpe_n FROM inpe_focos_ref GROUP BY ALL),
        raw AS (
            SELECT year(acq_date) AS year, {_norm('nm_uf')} AS uf_name, count(*) AS firms_raw_n
            FROM fire_detections_mun
            WHERE sensor = 'modis' AND satellite = 'Aqua' AND daynight = 'D' AND nm_uf IS NOT NULL GROUP BY ALL),
        clean AS (
            SELECT year(acq_date) AS year, {_norm('nm_uf')} AS uf_name, count(*) AS firms_clean_n
            FROM fire_detections_clean
            WHERE sensor = 'modis' AND satellite = 'Aqua' AND daynight = 'D' GROUP BY ALL)
        SELECT year, uf_name, inpe_n, firms_raw_n, firms_clean_n,
               round(firms_raw_n / inpe_n, 3) AS ratio_raw, round(firms_clean_n / inpe_n, 3) AS ratio_clean
        FROM inpe FULL JOIN raw USING (year, uf_name) FULL JOIN clean USING (year, uf_name)
        WHERE year BETWEEN (SELECT min(year) FROM inpe_focos_ref)
                       AND (SELECT max(year(acq_date)) FROM fire_detections WHERE sensor = 'modis')
        ORDER BY year, uf_name""")

    unmatched = con.execute("""SELECT DISTINCT uf_name FROM crosscheck_inpe
                               WHERE inpe_n IS NULL OR firms_raw_n IS NULL""").fetchall()
    if unmatched:
        log_event("crosscheck_state_mismatch", f"state names present on one side only: {unmatched}",
                  level=logging.WARNING)

    national = con.execute("""
        SELECT year, sum(inpe_n), sum(firms_raw_n), sum(firms_clean_n),
               round(sum(firms_raw_n) / sum(inpe_n), 3), round(sum(firms_clean_n) / sum(inpe_n), 3)
        FROM crosscheck_inpe GROUP BY year ORDER BY year""").fetchall()
    for y, i, r, c, rr, rc in national:
        lo, hi = RATIO_BAND
        log_event("crosscheck_year", f"{y}: INPE {i:,} | FIRMS raw {r:,} ({rr}) | clean {c:,} ({rc})",
                  level=logging.INFO if lo <= rr <= hi else logging.WARNING,
                  year=y, inpe=i, firms_raw=r, firms_clean=c, ratio_raw=rr, ratio_clean=rc)

    flagged = con.execute(f"""
        SELECT year, uf_name, inpe_n, firms_raw_n, ratio_raw FROM crosscheck_inpe
        WHERE inpe_n >= {MIN_FOR_FLAG} AND (ratio_raw IS NULL OR ratio_raw NOT BETWEEN {RATIO_BAND[0]} AND {RATIO_BAND[1]})
        ORDER BY abs(ln(coalesce(ratio_raw, 0.001))) DESC""").fetchall()
    total = con.execute(f"SELECT count(*) FROM crosscheck_inpe WHERE inpe_n >= {MIN_FOR_FLAG}").fetchone()[0]
    log_event("crosscheck_flagged", f"{len(flagged)} of {total} state-years (INPE >= {MIN_FOR_FLAG} foci) outside "
              f"FIRMS/INPE ratio {RATIO_BAND}", level=logging.WARNING if flagged else logging.INFO,
              flagged=len(flagged), checked=total)
    for y, uf, i, r, rr in flagged[:25]:
        log_event("crosscheck_state", f"{y} {uf}: INPE {i:,} vs FIRMS {r:,} (ratio {rr})",
                  level=logging.WARNING, year=y, state=uf, inpe=i, firms_raw=r, ratio_raw=rr)

    # Daily completeness: days where the FIRMS Aqua daytime archive is missing most of what INPE saw.
    con.execute(f"""
        CREATE OR REPLACE TABLE incomplete_days AS
        WITH i AS (SELECT data_pas::DATE AS day, count(*) AS inpe_n FROM inpe_focos_ref GROUP BY 1),
             f AS (SELECT acq_date AS day, count(*) AS firms_n FROM fire_detections
                   WHERE sensor = 'modis' AND satellite = 'Aqua' AND daynight = 'D' GROUP BY 1)
        SELECT 'modis' AS sensor, 'Aqua' AS satellite, 'D' AS daynight, day, inpe_n, coalesce(firms_n, 0) AS firms_n,
               round(coalesce(firms_n, 0) / inpe_n, 3) AS ratio
        FROM i LEFT JOIN f USING (day)
        WHERE inpe_n >= {DAILY_MIN_INPE} AND coalesce(firms_n, 0) < {DAILY_MAX_RATIO} * inpe_n
          AND day <= (SELECT max(acq_date) FROM fire_detections WHERE sensor = 'modis')
        ORDER BY day""")
    days = con.execute("SELECT day, inpe_n, firms_n FROM incomplete_days").fetchall()
    missing = sum(i - f for _, i, f in days)
    log_event("crosscheck_incomplete_days", f"{len(days)} days where FIRMS Aqua-day has < {DAILY_MAX_RATIO:.0%} of "
              f"INPE's foci ({missing:,} foci short): " + ", ".join(str(d) for d, _, _ in days),
              level=logging.WARNING if days else logging.INFO, days=len(days), foci_short=missing)
    return national, flagged
