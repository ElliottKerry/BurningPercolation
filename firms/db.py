"""DuckDB schema and loading.

Idempotency is per load unit (one archive file or one API window): reloading a
unit deletes its previous rows first. Units never overlap in (sensor, date), so
no row-level primary key is needed on the ~50M-row detections table.
"""
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS fire_detections (
    sensor       VARCHAR NOT NULL,
    source       VARCHAR NOT NULL,   -- e.g. archive:modis, VIIRS_NOAA20_SP, MODIS_NRT
    latitude     DOUBLE  NOT NULL,
    longitude    DOUBLE  NOT NULL,
    acq_date     DATE    NOT NULL,
    acq_time     VARCHAR NOT NULL,   -- HHMM UTC as published
    acq_ts       TIMESTAMP,          -- acq_date + acq_time, UTC
    satellite    VARCHAR,
    instrument   VARCHAR,
    confidence   VARCHAR,            -- MODIS 0-100, VIIRS l/n/h
    version      VARCHAR,
    brightness   DOUBLE,             -- MODIS channel 21/22 (K)
    bright_t31   DOUBLE,             -- MODIS channel 31 (K)
    bright_ti4   DOUBLE,             -- VIIRS I-4 (K)
    bright_ti5   DOUBLE,             -- VIIRS I-5 (K)
    frp          DOUBLE,             -- fire radiative power (MW)
    scan         DOUBLE,
    track        DOUBLE,
    daynight     VARCHAR,
    type         INTEGER,            -- 0 veg fire, 1 volcano, 2 static land, 3 offshore (archive only)
    load_id      VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS load_units (
    unit_id      VARCHAR PRIMARY KEY,  -- archive:modis:2012 / api:VIIRS_NOAA20_SP:2019-03-01
    sensor       VARCHAR,
    kind         VARCHAR,              -- archive | api
    url          VARCHAR,
    status       VARCHAR,              -- done | failed | unavailable | empty
    rows         BIGINT,
    bytes        BIGINT,
    sha256       VARCHAR,
    raw_path     VARCHAR,
    date_from    DATE,
    date_to      DATE,
    started_at   TIMESTAMP,
    finished_at  TIMESTAMP,
    run_id       VARCHAR,
    error        VARCHAR
);
CREATE TABLE IF NOT EXISTS ingest_log (
    ts                      TIMESTAMP,
    run_id                  VARCHAR,
    level                   VARCHAR,
    event                   VARCHAR,
    unit                    VARCHAR,
    http_status             INTEGER,
    bytes                   BIGINT,
    rows                    BIGINT,
    duration_ms             DOUBLE,
    transactions_used       INTEGER,
    transactions_remaining  INTEGER,
    llm_tokens_in           BIGINT,
    llm_tokens_out          BIGINT,
    message                 VARCHAR,
    extra                   VARCHAR
);
CREATE TABLE IF NOT EXISTS runs (
    run_id              VARCHAR PRIMARY KEY,
    args                VARCHAR,
    started_at          TIMESTAMP,
    finished_at         TIMESTAMP,
    status              VARCHAR,
    units_done          INTEGER,
    units_skipped       INTEGER,
    units_failed        INTEGER,
    rows_loaded         BIGINT,
    bytes_downloaded    BIGINT,
    api_calls           INTEGER,
    transactions_used   INTEGER,
    llm_tokens_in       BIGINT,
    llm_tokens_out      BIGINT
);
"""

# Canonical columns and how to derive each from a raw CSV (all read as VARCHAR).
_NUM = "TRY_CAST(NULLIF(TRIM({c}), '') AS DOUBLE)"
_COLUMN_EXPR = {
    "latitude": _NUM, "longitude": _NUM, "brightness": _NUM, "bright_t31": _NUM,
    "bright_ti4": _NUM, "bright_ti5": _NUM, "frp": _NUM, "scan": _NUM, "track": _NUM,
    "satellite": "{c}", "instrument": "{c}", "confidence": "{c}", "version": "{c}", "daynight": "{c}",
    "type": "TRY_CAST({c} AS INTEGER)",
}


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def connect(path: Path = config.DB_PATH, read_only: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        con.execute(SCHEMA)
    return con


def unit_status(con, unit_id: str):
    row = con.execute("SELECT status, sha256, rows FROM load_units WHERE unit_id = ?", [unit_id]).fetchone()
    return row  # (status, sha256, rows) or None


def record_unit(con, unit_id, *, run_id, **fields):
    fields.setdefault("finished_at", utcnow())
    cols = ["unit_id", "run_id", *fields]
    con.execute(
        f"INSERT OR REPLACE INTO load_units ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
        [unit_id, run_id, *fields.values()],
    )


def load_csv(con, csv_path: Path, *, sensor: str, source: str, load_id: str,
             also_delete: tuple[str, list] | None = None) -> tuple[int, int]:
    """Replace all rows of `load_id` (plus any rows matching `also_delete`, a WHERE clause and params)
    with the contents of csv_path, in one transaction. Returns (rows_inserted, rows_replaced)."""
    rel = f"read_csv('{csv_path.as_posix()}', header=true, all_varchar=true, auto_detect=true)"
    present = {r[0].lower() for r in con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()}
    missing = {"latitude", "longitude", "acq_date", "acq_time"} - present
    if missing:
        raise ValueError(f"{csv_path.name}: missing required columns {sorted(missing)}; got {sorted(present)}")

    selects = [
        f"'{sensor}' AS sensor", f"'{source}' AS source",
        "TRY_CAST(acq_date AS DATE) AS acq_date",
        "LPAD(TRIM(acq_time), 4, '0') AS acq_time",
        "TRY_CAST(acq_date AS DATE) + TO_MINUTES(TRY_CAST(SUBSTR(LPAD(TRIM(acq_time),4,'0'),1,2) AS INTEGER) * 60"
        " + TRY_CAST(SUBSTR(LPAD(TRIM(acq_time),4,'0'),3,2) AS INTEGER)) AS acq_ts",
    ]
    for col, expr in _COLUMN_EXPR.items():
        selects.append(f"{expr.format(c=col) if col in present else 'NULL'} AS {col}")
    selects.append(f"'{load_id}' AS load_id")

    con.execute("BEGIN")
    try:
        replaced = con.execute("DELETE FROM fire_detections WHERE load_id = ?", [load_id]).fetchone()[0]
        if also_delete:
            replaced += con.execute(f"DELETE FROM fire_detections WHERE {also_delete[0]}", also_delete[1]).fetchone()[0]
        cols = ["sensor", "source", "acq_date", "acq_time", "acq_ts", *_COLUMN_EXPR, "load_id"]
        inserted = con.execute(
            f"INSERT INTO fire_detections ({', '.join(cols)}) SELECT {', '.join(selects)} FROM {rel} "
            "WHERE TRY_CAST(latitude AS DOUBLE) IS NOT NULL"
        ).fetchone()[0]
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return inserted, replaced


def max_date(con, sensor: str):
    return con.execute("SELECT max(acq_date) FROM fire_detections WHERE sensor = ?", [sensor]).fetchone()[0]


def safe_rollback(con):
    """Roll back if a transaction is open; DuckDB's Python API has no in_transaction flag,
    and ROLLBACK with no open transaction raises, which must never mask the original error."""
    try:
        con.execute("ROLLBACK")
    except duckdb.Error:
        pass
