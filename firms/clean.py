"""Cleaned analysis table built from the raw detections (raw rows are never modified).

Rules (RULES_VERSION v1), applied in this order so each exclusion is counted once:
  static    FIRMS type 2/3 (static land source / offshore), or inside a 0.01 deg cell where
            FIRMS has ever flagged a static source (catches API rows, which lack `type`)
  low_conf  MODIS confidence < 30, VIIRS confidence 'l'
  bad_frp   FRP negative
  outside   not inside any IBGE municipality (offshore / across the border)
  duplicate exact repeat of (sensor, satellite, lat, lon, date, time); first kept

`sensor_days` holds, per sensor, satellite and day/night, every calendar day from the first to the
last observation with its raw count: days with zero detections nationwide are almost
certainly outages and should be treated as missing, not as "no fire".
"""
import logging
import time

from .log import log_event

RULES_VERSION = "v1"
STATIC_CELL = 0.01  # degrees (~1.1 km)

_CELL = f"floor(latitude/{STATIC_CELL})::INT AS sy, floor(longitude/{STATIC_CELL})::INT AS sx"


def build_sensor_days(con):
    t0 = time.perf_counter()
    con.execute("""
        CREATE OR REPLACE TABLE sensor_days AS
        WITH counts AS (
            SELECT sensor, satellite, daynight, acq_date AS day, count(*) AS n_raw FROM fire_detections GROUP BY ALL),
        span AS (SELECT sensor, satellite, daynight, min(day) d0, max(day) d1 FROM counts GROUP BY ALL),
        cal AS (SELECT sensor, satellite, daynight, unnest(generate_series(d0, d1, INTERVAL 1 DAY))::DATE AS day FROM span)
        SELECT cal.sensor, cal.satellite, cal.daynight, cal.day, coalesce(n_raw, 0) AS n_raw, n_raw IS NOT NULL AS observed
        FROM cal LEFT JOIN counts USING (sensor, satellite, daynight, day)
        ORDER BY ALL""")
    for sensor, sat, days, missing in con.execute("""
            SELECT sensor, satellite || '/' || daynight, count(*), count(*) FILTER (WHERE NOT observed)
            FROM sensor_days GROUP BY ALL ORDER BY ALL""").fetchall():
        log_event("sensor_days", f"{sensor}/{sat}: {days:,} days, {missing} with no detections (likely outage)",
                  sensor=sensor, satellite=sat, rows=days, missing_days=missing)
    log_event("sensor_days_built", "sensor_days built", duration_ms=round((time.perf_counter() - t0) * 1000))


def build_clean(con):
    t0 = time.perf_counter()
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE static_cells AS
        SELECT DISTINCT {_CELL} FROM fire_detections WHERE type = 2""")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE flagged AS
        WITH r AS (
            SELECT f.*,
              coalesce(f.type IN (2, 3), false) OR s.sy IS NOT NULL                  AS is_static,
              coalesce(CASE WHEN f.sensor = 'modis' THEN TRY_CAST(f.confidence AS INT) < 30
                            ELSE f.confidence = 'l' END, false)                      AS is_low_conf,
              coalesce(f.frp < 0, false)                                             AS is_bad_frp,
              f.cd_mun IS NULL                                                       AS is_outside
            FROM (SELECT *, {_CELL} FROM fire_detections_mun) f
            LEFT JOIN static_cells s USING (sy, sx))
        -- duplicates are ranked only among rows that pass every other rule, in a fixed order,
        -- so exactly one copy of each surviving detection is kept on every run
        SELECT *,
          NOT (is_static OR is_low_conf OR is_bad_frp OR is_outside) AND
          row_number() OVER (PARTITION BY sensor, satellite, latitude, longitude, acq_date, acq_time,
                                          (is_static OR is_low_conf OR is_bad_frp OR is_outside)
                             ORDER BY frp DESC NULLS LAST, confidence DESC NULLS LAST, load_id, source) > 1
                                                                                     AS is_duplicate
        FROM r""")

    total = con.execute("SELECT count(*) FROM flagged").fetchone()[0]
    log_event("clean_start", f"rules {RULES_VERSION}: {total:,} raw detections", rows=total, rules=RULES_VERSION)
    remaining, cond = total, "true"
    for rule in ("is_static", "is_low_conf", "is_bad_frp", "is_outside", "is_duplicate"):
        dropped = con.execute(f"SELECT count(*) FROM flagged WHERE {cond} AND {rule}").fetchone()[0]
        remaining -= dropped
        cond += f" AND NOT {rule}"
        log_event("clean_rule", f"{rule[3:]}: dropped {dropped:,} ({dropped / total:.2%}), {remaining:,} remain",
                  rule=rule[3:], rows=dropped, remaining=remaining)

    con.execute(f"""
        CREATE OR REPLACE TABLE fire_detections_clean AS
        SELECT * EXCLUDE (sy, sx, is_static, is_low_conf, is_bad_frp, is_outside, is_duplicate),
               '{RULES_VERSION}' AS rules_version
        FROM flagged WHERE {cond}""")
    con.execute("DROP TABLE flagged; DROP TABLE static_cells")
    kept = con.execute("SELECT count(*) FROM fire_detections_clean").fetchone()[0]
    assert kept == remaining, (kept, remaining)
    log_event("clean_done", f"fire_detections_clean: {kept:,} rows ({kept / total:.2%} of raw)",
              rows=kept, rules=RULES_VERSION, duration_ms=round((time.perf_counter() - t0) * 1000))
    return total, kept


# The most trustworthy subset: one satellite (Aqua), one overpass (~13:30 local, daytime),
# 2003 onward (Aqua launched mid-2002), validated against INPE's independent series.
AQUA_FILTER = "sensor = 'modis' AND satellite = 'Aqua' AND daynight = 'D' AND year({d}) BETWEEN 2003 AND 2024"


def build_aqua_views(con):
    """modis_aqua_day_clean: cleaned Aqua daytime detections, excluding days the archive is known to
    be missing. modis_aqua_day_calendar: every day in the period with a `usable` flag, so a day
    with no fires can be told apart from a day with no data."""
    has_incomplete = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'incomplete_days'").fetchone()[0]
    excluded = "SELECT \"day\" FROM incomplete_days" if has_incomplete else "SELECT NULL::DATE WHERE false"
    con.execute(f"""
        CREATE OR REPLACE VIEW modis_aqua_day_clean AS
        SELECT * FROM fire_detections_clean
        WHERE {AQUA_FILTER.format(d='acq_date')} AND acq_date NOT IN ({excluded})""")
    con.execute(f"""
        CREATE OR REPLACE VIEW modis_aqua_day_calendar AS
        SELECT sd."day",
               sd.n_raw,
               sd.observed,
               sd."day" IN ({excluded}) AS incomplete,
               sd.observed AND sd."day" NOT IN ({excluded}) AS usable
        FROM sensor_days sd
        WHERE {AQUA_FILTER.format(d='sd."day"')}""")
    n, days, usable = con.execute("""
        SELECT (SELECT count(*) FROM modis_aqua_day_clean), count(*), count(*) FILTER (WHERE usable)
        FROM modis_aqua_day_calendar""").fetchone()
    log_event("aqua_views", f"modis_aqua_day_clean: {n:,} detections; calendar: {usable:,} of {days:,} days usable"
              + ("" if has_incomplete else " (incomplete_days missing: run without --skip-inpe)"),
              level=logging.INFO if has_incomplete else logging.WARNING, rows=n, days=days, usable_days=usable)
    return n, days, usable
