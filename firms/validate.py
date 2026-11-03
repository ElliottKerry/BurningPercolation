"""Post-load checks; every result is logged and returned for printing."""
import logging

from . import config
from .log import log_event

W, S, E, N = config.BRAZIL_BBOX


def run(con):
    results = {}

    per_year = con.execute("""
        SELECT sensor, year(acq_date) AS yr, count(*) AS n, sum(CASE WHEN source LIKE 'archive:%' THEN 1 ELSE 0 END) AS archive_n
        FROM fire_detections GROUP BY ALL ORDER BY sensor, yr""").fetchall()
    results["per_year"] = per_year
    for sensor, yr, n, a in per_year:
        log_event("validate_count", f"{sensor} {yr}: {n:,} rows ({a:,} archive, {n - a:,} api)",
                  sensor=sensor, year=yr, rows=n)

    coverage = con.execute("""
        SELECT sensor, min(acq_date), max(acq_date), count(DISTINCT acq_date),
               date_diff('day', min(acq_date), max(acq_date)) + 1 AS span
        FROM fire_detections GROUP BY sensor ORDER BY sensor""").fetchall()
    results["coverage"] = coverage
    for sensor, dmin, dmax, days, span in coverage:
        log_event("validate_coverage", f"{sensor}: {dmin}..{dmax}, {days:,} days with detections of {span:,}",
                  sensor=sensor, date_from=dmin, date_to=dmax, days_with_data=days, span_days=span)

    # longest runs of calendar days with no detections at all (Brazil almost always has some fire)
    gaps = con.execute("""
        WITH d AS (SELECT DISTINCT sensor, acq_date FROM fire_detections),
        g AS (SELECT sensor, acq_date, lead(acq_date) OVER (PARTITION BY sensor ORDER BY acq_date) AS nxt FROM d)
        SELECT sensor, acq_date AS last_before, nxt AS first_after, date_diff('day', acq_date, nxt) - 1 AS missing
        FROM g WHERE date_diff('day', acq_date, nxt) > 3 ORDER BY missing DESC LIMIT 20""").fetchall()
    results["gaps"] = gaps
    for sensor, a, b, m in gaps:
        log_event("validate_gap", f"{sensor}: no detections for {m} days between {a} and {b}",
                  level=logging.WARNING, sensor=sensor, missing_days=m)

    bad = con.execute(f"""
        SELECT
          count(*) FILTER (WHERE latitude NOT BETWEEN {S} AND {N} OR longitude NOT BETWEEN {W} AND {E}) AS out_of_bbox,
          count(*) FILTER (WHERE acq_ts IS NULL) AS null_ts,
          count(*) FILTER (WHERE frp IS NULL) AS null_frp,
          count(*) FILTER (WHERE frp < 0) AS negative_frp,
          count(*) FILTER (WHERE acq_date > current_date) AS future
        FROM fire_detections""").fetchone()
    names = ("out_of_bbox", "null_ts", "null_frp", "negative_frp", "future")
    results["anomalies"] = dict(zip(names, bad))
    log_event("validate_anomalies", ", ".join(f"{k}={v:,}" for k, v in results["anomalies"].items()),
              level=logging.WARNING if any(bad) else logging.INFO, **results["anomalies"])

    dups = con.execute("""
        SELECT count(*) - count(DISTINCT (sensor, latitude, longitude, acq_date, acq_time, satellite))
        FROM fire_detections""").fetchone()[0]
    results["duplicate_rows"] = dups
    log_event("validate_duplicates", f"{dups:,} rows share (sensor, lat, lon, date, time, satellite)",
              level=logging.WARNING if dups else logging.INFO, duplicates=dups)

    total = con.execute("SELECT count(*) FROM fire_detections").fetchone()[0]
    results["total"] = total
    log_event("validate_total", f"{total:,} detections in database", rows=total)
    return results
