"""Yearly per-country archive CSVs (no MAP_KEY needed)."""
import logging

from . import config, db
from .client import Client, sha256_of
from .log import RUN_ID, log_event


def units(sensors, from_year: int | None, to_year: int):
    for s in sensors:
        if not s.archive_slug:
            continue
        for year in range(max(s.first_year, from_year or 0), to_year + 1):
            url = config.ARCHIVE_URL.format(slug=s.archive_slug, year=year)
            yield s, year, f"archive:{s.name}:{year}", url


def discover(client: Client, sensors, from_year, to_year):
    found = []
    for s, year, unit, url in units(sensors, from_year, to_year):
        status, size = client.head(url, unit)
        ok = status == 200
        log_event("discover", f"{unit} {'available' if ok else 'not published'} ({status})",
                  unit=unit, http_status=status, bytes=size if ok else None)
        if ok:
            found.append((s, year, unit, url, size))
    return found


def run(con, client: Client, sensors, from_year, to_year, stats, dry_run=False):
    found = discover(client, sensors, from_year, to_year)
    total = sum(f[4] or 0 for f in found)
    log_event("archive_plan", f"{len(found)} archive files, {total / 1e9:.2f} GB",
              files=len(found), bytes=total)
    if dry_run:
        return found

    for s, year, unit, url, size in found:
        raw = config.RAW_DIR / "archive" / url.rsplit("/", 1)[1]
        prev = db.unit_status(con, unit)
        if prev and prev[0] == "done" and raw.exists() and raw.stat().st_size == size and sha256_of(raw) == prev[1]:
            log_event("skip", f"{unit} already loaded ({prev[2]:,} rows)", unit=unit, rows=prev[2])
            stats["units_skipped"] += 1
            continue

        started = db.utcnow()
        try:
            if raw.exists() and raw.stat().st_size == size:
                digest, n = sha256_of(raw), size
                log_event("cache_hit", f"{raw.name} already downloaded", unit=unit, bytes=n)
            else:
                n, digest = client.download(url, raw, unit)
            inserted, replaced = db.load_csv(con, raw, sensor=s.name, source=f"archive:{s.archive_slug}",
                                             load_id=unit)
            dmin, dmax = con.execute("SELECT min(acq_date), max(acq_date) FROM fire_detections WHERE load_id=?",
                                     [unit]).fetchone()
            db.record_unit(con, unit, run_id=RUN_ID, sensor=s.name, kind="archive", url=url, status="done",
                           rows=inserted, bytes=n, sha256=digest, raw_path=str(raw), date_from=dmin,
                           date_to=dmax, started_at=started, error=None)
            log_event("loaded", f"{unit}: {inserted:,} rows ({dmin}..{dmax}), replaced {replaced:,}",
                      unit=unit, rows=inserted, bytes=n, rows_replaced=replaced)
            stats["units_done"] += 1
            stats["rows_loaded"] += inserted
        except Exception as e:
            stats["units_failed"] += 1
            db.record_unit(con, unit, run_id=RUN_ID, sensor=s.name, kind="archive", url=url, status="failed",
                           started_at=started, error=repr(e))
            log_event("unit_failed", f"{unit}: {e!r}", level=logging.ERROR, exc_info=True, unit=unit)
    return found
