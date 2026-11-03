"""FIRMS country API (MAP_KEY-billed) for everything the yearly archive doesn't cover."""
import csv
import io
import logging
from datetime import date, timedelta

from . import config, db
from .client import Client
from .log import RUN_ID, log_event


class FirmsApiError(RuntimeError):
    pass


def availability(client: Client) -> dict[str, tuple[date, date]]:
    url = f"{config.FIRMS_BASE}/api/data_availability/csv/{client.key}/all"
    r = client.api_get(url, "api:data_availability")
    text = r.text.strip()
    if not text.lower().startswith("data_id"):
        raise FirmsApiError(f"data_availability returned: {text[:200]}")
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        out[row["data_id"]] = (date.fromisoformat(row["min_date"]), date.fromisoformat(row["max_date"]))
    log_event("availability", "; ".join(f"{k} {a}..{b}" for k, (a, b) in sorted(out.items())), sources=len(out))
    return out


def plan_windows(con, sensors, avail, end: date):
    """Yield (sensor, source, window_start, days) covering each sensor from the day after its
    archive coverage (or the source's first day) up to `end`, preferring SP over NRT."""
    for s in sensors:
        archive_max = con.execute(
            "SELECT max(date_to) FROM load_units WHERE sensor=? AND kind='archive' AND status='done'",
            [s.name]).fetchone()[0]
        sp, nrt = avail.get(s.api_sp), avail.get(s.api_nrt)
        starts = [d[0] for d in (sp, nrt) if d]
        if not starts:
            log_event("no_source", f"{s.name}: neither {s.api_sp} nor {s.api_nrt} is available",
                      level=logging.WARNING)
            continue
        start = archive_max + timedelta(days=1) if archive_max else min(starts)
        segments = []
        if sp and start <= sp[1]:
            segments.append((s.api_sp, max(start, sp[0]), min(end, sp[1])))
            start = sp[1] + timedelta(days=1)
        if nrt and start <= end:
            segments.append((s.api_nrt, max(start, nrt[0]), min(end, nrt[1])))
        for source, a, b in segments:
            log_event("api_segment", f"{s.name}: {source} {a}..{b}", sensor=s.name, source=source,
                      date_from=a, date_to=b)
            d = a
            while d <= b:
                days = min(config.API_MAX_DAYS, (b - d).days + 1)
                yield s, source, d, days
                d += timedelta(days=days)


def run(con, client: Client, sensors, stats, dry_run=False):
    avail = availability(client)
    end = config.today()
    windows = list(plan_windows(con, sensors, avail, end))
    log_event("api_plan", f"{len(windows)} API windows (~{len(windows)} transactions at 1/call)",
              windows=len(windows))
    if dry_run:
        return windows

    for s, source, d0, days in windows:
        d1 = d0 + timedelta(days=days - 1)
        unit = f"api:{source}:{d0.isoformat()}:{days}"
        prev = db.unit_status(con, unit)
        # windows touching the last few days are refetched: NRT is still being filled in
        if prev and prev[0] in ("done", "empty") and d1 < end - timedelta(days=2):
            log_event("skip", f"{unit} already loaded ({prev[2]:,} rows)", level=logging.DEBUG, unit=unit)
            stats["units_skipped"] += 1
            continue
        url = f"{config.FIRMS_BASE}/api/country/csv/{client.key}/{source}/{config.COUNTRY}/{days}/{d0.isoformat()}"
        started = db.utcnow()
        try:
            r = client.api_get(url, unit)
            body = r.text
            if r.status_code != 200 or not body.lstrip().lower().startswith(("latitude", "country_id")):
                raise FirmsApiError(f"HTTP {r.status_code}: {body[:200].strip()}")
            raw = config.RAW_DIR / "api" / source / f"{d0.isoformat()}_{days}d.csv"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text(body, encoding="utf-8")
            nbytes = len(body.encode())
            client.bytes_downloaded += nbytes
            # clear any API rows for this sensor/date span from earlier, differently-aligned windows
            inserted, replaced = db.load_csv(
                con, raw, sensor=s.name, source=source, load_id=unit,
                also_delete=("sensor = ? AND acq_date BETWEEN ? AND ? AND load_id LIKE 'api:%'", [s.name, d0, d1]),
            ) if body.count("\n") > 1 else (0, 0)
            status = "done" if inserted else "empty"
            db.record_unit(con, unit, run_id=RUN_ID, sensor=s.name, kind="api", url=url.replace(client.key, "***"),
                           status=status, rows=inserted, bytes=nbytes, sha256=None, raw_path=str(raw),
                           date_from=d0, date_to=d1, started_at=started, error=None)
            log_event("loaded", f"{unit}: {inserted:,} rows, replaced {replaced:,}", unit=unit,
                      rows=inserted, bytes=nbytes, rows_replaced=replaced)
            stats["units_done"] += 1
            stats["rows_loaded"] += inserted
        except Exception as e:
            stats["units_failed"] += 1
            db.record_unit(con, unit, run_id=RUN_ID, sensor=s.name, kind="api", url=url.replace(client.key, "***"),
                           status="failed", date_from=d0, date_to=d1, started_at=started, error=repr(e))
            log_event("unit_failed", f"{unit}: {e!r}", level=logging.ERROR, exc_info=True, unit=unit)
            if isinstance(e, FirmsApiError) and "invalid" in str(e).lower():
                raise  # bad key / bad endpoint: every further call would fail the same way
    return windows
