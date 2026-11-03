"""One-off backfill of the full NASA FIRMS active-fire record for Brazil into DuckDB.

    python backfill.py --dry-run                       # discover + estimate, write nothing
    python backfill.py --sensors modis --from-year 2023
    python backfill.py                                 # everything, resumable
"""
import argparse
import json
import logging
import sys

from firms import api, archive, config, db, log, validate
from firms.client import Client
from firms.log import RUN_ID, log_event


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sensors", nargs="+", choices=sorted(config.SENSORS), default=sorted(config.SENSORS))
    p.add_argument("--from-year", type=int, help="skip archive years before this")
    p.add_argument("--dry-run", action="store_true", help="discover files/windows and estimate cost only")
    p.add_argument("--skip-archive", action="store_true")
    p.add_argument("--skip-api", action="store_true")
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true", help="show per-request debug lines on the console")
    return p.parse_args()


def main():
    args = parse_args()
    log_path = log.setup(args.verbose)
    key = config.map_key()
    log.register_secret(key)
    sensors = [config.SENSORS[s] for s in args.sensors]
    stats = dict(units_done=0, units_skipped=0, units_failed=0, rows_loaded=0)

    con = db.connect()
    if not args.dry_run:
        log.attach_db(con)
        con.execute("INSERT INTO runs (run_id, args, started_at, status) VALUES (?, ?, ?, 'running')",
                    [RUN_ID, json.dumps(vars(args)), db.utcnow()])
    log_event("run_start", f"run {RUN_ID}; log file {log_path}", args=vars(args), map_key_present=bool(key))

    client = Client(key)
    status = "ok"
    try:
        if not args.skip_archive:
            archive.run(con, client, sensors, args.from_year, config.today().year, stats, args.dry_run)
        if not args.skip_api:
            if key:
                client.quota.status()
                log_event("quota", f"MAP_KEY window usage at start: {client.quota.current}/{client.quota.limit}",
                          transactions_used=0, transactions_remaining=client.quota.remaining())
                api.run(con, client, sensors, stats, args.dry_run)
            else:
                log_event("api_skipped", "FIRMS_MAP_KEY not set (.env or env var): skipping API gap-fill "
                          "(NOAA-21 and everything after the archive)", level=logging.WARNING)
        if not args.dry_run and not args.no_validate:
            validate.run(con)
    except KeyboardInterrupt:
        status = "interrupted"
        log_event("run_interrupted", "interrupted by user; rerun to resume", level=logging.WARNING)
    except Exception as e:
        status = "failed"
        log_event("run_failed", repr(e), level=logging.ERROR, exc_info=True)
    finally:
        q = client.quota
        totals = dict(stats, bytes_downloaded=client.bytes_downloaded,
                      api_calls=q.api_calls if q else 0, transactions_used=q.total_used if q else 0,
                      llm_tokens_in=0, llm_tokens_out=0)  # the pipeline makes no LLM calls
        if stats["units_failed"] and status == "ok":
            status = "partial"
        log_event("run_end", f"status={status}", status=status, units_done=stats["units_done"],
                  units_skipped=stats["units_skipped"], units_failed=stats["units_failed"],
                  rows=totals["rows_loaded"], bytes=totals["bytes_downloaded"], api_calls=totals["api_calls"],
                  transactions_used=totals["transactions_used"], llm_tokens_in=0, llm_tokens_out=0)
        if not args.dry_run:
            con.execute(f"""UPDATE runs SET finished_at=?, status=?, {', '.join(f'{k}=?' for k in totals)}
                            WHERE run_id=?""", [db.utcnow(), status, *totals.values(), RUN_ID])
        con.close()
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
