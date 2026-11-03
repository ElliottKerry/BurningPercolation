"""Data-quality pass: build the cleaned table + outage calendar, then cross-check against INPE.

    python quality.py                  # clean + INPE cross-check
    python quality.py --skip-inpe      # rebuild cleaned table only

Outputs: fire_detections_clean, sensor_days, inpe_focos_ref, crosscheck_inpe, incomplete_days,
and the views modis_aqua_day_clean / modis_aqua_day_calendar.
Run after backfill.py and assign_areas.py.
"""
import argparse
import json
import logging
import sys

from firms import areas, clean, config, crosscheck, db, log
from firms.client import Client
from firms.log import RUN_ID, log_event


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-inpe", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    log_path = log.setup(args.verbose)
    con = db.connect()
    log.attach_db(con)
    con.execute("INSERT INTO runs (run_id, args, started_at, status) VALUES (?, ?, ?, 'running')",
                [RUN_ID, json.dumps({"command": "quality", **vars(args)}), db.utcnow()])
    log_event("run_start", f"quality run {RUN_ID}; log file {log_path}", args=vars(args))
    client, status, kept = Client(None), "ok", 0
    try:
        areas.ensure_schema(con)  # loads spatial; fire_detections_mun is needed below
        clean.build_sensor_days(con)
        _, kept = clean.build_clean(con)
        if not args.skip_inpe:
            crosscheck.load_inpe(con, client, config.today().year)
            crosscheck.compare(con)
        clean.build_aqua_views(con)
    except KeyboardInterrupt:
        status = "interrupted"
        log_event("run_interrupted", "interrupted", level=logging.WARNING)
    except Exception as e:
        status = "failed"
        log_event("run_failed", repr(e), level=logging.ERROR, exc_info=True)
    finally:
        log_event("run_end", f"status={status}", status=status, rows=kept, bytes=client.bytes_downloaded,
                  transactions_used=0, llm_tokens_in=0, llm_tokens_out=0)
        con.execute("""UPDATE runs SET finished_at=?, status=?, rows_loaded=?, bytes_downloaded=?,
                       api_calls=0, transactions_used=0, llm_tokens_in=0, llm_tokens_out=0 WHERE run_id=?""",
                    [db.utcnow(), status, kept, client.bytes_downloaded, RUN_ID])
        con.close()
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
