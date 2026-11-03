"""Download IBGE municipal and biome boundaries; map every fire detection to its
municipality/state/region and biome.

    python assign_areas.py            # incremental: only locations not yet assigned
    python assign_areas.py --reload   # reload boundaries and reassign everything

Query the result through the `fire_detections_mun` view.
"""
import argparse
import json
import logging
import sys

from firms import areas, db, log
from firms.client import Client
from firms.log import RUN_ID, log_event


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reload", action="store_true", help="reload boundaries and redo all assignments")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    log_path = log.setup(args.verbose)
    con = db.connect()
    log.attach_db(con)
    con.execute("INSERT INTO runs (run_id, args, started_at, status) VALUES (?, ?, ?, 'running')",
                [RUN_ID, json.dumps({"command": "assign_areas", **vars(args)}), db.utcnow()])
    log_event("run_start", f"assign_areas run {RUN_ID}; log file {log_path}", args=vars(args))
    client, status, result = Client(None), "ok", {}
    try:
        areas.ensure_schema(con)
        sha = areas.ensure_files(client)
        areas.load_municipalities(con, sha, force=args.reload)
        areas.build_grid(con, areas.MUNICIPALITIES)
        result = areas.assign(con, areas.MUNICIPALITIES)
        areas.load_biomes(con, areas.ensure_biome_files(client), force=args.reload)
        areas.build_grid(con, areas.BIOMES)
        result["assigned"] += areas.assign(con, areas.BIOMES)["assigned"]
        areas.validate(con)
    except KeyboardInterrupt:
        status = "interrupted"
        log_event("run_interrupted", "interrupted; rerun to resume", level=logging.WARNING)
    except Exception as e:
        status = "failed"
        log_event("run_failed", repr(e), level=logging.ERROR, exc_info=True)
    finally:
        log_event("run_end", f"status={status}", status=status, rows=result.get("assigned"),
                  bytes=client.bytes_downloaded, transactions_used=0, llm_tokens_in=0, llm_tokens_out=0)
        con.execute("""UPDATE runs SET finished_at=?, status=?, rows_loaded=?, bytes_downloaded=?,
                       api_calls=0, transactions_used=0, llm_tokens_in=0, llm_tokens_out=0 WHERE run_id=?""",
                    [db.utcnow(), status, result.get("assigned", 0), client.bytes_downloaded, RUN_ID])
        con.close()
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
