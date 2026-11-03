"""Structured logging: console + JSONL file + mirror into the DuckDB ingest_log table.

Use `log_event(event, level=..., **fields)` for anything important. Recognised
fields (unit, http_status, bytes, rows, duration_ms, transactions_used,
transactions_remaining, llm_tokens_in, llm_tokens_out) become columns in
ingest_log; everything else is kept in the JSONL record and the `extra` column.
"""
import json
import queue
import logging
import re
import sys
import threading
import traceback
import uuid
from datetime import datetime, timezone

from . import config

LOGGER = logging.getLogger("firms")
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]

DB_COLUMNS = ("unit", "http_status", "bytes", "rows", "duration_ms",
              "transactions_used", "transactions_remaining", "llm_tokens_in", "llm_tokens_out")

_secrets: list[str] = []


def register_secret(value: str | None) -> None:
    if value:
        _secrets.append(value)


def redact(text: str) -> str:
    for s in _secrets:
        text = text.replace(s, "***MAP_KEY***")
    # belt and braces: FIRMS keys are 32 hex chars in the URL path / query
    return re.sub(r"(MAP_KEY=|/csv/|/json/)[0-9a-fA-F]{32}", r"\1***MAP_KEY***", text)


class _JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "run_id": RUN_ID,
            "level": record.levelname,
            "event": getattr(record, "event", "message"),
            "message": redact(record.getMessage()),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["traceback"] = redact("".join(traceback.format_exception(*record.exc_info)))
        return json.dumps(payload, default=str)


class _ConsoleFormatter(logging.Formatter):
    def format(self, record):
        fields = getattr(record, "fields", {})
        tail = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
        base = f"{datetime.now():%H:%M:%S} {record.levelname:<7} {getattr(record, 'event', '-'):<22} {redact(record.getMessage())}"
        out = f"{base} {tail}".rstrip()
        if record.exc_info:
            out += "\n" + redact("".join(traceback.format_exception(*record.exc_info)))
        return out


class _DuckDBHandler(logging.Handler):
    """Mirrors records into ingest_log. Attached once the DB connection exists.

    Only the main thread touches DuckDB. Records from other threads (e.g. parallel
    downloads) are queued and written by the main thread at its next log call or at
    flush(); writing from worker threads, even through separate cursors, deadlocked
    against the main thread's load transactions."""

    def __init__(self, con):
        super().__init__(logging.INFO)
        self.con = con
        self.pending = queue.SimpleQueue()

    def _write(self, record):
        fields = dict(getattr(record, "fields", {}))
        cols = {c: fields.pop(c, None) for c in DB_COLUMNS}
        msg = redact(record.getMessage())
        if record.exc_info:
            msg += "\n" + redact("".join(traceback.format_exception(*record.exc_info)))
        self.con.execute(
            f"INSERT INTO ingest_log (ts, run_id, level, event, message, extra, {', '.join(DB_COLUMNS)}) "
            f"VALUES (?, ?, ?, ?, ?, ?, {', '.join('?' * len(DB_COLUMNS))})",
            [datetime.fromtimestamp(record.created, timezone.utc).replace(tzinfo=None), RUN_ID,
             record.levelname, getattr(record, "event", "message"), msg,
             json.dumps(fields, default=str) if fields else None, *cols.values()],
        )

    def emit(self, record):
        if threading.current_thread() is not threading.main_thread():
            self.pending.put(record)
            return
        try:
            self.flush()
            self._write(record)
        except Exception:  # never let logging kill the run
            self.handleError(record)

    def flush(self):
        if threading.current_thread() is not threading.main_thread():
            return
        while True:
            try:
                rec = self.pending.get_nowait()
            except queue.Empty:
                return
            try:
                self._write(rec)
            except Exception:
                self.handleError(rec)


def setup(verbose: bool = False):
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(_ConsoleFormatter())
    path = config.LOG_DIR / f"firms_{RUN_ID}.jsonl"
    fileh = logging.FileHandler(path, encoding="utf-8")
    fileh.setLevel(logging.DEBUG)
    fileh.setFormatter(_JsonFormatter())
    LOGGER.addHandler(console)
    LOGGER.addHandler(fileh)
    return path


def attach_db(con):
    LOGGER.addHandler(_DuckDBHandler(con))


def flush_db():
    """Write queued worker-thread records; call from the main thread (e.g. before closing the DB)."""
    for h in LOGGER.handlers:
        if isinstance(h, _DuckDBHandler):
            h.flush()


def log_event(event: str, message: str = "", level: int = logging.INFO, exc_info=None, **fields):
    LOGGER.log(level, message or event, extra={"event": event, "fields": fields}, exc_info=exc_info)
