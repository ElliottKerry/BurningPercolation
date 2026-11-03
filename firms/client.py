"""HTTP client: retries, backoff, streaming downloads and FIRMS MAP_KEY quota accounting."""
import hashlib
import logging
import time
from pathlib import Path

import requests

from . import config
from .log import log_event, redact


class QuotaTracker:
    """Tracks FIRMS transactions via /mapserver/mapkey_status (that endpoint itself is free)."""

    def __init__(self, session: requests.Session, key: str):
        self.session, self.key = session, key
        self.limit = None
        self.current = None
        self.total_used = 0
        self.api_calls = 0

    def status(self) -> dict:
        r = self.session.get(f"{config.FIRMS_BASE}/mapserver/mapkey_status/",
                             params={"MAP_KEY": self.key}, timeout=config.HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        self.limit = int(data.get("transaction_limit") or 5000)
        self.current = int(data.get("current_transactions") or 0)
        return data

    def remaining(self):
        return None if self.limit is None else self.limit - self.current

    def wait_for_headroom(self, needed: int = 1):
        self.status()
        while self.current + needed > self.limit - config.QUOTA_SAFETY_MARGIN:
            log_event("quota_wait", f"quota near limit ({self.current}/{self.limit}); sleeping 60s",
                      level=logging.WARNING, transactions_remaining=self.remaining())
            time.sleep(60)
            self.status()

    def account(self, unit: str, before: int):
        """Re-read the counter after a call and log what that call cost."""
        self.status()
        # the 10-minute window may have rolled over between readings; never report negative cost
        used = max(self.current - before, 0) if self.current >= before else self.current
        self.total_used += used
        self.api_calls += 1
        log_event("quota", f"{unit}: {used} transaction(s); {self.current}/{self.limit} in window",
                  unit=unit, transactions_used=used, transactions_remaining=self.remaining(),
                  total_transactions=self.total_used)
        return used


class Client:
    def __init__(self, key: str | None):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "firms-brazil-backfill/1.0"
        self.key = key
        self.quota = QuotaTracker(self.session, key) if key else None
        self.bytes_downloaded = 0

    def _request(self, method: str, url: str, unit: str, **kw) -> requests.Response:
        for attempt in range(1, config.MAX_RETRIES + 1):
            t0 = time.perf_counter()
            try:
                r = self.session.request(method, url, timeout=config.HTTP_TIMEOUT, **kw)
            except requests.RequestException as e:
                status, err = None, repr(e)
            else:
                status, err = r.status_code, None
                if status < 500 and status != 429:
                    log_event("http", f"{method} {redact(url)}", level=logging.DEBUG, unit=unit,
                              http_status=status, duration_ms=round((time.perf_counter() - t0) * 1000, 1),
                              attempt=attempt)
                    return r
                r.close()
            delay = min(2 ** attempt * 5, 300)
            log_event("http_retry", f"{method} {redact(url)} failed ({status or err}); retry {attempt}/"
                      f"{config.MAX_RETRIES} in {delay}s", level=logging.WARNING, unit=unit,
                      http_status=status, attempt=attempt)
            time.sleep(delay)
        raise RuntimeError(f"giving up on {redact(url)} after {config.MAX_RETRIES} attempts")

    def head(self, url: str, unit: str):
        r = self._request("HEAD", url, unit, allow_redirects=True)
        size = r.headers.get("Content-Length")
        return r.status_code, int(size) if size else None

    def download(self, url: str, dest: Path, unit: str) -> tuple[int, str]:
        """Stream url to dest (atomically). Returns (bytes, sha256)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        t0 = time.perf_counter()
        r = self._request("GET", url, unit, stream=True)
        r.raise_for_status()
        h, n = hashlib.sha256(), 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                h.update(chunk)
                n += len(chunk)
        expected = r.headers.get("Content-Length")
        if expected and int(expected) != n:
            tmp.unlink(missing_ok=True)
            raise IOError(f"truncated download: got {n} of {expected} bytes")
        tmp.replace(dest)
        self.bytes_downloaded += n
        secs = time.perf_counter() - t0
        log_event("download", f"{dest.name} {n / 1e6:.1f} MB in {secs:.1f}s", unit=unit, http_status=r.status_code,
                  bytes=n, duration_ms=round(secs * 1000, 1), sha256=h.hexdigest())
        return n, h.hexdigest()

    def api_get(self, url: str, unit: str) -> requests.Response:
        """GET a MAP_KEY-billed FIRMS API url with quota wait + per-call transaction accounting."""
        self.quota.wait_for_headroom()
        before = self.quota.current
        r = self._request("GET", url, unit)
        self.quota.account(unit, before)
        return r


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
