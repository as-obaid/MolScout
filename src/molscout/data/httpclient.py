"""A polite HTTP client: an honest User-Agent, timeouts, per-host pacing and retries with backoff."""

from __future__ import annotations

import http.client
import json
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

USER_AGENT = "MolScout/0.1 (+https://github.com/as-obaid/MolScout)"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_AFTER = 60.0
CHUNK_BYTES = 1 << 16
ERROR_BODY_BYTES = 64 << 10


class NetworkError(OSError):
    """A request failed at the network level (DNS, TLS, timeout, reset) after every retry."""


@dataclass(frozen=True, slots=True)
class Response:
    status: int
    url: str
    headers: Mapping[str, str]
    body: bytes
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> object:
        return json.loads(self.body.decode("utf-8"))


def retry_delay(attempt: int, retry_after: str | None, backoff: float) -> float:
    """Seconds to wait before retrying after failed attempt `attempt` (1-based).

    A server's Retry-After (seconds or an HTTP date) wins, capped at MAX_RETRY_AFTER; otherwise
    the wait doubles from `backoff`.
    """
    if retry_after:
        seconds = _retry_after_seconds(retry_after.strip())
        if seconds is not None and math.isfinite(seconds):
            return min(max(seconds, 0.0), MAX_RETRY_AFTER)
    return backoff * 2 ** (attempt - 1)


class HostPacer:
    """Spaces request starts to the same host at least `interval` seconds apart, across threads."""

    def __init__(
        self,
        intervals: Mapping[str, float],
        default: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._intervals = dict(intervals)
        self._default = default
        self._clock = clock
        self._sleep = sleep
        self._next: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> None:
        interval = self._intervals.get(host, self._default)
        with self._lock:
            now = self._clock()
            start = max(now, self._next.get(host, now))
            self._next[host] = start + interval
        if start > now:
            self._sleep(start - now)


class PoliteClient:
    """GET with retries on 429/5xx and network errors; any other status is returned, not raised."""

    def __init__(
        self,
        *,
        intervals: Mapping[str, float] | None = None,
        default_interval: float = 1.0,
        timeout: float = 30.0,
        attempts: int = 4,
        backoff: float = 1.0,
        user_agent: str = USER_AGENT,
        opener: urllib.request.OpenerDirector | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if attempts < 1:
            raise ValueError("attempts must be at least 1")
        self._pacer = HostPacer(intervals or {}, default_interval, sleep=sleep)
        self._timeout = timeout
        self._attempts = attempts
        self._backoff = backoff
        self._user_agent = user_agent
        self._opener = opener or urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
        self._sleep = sleep

    def get(
        self,
        url: str,
        *,
        accept: str = "application/json",
        max_bytes: int = 32 << 20,
        timeout: float | None = None,
    ) -> Response:
        """Fetch url. A body longer than max_bytes is cut there and marked truncated."""
        host = urllib.parse.urlsplit(url).hostname or ""
        request = urllib.request.Request(url, headers={"User-Agent": self._user_agent, "Accept": accept})
        for attempt in range(1, self._attempts + 1):
            self._pacer.wait(host)
            try:
                response = self._open(request, max_bytes, timeout or self._timeout)
            except (OSError, http.client.HTTPException) as exc:
                if attempt == self._attempts:
                    raise NetworkError(f"{host}: {_describe(exc)}") from exc
                self._sleep(retry_delay(attempt, None, self._backoff))
                continue
            if response.status in RETRY_STATUSES and attempt < self._attempts:
                self._sleep(retry_delay(attempt, response.headers.get("retry-after"), self._backoff))
                continue
            return response
        raise AssertionError("unreachable")

    def _open(self, request: urllib.request.Request, max_bytes: int, timeout: float) -> Response:
        try:
            with self._opener.open(request, timeout=timeout) as raw:
                body, truncated = _read_capped(raw, max_bytes)
                headers = _lower_keys(raw.headers)
                if not truncated:
                    _check_length(body, headers)
                return Response(raw.status, raw.url, headers, body, truncated)
        except urllib.error.HTTPError as err:
            try:
                body = _read_capped(err, ERROR_BODY_BYTES)[0] if err.fp is not None else b""
            finally:
                err.close()
            url = getattr(err, "url", None) or request.full_url
            return Response(err.code, url, _lower_keys(err.headers), body)


def _check_length(body: bytes, headers: Mapping[str, str]) -> None:
    """Raise IncompleteRead if fewer bytes arrived than Content-Length promised.

    http.client's read(n) returns b"" on a connection closed early instead of raising, so a cut
    download would otherwise look complete. Raising here sends it down the retry path.
    """
    declared = headers.get("content-length", "")
    if declared.isdigit() and "content-encoding" not in headers and len(body) < int(declared):
        raise http.client.IncompleteRead(body, int(declared) - len(body))


def _read_capped(raw: object, max_bytes: int) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    size = 0
    while chunk := raw.read(CHUNK_BYTES):  # type: ignore[attr-defined]
        chunks.append(chunk)
        size += len(chunk)
        if size > max_bytes:
            return b"".join(chunks)[:max_bytes], True
    return b"".join(chunks), False


def _lower_keys(headers: object) -> dict[str, str]:
    if headers is None:
        return {}
    return {str(key).lower(): str(value) for key, value in headers.items()}  # type: ignore[attr-defined]


def _retry_after_seconds(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return (when - datetime.now(UTC)).total_seconds()


def _describe(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.URLError) and not isinstance(exc, urllib.error.HTTPError):
        return f"{type(exc.reason).__name__}: {exc.reason}"
    return f"{type(exc).__name__}: {exc}"
