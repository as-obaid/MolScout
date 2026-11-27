import io
import urllib.error
from email.message import Message

import pytest

from molscout.data.httpclient import USER_AGENT, HostPacer, NetworkError, PoliteClient, retry_delay


def test_retry_delay_doubles_without_retry_after():
    assert [retry_delay(attempt, None, 1.0) for attempt in (1, 2, 3)] == [1.0, 2.0, 4.0]


def test_retry_delay_honours_retry_after_up_to_a_cap():
    assert retry_delay(1, "7", 1.0) == 7.0
    assert retry_delay(1, "3600", 1.0) == 60.0
    assert retry_delay(1, "Wed, 21 Oct 2015 07:28:00 GMT", 1.0) == 0.0  # a date in the past
    assert retry_delay(2, "soon", 1.0) == 2.0
    assert retry_delay(3, "nan", 1.0) == 4.0


def test_host_pacer_spaces_requests_per_host():
    now = [100.0]
    waits = []

    def sleep(seconds):
        waits.append(round(seconds, 3))

    pacer = HostPacer({"api.openalex.org": 0.15}, 1.0, clock=lambda: now[0], sleep=sleep)
    pacer.wait("api.openalex.org")
    pacer.wait("api.openalex.org")
    pacer.wait("publisher.example")
    pacer.wait("publisher.example")
    assert waits == [0.15, 1.0]


class FakeRaw(io.BytesIO):
    def __init__(self, body, status=200, url="https://x/final", headers=None):
        super().__init__(body)
        self.status = status
        self.url = url
        message = Message()
        for key, value in (headers or {}).items():
            message[key] = value
        self.headers = message


def http_error(code, body=b"", headers=None):
    message = Message()
    for key, value in (headers or {}).items():
        message[key] = value
    return urllib.error.HTTPError("https://x/a", code, "error", message, io.BytesIO(body))


class FakeOpener:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def client(opener, **kwargs):
    return PoliteClient(opener=opener, sleep=lambda _: None, default_interval=0.0, **kwargs)


def test_get_sends_user_agent_and_returns_body():
    opener = FakeOpener(FakeRaw(b'{"a": 1}', headers={"Content-Type": "application/json"}))
    response = client(opener).get("https://data.rcsb.org/rest/v1/core/entry/6V5L")
    assert response.ok and response.json() == {"a": 1}
    assert response.headers["content-type"] == "application/json"
    request, timeout = opener.requests[0]
    assert request.get_header("User-agent") == USER_AGENT
    assert request.get_header("Accept") == "application/json"
    assert timeout == 30.0


def test_get_retries_429_and_5xx_then_succeeds():
    sleeps = []
    opener = FakeOpener(http_error(429, headers={"Retry-After": "2"}), http_error(503), FakeRaw(b"ok"))
    response = PoliteClient(opener=opener, sleep=sleeps.append, default_interval=0.0).get("https://x/a")
    assert response.body == b"ok"
    assert sleeps == [2.0, 2.0]


def test_get_returns_client_errors_without_retrying():
    opener = FakeOpener(http_error(403, b"<html>Just a moment...</html>", {"cf-mitigated": "challenge"}))
    response = client(opener).get("https://x/a")
    assert (response.status, response.ok, response.headers["cf-mitigated"]) == (403, False, "challenge")
    assert response.body.startswith(b"<html>")
    assert len(opener.requests) == 1


def test_get_returns_the_last_5xx_once_attempts_run_out():
    opener = FakeOpener(http_error(503), http_error(503))
    assert client(opener, attempts=2).get("https://x/a").status == 503


def test_get_raises_network_error_after_every_attempt_fails():
    opener = FakeOpener(*(urllib.error.URLError(TimeoutError("timed out")) for _ in range(3)))
    with pytest.raises(NetworkError, match="x: TimeoutError"):
        client(opener, attempts=3).get("https://x/a")
    assert len(opener.requests) == 3


def test_get_retries_a_body_shorter_than_its_content_length():
    short = FakeRaw(b"%PDF-1.4 cut", headers={"Content-Length": "5000"})
    full = FakeRaw(b"%PDF-1.4 whole", headers={"Content-Length": "14"})
    response = client(FakeOpener(short, full)).get("https://x/a.pdf")
    assert response.body == b"%PDF-1.4 whole"


def test_get_gives_up_on_bodies_that_stay_short():
    opener = FakeOpener(*(FakeRaw(b"cut", headers={"Content-Length": "10"}) for _ in range(2)))
    with pytest.raises(NetworkError, match="IncompleteRead"):
        client(opener, attempts=2).get("https://x/a.pdf")


def test_get_cuts_bodies_over_max_bytes():
    response = client(FakeOpener(FakeRaw(b"a" * 100))).get("https://x/a", max_bytes=10)
    assert response.truncated and response.body == b"a" * 10


def test_attempts_must_be_positive():
    with pytest.raises(ValueError, match="attempts"):
        PoliteClient(attempts=0)
