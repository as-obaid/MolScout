import importlib.util
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BIOMINER = REPO / "benchmarks" / "tools" / "complete_systems" / "biominer"


def load_run():
    spec = importlib.util.spec_from_file_location("biominer_run", BIOMINER / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(BIOMINER))  # run.py imports its sibling stage.py by name
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(BIOMINER))
        sys.modules.pop("stage", None)  # leave no folder-local module name behind for other tests
    return module


run = load_run()


def sleeper(seconds):
    def predict(paper):
        time.sleep(seconds)
        return ["done"]

    return predict


# per-paper limit

def test_paper_that_outlives_the_limit_raises_timeout_and_disarms_the_timer():
    bounded = run.limited(sleeper(5), 0.2, probe=lambda: [])
    start = time.monotonic()
    with pytest.raises(run.PaperTimeout):
        bounded("paper")
    assert time.monotonic() - start < 2
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert signal.getsignal(signal.SIGALRM) == signal.SIG_DFL


def test_timeout_with_servers_still_answering_stays_a_paper_error():
    paper = type("P", (), {"paper_id": "p1"})()
    with pytest.raises(run.PaperTimeout):
        run.limited(sleeper(5), 0.1, probe=lambda: [])(paper)


def test_timeout_with_a_dead_server_stops_the_run_with_a_message():
    paper = type("P", (), {"paper_id": "p1"})()
    with pytest.raises(SystemExit) as info:
        run.limited(sleeper(5), 0.1, probe=lambda: ["mineru"])(paper)
    assert "mineru" in str(info.value) and "p1" in str(info.value)
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_other_errors_do_not_probe_the_servers():
    def boom(paper):
        raise ValueError("x")

    def probe():
        raise AssertionError("probed")

    with pytest.raises(ValueError):
        run.limited(boom, 30, probe=probe)("paper")


def test_unresponsive_servers_names_every_server_when_nothing_listens(monkeypatch):
    monkeypatch.setattr(run, "VLLM_PORT", free_port())
    monkeypatch.setattr(run, "LITSERVE_PORTS", {"mineru": free_port()})
    assert run.unresponsive_servers(timeout=1) == ["vllm", "mineru"]


def test_timeout_is_an_exception_paper_runner_records():
    assert issubclass(run.PaperTimeout, TimeoutError) and issubclass(run.PaperTimeout, Exception)


def test_normal_paper_returns_and_clears_timer_and_restores_handler():
    marker = lambda signum, frame: None  # noqa: E731
    previous = signal.signal(signal.SIGALRM, marker)
    try:
        assert run.limited(sleeper(0), 30)("paper") == ["done"]
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
        assert signal.getsignal(signal.SIGALRM) is marker
    finally:
        signal.signal(signal.SIGALRM, previous)


def test_error_in_paper_also_clears_timer():
    def boom(paper):
        raise ValueError("x")

    with pytest.raises(ValueError):
        run.limited(boom, 30)("paper")
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_limit_is_per_call():
    bounded = run.limited(sleeper(0.1), 0.5)
    assert bounded("a") == ["done"] and bounded("b") == ["done"]


# hang watchdog

class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def test_watch_is_quiet_when_idle_or_within_two_limits():
    clock, calls = Clock(), []
    watch = run.HangWatch(10, lambda: calls.append(1), clock)
    clock.now = 1000
    assert watch.poll() is False  # no paper in progress
    watch.begin()
    clock.now += 20  # exactly 2 x limit: not yet
    assert watch.poll() is False and calls == []


def test_watch_gives_up_past_two_limits():
    clock, calls = Clock(), []
    watch = run.HangWatch(10, lambda: calls.append(1), clock)
    watch.begin()
    clock.now += 20.5
    assert watch.poll() is True and calls == [1]


def test_watch_forgets_a_finished_paper():
    clock, calls = Clock(), []
    watch = run.HangWatch(10, lambda: calls.append(1), clock)
    watch.begin()
    watch.end()
    clock.now += 999
    assert watch.poll() is False and calls == []


def test_paper_limit_marks_the_watch_during_a_paper_only():
    watch = run.HangWatch(10, lambda: None)
    with run.paper_limit(30, watch):
        assert watch.started is not None
    assert watch.started is None


def test_watch_run_loops_until_it_gives_up():
    clock, calls, sleeps = Clock(), [], []
    watch = run.HangWatch(10, lambda: calls.append(1), clock)
    watch.begin()

    def sleep(interval):
        sleeps.append(interval)
        clock.now += 10

    watch.run(interval=7, sleep=sleep)
    assert calls == [1] and sleeps == [7, 7, 7]  # 30 s > 2 x 10 s after the third


def test_start_runs_a_daemon_thread():
    watch = run.HangWatch(10, lambda: None)
    started = []
    watch.run = lambda: started.append(threading.current_thread())
    watch.start()
    deadline = time.monotonic() + 2
    while not started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert started and started[0].daemon


def test_abandon_stops_servers_discards_work_and_exits_distinctively(tmp_path, monkeypatch):
    work = tmp_path / "biominer-x"
    work.mkdir()
    events = []
    servers = type("S", (), {"stop": lambda self: events.append("stop")})()
    monkeypatch.setattr(run, "discard", lambda path: events.append(("discard", path)))
    run.abandon(servers, work, keep=False, exit_fn=lambda code: events.append(("exit", code)))
    assert events == ["stop", ("discard", work), ("exit", run.HANG_EXIT_CODE)]
    assert run.HANG_EXIT_CODE not in (0, 1, 2, 124, 137, 143)


def test_abandon_keeps_work_when_asked_and_exits_even_if_stop_fails(tmp_path, monkeypatch):
    events = []

    class Servers:
        def stop(self):
            raise RuntimeError("stuck")

    monkeypatch.setattr(run, "discard", lambda path: events.append("discard"))
    with pytest.raises(RuntimeError):
        run.abandon(Servers(), tmp_path, keep=True, exit_fn=lambda code: events.append(code))
    assert events == [run.HANG_EXIT_CODE]


# trash and sweep

def wait_gone(path):
    deadline = time.monotonic() + 10
    while path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    return not path.exists()


def test_discard_renames_at_once_and_deletes_in_the_background(tmp_path):
    work = tmp_path / "biominer-abc"
    (work / "sub").mkdir(parents=True)
    (work / "sub" / "f").write_text("x")
    trash = run.discard(work)
    assert trash == tmp_path / "biominer-abc.trash"
    assert not work.exists()
    assert wait_gone(trash)


def test_discard_of_an_existing_trash_deletes_it_in_place(tmp_path):
    trash = tmp_path / "biominer-abc.trash"
    trash.mkdir()
    assert run.discard(trash) == trash
    assert wait_gone(trash)


def test_discard_falls_back_to_in_place_when_rename_fails(tmp_path, monkeypatch):
    work = tmp_path / "biominer-abc"
    work.mkdir()

    def refuse(src, dst):
        raise OSError("no")

    monkeypatch.setattr(run.os, "rename", refuse)
    assert run.discard(work) == work
    assert wait_gone(work)


def test_sweep_removes_only_this_users_biominer_dirs_and_trash(tmp_path):
    (tmp_path / "biominer-old").mkdir()
    (tmp_path / "biominer-older.trash").mkdir()
    (tmp_path / "biominer-keep").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "biominer-file").write_text("x")  # not a folder
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "biominer-link").symlink_to(outside)  # never followed
    swept = run.sweep_stale(tmp_path, keep=tmp_path / "biominer-keep")
    assert sorted(p.name for p in swept) == ["biominer-old.trash", "biominer-older.trash"]
    assert wait_gone(tmp_path / "biominer-old.trash") and wait_gone(tmp_path / "biominer-older.trash")
    assert (tmp_path / "biominer-keep").is_dir() and (tmp_path / "other").is_dir()
    assert outside.is_dir() and (tmp_path / "biominer-file").is_file()


def test_sweep_leaves_another_users_folders(tmp_path):
    (tmp_path / "biominer-theirs").mkdir()
    assert run.sweep_stale(tmp_path, uid=os.getuid() + 1) == []
    assert (tmp_path / "biominer-theirs").is_dir()


def test_sweep_of_a_missing_parent_is_empty(tmp_path):
    assert run.sweep_stale(tmp_path / "nope") == []


# ports

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_busy_port_is_reported_and_free_port_is_not():
    free = free_port()
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        busy = holder.getsockname()[1]
        assert run.port_free(busy) is False
        assert run.busy_ports((busy, free)) == [busy]
    assert run.busy_ports((free,)) == []


def test_time_wait_does_not_count_as_busy():
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        client = socket.create_connection(("127.0.0.1", port))
        conn, _ = server.accept()
        conn.close()  # the closing side keeps the port in TIME_WAIT
        client.close()
    assert run.port_free(port) is True


def test_require_free_ports_exits_with_a_clear_message(monkeypatch):
    monkeypatch.setattr(run, "busy_ports", lambda: [8613])
    with pytest.raises(SystemExit) as info:
        run.require_free_ports()
    assert "8613" in str(info.value) and "in use" in str(info.value)
    monkeypatch.setattr(run, "busy_ports", lambda: [])
    run.require_free_ports()
