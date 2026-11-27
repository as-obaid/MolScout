"""Helpers every stage worker shares (_readers_common.py): per-stage errors, file locks, stale downstream
files, and the reads merge under real concurrent processes."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

WORKERS = Path(__file__).resolve().parents[2] / "src" / "molscout" / "pipeline" / "workers"
sys.path.insert(0, str(WORKERS))
try:
    common = importlib.import_module("_readers_common")
finally:
    sys.path.remove(str(WORKERS))

READERS = ("molglyph", "molscribe", "molnextr")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


# --- errors.jsonl ------------------------------------------------------------------------------


def test_replace_stage_errors_keeps_other_stages_and_lines_it_cannot_read(tmp_path):
    path = tmp_path / "errors.jsonl"
    path.write_text('{"stage": "segment", "error": "old"}\nnot json\n{"stage": "render", "error": "x"}\n\n["a"]\n')
    common.replace_stage_errors(tmp_path, "segment", [{"stage": "segment", "error": "new"}])
    assert path.read_text().splitlines() == [
        "not json", '{"stage": "render", "error": "x"}', '["a"]', '{"stage": "segment", "error": "new"}']


def test_clear_stage_errors_removes_the_file_when_nothing_is_left(tmp_path):
    common.append_error(tmp_path, "segment", "p", "boom")
    common.clear_stage_errors(tmp_path, "segment")
    assert not (tmp_path / "errors.jsonl").exists()
    common.clear_stage_errors(tmp_path / "new", "segment")
    assert not (tmp_path / "new" / "errors.jsonl").exists()


# --- locks -------------------------------------------------------------------------------------


def refuse(handle, operation):
    raise OSError(37, "No locks available")


def test_file_lock_falls_back_to_lockf_when_flock_is_refused(tmp_path, monkeypatch):
    fcntl = pytest.importorskip("fcntl")
    calls, real = [], fcntl.lockf

    def lockf(handle, operation):
        calls.append(operation)
        real(handle, operation)

    monkeypatch.setattr(fcntl, "flock", refuse)
    monkeypatch.setattr(fcntl, "lockf", lockf)
    with common.file_lock(tmp_path / ".lock"):
        pass
    assert calls == [fcntl.LOCK_EX, fcntl.LOCK_UN]


def test_file_lock_fails_loudly_when_no_lock_works(tmp_path, monkeypatch):
    fcntl = pytest.importorskip("fcntl")
    monkeypatch.setattr(fcntl, "flock", refuse)
    monkeypatch.setattr(fcntl, "lockf", refuse)
    with pytest.raises(common.LockError, match="flock.*lockf"), common.file_lock(tmp_path / ".lock"):
        pass


# --- stale downstream files --------------------------------------------------------------------


def test_remove_outputs_deletes_what_a_rewritten_stage_made_stale(tmp_path):
    for name in ("regions.jsonl", "reads.jsonl", ".reads.lock", "voted.jsonl", "layout.jsonl"):
        (tmp_path / name).write_text("")
    common.remove_outputs(tmp_path, common.AFTER_REGIONS)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["layout.jsonl", "regions.jsonl"]
    common.remove_outputs(tmp_path, common.AFTER_PAGES)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["layout.jsonl"]


# --- three readers at once ---------------------------------------------------------------------

MERGER = """
import sys
sys.path.insert(0, sys.argv[1])
import _readers_common as common
paper_dir, reader, rounds = sys.argv[2], sys.argv[3], int(sys.argv[4])
for n in range(rounds):
    common.merge_reads(paper_dir, reader, [common.read_record(f"p1-r{k}", reader, f"C{n}") for k in range(5)])
    common.append_error(paper_dir, "read_" + reader, "p1", f"round {n}")
    common.clear_stage_errors(paper_dir, "read_" + reader)
common.append_error(paper_dir, "read_" + reader, "p1", "last")
"""


def test_three_readers_merging_at_once_keep_every_readers_reads_and_errors(tmp_path):
    processes = [subprocess.Popen([sys.executable, "-c", MERGER, str(WORKERS), str(tmp_path), reader, "30"])
                 for reader in READERS]
    assert [process.wait(timeout=120) for process in processes] == [0, 0, 0]
    reads = read_jsonl(tmp_path / "reads.jsonl")
    assert sorted({r["reader"] for r in reads}) == sorted(READERS)
    assert len(reads) == 15 and {r["smiles"] for r in reads} == {"C29"}
    errors = read_jsonl(tmp_path / "errors.jsonl")
    assert sorted(e["stage"] for e in errors) == sorted(f"read_{reader}" for reader in READERS)
    assert {e["error"] for e in errors} == {"last"}
