"""The shared whole-PDF loop that every complete system's run.py calls."""

import csv
import json
import os
import signal
import socket
import sys
from pathlib import Path

import pytest

from molscout.predictions import read_predictions

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "tools"))
import paper_runner  # noqa: E402
from paper_runner import Molecule  # noqa: E402

ANSWERS = {
    "p1": [Molecule("CCO", 1, (10.0, 20.0, 110.0, 120.5), 0.9), Molecule("c1ccccc1", 2)],
    "p2": [],
    "p3": [Molecule("C")],
}


def make_papers(tmp_path, ids=("p1", "p2", "p3")):
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir(exist_ok=True)
    listing = tmp_path / "papers.csv"
    with listing.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["paper_id", "pdf"])
        for paper in ids:
            (pdfs / f"{paper}.pdf").write_bytes(b"%PDF-1.4\n")
            writer.writerow([paper, str(pdfs / f"{paper}.pdf")])
    return listing


def parse(listing, output, resume=None):
    argv = ["--papers", str(listing), "--dataset", "biovista", "--tool", "fake 1.0", "--output", str(output)]
    return paper_runner.base_parser("test").parse_args(argv + (["--resume", str(resume)] if resume else []))


def fake(answers=ANSWERS, fail=(), calls=None):
    def predict(paper):
        if calls is not None:
            calls.append(paper.paper_id)
        if paper.paper_id in fail:
            raise RuntimeError(f"boom {paper.paper_id}")
        return answers[paper.paper_id]

    return predict


def log_lines(checkpoint):
    return [json.loads(line) for line in paper_runner.papers_log_path(checkpoint).read_text().splitlines()]


def done_lines(checkpoint):
    return [entry for entry in log_lines(checkpoint) if not entry.get("attempted")]


def rows_of(path):
    return [(r.item_id, r.smiles, r.page, r.bbox, r.confidence) for r in read_predictions(path)]


def test_writes_rows_with_page_and_bbox_that_read_predictions_accepts(tmp_path):
    out = tmp_path / "predictions.csv"
    assert paper_runner.run_papers(fake(), parse(make_papers(tmp_path), out), warmup=False) == 0
    assert rows_of(out) == [
        ("p1", "CCO", 1, (10.0, 20.0, 110.0, 120.5), 0.9),
        ("p1", "c1ccccc1", 2, None, None),
        ("p3", "C", None, None, None),
    ]
    assert len({r.seconds for r in read_predictions(out) if r.item_id == "p1"}) == 1


def test_paper_with_no_molecules_has_timing_and_no_rows(tmp_path):
    out = tmp_path / "predictions.csv"
    paper_runner.run_papers(fake(), parse(make_papers(tmp_path), out), warmup=False)
    timing = json.loads(paper_runner.timing_path(out).read_text())
    assert timing["papers"] == 3 and set(timing["seconds"]) == {"p1", "p2", "p3"}
    assert all(seconds >= 0 for seconds in timing["seconds"].values())
    assert "p2" not in {row[0] for row in rows_of(out)}
    assert json.loads(paper_runner.errors_path(out).read_text()) == {"papers": 3, "failed": 0, "errors": {}}


def test_failed_paper_gets_no_rows_an_error_and_the_run_goes_on(tmp_path):
    out = tmp_path / "predictions.csv"
    assert paper_runner.run_papers(fake(fail={"p1"}), parse(make_papers(tmp_path), out), warmup=False) == 1
    errors = json.loads(paper_runner.errors_path(out).read_text())
    assert errors == {"papers": 3, "failed": 1, "errors": {"p1": "RuntimeError: boom p1"}}
    assert {row[0] for row in rows_of(out)} == {"p3"}
    assert "p1" in json.loads(paper_runner.timing_path(out).read_text())["seconds"]


@pytest.mark.parametrize(
    "bad",
    [Molecule("C", 0), Molecule("C", 1, (5.0, 0.0, 1.0, 2.0)), Molecule("C", 1, (0.0, 0.0, float("nan"), 2.0))],
)
def test_bad_page_or_bbox_fails_that_paper_only(tmp_path, bad):
    out = tmp_path / "predictions.csv"
    answers = {**ANSWERS, "p1": [Molecule("CCO", 1), bad]}
    assert paper_runner.run_papers(fake(answers), parse(make_papers(tmp_path), out), warmup=False) == 1
    errors = json.loads(paper_runner.errors_path(out).read_text())["errors"]
    assert list(errors) == ["p1"] and errors["p1"].startswith("ValueError")
    assert "p1" not in {row[0] for row in rows_of(out)}


def test_tuples_and_none_smiles_are_accepted(tmp_path):
    out = tmp_path / "predictions.csv"
    answers = {"p1": [("CCO", 3, None, None), (None, None, None, 0.5)], "p2": [], "p3": []}
    paper_runner.run_papers(fake(answers), parse(make_papers(tmp_path), out), warmup=False)
    assert rows_of(out) == [("p1", "CCO", 3, None, None), ("p1", "", None, None, 0.5)]


def test_warmup_runs_the_first_paper_once_more_untimed(tmp_path):
    calls = []
    paper_runner.run_papers(fake(calls=calls), parse(make_papers(tmp_path), tmp_path / "o.csv"))
    assert calls == ["p1", "p1", "p2", "p3"]


def test_consecutive_failures_stop_the_run_and_write_nothing(tmp_path):
    ids = [f"p{i}" for i in range(paper_runner.MAX_CONSECUTIVE_FAILURES)]
    out = tmp_path / "o.csv"
    with pytest.raises(RuntimeError, match="in a row"):
        paper_runner.run_papers(fake({}, fail=set(ids)), parse(make_papers(tmp_path, ids), out), warmup=False)
    for path in (out, paper_runner.errors_path(out), paper_runner.timing_path(out), Path(f"{out}.part")):
        assert not path.exists()


def test_read_papers_rejects_bad_lists(tmp_path):
    listing = make_papers(tmp_path)
    assert [p.paper_id for p in paper_runner.read_papers(listing)] == ["p1", "p2", "p3"]
    listing.write_text(listing.read_text() + f"p1,{tmp_path / 'pdfs' / 'p1.pdf'}\n")
    with pytest.raises(ValueError, match="p1"):
        paper_runner.read_papers(listing)
    listing.write_text(f"paper_id,pdf\np9,{tmp_path / 'missing.pdf'}\n")
    with pytest.raises(ValueError, match="missing.pdf"):
        paper_runner.read_papers(listing)
    listing.write_text("id,path\n")
    with pytest.raises(ValueError, match="paper_id,pdf"):
        paper_runner.read_papers(listing)


def test_resume_carries_on_after_a_stop(tmp_path):
    listing, checkpoint = make_papers(tmp_path), tmp_path / "checkpoint.csv"

    def stopping(paper):
        if paper.paper_id == "p2":
            raise KeyboardInterrupt
        return ANSWERS[paper.paper_id]

    with pytest.raises(KeyboardInterrupt):
        paper_runner.run_papers(stopping, parse(listing, tmp_path / "a.csv", checkpoint), warmup=False)
    assert not (tmp_path / "a.csv").exists()
    calls = []
    paper_runner.run_papers(fake(calls=calls), parse(listing, tmp_path / "b.csv", checkpoint), warmup=False)
    assert calls == ["p2", "p3"]
    paper_runner.run_papers(fake(), parse(listing, tmp_path / "c.csv"), warmup=False)
    assert rows_of(tmp_path / "b.csv") == rows_of(tmp_path / "c.csv")
    assert set(json.loads(paper_runner.timing_path(tmp_path / "b.csv").read_text())["seconds"]) == {"p1", "p2", "p3"}


def test_resume_drops_rows_of_unfinished_paper(tmp_path):
    listing, checkpoint = make_papers(tmp_path), tmp_path / "checkpoint.csv"
    paper_runner.run_papers(fake(), parse(listing, tmp_path / "a.csv", checkpoint), warmup=False)
    log = paper_runner.papers_log_path(checkpoint)
    lines = log.read_text().splitlines(keepends=True)
    lines = [line for line in lines if '"attempted"' not in line]
    log.write_text(lines[0] + lines[1][: len(lines[1]) // 2])  # p1 done; p2's line cut short; p3 has rows, no line
    calls = []
    paper_runner.run_papers(fake(calls=calls), parse(listing, tmp_path / "b.csv", checkpoint), warmup=False)
    assert calls == ["p2", "p3"]
    assert [row[0] for row in rows_of(tmp_path / "b.csv")] == ["p1", "p1", "p3"]


def test_resume_rejects_a_checkpoint_for_other_papers(tmp_path):
    checkpoint = tmp_path / "checkpoint.csv"
    paper_runner.run_papers(fake(), parse(make_papers(tmp_path), tmp_path / "a.csv", checkpoint), warmup=False)
    with pytest.raises(ValueError, match="p1"):
        paper_runner.run_papers(fake(), parse(make_papers(tmp_path, ["p2", "p3"]), tmp_path / "b.csv", checkpoint))


def test_sigterm_exits_143_and_keeps_finished_papers(tmp_path):
    listing, checkpoint = make_papers(tmp_path), tmp_path / "checkpoint.csv"

    def terminated(paper):
        if paper.paper_id == "p2":
            os.kill(os.getpid(), signal.SIGTERM)
        return ANSWERS[paper.paper_id]

    with pytest.raises(SystemExit) as stop:
        paper_runner.run_papers(terminated, parse(listing, tmp_path / "a.csv", checkpoint), warmup=False)
    assert stop.value.code == 143
    done = [entry["item_id"] for entry in done_lines(checkpoint)]
    assert done == ["p1"]


def test_resume_holds_back_a_stopping_streak_and_runs_those_papers_again(tmp_path):
    ids = [f"p{i}" for i in range(paper_runner.MAX_CONSECUTIVE_FAILURES + 1)]
    listing, checkpoint = make_papers(tmp_path, ids), tmp_path / "checkpoint.csv"
    answers = {"p0": [Molecule("C")], **{p: [Molecule("N")] for p in ids[1:]}}
    with pytest.raises(RuntimeError, match="in a row"):
        paper_runner.run_papers(fake(answers, fail=set(ids[1:])), parse(listing, tmp_path / "a.csv", checkpoint), warmup=False)
    done = [entry["item_id"] for entry in done_lines(checkpoint)]
    assert done == ["p0"]
    calls = []
    paper_runner.run_papers(fake(answers, calls=calls), parse(listing, tmp_path / "b.csv", checkpoint), warmup=False)
    assert calls == ids[1:]


def test_resume_logs_failures_once_a_paper_succeeds_and_does_not_rerun_them(tmp_path):
    listing, checkpoint = make_papers(tmp_path), tmp_path / "checkpoint.csv"
    out = tmp_path / "a.csv"
    assert paper_runner.run_papers(fake(fail={"p1", "p2"}), parse(listing, out, checkpoint), warmup=False) == 2
    entries = done_lines(checkpoint)
    assert [(e["item_id"], e["rows"], e["error"]) for e in entries] == [
        ("p1", 0, "RuntimeError: boom p1"),
        ("p2", 0, "RuntimeError: boom p2"),
        ("p3", 1, None),
    ]
    calls = []
    paper_runner.run_papers(fake(fail={"p1", "p2"}, calls=calls), parse(listing, tmp_path / "b.csv", checkpoint))
    assert calls == []
    assert json.loads(paper_runner.errors_path(tmp_path / "b.csv").read_text())["failed"] == 2


def streak_setup(tmp_path):
    ids = [f"p{i}" for i in range(paper_runner.MAX_CONSECUTIVE_FAILURES + 2)]
    answers = {p: [Molecule("N")] for p in ids}
    return ids, make_papers(tmp_path, ids), tmp_path / "checkpoint.csv", answers


def test_a_deterministic_streak_is_recorded_as_failed_on_resume(tmp_path):
    ids, listing, checkpoint, answers = streak_setup(tmp_path)
    streak = set(ids[: paper_runner.MAX_CONSECUTIVE_FAILURES])
    with pytest.raises(RuntimeError, match="in a row"):
        paper_runner.run_papers(fake(answers, fail=streak), parse(listing, tmp_path / "a.csv", checkpoint), warmup=False)
    assert done_lines(checkpoint) == []
    assert [e["item_id"] for e in log_lines(checkpoint)] == ids[: paper_runner.MAX_CONSECUTIVE_FAILURES]
    out = tmp_path / "b.csv"
    assert paper_runner.run_papers(fake(answers, fail=streak), parse(listing, out, checkpoint), warmup=False) == len(streak)
    errors = json.loads(paper_runner.errors_path(out).read_text())
    assert errors["failed"] == len(streak) and set(errors["errors"]) == streak
    assert set(json.loads(paper_runner.timing_path(out).read_text())["seconds"]) == set(ids)
    assert {row[0] for row in rows_of(out)} == set(ids) - streak


def test_a_transient_streak_yields_its_molecules_on_resume(tmp_path):
    ids, listing, checkpoint, answers = streak_setup(tmp_path)
    streak = set(ids[: paper_runner.MAX_CONSECUTIVE_FAILURES])
    with pytest.raises(RuntimeError, match="in a row"):
        paper_runner.run_papers(fake(answers, fail=streak), parse(listing, tmp_path / "a.csv", checkpoint), warmup=False)
    out = tmp_path / "b.csv"
    assert paper_runner.run_papers(fake(answers), parse(listing, out, checkpoint), warmup=False) == 0
    assert json.loads(paper_runner.errors_path(out).read_text())["errors"] == {}
    assert {row[0] for row in rows_of(out)} == set(ids)


def test_a_fresh_streak_on_resume_still_stops_the_run(tmp_path):
    n = paper_runner.MAX_CONSECUTIVE_FAILURES
    ids = [f"p{i}" for i in range(2 * n)]
    listing, checkpoint = make_papers(tmp_path, ids), tmp_path / "checkpoint.csv"
    answers = {p: [Molecule("N")] for p in ids}
    with pytest.raises(RuntimeError, match="in a row"):
        paper_runner.run_papers(fake(answers, fail=set(ids[:n])), parse(listing, tmp_path / "a.csv", checkpoint), warmup=False)
    calls = []
    with pytest.raises(RuntimeError, match="in a row"):  # the first n fail again (final); the next n are first attempts
        paper_runner.run_papers(
            fake(answers, fail=set(ids), calls=calls), parse(listing, tmp_path / "b.csv", checkpoint), warmup=False
        )
    assert calls == ids
    assert [e["item_id"] for e in done_lines(checkpoint)] == ids[:n]


def test_done_lines_name_the_host_and_attempts_are_logged_before_the_paper_runs(tmp_path):
    listing, checkpoint = make_papers(tmp_path), tmp_path / "checkpoint.csv"
    seen = []

    def spy(paper):
        seen.append([e["item_id"] for e in log_lines(checkpoint) if e.get("attempted")])
        return ANSWERS[paper.paper_id]

    paper_runner.run_papers(spy, parse(listing, tmp_path / "a.csv", checkpoint), warmup=False)
    assert seen == [["p1"], ["p1", "p2"], ["p1", "p2", "p3"]]
    assert {e["host"] for e in done_lines(checkpoint)} == {socket.gethostname()}


class IndexLike:
    """Stands in for numpy.int64: not an int, but it has __index__."""

    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


def test_integral_page_objects_are_accepted_and_written_as_plain_ints(tmp_path):
    out = tmp_path / "predictions.csv"
    answers = {"p1": [Molecule("CCO", IndexLike(7))], "p2": [], "p3": []}
    assert paper_runner.run_papers(fake(answers), parse(make_papers(tmp_path), out), warmup=False) == 0
    assert rows_of(out) == [("p1", "CCO", 7, None, None)]


@pytest.mark.parametrize("page", [True, 1.0, "2", IndexLike(0)])
def test_bool_float_text_and_small_pages_are_still_rejected(tmp_path, page):
    out = tmp_path / "predictions.csv"
    answers = {"p1": [Molecule("CCO", page)], "p2": [], "p3": []}
    assert paper_runner.run_papers(fake(answers), parse(make_papers(tmp_path), out), warmup=False) == 1
    assert json.loads(paper_runner.errors_path(out).read_text())["errors"]["p1"].startswith("ValueError")
