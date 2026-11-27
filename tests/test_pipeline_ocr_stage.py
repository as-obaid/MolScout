"""The ocr_labels stage in the runner: where it runs, what it is given, the counts its worker writes, resume and keys."""

import json
from pathlib import Path

import pytest
from pipeline_helpers import FAKE_STEPS, fake_config, make_papers, ran

from molscout.pipeline import config as pc
from molscout.pipeline import stages as st
from molscout.pipeline.runner import _worker_counts, run

REPO = Path(__file__).resolve().parents[1]

READS = ["read_molglyph", "read_molscribe", "read_molnextr"]


def go(tmp_path, config, out="out"):
    messages = []
    summary = run(make_papers(tmp_path), tmp_path / out, config, steps=FAKE_STEPS, echo=messages.append)
    return summary, messages


def test_the_stage_runs_after_segment_when_the_config_names_its_worker(tmp_path):
    assert "ocr_labels" not in fake_config(tmp_path).stage_names()
    config = fake_config(tmp_path, ocr=True)
    assert config.stage_names()[:5] == ["render", "parse", "segment", "ocr_labels", READS[0]]
    assert "ocr_labels" in config.worker_stages() and config.output("ocr_labels") == "ocr.jsonl"
    assert st.is_upstream("ocr_labels")
    assert st.dependencies("ocr_labels", config) == ("render", "parse", "segment")
    assert st.dependencies("link", config) == ("vote", "parse", "ocr_labels")
    assert st.dependencies("link", fake_config(tmp_path)) == ("vote", "parse")


def test_the_packaged_configs_run_label_ocr_on_the_cpu_in_the_mineru_environment(monkeypatch):
    monkeypatch.setenv("MOLSCOUT_STORE", "/store")
    for path in (pc.DEFAULT_CONFIG, REPO / "benchmarks" / "tools" / "complete_systems" / "molscout" / "pipeline.yaml"):
        config = pc.load_pipeline_config(path, device="gpu")
        spec = config.workers["ocr_labels"]
        assert spec.python == Path("/store/envs/mineru/bin/python") and spec.worker == pc.WORKERS_DIR / "ocr_labels.py"
        assert config.worker_args("ocr_labels") == ["--device", "cpu"]
        assert spec.env["MINERU_HOME"] == "/store/models/mineru"


def test_allow_empty_may_name_the_stage(tmp_path):
    assert "ocr_labels" in fake_config(tmp_path, "allow_empty: [ocr_labels]\n").allow_empty


def test_the_runner_records_areas_in_lines_out_and_the_workers_counts(tmp_path):
    summary, _ = go(tmp_path, fake_config(tmp_path, ocr=True))
    out = tmp_path / "out"
    assert summary.ok and ran(out, "ocr_labels") == ["p1", "p2"]
    stage = summary.papers["p1"]["stages"]["ocr_labels"]
    assert stage["records_in"] == 2 and stage["records_out"] == 2  # one area per drawing, one line per area
    assert stage["counts"] == {"in": 2, "out": 2, "dropped_no_text": 0, "dropped_failed": 0, "lines": 2}
    assert summary.papers["p1"]["funnel"]["ocr_lines"] == 2
    assert (out / "papers" / "p1" / "ocr.jsonl").is_file()


def test_counts_that_do_not_add_up_are_an_error(tmp_path):
    summary, _ = go(tmp_path, fake_config(tmp_path, ocr=True, worker_args={"ocr_labels": ["--bad-counts", "p2"]}))
    assert summary.papers["p1"]["status"] == "ok"
    assert summary.papers["p2"]["errors"] == [
        {"stage": "ocr_labels", "error": "counts do not add up: in 2 != out 3 + dropped 0"}
    ]


@pytest.mark.parametrize(("text", "problem"), [
    ("[1, 2]", "are not a JSON object of integers"),
    ('{"in": true}', "are not a JSON object of integers"),
    ("{in: 2", "cannot read the worker's counts"),
])
def test_unreadable_counts_are_a_problem(tmp_path, text, problem):
    assert _worker_counts(tmp_path, "ocr_labels") == (None, None)  # a worker that writes no counts
    (tmp_path / ".counts").mkdir()
    (tmp_path / ".counts" / "ocr_labels.json").write_text(text)
    counts, found = _worker_counts(tmp_path, "ocr_labels")
    assert counts is None and problem in found


def test_no_text_in_the_figures_is_not_an_error(tmp_path):
    summary, _ = go(tmp_path, fake_config(tmp_path, ocr=True, worker_args={"ocr_labels": ["--empty", "p1"]}))
    assert summary.ok and summary.papers["p1"]["stages"]["ocr_labels"]["records_out"] == 0


def test_a_paper_the_worker_failed_stops(tmp_path):
    summary, _ = go(tmp_path, fake_config(tmp_path, ocr=True, worker_args={"ocr_labels": ["--fail", "p2"]}))
    assert summary.papers["p2"]["status"] == "stopped" and summary.papers["p2"]["stopped_at"] == "ocr_labels"
    assert summary.papers["p1"]["status"] == "ok"


def test_a_rerun_resumes_it_and_a_changed_worker_reruns_link_but_not_the_readers(tmp_path):
    go(tmp_path, fake_config(tmp_path, ocr=True))
    summary, _ = go(tmp_path, fake_config(tmp_path, ocr=True))
    assert summary.stages["ocr_labels"]["resumed"] == 2 and ran(tmp_path / "out", "ocr_labels") == ["p1", "p2"]
    changed = fake_config(tmp_path, ocr=True, worker_args={"ocr_labels": ["--url", "changed"]})
    summary, _ = go(tmp_path, changed)
    resumed = {stage: record["resumed"] for stage, record in summary.stages.items()}
    assert resumed["ocr_labels"] == 0 and resumed["link"] == 0
    assert all(resumed[stage] == 2 for stage in ("render", "parse", "segment", *READS, "vote"))


def test_stale_counts_are_removed_before_the_worker_runs(tmp_path):
    go(tmp_path, fake_config(tmp_path, ocr=True))
    counts = tmp_path / "out" / "papers" / "p1" / ".counts" / "ocr_labels.json"
    counts.write_text(json.dumps({"in": 9, "out": 0}))
    summary, _ = go(tmp_path, fake_config(tmp_path, ocr=True, worker_args={"ocr_labels": ["--fail", "p1"]}))
    assert not counts.exists() and summary.papers["p1"]["stages"]["ocr_labels"]["counts"] is None
