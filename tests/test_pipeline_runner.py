"""The pipeline runner with fake workers and fake in-process steps: stage order, resume, sharing and failures."""

import csv
import json
import sys

import pytest
from pipeline_helpers import FAKE_STEPS, dead_proxy, fake_config, make_papers, models_endpoint, pipeline_yaml, ran

from molscout.pipeline.config import load_pipeline_config
from molscout.pipeline.runner import PipelineError, _endpoint_problem, run
from molscout.predictions import read_predictions

READS = ["read_molglyph", "read_molscribe", "read_molnextr"]
RULES = ["render", "parse", "segment", *READS, "vote", "link", "finalize", "export"]


def go(tmp_path, config, papers=None, out="out", **kwargs):
    papers = papers or make_papers(tmp_path)
    messages = []
    summary = run(papers, tmp_path / out, config, steps=FAKE_STEPS, echo=messages.append, **kwargs)
    return summary, messages


def stages_json(out):
    return json.loads((out / "stages.json").read_text())


def test_rules_run_end_to_end(tmp_path):
    summary, _ = go(tmp_path, fake_config(tmp_path), dataset="biovista", tool="MolScout test")
    out = tmp_path / "out"
    assert summary.ok and list(summary.papers) == ["p1", "p2"]
    data = stages_json(out)
    assert list(data["stages"]) == RULES
    assert data["stages"]["segment"] == {
        "records_in": 4, "records_out": 4, "seconds": data["stages"]["segment"]["seconds"], "peak_gpu_mb": None,
        "errors": 0, "diagnostics": {}, "papers": 2, "resumed": 0,
    }
    assert data["stages"]["read_molnextr"]["records_out"] == 4
    assert data["stages"]["finalize"]["records_out"] == 6 and data["stages"]["export"]["records_out"] == 4
    assert data["per_paper"]["p1"]["stages"]["vote"]["counts"] == {
        "in": 2, "out": 2, "cores": 0, "dropped_not_structure": 0, "dropped_no_reads": 0,
        "dropped_no_usable_read": 0, "dropped_tie": 0, "dropped_slot_labels_disagree": 0,
    }
    assert data["per_paper"]["p1"]["stages"]["finalize"]["counts"]["accepted"] == 2
    assert data["funnel"]["voted_agree_2"] == 4 and data["funnel"]["status_review"] == 2
    assert data["funnel"]["exported"] == 4 and data["funnel"]["tables"] == 2
    assert data["config"]["workers"]["parse"]["worker"] == "fake_worker.py"
    assert (out / "papers" / "p1" / "stages.json").is_file()
    with (out / "molecules.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6 and {row["status"] for row in rows} == {"accepted", "review"}
    assert rows[0]["bbox"] == "1.0,2.0,30.0,40.0"
    predictions = read_predictions(out / "predictions.csv")
    assert len(predictions) == 4 and {p.tool for p in predictions} == {"MolScout test"}
    assert predictions[0].seconds == pytest.approx(summary.papers["p1"]["seconds"])


def test_each_worker_runs_once_for_all_papers(tmp_path):
    go(tmp_path, fake_config(tmp_path))
    out = tmp_path / "out"
    for stage in ("parse", "segment", *READS):
        assert ran(out, stage) == ["p1", "p2"]
        assert len((out / "logs" / f"{stage}.manifest.jsonl").read_text().splitlines()) == 2


def test_export_accepted_and_review(tmp_path):
    summary, _ = go(tmp_path, fake_config(tmp_path, export="accepted+review"), dataset="biovista")
    assert summary.stages["export"]["records_out"] == 6
    assert len(read_predictions(tmp_path / "out" / "predictions.csv")) == 6


def test_a_second_run_resumes_every_stage(tmp_path):
    config = fake_config(tmp_path)
    go(tmp_path, config)
    summary, messages = go(tmp_path, config)
    out = tmp_path / "out"
    assert ran(out, "segment") == ["p1", "p2"]  # not again
    assert all(stage["resumed"] == 2 for stage in summary.stages.values())
    assert "segment: 0 paper(s) to run, 2 done before" in messages
    assert summary.stages["segment"]["records_out"] == 4


def test_a_changed_fix_reruns_only_the_stages_it_reaches(tmp_path):
    papers = make_papers(tmp_path)
    go(tmp_path, fake_config(tmp_path), papers)
    summary, _ = go(tmp_path, fake_config(tmp_path, fixes_off=("junk",)), papers)
    resumed = {stage: record["resumed"] for stage, record in summary.stages.items()}
    assert all(resumed[stage] == 2 for stage in ("render", "parse", "segment", *READS))
    assert all(resumed[stage] == 0 for stage in ("vote", "link", "finalize", "export"))  # they all take the fixes


def test_a_new_paper_runs_alone(tmp_path):
    config = fake_config(tmp_path)
    go(tmp_path, config, make_papers(tmp_path, ("p1",)))
    go(tmp_path, config, make_papers(tmp_path, ("p1", "p2")))
    assert ran(tmp_path / "out", "parse") == ["p1", "p2"]


def test_a_worker_crash_stops_every_paper_it_had(tmp_path):
    config = fake_config(tmp_path, worker_args={"segment": ["--crash"]})
    summary, messages = go(tmp_path, config, dataset="biovista")
    assert summary.failed == ["p1", "p2"]
    assert all(record["status"] == "stopped" and record["stopped_at"] == "segment" for record in summary.papers.values())
    assert "exited with status 1 (a model or setup failure)" in summary.papers["p1"]["errors"][0]["error"]
    assert "read_molglyph" not in summary.papers["p1"]["stages"]
    assert any("segment" in message and "status 1" in message for message in messages)
    assert read_predictions(tmp_path / "out" / "predictions.csv") == ()


def test_a_crashed_stage_runs_again_on_resume(tmp_path):
    papers = make_papers(tmp_path)
    go(tmp_path, fake_config(tmp_path, worker_args={"segment": ["--crash"]}), papers)
    summary, _ = go(tmp_path, fake_config(tmp_path), papers)
    assert summary.ok and summary.stages["segment"]["resumed"] == 0 and summary.stages["parse"]["resumed"] == 2


def test_a_paper_the_worker_fails_stops_alone(tmp_path):
    summary, _ = go(tmp_path, fake_config(tmp_path, worker_args={"segment": ["--fail", "p2"]}))
    assert summary.papers["p1"]["status"] == "ok"
    p2 = summary.papers["p2"]
    assert p2["status"] == "stopped" and p2["stopped_at"] == "segment"
    assert [e["error"] for e in p2["errors"]] == ["fake failure", "the worker wrote no regions.jsonl for this paper"]


def test_a_reader_that_fails_a_paper_leaves_the_other_readers(tmp_path):
    summary, _ = go(tmp_path, fake_config(tmp_path, worker_args={"read_molscribe": ["--fail", "p2"]}))
    p2 = summary.papers["p2"]
    assert p2["status"] == "failed" and p2["stopped_at"] is None
    assert [e["error"] for e in p2["errors"]] == ["fake failure", "2 record(s) in, none out"]
    assert p2["funnel"]["reads_molscribe"] == 0 and p2["funnel"]["voted"] == 2


def test_a_paper_whose_every_vote_fails_is_not_failed_by_the_vote(tmp_path):
    config = fake_config(tmp_path, fixes_off=("vote",), worker_args={"read_molglyph": ["--empty", "p2"]})
    summary, _ = go(tmp_path, config)
    vote = summary.papers["p2"]["stages"]["vote"]
    assert (vote["records_in"], vote["records_out"], vote["errors"]) == (2, 0, [])


def test_input_without_output_is_a_recorded_error(tmp_path):
    summary, _ = go(tmp_path, fake_config(tmp_path, worker_args={"segment": ["--empty", "p2"]}))
    p2 = summary.papers["p2"]
    assert p2["status"] == "failed" and p2["stopped_at"] is None
    assert p2["errors"] == [{"stage": "segment", "error": "2 record(s) in, none out"}]
    assert summary.failed == ["p2"]


def test_allow_empty_accepts_a_stage_without_output(tmp_path):
    config = fake_config(tmp_path, "allow_empty: [segment]\n", worker_args={"segment": ["--empty", "p2"]})
    summary, _ = go(tmp_path, config)
    assert summary.ok


def test_a_step_that_raises_stops_the_paper_and_logs_the_traceback(tmp_path):
    def broken(paper, paper_dir, config):
        if paper.paper_id == "p1":
            raise ValueError("table 3 has 4 rows but 6 labels")
        FAKE_STEPS["link"](paper, paper_dir, config)

    papers = make_papers(tmp_path)
    summary = run(papers, tmp_path / "out", fake_config(tmp_path), steps={**FAKE_STEPS, "link": broken}, echo=lambda m: None)
    assert summary.papers["p1"]["errors"] == [{"stage": "link", "error": "ValueError: table 3 has 4 rows but 6 labels"}]
    assert summary.papers["p1"]["status"] == "stopped" and summary.papers["p2"]["status"] == "ok"
    assert "Traceback" in (tmp_path / "out" / "logs" / "link.log").read_text()


def test_errors_a_step_writes_are_recorded(tmp_path):
    def short_tables(paper, paper_dir, config):
        FAKE_STEPS["link"](paper, paper_dir, config)
        with (paper_dir / "errors.jsonl").open("a") as handle:
            handle.write(json.dumps({"stage": "link", "error": "table t1: 5 rows, 7 labels"}) + "\n")

    papers = make_papers(tmp_path, ("p1",))
    summary = run(papers, tmp_path / "out", fake_config(tmp_path), steps={**FAKE_STEPS, "link": short_tables}, echo=lambda m: None)
    assert summary.papers["p1"]["status"] == "failed"
    assert summary.papers["p1"]["errors"] == [{"stage": "link", "error": "table t1: 5 rows, 7 labels"}]
    assert summary.stages["export"]["records_out"] == 2  # the paper goes on


def test_a_reader_that_overwrites_the_others_stops_the_paper(tmp_path):
    summary, _ = go(tmp_path, fake_config(tmp_path, worker_args={"read_molnextr": ["--overwrite"]}))
    record = summary.papers["p1"]
    assert record["stopped_at"] == "vote"
    assert "read_molglyph wrote 2 read(s) but 0 are left" in record["errors"][0]["error"]


def test_a_rerun_reader_replaces_only_its_own_reads(tmp_path):
    papers = make_papers(tmp_path, ("p1",))
    go(tmp_path, fake_config(tmp_path), papers)
    marker = tmp_path / "out" / "papers" / "p1" / ".done" / "read_molscribe.json"
    marker.unlink()
    summary, _ = go(tmp_path, fake_config(tmp_path), papers)
    assert summary.ok and summary.stages["read_molscribe"]["resumed"] == 0
    reads = (tmp_path / "out" / "papers" / "p1" / "reads.jsonl").read_text().splitlines()
    assert len(reads) == 6


def test_vote_off_keeps_the_primary_reader(tmp_path):
    summary, _ = go(tmp_path, fake_config(tmp_path, fixes_off=("vote",)))
    assert summary.funnel["voted_agree_1"] == 4


def test_missing_worker_python_stops_the_papers(tmp_path):
    path = pipeline_yaml(tmp_path)
    path.write_text(path.read_text().replace(json.dumps(sys.executable), '"/no/such/python"', 1))
    summary, _ = go(tmp_path, load_pipeline_config(path))
    assert summary.papers["p1"]["stopped_at"] == "parse"
    assert summary.papers["p1"]["errors"][0]["error"] == "worker Python not found: /no/such/python"


def test_the_endpoint_check_reaches_a_local_server_past_a_proxy(monkeypatch):
    dead_proxy(monkeypatch)
    with models_endpoint() as url:
        assert _endpoint_problem(url) is None


def test_32b_joiner_needs_an_answering_endpoint(tmp_path):
    config = fake_config(tmp_path, joiner="32b", vllm_url="http://127.0.0.1:9/v1")
    summary, _ = go(tmp_path, config)
    assert summary.papers["p1"]["stopped_at"] == "join_32b"
    assert "does not answer" in summary.papers["p1"]["errors"][0]["error"]


def test_32b_joiner_runs_its_worker(tmp_path):
    config = fake_config(tmp_path, "check_endpoint: false\n", joiner="32b", vllm_url="http://127.0.0.1:9/v1")
    summary, _ = go(tmp_path, config, dataset="biovista")
    assert summary.ok and "link" not in summary.stages
    assert summary.funnel["source_32b"] == 4 and summary.stages["export"]["records_out"] == 4


def test_a_32b_join_that_finds_nothing_is_not_an_error(tmp_path):
    config = fake_config(tmp_path, "check_endpoint: false\n", joiner="32b", vllm_url="http://127.0.0.1:9/v1",
                         worker_args={"join_32b": ["--empty", "p2"]})
    summary, _ = go(tmp_path, config)
    join = summary.papers["p2"]["stages"]["join_32b"]
    assert (join["records_out"], join["errors"]) == (0, [])


def test_a_resumed_stage_drops_a_none_out_error_the_rule_now_allows(tmp_path):
    papers = make_papers(tmp_path)
    go(tmp_path, fake_config(tmp_path), papers)
    marker = tmp_path / "out" / "papers" / "p2" / ".done" / "vote.json"
    data = json.loads(marker.read_text())
    marker.write_text(json.dumps({**data, "errors": ["2 record(s) in, none out"]}))
    summary, _ = go(tmp_path, fake_config(tmp_path), papers)
    assert summary.papers["p2"]["stages"]["vote"]["resumed"] and summary.papers["p2"]["status"] == "ok"
    assert summary.papers["p2"]["errors"] == []


def test_both_joiner_merges(tmp_path):
    config = fake_config(tmp_path, "check_endpoint: false\n", joiner="both", vllm_url="http://127.0.0.1:9/v1")
    summary, _ = go(tmp_path, config)
    assert summary.ok
    assert summary.stages["merge"]["records_in"] == 10 and summary.stages["merge"]["records_out"] == 10


def test_worker_seconds_are_shared_by_records_in(tmp_path):
    summary, _ = go(tmp_path, fake_config(tmp_path))
    p1, p2 = (summary.papers[p]["stages"]["segment"]["seconds"] for p in ("p1", "p2"))
    assert p1 == pytest.approx(p2) and p1 + p2 == pytest.approx(summary.stages["segment"]["seconds"])


def test_shared_upstream_is_computed_once_for_several_configs(tmp_path):
    shared = tmp_path / "shared"
    papers = make_papers(tmp_path)
    go(tmp_path, fake_config(tmp_path, shared=shared), papers, out="full")
    summary, _ = go(tmp_path, fake_config(tmp_path, shared=shared, fixes_off=("vote", "junk")), papers, out="ladder")
    assert ran(tmp_path / "full", "segment") == ["p1", "p2"] and ran(tmp_path / "ladder", "segment") == []
    assert summary.stages["segment"]["resumed"] == 2 and summary.stages["vote"]["resumed"] == 0
    linked = tmp_path / "ladder" / "papers" / "p1" / "regions.jsonl"
    assert linked.is_symlink() and linked.resolve().parent.name == "p1"
    assert linked.resolve().parent.parent.parent == shared.resolve()
    assert summary.funnel["voted_agree_1"] == 4


def test_a_held_shared_folder_stops_the_run(tmp_path):
    shared = tmp_path / "shared"
    config = fake_config(tmp_path, shared=shared)
    papers = make_papers(tmp_path, ("p1",))
    from molscout.pipeline import stages as st

    folder = shared / st.upstream_key(st.stage_keys(config, papers[0].sha256()))[:16] / "p1"
    (folder / ".lock").mkdir(parents=True)
    (folder / ".lock" / "owner.json").write_text('{"host": "elsewhere", "pid": 1, "job": "42"}')
    with pytest.raises(PipelineError, match="elsewhere"):
        go(tmp_path, config, papers)


def test_worker_status_4_goes_on_and_counts_the_failed_papers(tmp_path):
    summary, messages = go(tmp_path, fake_config(tmp_path, worker_args={"segment": ["--fail", "p2", "--exit", "4"]}))
    assert summary.papers["p1"]["status"] == "ok"
    assert summary.papers["p2"]["status"] == "stopped"  # it has no regions.jsonl
    assert summary.papers["p2"]["errors"][0] == {"stage": "segment", "error": "fake failure"}
    assert "segment: the worker reports that some papers failed; their errors are recorded" in messages


def test_worker_status_3_stops_every_paper_and_says_why(tmp_path):
    config = fake_config(tmp_path, "check_endpoint: false\n", joiner="32b", vllm_url="http://127.0.0.1:9/v1",
                         worker_args={"join_32b": ["--exit", "3"]})
    summary, _ = go(tmp_path, config)
    assert all(record["stopped_at"] == "join_32b" for record in summary.papers.values())
    assert "status 3 (the 32B server was lost)" in summary.papers["p1"]["errors"][-1]["error"]


def test_counts_that_do_not_add_up_are_an_error(tmp_path):
    def leaky(paper, paper_dir, config):
        FAKE_STEPS["finalize"](paper, paper_dir, config)
        return {"in": 3, "out": 1, "dropped_duplicates": 1}

    papers = make_papers(tmp_path, ("p1",))
    summary = run(papers, tmp_path / "out", fake_config(tmp_path), steps={**FAKE_STEPS, "finalize": leaky}, echo=lambda m: None)
    record = summary.papers["p1"]
    assert record["errors"] == [{"stage": "finalize", "error": "counts do not add up: in 3 != out 1 + dropped 1"}]
    assert record["stages"]["finalize"]["counts"] == {"in": 3, "out": 1, "dropped_duplicates": 1}


def test_errors_are_filed_under_the_stage_they_name(tmp_path):
    def link(paper, paper_dir, config):
        FAKE_STEPS["link"](paper, paper_dir, config)
        with (paper_dir / "errors.jsonl").open("a") as handle:
            handle.write(json.dumps({"stage": "vote", "error": "region p1-r1: readers disagree"}) + "\n")
            handle.write(json.dumps({"stage": "elsewhere", "error": "odd"}) + "\n")

    papers = make_papers(tmp_path, ("p1",))
    summary = run(papers, tmp_path / "out", fake_config(tmp_path), steps={**FAKE_STEPS, "link": link}, echo=lambda m: None)
    stages = summary.papers["p1"]["stages"]
    assert stages["vote"]["errors"] == ["region p1-r1: readers disagree"]
    assert stages["link"]["errors"] == ["[elsewhere] odd"]


def test_diagnostic_lines_are_counted_not_failures(tmp_path):
    def join(paper, paper_dir, config):
        FAKE_STEPS["link"](paper, paper_dir, config)
        lines = [
            {"stage": "link", "paper_id": paper.paper_id, "kind": "malformed", "count": 3, "message": "3 entries"},
            {"stage": "link", "paper_id": paper.paper_id, "kind": "truncated", "message": "max_tokens"},
            {"stage": "link", "paper_id": paper.paper_id, "kind": "unusable", "message": "no answer"},
        ]
        with (paper_dir / "errors.jsonl").open("a") as handle:
            handle.writelines(json.dumps(line) + "\n" for line in lines)

    papers = make_papers(tmp_path, ("p1",))
    summary = run(papers, tmp_path / "out", fake_config(tmp_path), steps={**FAKE_STEPS, "link": join}, echo=lambda m: None)
    link = summary.papers["p1"]["stages"]["link"]
    assert link["diagnostics"] == {"malformed": 3, "truncated": 1}
    assert link["errors"] == ["no answer"] and summary.papers["p1"]["status"] == "failed"
    assert summary.stages["link"]["diagnostics"] == {"malformed": 3, "truncated": 1}


def test_new_error_lines_are_found_by_content(tmp_path):
    from collections import Counter

    from molscout.pipeline.runner import _new_errors

    path = tmp_path / "errors.jsonl"
    old = json.dumps({"stage": "segment", "error": "old"})
    path.write_text(json.dumps({"stage": "segment", "error": "new"}) + "\n")  # the worker cleared its old line
    assert _new_errors(tmp_path, Counter([old])) == [("segment", "new", None, 1)]


def test_regions_the_gate_rejected_are_no_input_for_link_or_a_failed_vote(tmp_path):
    summary, _ = go(tmp_path, fake_config(tmp_path, worker_args={"segment": ["--reject", "p2"]}))
    p2 = summary.papers["p2"]
    assert p2["status"] == "ok" and p2["errors"] == []
    assert p2["stages"]["link"]["records_in"] == 1  # the table only, not the two rejected regions
    assert p2["stages"]["vote"]["counts"]["dropped_not_structure"] == 2
    assert p2["funnel"]["vote_failed"] == 0 and p2["funnel"]["structures"] == 0


def test_link_input_counts_unvoted_structures_but_not_rejected_regions(tmp_path):
    from molscout.pipeline.stages import records_in

    config = fake_config(tmp_path)
    votes = [{"region_id": "r1", "smiles": None, "reason": "not a structure"},
             {"region_id": "r2", "smiles": None, "reason": "no usable read"},
             {"region_id": "r3", "smiles": "CCO", "reason": None}]
    (tmp_path / "voted.jsonl").write_text("".join(json.dumps(v) + "\n" for v in votes))
    (tmp_path / "layout.jsonl").write_text(json.dumps({"type": "text"}) + "\n")
    assert records_in("link", tmp_path, config) == 2
