"""Loading result folders, refusing inconsistent ones, the W&B payload, and an offline end-to-end upload.

Result folders are built on the hand-made crop references (c1 to c7, c5 unreadable) and scored
by the real scorer, as the harness would.
"""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from molscout.predictions import Prediction, write_predictions
from molscout.report.analysis import (
    InconsistentResults,
    ResultsError,
    check_consistency,
    check_references,
    load_run,
    load_runs,
)
from molscout.runs import score_run
from molscout.scoring import write_scores

REPO = Path(__file__).resolve().parents[1]
REFERENCES = Path(__file__).parent / "fixtures" / "handmade" / "crops_references"
CROPS = ("c1", "c2", "c3", "c4", "c5", "c6", "c7")
COMMIT = "4cbaee3e2d4abdb94afaac495b0922ecf3c159fc"
# MolScribe: 3 of 6 right (c2 has the wrong enantiomer, c6 crashed, c7 is wrong); MolNexTR: 4 of 6.
MOLSCRIBE = {
    "c1": "OCC",
    "c2": "C[C@@H](N)C(=O)O",
    "c3": "C1=CC=CC=C1",
    "c4": "*c1ccccc1",
    "c5": "C",
    "c6": "",
    "c7": "C1CC",
}
MOLNEXTR = {"c1": "CCO", "c2": "C[C@H](N)C(=O)O", "c3": "c1ccccc1", "c4": "C(", "c5": "", "c6": "CC(=O)O", "c7": "CCO"}
RESOURCES = {
    "tool_peak_rss_mib": 3072.0,
    "tool_cpu_seconds": 41.5,
    "gpu": {
        "name": "NVIDIA H200",
        "peak_memory_mib": 6144.0,
        "mean_utilization_pct": 37.5,
        "samples": 12,
        "interval_seconds": 1.0,
    },
}


@pytest.fixture
def repo(tmp_path):
    """A repository root with the USPTO references and one tiny image per crop under data/raw/uspto."""
    root = tmp_path / "repo"
    shutil.copytree(REFERENCES, root / "data" / "raw" / "uspto" / "USPTO_mol_ref")
    images = root / "data" / "raw" / "uspto" / "USPTO"
    images.mkdir(parents=True)
    for crop in CROPS:
        Image.new("RGB", (8, 6), "white").save(images / f"{crop}.png")
    return root


def make_result(repo, tool, answers, *, commit=COMMIT, dirty=False, crashed=(), resources=RESOURCES):
    """A `<tool>__uspto` result folder as the harness writes it, scored by the real scorer."""
    folder = repo / "benchmarks" / "results" / f"{tool}__uspto"
    folder.mkdir(parents=True)
    label = f"{tool.capitalize()} 1.0 (test)"
    rows = [
        Prediction("uspto", crop, smiles, None, None, None, label, 0.2 + n / 10)
        for n, (crop, smiles) in enumerate(answers.items())
    ]
    write_predictions(folder / "predictions.csv", rows)
    write_scores(
        folder / "scores.json", score_run(folder / "predictions.csv", references=repo / "data/raw/uspto/USPTO_mol_ref")
    )
    meta = {
        "run": f"{tool}__uspto",
        "tool": {
            "tool": tool,
            "name": tool.capitalize(),
            "version": "1.0 (test)",
            "checkpoints": [{"path": f"/store/models/{tool}/model.pth", "sha256": "c" * 64}],
        },
        "dataset": "uspto",
        "items": len(answers),
        "tool_errors": len(crashed),
        "git": {"commit": commit, "dirty": dirty, "dirty_paths": ["src/molscout/x.py"] if dirty else []},
        "environment": {"sha256": "e" * 64},
        "hardware": {
            "cpu_model": "Test CPU",
            "gpus": [{"name": "NVIDIA H200", "memory": "143771 MiB", "driver": "570"}],
        },
        "slurm": {"job": "1", "partition": "gpu-short"},
        **({"resources": resources} if resources is not None else {}),
    }
    (folder / "meta.json").write_text(json.dumps(meta))
    (folder / "config.yaml").write_text(f"tool: {tool}\ndataset: uspto\n")
    errors = {
        "images": len(answers),
        "failed": len(crashed),
        "errors": {crop: "RuntimeError: boom" for crop in crashed},
    }
    (folder / "errors.json").write_text(json.dumps(errors))
    return folder


def two_tools(repo, **molnextr):
    make_result(repo, "molscribe", MOLSCRIBE, crashed=("c6",))
    make_result(repo, "molnextr", MOLNEXTR, **molnextr)
    return repo / "benchmarks" / "results"


def test_load_runs_reads_crop_folders_in_tool_order_and_skips_the_rest(repo):
    results = two_tools(repo)
    for leftover in (".staging-molscribe__uspto-1", "biominer__internal"):
        (results / leftover).mkdir()
        (results / leftover / "scores.json").write_text("{}")
    runs = load_runs(results)
    assert [run.name for run in runs] == ["molscribe__uspto", "molnextr__uspto"]
    assert runs[0].errors == {"c6": "RuntimeError: boom"}
    assert check_consistency(runs) == [] and check_references(runs, repo) == []


def test_load_run_refuses_incomplete_or_mismatched_folders(repo):
    folder = make_result(repo, "molscribe", MOLSCRIBE)
    (folder / "config.yaml").unlink()
    with pytest.raises(ResultsError, match="missing config.yaml"):
        load_run(folder)
    (folder / "config.yaml").write_text("tool: molscribe\n")
    with (folder / "predictions.csv").open("a") as handle:
        handle.write('uspto,c8,C,,,,"Molscribe 1.0 (test)",0.1\n')
    with pytest.raises(ResultsError, match="another predictions.csv"):
        load_run(folder)
    renamed = folder.with_name("molvec__uspto")
    folder.rename(renamed)
    with pytest.raises(ResultsError, match="the folder is molvec on uspto"):
        load_run(renamed)


def test_consistency_refuses_mixed_commits_uncommitted_code_and_different_references(repo):
    results = two_tools(repo, commit="0" * 40, dirty=True)
    scores = results / "molnextr__uspto" / "scores.json"
    report = json.loads(scores.read_text())
    report["inputs"]["references"]["sha256"] = "f" * 64
    scores.write_text(json.dumps(report))
    problems = check_consistency(load_runs(results))
    assert problems == [
        "molnextr__uspto ran with uncommitted changes: src/molscout/x.py",
        f"the runs come from 2 commits: {COMMIT[:12]} (molscribe__uspto); 000000000000 (molnextr__uspto)",
        f"uspto: the tools were scored against 2 reference sets: "
        f"{report_sha(results, 'molscribe')[:12]} (molscribe); ffffffffffff (molnextr)",
    ]


def test_consistency_refuses_runs_that_do_not_record_their_commit(repo):
    make_result(repo, "molscribe", MOLSCRIBE, commit=None)
    meta_path = repo / "benchmarks/results/molscribe__uspto/meta.json"
    meta = json.loads(meta_path.read_text())
    meta["git"]["dirty"] = None
    meta_path.write_text(json.dumps(meta))
    assert check_consistency(load_runs(repo / "benchmarks" / "results")) == [
        "molscribe__uspto does not record whether its code was committed",
        "no git commit recorded for molscribe__uspto",
    ]


def test_references_that_changed_since_scoring_are_reported(repo):
    results = two_tools(repo)
    (repo / "data/raw/uspto/USPTO_mol_ref/c7.mol").write_text((REFERENCES / "c1.MOL").read_text())
    [problem] = check_references(load_runs(results), repo)
    assert problem.startswith("uspto: ") and problem.endswith(
        "than molscribe__uspto, molnextr__uspto was scored against"
    )


@pytest.fixture
def publisher():
    """molscout.report.wandb_publish; skips where the report extra (wandb, plotly) is not installed."""
    pytest.importorskip("wandb")
    pytest.importorskip("plotly")
    from molscout.report import wandb_publish

    return wandb_publish


def test_payload_for_one_folder_names_the_run_and_carries_config_and_summary(repo, publisher):
    two_tools(repo)
    benchmark = publisher.prepare(repo / "benchmarks" / "results", repo)
    run = benchmark.runs[0]
    assert publisher.run_id(run) == f"molscribe-uspto-{run.predictions_sha256[:10]}"
    assert publisher.run_config(run) == {
        "tool": "molscribe",
        "name": "Molscribe",
        "version": "1.0 (test)",
        "dataset": "uspto",
        "git_commit": COMMIT,
        "checkpoints": [{"file": "model.pth", "sha256": "c" * 64}],
        "environment_sha256": "e" * 64,
        "device": "NVIDIA H200",
        "slurm_partition": "gpu-short",
    }
    summary = benchmark.metrics[("molscribe", "uspto")]
    assert summary["accuracy/stereo_aware"] == 0.5 and summary["items/scored"] == 6
    assert (summary["outcome/correct"], summary["outcome/stereo_only"], summary["outcome/crashed"]) == (3, 1, 1)
    assert summary["resources/gpu_peak_memory_gib"] == 6.0 and summary["items/crashed"] == 1
    assert publisher.summary_run_id(benchmark.runs) == publisher.summary_run_id(benchmark.runs[::-1])


def test_tables_hold_every_scored_crop_and_a_seeded_failure_sample(repo, publisher):
    two_tools(repo)
    benchmark = publisher.prepare(repo / "benchmarks" / "results", repo)
    columns, rows = publisher.prediction_rows(benchmark)
    assert columns == [
        "dataset",
        "item_id",
        "reference",
        "molscribe/smiles",
        "molscribe/outcome",
        "molnextr/smiles",
        "molnextr/outcome",
    ]
    assert [row[1] for row in rows] == ["c1", "c2", "c3", "c4", "c6", "c7"]
    assert rows[4][3:] == ["", "crashed", "CC(=O)O", "correct"]
    columns, rows = publisher.failure_rows(benchmark, 3, repo)
    assert columns[:4] == ["dataset", "item_id", "image", "reference"] and len(rows) == 3
    assert all(row[2] == repo / "data/raw/uspto/USPTO" / f"{row[1]}.png" for row in rows)
    columns, rows = publisher.leaderboard_rows(benchmark)
    assert columns[:3] == ["tool", "dataset", "accuracy/stereo_aware"] and [row[0] for row in rows] == [
        "molscribe",
        "molnextr",
    ]


def test_upload_refuses_an_incomplete_meta_before_touching_wandb(repo, publisher, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb-dir"))
    results = two_tools(repo)
    meta_path = results / "molnextr__uspto" / "meta.json"
    meta = json.loads(meta_path.read_text())
    del meta["slurm"]
    meta_path.write_text(json.dumps(meta))
    assert upload(results, repo, "--offline", "--allow-inconsistent") == 1
    assert "molnextr__uspto: meta.json lacks 'slurm'" in capsys.readouterr().err
    assert not (tmp_path / "wandb-dir").exists()


def test_upload_refuses_inconsistent_results_before_touching_wandb(repo, publisher, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb-dir"))
    results = two_tools(repo, commit="0" * 40)
    assert upload(results, repo, "--offline") == 1
    assert "2 commits" in capsys.readouterr().err
    assert not (tmp_path / "wandb-dir").exists()


def test_offline_upload_logs_one_run_per_folder_and_one_summary(repo, publisher, tmp_path, monkeypatch):
    wandb_dir = tmp_path / "wandb-dir"
    wandb_dir.mkdir()
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(wandb_dir))
    results = two_tools(repo)
    assert upload(results, repo, "--offline", "--images-per-dataset", "2") == 0
    offline = wandb_dir / "wandb"
    assert len(list(offline.glob("offline-run-*"))) == 3
    [eval_dir] = offline.glob("offline-run-*-molscribe-uspto-*")
    logged = logged_values(eval_dir)
    assert logged["summary"]["accuracy/stereo_aware"] == 0.5 and logged["summary"]["outcome/crashed"] == 1
    assert {"accuracy/stereo_aware_ci_low", "speed/s_per_crop_mean", "resources/peak_rss_gib", "items/excluded"} <= set(
        logged["summary"]
    )
    assert logged["config"]["tool"] == "molscribe" and logged["config"]["device"] == "NVIDIA H200"
    assert logged["history"] == {}  # an eval run has a summary, not panels
    assert logged["run"] == {"tags": ["molscribe", "uspto"], "group": "structure-readers", "job_type": "eval"}
    [analysis_dir] = offline.glob("offline-run-*-summary-*")
    analysis = logged_values(analysis_dir)
    assert analysis["run"] == {"tags": [], "group": "", "job_type": "analysis"}  # nothing carried over from eval runs
    history = analysis["history"]
    figures = {key.removesuffix("/_type") for key, value in history.items() if value == "plotly-file"}
    assert figures == {
        "accuracy/by_dataset",
        "accuracy/stereo_penalty",
        "accuracy/vs_speed",
        "outcomes/by_tool",
        "speed/time_per_crop",
        "resources/by_tool",
        "agreement/pairwise",
        "agreement/consensus",
    }
    assert {key.removesuffix("/_type") for key, value in history.items() if value == "table-file"} == {
        "leaderboard",
        "predictions",
        "failures",
    }
    plots = list((analysis_dir / "files" / "media" / "plotly").rglob("*.plotly.json"))
    assert len(plots) == len(figures) and not any("bdata" in path.read_text() for path in plots)


def logged_values(run_dir):
    """Summary, config and history values from an offline run's transaction log (LevelDB-style records)."""
    from wandb.proto import wandb_internal_pb2

    data = next(run_dir.glob("*.wandb")).read_bytes()
    assert data[:4] == b":W&B"
    values = {"summary": {}, "config": {}, "history": {}}
    position, block, pending = 7, 32768, b""
    while position + 7 <= len(data):
        remaining = block - position % block
        length, kind = int.from_bytes(data[position + 4 : position + 6], "little"), data[position + 6]
        if remaining < 7 or (length == 0 and kind == 0):
            position += remaining
            continue
        pending += data[position + 7 : position + 7 + length]
        position += 7 + length
        if kind in (1, 4):  # a full record, or the last chunk of one
            record = wandb_internal_pb2.Record()
            record.ParseFromString(pending)
            pending = b""
            kind_of = record.WhichOneof("record_type")
            if kind_of == "run":
                run = record.run
                values["run"] = {"tags": sorted(run.tags), "group": run.run_group, "job_type": run.job_type}
                continue
            if kind_of == "summary":
                items = record.summary.update
            elif kind_of == "config":
                items = record.config.update
            elif kind_of == "history":
                items = [item for item in record.history.item if not item.key.startswith("_")]
            else:
                continue
            for item in items:
                values[kind_of][item.key or "/".join(item.nested_key)] = json.loads(item.value_json)
    return values


def report_sha(results, tool):
    return json.loads((results / f"{tool}__uspto" / "scores.json").read_text())["inputs"]["references"]["sha256"]


def upload(results, repo, *args):
    """Run scripts/wandb_upload.py's main in-process."""
    spec = importlib.util.spec_from_file_location("wandb_upload", REPO / "scripts" / "wandb_upload.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = [
        "--results",
        str(results),
        "--repo-root",
        str(repo),
        "--entity",
        "test",
        "--project",
        "molscout-test",
        *args,
    ]
    return module.main(argv)


def test_inconsistent_results_list_every_problem():
    error = InconsistentResults(["a", "b"])
    assert error.problems == ("a", "b") and str(error).endswith("  - a\n  - b")
    assert isinstance(error, ResultsError)
