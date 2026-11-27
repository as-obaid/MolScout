"""Publishing MolScout's pipeline runs to W&B, with wandb mocked: nothing is uploaded."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from test_molscout_tool import molscout_config, ws  # noqa: F401  (ws is a fixture)

from molscout.bench.harness import run_benchmark
from molscout.report.analysis import ResultsError

pytest.importorskip("wandb")
pytest.importorskip("plotly")
from molscout.report import wandb_pipeline as wp  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def results(ws):  # noqa: F811
    for dataset in ("biovista", "internal"):
        run_benchmark(molscout_config(ws, dataset), repo_root=ws.root, results_root=ws.results)
    return ws.results


def prepare(ws, results):  # noqa: F811
    return wp.prepare_pipeline(results, ws.root, allow_inconsistent=True, warn=lambda _: None)


def test_runs_metrics_and_config(ws, results):  # noqa: F811
    benchmark = prepare(ws, results)
    assert [run.name for run in benchmark.runs] == ["molscout_fake__biovista", "molscout_fake__internal"]
    metrics = benchmark.metrics["molscout_fake__biovista"]
    assert {"micro/f1", "dev/f1", "speed/s_per_paper_mean", "pipe/funnel/exported"} <= set(metrics)
    assert metrics["pipe/stage/segment/records_out"] == 4 and metrics["pipe/papers_failed"] == 0
    assert "pipe/stage/segment/peak_gpu_mb" not in metrics  # no nvidia-smi: unknown, so not logged
    internal = benchmark.metrics["molscout_fake__internal"]
    assert internal["pipe/papers_failed"] == 0 and internal["pipe/stage/segment/errors"] == 0
    assert internal["pipe/funnel/exported"] == 2 and "test/f1" not in internal
    config = benchmark.configs["molscout_fake__internal"]
    assert (config["variant"], config["pipeline"]["joiner"], config["subset"]) == ("fake", "rules", ["p1"])
    assert config["pipeline"]["workers"]["segment"]["env"]  # an environment name, never a path
    assert "/" not in json.dumps(config["pipeline"]["workers"])


def test_the_ladder_has_one_f1_and_one_dev_f1_per_run(ws, results):  # noqa: F811
    benchmark = prepare(ws, results)
    columns, rows = wp.ladder_rows(benchmark)
    assert columns == [
        "variant", "dataset", "fixes", "joiner", "export", "papers", "precision", "recall", "f1", "dev_f1",
        "seconds_per_paper", "failed_papers",
    ]
    ladder = {row[1]: dict(zip(columns, row, strict=True)) for row in rows}
    assert ladder["biovista"]["f1"] == benchmark.metrics["molscout_fake__biovista"]["micro/f1"]
    assert ladder["biovista"]["dev_f1"] == benchmark.metrics["molscout_fake__biovista"]["dev/f1"]
    assert not any(key.startswith("repaired/") for metrics in benchmark.metrics.values() for key in metrics)


def test_paper_table_holds_counts_and_statuses_only(ws, results):  # noqa: F811
    benchmark = prepare(ws, results)
    columns, rows = benchmark.papers["molscout_fake__internal"]
    record = dict(zip(columns, rows[0], strict=True))
    assert (record["paper"], record["status"], record["errors"], record["regions"]) == ("p1", "ok", 0, 2)
    assert all(not isinstance(value, str) or value in ("fake", "internal", "p1", "ok") for value in rows[0])


def test_internal_payloads_never_hold_smiles(ws, results):  # noqa: F811
    benchmark = prepare(ws, results)
    [internal] = [run for run in benchmark.runs if run.dataset == "internal"]
    assert {p.smiles for p in internal.run.predictions} == {"CCO"}
    payload = [benchmark.configs[internal.name], benchmark.metrics[internal.name], benchmark.papers[internal.name]]
    assert "CCO" not in json.dumps(payload)
    with pytest.raises(ResultsError, match="SMILES"):
        wp.check_private(internal, {"note": "read CCO here"})
    [public] = [run for run in benchmark.runs if run.dataset == "biovista"]
    wp.check_private(public, {"note": "CCO"})  # BioVista is public


def test_internal_artifact_leaves_out_predictions_and_messages(ws, results):  # noqa: F811
    benchmark = prepare(ws, results)
    [internal] = [run for run in benchmark.runs if run.dataset == "internal"]
    artifact = wp.artifact_for(internal)
    assert "predictions.csv" not in artifact.manifest.entries and "stages.json" in artifact.manifest.entries
    public = wp.artifact_for(next(run for run in benchmark.runs if run.dataset == "biovista"))
    assert "predictions.csv" in public.manifest.entries


def test_publish_logs_eval_runs_then_the_summary(ws, results, monkeypatch):  # noqa: F811
    import wandb

    benchmark = prepare(ws, results)
    run = MagicMock()
    inits = []
    monkeypatch.setattr(wp.wandb, "init", lambda **kwargs: inits.append(kwargs) or run)
    wp.publish_pipeline(benchmark, entity="e", project="p", repo_root=ws.root, echo=lambda _: None)
    assert [(i["name"], i["group"], i["job_type"]) for i in inits] == [
        ("molscout_fake/biovista", "pipeline", "eval"),
        ("molscout_fake/internal", "pipeline", "eval"),
        ("summary-pipeline", "pipeline", "analysis"),
    ]
    logged = {key: value for call in run.__enter__.return_value.log.call_args_list for key, value in call.args[0].items()}
    assert {"pipe/papers", "pipe/ladder", "pipe/stages"} <= set(logged)
    figures = [key for key, value in logged.items() if isinstance(value, wandb.Html)]
    assert sorted(figures) == ["pipe/funnel_biovista", "pipe/funnel_internal", "pipe/ladder_biovista", "pipe/ladder_internal"]


def test_publish_builds_everything_before_the_first_run(ws, results, monkeypatch):  # noqa: F811
    benchmark = prepare(ws, results)
    started = []
    monkeypatch.setattr(wp.wandb, "init", lambda **kwargs: started.append(kwargs))
    monkeypatch.setattr(wp, "ladder_rows", lambda _: (_ for _ in ()).throw(ResultsError("cannot build")))
    with pytest.raises(ResultsError):
        wp.publish_pipeline(benchmark, entity="e", project="p", repo_root=ws.root, echo=lambda _: None)
    assert started == []


def test_upload_script_publishes_the_pipeline_group(ws, results, monkeypatch):  # noqa: F811
    spec = importlib.util.spec_from_file_location("wandb_upload", REPO / "scripts" / "wandb_upload.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    published = []
    monkeypatch.setattr(wp, "publish_pipeline", lambda benchmark, **kwargs: published.append(benchmark))
    argv = ["--benchmark", "pipeline", "--results", str(results), "--repo-root", str(ws.root), "--entity", "e",
            "--project", "p", "--allow-inconsistent"]
    assert module.main(argv) == 0
    assert [run.name for run in published[0].runs] == ["molscout_fake__biovista", "molscout_fake__internal"]


def test_no_molscout_runs_is_an_error(tmp_path):
    with pytest.raises(ResultsError, match="no MolScout"):
        wp.prepare_pipeline(tmp_path, tmp_path)
