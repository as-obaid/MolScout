"""MolScout as a complete system in the harness: its run.py with fake workers and steps, scored like any other."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import yaml
from pipeline_helpers import dead_proxy, models_endpoint, pipeline_yaml
from test_bench_papers import Workspace

from molscout.bench.harness import run_benchmark
from molscout.data.biovista_dev import DevPaper, write_dev_manifest
from molscout.predictions import read_predictions

TESTS = Path(__file__).resolve().parent
TOOL = TESTS.parent / "benchmarks" / "tools" / "complete_systems" / "molscout"
WRAPPER = f'''import importlib.util, sys
from pathlib import Path
sys.path.insert(0, {str(TESTS)!r})
spec = importlib.util.spec_from_file_location("molscout_tool_run", Path(__file__).with_name("real_run.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from pipeline_helpers import FAKE_STEPS
sys.exit(module.main(steps=FAKE_STEPS))
'''


def load_run_module():
    spec = importlib.util.spec_from_file_location("molscout_tool_run", TOOL / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ws(tmp_path):
    workspace = Workspace(tmp_path / "repo")
    dev = (DevPaper("1_aaaa", "aaaa", "drawn", 3, 2, 1), DevPaper("3_cccc", "cccc", "table", 1, 0, 1))
    write_dev_manifest(workspace.biovista / "biovista_dev.csv", dev)
    tool = workspace.root / "tools" / "molscout_fake"
    tool.mkdir(parents=True)
    shutil.copy(TOOL / "run.py", tool / "real_run.py")
    shutil.copy(TOOL / "vllm_server.py", tool / "vllm_server.py")
    shutil.copy(pipeline_yaml(tmp_path), tool / "pipeline.yaml")
    (tool / "run.py").write_text(WRAPPER)
    return workspace


def molscout_config(ws, dataset, *args):
    path = ws.config(dataset)
    data = yaml.safe_load(path.read_text())
    data.update(tool="molscout_fake", name="MolScout", run_dir="tools/molscout_fake", subset="dev")
    data["args"] = ["--work", str(ws.root / "work" / dataset), *args]
    target = path.with_name(f"molscout_fake__{dataset}.yaml")
    target.write_text(yaml.safe_dump(data, sort_keys=False))
    return target


def test_the_harness_runs_and_scores_molscout_on_the_dev_papers(ws):
    folder = run_benchmark(molscout_config(ws, "biovista"), repo_root=ws.root, results_root=ws.results)
    assert sorted(path.name for path in folder.iterdir()) == [
        "config.yaml", "errors.json", "meta.json", "predictions.csv", "scores.json", "stages.json", "timing.json",
    ]
    scores = json.loads((folder / "scores.json").read_text())
    assert list(scores["scores"]["papers"]) == ["1_aaaa", "3_cccc"]
    assert scores["scores"]["groups"]["dev"]["counts"]["tp"] == 1  # CCO in 1_aaaa
    predictions = read_predictions(folder / "predictions.csv")
    assert {p.smiles for p in predictions} == {"CCO"} and {p.tool for p in predictions} == {"MolScout 1.0 (test)"}
    stages = json.loads((folder / "stages.json").read_text())
    assert set(stages["per_paper"]) == {"1_aaaa", "3_cccc"} and stages["stages"]["export"]["records_out"] == 4
    timing = json.loads((folder / "timing.json").read_text())
    assert timing["seconds"]["1_aaaa"] == pytest.approx(stages["per_paper"]["1_aaaa"]["seconds"])
    assert json.loads((folder / "errors.json").read_text()) == {"papers": 2, "failed": 0, "errors": {}}


def set_worker_args(ws, stage, *extra):
    tool = ws.root / "tools" / "molscout_fake"
    old = f'"--stage", "{stage}"'
    text = (tool / "pipeline.yaml").read_text().replace(old, ", ".join([old, *(f'"{arg}"' for arg in extra)]))
    (tool / "pipeline.yaml").write_text(text)


def test_internal_stages_keep_counts_and_cut_errors_to_types(ws):
    set_worker_args(ws, "segment", "--empty", "p1")
    folder = run_benchmark(molscout_config(ws, "internal"), repo_root=ws.root, results_root=ws.results)
    record = json.loads((folder / "stages.json").read_text())["per_paper"]["p1"]
    assert record["status"] == "failed" and record["errors"] == [{"stage": "segment", "error": "error"}]
    assert record["stages"]["segment"]["errors"] == ["error"] and record["stages"]["segment"]["records_in"] == 2


def test_a_stopped_paper_is_a_crashed_paper_for_the_harness(ws):
    set_worker_args(ws, "segment", "--fail", "1_aaaa")
    folder = run_benchmark(molscout_config(ws, "biovista"), repo_root=ws.root, results_root=ws.results)
    errors = json.loads((folder / "errors.json").read_text())
    assert errors == {
        "papers": 2, "failed": 1, "errors": {"1_aaaa": "segment: the worker wrote no regions.jsonl for this paper"}
    }
    assert {p.item_id for p in read_predictions(folder / "predictions.csv")} == {"3_cccc"}
    assert json.loads((folder / "meta.json").read_text())["tool_errors"] == 1


def test_public_stages_keep_exception_types():
    module = load_run_module()
    stages = {
        "per_paper": {
            "1": {
                "errors": [{"stage": "link", "error": "ValueError: compound 7 is CCO"}, {"stage": "x", "error": "free text"}],
                "stages": {"link": {"records_in": 1, "errors": ["ValueError: compound 7 is CCO"]}},
            }
        }
    }
    cut = module.public_stages(stages)["per_paper"]["1"]
    assert [e["error"] for e in cut["errors"]] == ["ValueError", "error"]
    assert cut["stages"]["link"] == {"records_in": 1, "errors": ["ValueError"]}


def test_run_py_needs_a_work_folder(tmp_path, capsys):
    module = load_run_module()
    base = ["--papers", "p.csv", "--dataset", "biovista", "--tool", "MolScout 0", "--output", "o.csv"]
    with pytest.raises(SystemExit):
        module.main(base)
    assert "--work" in capsys.readouterr().err


def test_the_vllm_server_is_serve_32b_on_its_own_port(tmp_path):
    spec = importlib.util.spec_from_file_location("vllm_server", TOOL / "vllm_server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = module.command(tmp_path / "serve_32b.sh", tmp_path / "work", 8623, 0.8)
    assert argv == ["bash", str(tmp_path / "serve_32b.sh"), "--work", str(tmp_path / "work"), "--port", "8623",
                    "--gpu-memory-utilization", "0.8"]
    assert module.PORT != 8613  # BioMiner's own runs serve there


def test_the_vllm_server_wait_reaches_localhost_past_a_proxy(monkeypatch):
    spec = importlib.util.spec_from_file_location("vllm_server", TOOL / "vllm_server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dead_proxy(monkeypatch)
    with models_endpoint(module.SERVED_MODEL) as url:
        server = module.VllmServer([], "unused.log", port=int(url.rsplit(":", 1)[1].split("/")[0]))
        assert server.ready()


def test_fix_lists(tmp_path):
    module = load_run_module()
    assert module.names(None) is None and module.names("none") == () and module.names("") == ()
    assert module.names("labels, tables") == ("labels", "tables")


def test_dev_runs_stay_out_of_the_complete_system_benchmark(ws):
    from molscout.report.paper_analysis import load_paper_runs, paper_metrics, system_name

    run_benchmark(molscout_config(ws, "internal"), repo_root=ws.root, results_root=ws.results)
    assert load_paper_runs(ws.results) == ()
    [run] = load_paper_runs(ws.results, subsets=True)
    metrics = paper_metrics(run)
    assert "dev/f1" in metrics and "test/f1" not in metrics and metrics["items/papers"] == 1
    assert (system_name("molscout_l3_core"), system_name("molscout"), system_name("biominer")) == (
        "MolScout l3_core", "MolScout", "BioMiner",
    )
