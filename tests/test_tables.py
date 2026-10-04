import json
import subprocess
import sys
from pathlib import Path

import pytest

from molscout.scoring import build_report, write_scores
from molscout.scoring.stats import proportion
from molscout.tables import DATASETS, fill_type1, load_runs

DOC = """# Benchmarking

## Type 1: Structure readers

### Setup

| Tool | Design | Output | Hardware | Version |
|:-----|:-------|:-------|:---------|:--------|
| [MolScribe](https://example.org/molscribe) | Swin Transformer | Graph → SMILES | Explorer GPU | — |
| [MolVec](https://example.org/molvec) | Rule-based vectorization | Molfile → SMILES | Local CPU | — |

### Results

**Exact match, stereo-aware (%)**

| Tool | USPTO | UOB | JPO | CLEF | MolRecBench-Wild |
|:-----|------:|----:|----:|-----:|-----------------:|
| MolScribe | — | — | — | — | — |
| MolVec | — | — | — | — | — |

**Exact match, stereo-stripped (%)**

| Tool | USPTO | UOB | JPO | CLEF | MolRecBench-Wild |
|:-----|------:|----:|----:|-----:|-----------------:|
| MolScribe | — | — | — | — | — |
| MolVec | — | — | — | — | — |

**Valid output and speed, all datasets pooled**

| Tool | Valid output (%) | s / crop |
|:-----|-----------------:|---------:|
| MolScribe | — | — |
| MolVec | — | — |

## Type 2: Complete systems

### Setup

| System | Detect | Recognize | Extra | Hardware | Version |
|:-------|:-------|:----------|:------|:---------|:--------|
| [OpenChemIE](https://example.org/oc) | MolDet | [MolScribe](https://example.org/molscribe) | Coreference | Explorer GPU | — |
"""


def make_run(
    results: Path,
    tool: str,
    dataset: str,
    *,
    correct: int = 9,
    stripped: int = 10,
    valid: int = 9,
    trials: int = 10,
    seconds: float = 1.0,
    folder: str | None = None,
    version: str = "1.1.1 (7296a30)",
    hardware: dict | None = None,
) -> None:
    folder_path = results / (folder or f"{tool}__{dataset}")
    folder_path.mkdir(parents=True)
    predictions = folder_path / "predictions.csv"
    predictions.write_text("dataset,item_id,smiles,page,bbox,confidence,tool,seconds\n")
    scores = {
        "kind": "crop",
        "items_scored": trials,
        "accuracy": proportion(correct, trials),
        "accuracy_stereo_stripped": proportion(stripped, trials),
        "valid_output_rate": proportion(valid, trials),
        "seconds_per_item": {"mean": seconds / trials, "median": seconds / trials, "items": trials, "total": seconds},
    }
    report = build_report(scores, dataset=dataset, tool=tool, predictions_path=predictions, inputs={})
    write_scores(folder_path / "scores.json", report)
    hardware = hardware or {"gpus": [{"name": "NVIDIA H200"}], "cpus": 8, "cluster": "explorer"}
    meta = {"tool": {"version": version}, "hardware": hardware}
    (folder_path / "meta.json").write_text(json.dumps(meta))


def row(markdown: str, table: int, tool: str) -> str:
    lines = [line for line in markdown.splitlines() if line.startswith(f"| {tool}") or line.startswith(f"| [{tool}]")]
    return lines[table]


def test_one_run_fills_one_cell(tmp_path):
    make_run(tmp_path, "molscribe", "uspto", correct=9, stripped=10)
    out = fill_type1(DOC, load_runs(tmp_path))
    assert row(out, 1, "MolScribe") == "| MolScribe | 90.0 | — | — | — | — |"
    assert row(out, 2, "MolScribe") == "| MolScribe | 100.0 | — | — | — | — |"
    assert row(out, 3, "MolScribe") == "| MolScribe | — | — |"


def test_pooled_row_needs_all_five_datasets(tmp_path):
    for dataset in DATASETS[:4]:
        make_run(tmp_path, "molscribe", dataset, valid=9, seconds=1.0)
    assert row(fill_type1(DOC, load_runs(tmp_path)), 3, "MolScribe") == "| MolScribe | — | — |"
    make_run(tmp_path, "molscribe", DATASETS[4], valid=5, seconds=3.0)
    assert row(fill_type1(DOC, load_runs(tmp_path)), 3, "MolScribe") == "| MolScribe | 82.0 | 0.140 |"


def test_setup_row_gets_hardware_and_version_only_for_tools_with_runs(tmp_path):
    make_run(tmp_path, "molscribe", "uspto")
    out = fill_type1(DOC, load_runs(tmp_path))
    assert row(out, 0, "MolScribe") == (
        "| [MolScribe](https://example.org/molscribe) | Swin Transformer | Graph → SMILES"
        " | Explorer, NVIDIA H200 | 1.1.1 (7296a30) |"
    )
    assert row(out, 0, "MolVec") == "| [MolVec](https://example.org/molvec) | Rule-based vectorization | Molfile → SMILES | Local CPU | — |"
    assert out.split("## Type 2")[1] == DOC.split("## Type 2")[1]


def test_differing_runs_are_joined_and_cpu_runs_name_their_cores(tmp_path):
    make_run(tmp_path, "molvec", "uspto", version="0.9", hardware={"gpus": [], "cpus": 8, "cluster": "explorer"})
    make_run(tmp_path, "molvec", "uob", version="1.0", hardware={"gpus": [], "cpus": 4, "cluster": "explorer"})
    out = fill_type1(DOC, load_runs(tmp_path))
    assert row(out, 0, "MolVec").endswith("| Explorer, CPU, 8 cores / Explorer, CPU, 4 cores | 0.9 / 1.0 |")


def test_no_runs_changes_nothing(tmp_path):
    assert fill_type1(DOC, load_runs(tmp_path)) == DOC


def test_fill_is_idempotent(tmp_path):
    make_run(tmp_path, "molscribe", "uspto")
    once = fill_type1(DOC, load_runs(tmp_path))
    assert fill_type1(once, load_runs(tmp_path)) == once


def test_folder_and_dataset_must_agree(tmp_path):
    make_run(tmp_path, "molscribe", "uspto", folder="molscribe__uob")
    with pytest.raises(ValueError, match="uob"):
        load_runs(tmp_path)


def test_script_reports_missing_runs_and_leaves_an_unchanged_document(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    doc = tmp_path / "Benchmarking.md"
    doc.write_text(DOC, encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_tables.py"
    result = subprocess.run(
        [sys.executable, str(script), "--results", str(results), "--doc", str(doc)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "0 runs" in result.stdout
    assert "molscribe__uspto" in result.stdout
    assert doc.read_text(encoding="utf-8") == DOC
