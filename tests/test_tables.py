import json
import subprocess
import sys
from pathlib import Path

import pytest

from molscout.bench.meta import git_state, hardware
from molscout.scoring import build_report, write_scores
from molscout.scoring.stats import proportion
from molscout.tables import DATASETS, fill_type1, git_warnings, load_runs

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
| [BioMiner](https://example.org/bm) | MolDetv2 | MolGlyph | Agents | 4× H200 | — |
| [OpenChemIE](https://example.org/oc) | MolDet | MolScribe | Coreference | Explorer GPU | — |

### Results

**BioVista**

| System | Precision | Recall | F1 | Macro P | Macro R | Stripped P | Stripped R | PDFs | s / paper |
|:-------|----------:|-------:|---:|--------:|--------:|-----------:|-----------:|-----:|----------:|
| BioMiner | — | — | — | — | — | — | — | — | — |
| OpenChemIE | — | — | — | — | — | — | — | — | — |
| *BioMiner, published* | — | — | *52.8* | — | — | — | — | — | — |

**BioVista, drawn structures only**

Drawn papers: 159.

| System | Precision | Recall | F1 | Macro P | Macro R | Stripped P | Stripped R |
|:-------|----------:|-------:|---:|--------:|--------:|-----------:|-----------:|
| BioMiner | — | — | — | — | — | — | — |
| OpenChemIE | — | — | — | — | — | — | — |

**BioVista, without submitted versions**

| System | Precision | Recall | F1 | Macro P | Macro R | Stripped P | Stripped R |
|:-------|----------:|-------:|---:|--------:|--------:|-----------:|-----------:|
| BioMiner | — | — | — | — | — | — | — |
| OpenChemIE | — | — | — | — | — | — | — |

**Internal**

| System | Precision | Recall | F1 | Macro P | Macro R | Stripped P | Stripped R | s / paper |
|:-------|----------:|-------:|---:|--------:|--------:|-----------:|-----------:|----------:|
| BioMiner | — | — | — | — | — | — | — | — |
| OpenChemIE | — | — | — | — | — | — | — | — |

**Internal, by split**

| System | Dev P | Dev R | Dev F1 | Test P | Test R | Test F1 |
|:-------|------:|------:|-------:|-------:|-------:|--------:|
| BioMiner | — | — | — | — | — | — |
| OpenChemIE | — | — | — | — | — | — |

---

## Type 3: AI agent

| BioMiner | untouched |
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
    git: dict | None = None,
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
    meta = {"tool": {"version": version}, "hardware": real_hardware(hardware), "git": real_git(git)}
    (folder_path / "meta.json").write_text(json.dumps(meta))


def real_hardware(values: dict | None = None) -> dict:
    """What molscout.bench.meta.hardware() writes, with `values` swapped in; unknown keys fail the test."""
    values = {"gpus": [H200], "cpus_available": 8, "cluster": "explorer", **(values or {})}
    return swapped(hardware(), values)


def real_git(values: dict | None = None) -> dict:
    """What molscout.bench.meta.git_state() writes, with `values` swapped in; unknown keys fail the test."""
    values = {"commit": COMMIT, "dirty": False, "dirty_paths": [], **(values or {})}
    return swapped(git_state(Path("/"), []), values)


def swapped(record: dict, values: dict) -> dict:
    unknown = sorted(set(values) - set(record))
    assert not unknown, f"meta.json has no {unknown}; the fixture has drifted from molscout.bench.meta"
    return {**record, **values}


H200 = {"name": "NVIDIA H200", "memory": "143771 MiB", "driver": "570.86.15"}
COMMIT = "a" * 40


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
    make_run(tmp_path, "molvec", "uspto", version="0.9", hardware={"gpus": [], "cpus_available": 8})
    make_run(tmp_path, "molvec", "uob", version="1.0", hardware={"gpus": [], "cpus_available": 4})
    out = fill_type1(DOC, load_runs(tmp_path))
    assert row(out, 0, "MolVec").endswith("| Explorer, CPU, 8 cores / Explorer, CPU, 4 cores | 0.9 / 1.0 |")


def test_hardware_without_a_cluster_name_leaves_it_out(tmp_path):
    make_run(tmp_path, "molscribe", "uspto", hardware={"cluster": None})
    make_run(tmp_path, "molvec", "uspto", hardware={"cluster": None, "gpus": [], "cpus_available": 8})
    out = fill_type1(DOC, load_runs(tmp_path))
    assert row(out, 0, "MolScribe").endswith("| NVIDIA H200 | 1.1.1 (7296a30) |")
    assert row(out, 0, "MolVec").endswith("| CPU, 8 cores | 1.1.1 (7296a30) |")


def test_leftover_staging_and_old_folders_are_skipped(tmp_path):
    make_run(tmp_path, "molscribe", "uspto")
    make_run(tmp_path, "molscribe", "uspto", folder=".staging-molscribe__uob-123")
    make_run(tmp_path, "molscribe", "uob", folder=".old-molscribe__uob-456")
    assert list(load_runs(tmp_path)) == [("molscribe", "uspto")]


def test_git_warnings_name_uncommitted_runs_and_mixed_commits(tmp_path):
    make_run(tmp_path, "molscribe", "uspto")
    make_run(tmp_path, "molscribe", "uob", git={"dirty": True, "dirty_paths": ["src/molscout/runs.py"]})
    make_run(tmp_path, "molvec", "uspto", git={"commit": "b" * 40})
    make_run(tmp_path, "molvec", "uob", git={"commit": None, "dirty": None, "dirty_paths": None})
    warnings = git_warnings(load_runs(tmp_path))
    assert warnings == [
        "molscribe__uob ran with uncommitted changes: src/molscout/runs.py",
        "molvec__uob does not record whether its code was committed",
        f"runs come from 3 commits: {'a' * 12} (molscribe__uob, molscribe__uspto); "
        f"unknown (molvec__uob); {'b' * 12} (molvec__uspto)",
    ]


def test_one_clean_commit_gives_no_git_warnings(tmp_path):
    make_run(tmp_path, "molscribe", "uspto")
    make_run(tmp_path, "molvec", "uob")
    assert git_warnings(load_runs(tmp_path)) == []


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
    assert "warning" not in result.stderr
    assert doc.read_text(encoding="utf-8") == DOC


def test_script_warns_about_uncommitted_runs_and_mixed_commits_but_still_writes(tmp_path):
    results = tmp_path / "results"
    make_run(results, "molscribe", "uspto", git={"dirty": True, "dirty_paths": ["pyproject.toml"]})
    make_run(results, "molscribe", "uob", git={"commit": "b" * 40})
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
    assert result.stderr.splitlines() == [
        "make_tables: warning: molscribe__uspto ran with uncommitted changes: pyproject.toml",
        f"make_tables: warning: runs come from 2 commits: {'b' * 12} (molscribe__uob); {'a' * 12} (molscribe__uspto)",
    ]
    assert doc.read_text(encoding="utf-8") != DOC
