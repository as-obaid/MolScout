import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from molscout.predictions import Prediction, write_predictions
from molscout.runs import score_run
from molscout.scoring import write_scores
from molscout.tables import fill_type1, load_runs
from molscout.tables_papers import crash_warnings, fill_type2, git_warnings, load_paper_runs, missing_paper_runs

from test_tables import DOC as TYPE1_DOC, H200, real_git, real_hardware

FIXTURES = Path(__file__).parent / "fixtures"
TRUTH = "paperID,name of molecule,canonical_SMILES\np1,ethanol,CCO\np1,benzene,c1ccccc1\np2,ethylamine,CCN\n"
SPLIT = "paperID,split,molecules\np1,dev,2\np2,test,1\n"

DOC = TYPE1_DOC
DASH_ROW = "| — | — | — | — | — | — | — |"


def predictions(path: Path, dataset: str, tool: str, answers: dict[str, list[str]], seconds: float = 2.0) -> None:
    rows = [
        Prediction(dataset, paper, smiles, None, None, None, tool, seconds)
        for paper, found in answers.items()
        for smiles in found
    ]
    write_predictions(path, rows)


def make_paper_run(
    results: Path, tool: str, dataset: str, *, gpus: int = 1, version: str = "1.0 (abc1234)", folder: str | None = None
) -> Path:
    folder_path = results / (folder or f"{tool}__{dataset}")
    folder_path.mkdir(parents=True)
    path = folder_path / "predictions.csv"
    if dataset == "biovista":
        answers = {"1_aaaa": ["CCO", "c1ccccc1", "CCCCl", "CCC"], "2_bbbb": ["CCN"]}
        seconds = {"1_aaaa": 2.0, "2_bbbb": 4.0, "3_cccc": 6.0}
        predictions(path, dataset, tool, answers)
        report = score_run(
            path, references=FIXTURES / "biovista", papers=FIXTURES / "biovista" / "manifest.csv", paper_seconds=seconds
        )
    else:
        (results / "truth.csv").write_text(TRUTH)
        (results / "split.csv").write_text(SPLIT)
        predictions(path, dataset, tool, {"p1": ["CCO"], "p2": ["CCN", "CCC"]})
        report = score_run(path, ground_truth=results / "truth.csv", split=results / "split.csv")
    write_scores(folder_path / "scores.json", report)
    meta = {"tool": {"version": version}, "hardware": real_hardware({"gpus": [H200] * gpus}), "git": real_git(), "sources": []}
    (folder_path / "meta.json").write_text(json.dumps(meta))
    return folder_path


def row(markdown: str, table: int, label: str) -> str:
    lines = [ln for ln in markdown.splitlines() if ln.startswith(f"| {label} ") or ln.startswith(f"| [{label}]")]
    return lines[table]


def test_biovista_run_fills_the_three_biovista_tables_only(tmp_path):
    make_paper_run(tmp_path, "biominer", "biovista")
    scores = json.loads((tmp_path / "biominer__biovista" / "scores.json").read_text())["scores"]
    out = fill_type2(DOC, load_paper_runs(tmp_path))
    pct = lambda p: f"{p['value'] * 100:.1f}"  # noqa: E731
    group = scores["groups"]["all"]
    refs = json.loads((tmp_path / "biominer__biovista" / "scores.json").read_text())["inputs"]["references"]
    expected = [
        pct(group["micro"]["precision"]),
        pct(group["micro"]["recall"]),
        pct(group["micro"]["f1"]),
        pct(group["macro"]["precision"]),
        pct(group["macro"]["recall"]),
        pct(group["stereo_stripped"]["precision"]),
        pct(group["stereo_stripped"]["recall"]),
        str(refs["papers"]),
        "4.0",
    ]
    assert row(out, 1, "BioMiner") == "| BioMiner | " + " | ".join(expected) + " |"
    drawn = scores["drawn_only"]["groups"]["all"]
    assert row(out, 2, "BioMiner").split(" | ")[1] == pct(drawn["micro"]["precision"])
    without = scores["groups"]["without_submitted"]
    assert row(out, 3, "BioMiner").split(" | ")[3] == pct(without["micro"]["f1"])
    assert len(row(out, 2, "BioMiner").split("|")) == 10
    assert row(out, 4, "BioMiner") == "| BioMiner |" + " — |" * 8
    assert row(out, 5, "BioMiner") == "| BioMiner | — | — | — | — | — | — |"
    assert row(out, 1, "OpenChemIE") == "| OpenChemIE | — | — | — | — | — | — | — | — | — |"


def test_published_row_and_other_text_are_unchanged(tmp_path):
    make_paper_run(tmp_path, "biominer", "biovista")
    out = fill_type2(DOC, load_paper_runs(tmp_path))
    assert "| *BioMiner, published* | — | — | *52.8* | — | — | — | — | — | — |" in out
    assert "Drawn papers: 159." in out
    assert out.endswith("| BioMiner | untouched |\n")
    changed = [a for a, b in zip(DOC.splitlines(), out.splitlines()) if a != b]
    assert all(line.startswith("| BioMiner |") or line.startswith("| [BioMiner]") for line in changed)


def test_internal_run_fills_internal_and_split_tables(tmp_path):
    make_paper_run(tmp_path, "openchemie", "internal")
    scores = json.loads((tmp_path / "openchemie__internal" / "scores.json").read_text())["scores"]
    out = fill_type2(DOC, load_paper_runs(tmp_path))
    cells = [c.strip() for c in row(out, 4, "OpenChemIE").strip("|").split("|")]
    assert cells[0] == "OpenChemIE"
    assert cells[1] == f"{scores['groups']['all']['micro']['precision']['value'] * 100:.1f}"
    assert cells[-1] == "2.0"
    assert len(cells) == 9
    split = [c.strip() for c in row(out, 5, "OpenChemIE").strip("|").split("|")]
    dev, test = scores["groups"]["dev"]["micro"], scores["groups"]["test"]["micro"]
    assert split[1:] == [f"{x['value'] * 100:.1f}" for x in (
        dev["precision"], dev["recall"], dev["f1"], test["precision"], test["recall"], test["f1"])]
    assert row(out, 1, "OpenChemIE") == "| OpenChemIE | — | — | — | — | — | — | — | — | — |"


def test_setup_row_shows_hardware_and_version(tmp_path):
    make_paper_run(tmp_path, "openchemie", "biovista", version="1.0 (abc1234)")
    out = fill_type2(DOC, load_paper_runs(tmp_path))
    assert row(out, 0, "OpenChemIE") == (
        "| [OpenChemIE](https://example.org/oc) | MolDet | MolScribe | Coreference | Explorer, NVIDIA H200 | 1.0 (abc1234) |"
    )


def test_two_gpus_are_counted(tmp_path):
    make_paper_run(tmp_path, "biominer", "biovista", gpus=2)
    out = fill_type2(DOC, load_paper_runs(tmp_path))
    assert "| Explorer, 2× NVIDIA H200 | 1.0 (abc1234) |" in row(out, 0, "BioMiner")


def test_differing_runs_join_hardware_and_version(tmp_path):
    make_paper_run(tmp_path, "biominer", "biovista", gpus=2, version="1.0")
    make_paper_run(tmp_path, "biominer", "internal", gpus=1, version="1.1")
    out = fill_type2(DOC, load_paper_runs(tmp_path))
    assert "| Explorer, 2× NVIDIA H200 / Explorer, NVIDIA H200 | 1.0 / 1.1 |" in row(out, 0, "BioMiner")


def test_fill_type2_is_idempotent_and_leaves_type1_alone(tmp_path):
    make_paper_run(tmp_path, "biominer", "biovista")
    make_paper_run(tmp_path, "biominer", "internal")
    once = fill_type2(DOC, load_paper_runs(tmp_path))
    assert fill_type2(once, load_paper_runs(tmp_path)) == once
    assert once.split("## Type 2")[0] == DOC.split("## Type 2")[0]
    assert fill_type1(once, load_runs(tmp_path)) == once


def test_fill_type1_does_not_touch_type2(tmp_path):
    make_paper_run(tmp_path, "biominer", "biovista")
    assert fill_type1(DOC, load_runs(tmp_path)) == DOC


def test_load_runs_ignores_paper_folders(tmp_path):
    make_paper_run(tmp_path, "biominer", "biovista")
    make_paper_run(tmp_path, "decimer_ai", "internal")
    assert load_runs(tmp_path) == {}
    assert sorted(load_paper_runs(tmp_path)) == [("biominer", "biovista"), ("decimer_ai", "internal")]


def test_load_paper_runs_ignores_crop_and_hidden_folders(tmp_path):
    make_paper_run(tmp_path, "biominer", "biovista", folder=".staging-biominer__biovista")
    (tmp_path / "molscribe__uspto").mkdir()
    assert load_paper_runs(tmp_path) == {}


def test_missing_paper_runs_lists_absent_names(tmp_path):
    make_paper_run(tmp_path, "biominer", "biovista")
    missing = missing_paper_runs(load_paper_runs(tmp_path))
    assert "biominer__biovista" not in missing
    assert "biominer__internal" in missing
    assert len(missing) == 5


def test_git_warnings_cover_paper_runs(tmp_path):
    make_paper_run(tmp_path, "biominer", "biovista")
    meta_path = tmp_path / "biominer__biovista" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["git"].update(dirty=True, dirty_paths=["src/x.py"])
    meta_path.write_text(json.dumps(meta))
    assert git_warnings(load_paper_runs(tmp_path)) == ["biominer__biovista ran with uncommitted changes: src/x.py"]


def test_crash_warnings_name_each_run_with_crashed_papers(tmp_path):
    folder = make_paper_run(tmp_path, "biominer", "biovista")
    make_paper_run(tmp_path, "openchemie", "biovista")
    (folder / "errors.json").write_text(json.dumps({"papers": 3, "failed": 2, "errors": {"1_aaaa": "E: x", "2_bbbb": "E: y"}}))
    assert crash_warnings(load_paper_runs(tmp_path)) == [
        "biominer__biovista: 2 of 3 papers crashed; their labels count as missed"
    ]


def test_make_tables_script_warns_about_crashed_papers(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    folder = make_paper_run(results, "biominer", "biovista")
    (folder / "errors.json").write_text(json.dumps({"papers": 3, "failed": 1, "errors": {"1_aaaa": "E: x"}}))
    doc = tmp_path / "Benchmarking.md"
    doc.write_text(DOC)
    script = Path(__file__).parent.parent / "scripts" / "make_tables.py"
    done = subprocess.run(
        [sys.executable, str(script), "--results", str(results), "--doc", str(doc)], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert "biominer__biovista: 1 of 3 papers crashed; their labels count as missed" in done.stderr


def test_make_tables_script_fills_both_types(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    make_paper_run(results, "biominer", "biovista")
    doc = tmp_path / "Benchmarking.md"
    doc.write_text(DOC)
    script = Path(__file__).parent.parent / "scripts" / "make_tables.py"
    done = subprocess.run(
        [sys.executable, str(script), "--results", str(results), "--doc", str(doc)], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert "1 complete-system runs found" in done.stdout
    assert "missing Type 2: " in done.stdout
    assert "biominer__internal" in done.stdout
    assert row(doc.read_text(), 1, "BioMiner").count("—") == 0
    shutil.rmtree(results)


def test_missing_table_anchor_is_an_error():
    broken = DOC.replace("**Internal, by split**", "**Internal, split**")
    with pytest.raises(ValueError, match=r"table anchor not found: \*\*Internal, by split\*\*"):
        fill_type2(broken, {})


def test_anchor_without_a_table_is_an_error():
    broken = DOC.replace("**Internal, by split**", "**Internal, by split**\n\nnothing here", 1)
    broken = broken.split("**Internal, by split**")[0] + "**Internal, by split**\n"
    with pytest.raises(ValueError, match="no table after anchor"):
        fill_type2(broken, {})


def test_unknown_row_label_is_an_error_naming_label_and_table():
    broken = DOC.replace("| OpenChemIE | — | — | — | — | — | — | — |\n\n**BioVista, without", "| Decimer.ai | — | — | — | — | — | — | — |\n\n**BioVista, without")
    assert broken != DOC
    with pytest.raises(ValueError, match=r"Decimer\.ai.*\*\*BioVista, drawn structures only\*\*"):
        fill_type2(broken, {})
