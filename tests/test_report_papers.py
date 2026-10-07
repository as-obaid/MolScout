"""The complete-system (Type 2) report: loading paper result folders, refusing inconsistent ones, metrics, tables,
figures, the privacy rule for the Internal set, and an offline end-to-end upload.

Result folders are built on the BioVista fixture (three scored papers) and a hand-made Internal set, and scored by
the real scorer, as the harness would write them. The Internal SMILES are chosen so that none of them also occurs in
BioVista or in any other file, which lets the privacy tests search whole directory trees for them.
"""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from molscout.data.internal import GROUND_TRUTH_PATH, SPLIT_PATH
from molscout.predictions import Prediction, write_predictions
from molscout.report.analysis import InconsistentResults, ResultsError
from molscout.report.paper_analysis import (
    PAPER_METRIC_KEYS,
    check_paper_consistency,
    check_paper_references,
    load_paper_run,
    load_paper_runs,
    molecule_rows,
    paper_metrics,
    paper_result_folders,
    paper_rows,
    view_sizes,
)
from molscout.runs import score_run
from molscout.scoring import write_scores

REPO = Path(__file__).resolve().parents[1]
BIOVISTA = Path(__file__).parent / "fixtures" / "biovista"
COMMIT = "4cbaee3e2d4abdb94afaac495b0922ecf3c159fc"
SOURCE_COMMIT = "5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f80"
NAMES = {"biominer": "BioMiner", "decimer_ai": "DECIMER.ai", "openchemie": "OpenChemIE"}
BIOVISTA_PAPERS = ("1_aaaa", "2_bbbb", "3_cccc")
INTERNAL_PAPERS = ("p1", "p2")
# Labels: 1_aaaa drawn CCO and benzene, enumerated CCCCl; 2_bbbb drawn CCN; 3_cccc enumerated CCCl (7 rows, 5 readable).
BIOMINER_BIOVISTA = {
    "1_aaaa": ["CCO", "OCC", "c1ccccc1", "CCC", "C("],
    "2_bbbb": ["CCN", "CCO"],
}
OPENCHEMIE_BIOVISTA = {"1_aaaa": ["CCO", "CCCCl", "c1ccccc1"], "2_bbbb": ["CCN"], "3_cccc": ["CCCl"]}
# Internal truth: p1 (dev) aspirin and naphthalene; p2 (test) hexane. Raw outputs are written differently from canonical.
ASPIRIN, NAPHTHALENE, HEXANE, HEPTANE = "CC(=O)Oc1ccccc1C(=O)O", "c1ccc2ccccc2c1", "CCCCCC", "CCCCCCC"
INTERNAL_TRUTH = (
    "paperID,name of molecule,canonical_SMILES\n"
    f"p1,aspirin,{ASPIRIN}\np1,naphthalene,{NAPHTHALENE}\np2,hexane,{HEXANE}\n"
)
INTERNAL_SPLIT = "paperID,split,molecules\np1,dev,2\np2,test,1\n"
BIOMINER_INTERNAL = {"p1": ["OC(=O)c1ccccc1OC(C)=O"], "p2": ["C" * 6, HEPTANE]}
OPENCHEMIE_INTERNAL = {"p1": ["OC(=O)c1ccccc1OC(C)=O", "C1=CC2=CC=CC=C2C=C1"], "p2": [HEXANE]}
INTERNAL_SMILES = (ASPIRIN, NAPHTHALENE, HEXANE, HEPTANE, "OC(=O)c1ccccc1OC(C)=O", "C1=CC2=CC=CC=C2C=C1")
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
SOURCE = {"path": "/store/src/biominer", "commit": SOURCE_COMMIT, "dirty": False, "dirty_paths": [], "untracked": 2}


@pytest.fixture
def repo(tmp_path):
    """A repository root with the BioVista fixture and the Internal ground truth where the scorer looks for them."""
    root = tmp_path / "repo"
    shutil.copytree(BIOVISTA / "bioactivity_extraction", root / "data" / "raw" / "biovista" / "bioactivity_extraction")
    (root / "data" / "manifests").mkdir(parents=True)
    shutil.copy(BIOVISTA / "manifest.csv", root / "data" / "manifests" / "biovista_papers.csv")
    (root / GROUND_TRUTH_PATH).parent.mkdir(parents=True)
    (root / GROUND_TRUTH_PATH).write_text(INTERNAL_TRUTH)
    (root / SPLIT_PATH).write_text(INTERNAL_SPLIT)
    return root


def make_result(
    repo, tool, dataset, answers, *, commit=COMMIT, dirty=False, sources=(SOURCE,), crashed=(), gpus=1, resources=RESOURCES
):
    """A `<tool>__<dataset>` result folder as the harness writes it for a paper dataset, scored by the real scorer."""
    folder = repo / "benchmarks" / "results" / f"{tool}__{dataset}"
    folder.mkdir(parents=True)
    papers = BIOVISTA_PAPERS if dataset == "biovista" else INTERNAL_PAPERS
    seconds = {paper: 10.0 + 5 * n for n, paper in enumerate(papers)}
    label = f"{NAMES[tool]} 1.0 (test)"
    rows = [
        Prediction(dataset, paper, smiles, 1, None, None, label, seconds[paper])
        for paper in papers
        for smiles in answers.get(paper, [])
    ]
    write_predictions(folder / "predictions.csv", rows)
    (folder / "timing.json").write_text(json.dumps({"papers": len(papers), "seconds": seconds}))
    if dataset == "biovista":
        report = score_run(
            folder / "predictions.csv",
            references=repo / "data/raw/biovista",
            papers=repo / "data/manifests/biovista_papers.csv",
            paper_seconds=seconds,
        )
    else:
        report = score_run(
            folder / "predictions.csv", ground_truth=repo / GROUND_TRUTH_PATH, split=repo / SPLIT_PATH, paper_seconds=seconds
        )
    write_scores(folder / "scores.json", report)
    meta = {
        "run": f"{tool}__{dataset}",
        "tool": {
            "tool": tool,
            "name": NAMES[tool],
            "version": "1.0 (test)",
            "checkpoints": [{"path": f"/store/models/{tool}/model.pth", "sha256": "c" * 64}],
        },
        "dataset": dataset,
        "items": len(papers),
        "tool_errors": len(crashed),
        "inputs": {"pdfs": "/scratch/pdfs", "pdfs_sha256": "d" * 64, "papers": "/scratch/papers.csv", "papers_sha256": "f" * 64},
        "git": {"commit": commit, "dirty": dirty, "dirty_paths": ["src/molscout/x.py"] if dirty else []},
        "sources": [dict(source) for source in sources],
        "environment": {"sha256": "e" * 64},
        "hardware": {
            "cpu_model": "Test CPU",
            "gpus": [{"name": "NVIDIA H200", "memory": "143771 MiB", "driver": "570"}] * gpus,
        },
        "slurm": {"job": "1", "partition": "gpu-short"},
        "timing": {"item_seconds_total": sum(seconds.values())},
        **({"resources": resources} if resources is not None else {}),
    }
    (folder / "meta.json").write_text(json.dumps(meta))
    (folder / "config.yaml").write_text(f"tool: {tool}\ndataset: {dataset}\nargs:\n- --device\n- cuda\n")
    errors = {"papers": len(papers), "failed": len(crashed), "errors": {p: "RuntimeError: boom" for p in crashed}}
    (folder / "errors.json").write_text(json.dumps(errors))
    return folder


def two_systems(repo, **kwargs):
    for dataset, biominer, openchemie in (
        ("biovista", BIOMINER_BIOVISTA, OPENCHEMIE_BIOVISTA),
        ("internal", BIOMINER_INTERNAL, OPENCHEMIE_INTERNAL),
    ):
        make_result(repo, "biominer", dataset, biominer, gpus=2, **kwargs)
        make_result(repo, "openchemie", dataset, openchemie, **kwargs)
    return repo / "benchmarks" / "results"


# Loading


def test_paper_result_folders_are_the_paper_datasets_only(repo):
    results = two_systems(repo)
    for leftover in ("molscribe__uspto", ".staging-biominer__biovista-1"):
        (results / leftover).mkdir()
        (results / leftover / "scores.json").write_text("{}")
    (results / "decimer_ai__biovista").mkdir()  # no scores.json yet
    assert [folder.name for folder in paper_result_folders(results)] == [
        "biominer__biovista",
        "biominer__internal",
        "openchemie__biovista",
        "openchemie__internal",
    ]


def test_load_paper_runs_orders_systems_then_datasets(repo):
    make_result(repo, "openchemie", "biovista", OPENCHEMIE_BIOVISTA)
    make_result(repo, "decimer_ai", "internal", OPENCHEMIE_INTERNAL)
    make_result(repo, "biominer", "internal", BIOMINER_INTERNAL)
    make_result(repo, "decimer_ai", "biovista", OPENCHEMIE_BIOVISTA)
    runs = load_paper_runs(repo / "benchmarks" / "results")
    assert [run.name for run in runs] == [
        "biominer__internal",
        "decimer_ai__biovista",
        "decimer_ai__internal",
        "openchemie__biovista",
    ]
    assert runs[0].timing == {"p1": 10.0, "p2": 15.0} and runs[0].tool == "biominer" and runs[0].errors == {}


def test_load_paper_run_refuses_a_folder_without_timing_or_with_other_papers(repo):
    folder = make_result(repo, "biominer", "biovista", BIOMINER_BIOVISTA)
    (folder / "timing.json").write_text(json.dumps({"papers": 2, "seconds": {"1_aaaa": 1.0, "2_bbbb": 2.0}}))
    with pytest.raises(ResultsError, match="timing.json.*3_cccc"):
        load_paper_run(folder)
    (folder / "timing.json").unlink()
    with pytest.raises(ResultsError, match="missing timing.json"):
        load_paper_run(folder)
    (folder / "timing.json").write_text("not json")
    with pytest.raises(ResultsError, match="timing.json"):
        load_paper_run(folder)


def test_load_paper_run_makes_the_same_checks_as_a_crop_run(repo):
    folder = make_result(repo, "biominer", "biovista", BIOMINER_BIOVISTA)
    with (folder / "predictions.csv").open("a") as handle:
        handle.write('biovista,1_aaaa,CCCCC,,,,"BioMiner 1.0 (test)",1.0\n')
    with pytest.raises(ResultsError, match="another predictions.csv"):
        load_paper_run(folder)
    (folder / "config.yaml").unlink()
    with pytest.raises(ResultsError, match="missing config.yaml"):
        load_paper_run(folder)


def test_load_paper_run_refuses_a_crop_dataset_folder(repo, tmp_path):
    folder = tmp_path / "biominer__uspto"
    folder.mkdir()
    for name in ("predictions.csv", "scores.json", "meta.json", "config.yaml", "timing.json"):
        (folder / name).write_text("{}")
    with pytest.raises(ResultsError):
        load_paper_run(folder)


# Consistency


def test_consistent_runs_have_no_problems(repo):
    runs = load_paper_runs(two_systems(repo))
    assert check_paper_consistency(runs) == []
    assert check_paper_references(runs, repo) == []


def test_consistency_refuses_mixed_commits_uncommitted_code_and_dirty_sources(repo):
    make_result(repo, "biominer", "biovista", BIOMINER_BIOVISTA, dirty=True)
    make_result(repo, "openchemie", "biovista", OPENCHEMIE_BIOVISTA, commit="0" * 40)
    dirty_source = {**SOURCE, "dirty": True, "dirty_paths": ["setup.py", "a.py"]}
    make_result(repo, "decimer_ai", "biovista", OPENCHEMIE_BIOVISTA, sources=[SOURCE, dirty_source])
    problems = check_paper_consistency(load_paper_runs(repo / "benchmarks" / "results"))
    assert problems == [
        "biominer__biovista ran with uncommitted changes: src/molscout/x.py",
        "decimer_ai__biovista ran with uncommitted upstream code in biominer: setup.py, a.py",
        f"the runs come from 2 commits: {COMMIT[:12]} (biominer__biovista, decimer_ai__biovista); "
        "000000000000 (openchemie__biovista)",
    ]


def test_consistency_refuses_runs_that_do_not_record_their_commit(repo):
    folder = make_result(repo, "biominer", "biovista", BIOMINER_BIOVISTA, commit=None)
    meta = json.loads((folder / "meta.json").read_text())
    meta["git"]["dirty"] = None
    (folder / "meta.json").write_text(json.dumps(meta))
    assert check_paper_consistency(load_paper_runs(repo / "benchmarks" / "results")) == [
        "biominer__biovista does not record whether its code was committed",
        "no git commit recorded for biominer__biovista",
    ]


def test_consistency_refuses_runs_scored_against_different_references(repo):
    results = two_systems(repo)
    for name, key in (("openchemie__biovista", "references"), ("openchemie__internal", "ground_truth_sha256")):
        path = results / name / "scores.json"
        report = json.loads(path.read_text())
        if key == "references":
            report["inputs"]["references"]["sha256"] = "f" * 64
        else:
            report["inputs"][key] = "f" * 64
        path.write_text(json.dumps(report))
    problems = check_paper_consistency(load_paper_runs(results))
    assert len(problems) == 2
    assert problems[0].startswith("biovista: the tools were scored against 2 reference sets: ")
    assert problems[0].endswith("ffffffffffff (openchemie)")
    assert problems[1].startswith("internal: the tools were scored against 2 reference sets: ")


def test_references_that_changed_since_scoring_are_reported(repo):
    runs = load_paper_runs(two_systems(repo))
    label = repo / "data/raw/biovista/bioactivity_extraction/labels/3_cccc_structure.csv"
    label.write_text(label.read_text().replace("CCCl", "CCCBr"))
    (repo / GROUND_TRUTH_PATH).write_text(INTERNAL_TRUTH.replace("aspirin", "salicylate"))
    biovista, internal = check_paper_references(runs, repo)
    assert biovista.startswith("biovista: ") and biovista.endswith(
        "than biominer__biovista, openchemie__biovista was scored against"
    )
    assert internal.startswith("internal: ") and "ground truth" in internal and "biominer__internal" in internal


def test_a_missing_reference_file_is_a_problem_not_a_crash(repo):
    runs = load_paper_runs(two_systems(repo))
    (repo / SPLIT_PATH).unlink()
    shutil.rmtree(repo / "data/raw/biovista")
    biovista, internal = check_paper_references(runs, repo)
    assert biovista.startswith("biovista: cannot read the references under ")
    assert internal.startswith("internal: cannot read ") and "internal_split.csv" in internal


# Metrics and tables


def approx(value):
    return pytest.approx(value, abs=1e-9)


def test_biovista_metrics_match_hand_computed_values(repo):
    runs = load_paper_runs(two_systems(repo))
    biominer = paper_metrics(runs[0])
    assert runs[0].name == "biominer__biovista"
    # All labels: 1_aaaa tp 2 fp 2 (CCC, C( ) fn 1; 2_bbbb tp 1 fp 1; 3_cccc fn 1. Pooled tp 3, fp 3, fn 2.
    assert (biominer["counts/tp"], biominer["counts/fp"], biominer["counts/fn"]) == (3, 3, 2)
    assert biominer["micro/precision"] == approx(0.5) and biominer["micro/recall"] == approx(0.6)
    assert biominer["micro/f1"] == approx(3 / 5.5)
    assert 0 < biominer["micro/precision_ci_low"] < 0.5 < biominer["micro/precision_ci_high"] < 1
    assert biominer["micro/f1_ci_low"] < biominer["micro/f1"] < biominer["micro/f1_ci_high"]
    assert biominer["macro/precision"] == approx(1 / 3) and biominer["macro/recall"] == approx(5 / 9)
    assert "macro/precision_ci_low" in biominer and "macro/recall_ci_high" in biominer
    assert biominer["stripped/precision"] == approx(0.5) and biominer["stripped/f1"] == approx(3 / 5.5)
    assert biominer["valid_output_rate"] == approx(6 / 7)  # 7 rows, "C(" invalid
    assert (biominer["speed/s_per_paper_mean"], biominer["speed/s_per_paper_median"]) == (15.0, 15.0)
    assert biominer["resources/gpu_peak_memory_gib"] == 6.0 and biominer["resources/peak_rss_gib"] == 3.0
    assert biominer["resources/cpu_seconds"] == 41.5 and biominer["resources/gpu_mean_utilization_pct"] == 37.5
    assert (biominer["items/papers"], biominer["items/molecules"]) == (3, 5)
    assert (biominer["items/papers_without_output"], biominer["items/crashed"]) == (1, 0)
    # Drawn only (1_aaaa, 2_bbbb): nothing ignored, tp 3 fp 3 fn 0.
    assert biominer["drawn/precision"] == approx(0.5) and biominer["drawn/recall"] == approx(1.0)
    assert biominer["drawn/f1"] == approx(3 / 4.5)
    assert biominer["drawn/macro_precision"] == approx(0.5) and biominer["drawn/macro_recall"] == approx(1.0)
    # Without the submitted version (1_aaaa, 3_cccc): tp 2 fp 2 fn 2.
    assert biominer["without_submitted/precision"] == approx(0.5)
    assert biominer["without_submitted/recall"] == approx(0.5) and biominer["without_submitted/f1"] == approx(0.5)
    assert not any(key.startswith(("dev/", "test/")) for key in biominer)
    assert set(biominer) <= set(PAPER_METRIC_KEYS)

    openchemie = paper_metrics(runs[2])
    assert runs[2].name == "openchemie__biovista"
    assert (openchemie["micro/precision"], openchemie["micro/recall"], openchemie["micro/f1"]) == (1.0, 1.0, 1.0)
    assert openchemie["items/papers_without_output"] == 0
    assert openchemie["drawn/precision"] == 1.0  # CCCCl is an enumerated label: ignored in the drawn-only view


def test_internal_metrics_report_dev_test_and_all_papers(repo):
    runs = load_paper_runs(two_systems(repo))
    biominer = paper_metrics(runs[1])
    assert runs[1].name == "biominer__internal"
    # p1 (dev): tp 1 fn 1; p2 (test): tp 1 fp 1. All: tp 2 fp 1 fn 1.
    assert (biominer["counts/tp"], biominer["counts/fp"], biominer["counts/fn"]) == (2, 1, 1)
    assert biominer["micro/precision"] == approx(2 / 3) and biominer["micro/f1"] == approx(2 / 3)
    assert (biominer["dev/precision"], biominer["dev/recall"], biominer["dev/f1"]) == (1.0, 0.5, approx(2 / 3))
    assert (biominer["test/precision"], biominer["test/recall"], biominer["test/f1"]) == (0.5, 1.0, approx(2 / 3))
    assert biominer["items/papers"] == 2 and biominer["items/molecules"] == 3
    assert not any(key.startswith(("drawn/", "without_submitted/")) for key in biominer)


def test_a_run_without_resources_leaves_those_keys_out_and_counts_crashed_papers(repo):
    make_result(repo, "biominer", "biovista", BIOMINER_BIOVISTA, resources=None, crashed=("3_cccc",))
    [run] = load_paper_runs(repo / "benchmarks" / "results")
    metrics = paper_metrics(run)
    assert not any(key.startswith("resources/") for key in metrics) and metrics["items/crashed"] == 1


def test_paper_rows_hold_every_paper_with_its_drawn_counts(repo):
    runs = load_paper_runs(two_systems(repo))
    columns, rows = paper_rows(runs)
    assert columns == [
        "system",
        "dataset",
        "paper",
        "molecules",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "seconds",
        "drawn_tp",
        "drawn_fp",
        "drawn_fn",
        "drawn_ignored",
    ]
    assert [(row[0], row[1], row[2]) for row in rows[:5]] == [
        ("BioMiner", "biovista", "1_aaaa"),
        ("BioMiner", "biovista", "2_bbbb"),
        ("BioMiner", "biovista", "3_cccc"),
        ("BioMiner", "internal", "p1"),
        ("BioMiner", "internal", "p2"),
    ]
    assert rows[0][3:] == [3, 2, 2, 1, 0.5, 2 / 3, 10.0, 2, 2, 0, 0]
    assert rows[2][3:] == [1, 0, 0, 1, 0.0, 0.0, 20.0, None, None, None, None]  # 3_cccc has no drawn label
    assert rows[3][3:] == [2, 1, 0, 1, 1.0, 0.5, 10.0, None, None, None, None]  # Internal: no drawn view
    openchemie = [row for row in rows if row[0] == "OpenChemIE" and row[2] == "1_aaaa"]
    assert openchemie[0][10:] == [2, 0, 0, 1]  # CCCCl is ignored in the drawn-only view
    assert sum(row[4] for row in rows if row[:2] == ["BioMiner", "biovista"]) == 3


def test_molecule_rows_list_each_output_and_each_missed_label(repo):
    runs = load_paper_runs(two_systems(repo))
    references = {
        "1_aaaa": ("CCO", "c1ccccc1", "CCCCl"),
        "2_bbbb": ("CCN",),
        "3_cccc": ("CCCl",),
    }
    columns, rows = molecule_rows(runs, references)
    assert columns == ["system", "paper", "smiles", "canonical", "outcome"]
    biominer = [row for row in rows if row[0] == "BioMiner"]
    assert biominer == [
        ["BioMiner", "1_aaaa", "CCO", "CCO", "tp"],  # "OCC" is the same structure: one row
        ["BioMiner", "1_aaaa", "c1ccccc1", "c1ccccc1", "tp"],
        ["BioMiner", "1_aaaa", "CCC", "CCC", "fp"],
        ["BioMiner", "1_aaaa", "C(", None, "invalid"],
        ["BioMiner", "1_aaaa", "CCCCl", "CCCCl", "fn"],
        ["BioMiner", "2_bbbb", "CCN", "CCN", "tp"],
        ["BioMiner", "2_bbbb", "CCO", "CCO", "fp"],
        ["BioMiner", "3_cccc", "CCCl", "CCCl", "fn"],
    ]
    assert {row[0] for row in rows} == {"BioMiner", "OpenChemIE"}  # BioVista runs only, never Internal
    counts = {outcome: sum(row[4] == outcome for row in biominer) for outcome in ("tp", "fp", "invalid", "fn")}
    assert counts == {"tp": 3, "fp": 2, "invalid": 1, "fn": 2}  # equal to scores.json: tp 3, fp 3 (with the invalid), fn 2


# The W&B payload


@pytest.fixture
def papers_publisher():
    """molscout.report.wandb_papers; skips where the report extra (wandb, plotly) is not installed."""
    pytest.importorskip("wandb")
    pytest.importorskip("plotly")
    from molscout.report import wandb_papers

    return wandb_papers


@pytest.fixture
def figures():
    pytest.importorskip("plotly")
    from molscout.report import paper_figures

    return paper_figures


def test_prepare_refuses_inconsistent_results_and_lets_development_override(repo, papers_publisher):
    results = two_systems(repo)
    make_result(repo, "decimer_ai", "biovista", OPENCHEMIE_BIOVISTA, commit="0" * 40)
    with pytest.raises(InconsistentResults, match="2 commits"):
        papers_publisher.prepare_papers(results, repo)
    warnings = []
    benchmark = papers_publisher.prepare_papers(results, repo, allow_inconsistent=True, warn=warnings.append)
    assert len(benchmark.runs) == 5 and warnings and all(w.startswith("allowed for development: ") for w in warnings)


def test_prepare_refuses_a_folder_that_holds_no_runs(tmp_path, repo, papers_publisher):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ResultsError, match="no paper-dataset result folders"):
        papers_publisher.prepare_papers(tmp_path / "empty", repo)


def test_run_config_adds_gpu_count_and_sources(repo, papers_publisher):
    benchmark = papers_publisher.prepare_papers(two_systems(repo), repo)
    biominer = benchmark.runs[0]
    assert papers_publisher.run_id(biominer) == f"biominer-biovista-{biominer.predictions_sha256[:10]}"
    assert benchmark.configs["biominer__biovista"] == {
        "tool": "biominer",
        "name": "BioMiner",
        "version": "1.0 (test)",
        "dataset": "biovista",
        "git_commit": COMMIT,
        "checkpoints": [{"file": "model.pth", "sha256": "c" * 64}],
        "environment_sha256": "e" * 64,
        "device": "NVIDIA H200",
        "slurm_partition": "gpu-short",
        "gpus": 2,
        "sources": [{"repo": "biominer", "commit": SOURCE_COMMIT, "dirty": False, "untracked": 2}],
    }
    assert benchmark.configs["openchemie__biovista"]["gpus"] == 1
    assert papers_publisher.summary_run_id(benchmark.runs) == papers_publisher.summary_run_id(benchmark.runs[::-1])
    assert papers_publisher.summary_run_id(benchmark.runs).startswith("summary-complete-systems-")


def test_prepare_refuses_an_incomplete_meta(repo, papers_publisher):
    results = two_systems(repo)
    meta_path = results / "openchemie__biovista" / "meta.json"
    meta = json.loads(meta_path.read_text())
    del meta["slurm"]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ResultsError, match="openchemie__biovista: meta.json lacks 'slurm'"):
        papers_publisher.prepare_papers(results, repo)


def test_tables_hold_the_leaderboard_every_paper_and_the_biovista_molecules(repo, papers_publisher):
    benchmark = papers_publisher.prepare_papers(two_systems(repo), repo)
    tables = papers_publisher.table_rows(benchmark, repo)
    assert sorted(tables) == ["leaderboard", "molecules", "papers"]
    columns, rows = tables["leaderboard"]
    assert columns[:2] == ["system", "dataset"] and columns[2:] == list(PAPER_METRIC_KEYS)
    assert [row[:2] for row in rows] == [
        ["BioMiner", "biovista"],
        ["BioMiner", "internal"],
        ["OpenChemIE", "biovista"],
        ["OpenChemIE", "internal"],
    ]
    assert rows[0][columns.index("micro/f1")] == approx(3 / 5.5) and rows[1][columns.index("drawn/f1")] is None
    assert len(tables["papers"][1]) == 10  # three BioVista papers and two Internal, for two systems
    molecules = tables["molecules"][1]
    assert {row[1] for row in molecules} <= set(BIOVISTA_PAPERS)
    assert sum(row[0] == "BioMiner" and row[4] == "tp" for row in molecules) == 3


def test_molecule_rows_that_disagree_with_scores_json_are_refused(repo, papers_publisher):
    results = two_systems(repo)
    label = repo / "data/raw/biovista/bioactivity_extraction/labels/3_cccc_structure.csv"
    label.write_text(label.read_text().replace("CCCl", "CCCBr"))
    benchmark = papers_publisher.prepare_papers(results, repo, allow_inconsistent=True, warn=lambda _: None)
    with pytest.raises(ResultsError, match="molecule rows disagree with scores.json"):
        papers_publisher.table_rows(benchmark, repo)


def test_internal_smiles_reach_no_table_and_no_artifact(repo, papers_publisher):
    benchmark = papers_publisher.prepare_papers(two_systems(repo), repo)
    for columns, rows in papers_publisher.table_rows(benchmark, repo).values():
        cells = [str(cell) for row in rows for cell in row] + columns
        assert not [smiles for smiles in INTERNAL_SMILES if smiles in cells]
    assert not any(
        row[1] in INTERNAL_PAPERS for row in papers_publisher.table_rows(benchmark, repo)["molecules"][1]
    )  # no Internal paper has molecule rows at all
    for run in benchmark.runs:
        artifact = papers_publisher.artifact_for(run)
        files = sorted(artifact.manifest.entries)
        if run.dataset == "internal":
            assert files == ["config.yaml", "errors.json", "meta.json", "scores.json", "timing.json"]
            for name, entry in artifact.manifest.entries.items():
                text = Path(entry.local_path).read_text()
                assert not [smiles for smiles in INTERNAL_SMILES if smiles in text], name
        else:
            assert files == ["config.yaml", "errors.json", "meta.json", "predictions.csv", "scores.json", "timing.json"]


def test_internal_artifact_keeps_error_types_but_not_error_text(repo, papers_publisher):
    folder = make_result(repo, "biominer", "internal", BIOMINER_INTERNAL, crashed=("p2",))
    errors = {"papers": 2, "failed": 1, "errors": {"p2": f"ValueError: cannot read molecule {ASPIRIN}"}}
    (folder / "errors.json").write_text(json.dumps(errors))
    [run] = load_paper_runs(repo / "benchmarks" / "results")
    artifact = papers_publisher.artifact_for(run)
    kept = json.loads(Path(artifact.manifest.entries["errors.json"].local_path).read_text())
    assert kept == {"papers": 2, "failed": 1, "errors": {"p2": "ValueError"}}


def test_figure_panels_cover_both_datasets_in_both_themes(repo, papers_publisher):
    benchmark = papers_publisher.prepare_papers(two_systems(repo), repo)
    light = papers_publisher.figure_panels(benchmark)
    dark = papers_publisher.figure_panels(benchmark, theme=papers_publisher.paper_figures.DARK)
    assert list(light) == [
        "viz/biovista_views",
        "viz/internal_views",
        "viz/recall_by_paper",
        "viz/f1_vs_time",
    ]
    assert list(dark) == list(light)
    for key in light:
        assert light[key].layout.height == dark[key].layout.height
        papers_publisher.panel_html(light[key], dark[key])  # both themes embed at one height
    assert light["viz/biovista_views"].data[0].marker.color != dark["viz/biovista_views"].data[0].marker.color


def test_biovista_only_results_get_no_internal_figure(repo, papers_publisher):
    make_result(repo, "biominer", "biovista", BIOMINER_BIOVISTA)
    benchmark = papers_publisher.prepare_papers(repo / "benchmarks" / "results", repo)
    assert list(papers_publisher.figure_panels(benchmark)) == [
        "viz/biovista_views",
        "viz/recall_by_paper",
        "viz/f1_vs_time",
    ]


def test_view_bars_show_precision_recall_and_f1_per_system_for_each_view(repo, figures):
    runs = load_paper_runs(two_systems(repo))
    metrics = {(run.tool, run.dataset): paper_metrics(run) for run in runs}
    sizes = view_sizes(runs)
    biovista = figures.view_bars(metrics, "biovista", sizes)
    biominer = [trace for trace in biovista.data if trace.name == "BioMiner"]
    assert len(biominer) == 3  # all labels, drawn only, without submitted: one facet each
    assert list(biominer[0].x) == ["Precision", "Recall", "F1"]
    assert list(biominer[0].y) == [approx(0.5), approx(0.6), approx(3 / 5.5)]
    assert list(biominer[1].y) == [approx(0.5), approx(1.0), approx(3 / 4.5)]
    assert biominer[0].error_y.array is not None and biominer[1].error_y.array is None  # only the first has CIs
    assert [trace.showlegend for trace in biominer] == [True, False, False]
    assert [a.text for a in biovista.layout.annotations[:3]] == [
        "All labels (3 papers)",
        "Drawn structures only (2 papers)",
        "Without submitted versions (2 papers)",
    ]
    internal = figures.view_bars(metrics, "internal", sizes)
    assert [list(t.y) for t in internal.data if t.name == "OpenChemIE"] == [[1.0, 1.0, 1.0]] * 3
    assert [a.text for a in internal.layout.annotations[:3]] == [
        "All papers (2 papers)",
        "Development papers (1 paper)",
        "Test papers (1 paper)",
    ]


def test_recall_dots_draw_one_dot_per_paper_and_system(repo, figures):
    columns, rows = paper_rows(load_paper_runs(two_systems(repo)))
    figure = figures.recall_by_paper((columns, rows))
    [biominer] = [trace for trace in figure.data if trace.name == "BioMiner"]
    assert sorted(biominer.y) == [approx(0.0), approx(2 / 3), approx(1.0)]
    assert len(biominer.y) == len(biominer.x) == 3
    assert "1_aaaa" in "".join(biominer.hovertext)
    assert [trace.name for trace in figure.data] == ["BioMiner", "OpenChemIE"]  # Internal papers are not drawn
    assert figures.recall_by_paper((columns, rows)).to_json() == figure.to_json()  # the jitter is seeded


def test_f1_against_time_has_one_point_per_run(repo, figures):
    runs = load_paper_runs(two_systems(repo))
    metrics = {(run.tool, run.dataset): paper_metrics(run) for run in runs}
    figure = figures.f1_vs_time(metrics)
    points = {(trace.name, trace.legendgroup): (trace.x[0], trace.y[0]) for trace in figure.data}
    assert points[("BioMiner", "biovista")] == (15.0, approx(3 / 5.5))
    assert points[("OpenChemIE", "internal")] == (12.5, 1.0)
    assert len(points) == 4
    assert figure.layout.xaxis.type == "log"


# Publishing


def test_publish_logs_one_html_page_per_figure_and_the_tables(repo, papers_publisher, monkeypatch):
    from unittest.mock import MagicMock

    import wandb

    benchmark = papers_publisher.prepare_papers(two_systems(repo), repo)
    run = MagicMock()
    inits = []
    monkeypatch.setattr(papers_publisher.wandb, "init", lambda **kwargs: inits.append(kwargs) or run)
    papers_publisher.publish_papers(benchmark, entity="e", project="p", repo_root=repo, echo=lambda _: None)
    [logged] = [call.args[0] for call in run.__enter__.return_value.log.call_args_list]
    keys = list(papers_publisher.figure_panels(benchmark))
    assert [key for key in logged if key in keys] == keys
    assert all(isinstance(logged[key], wandb.Html) for key in keys)
    assert all(key.startswith("viz/") for key in keys)
    light = papers_publisher.figure_panels(benchmark)
    dark = papers_publisher.figure_panels(benchmark, theme=papers_publisher.paper_figures.DARK)
    for key in keys:
        assert Path(logged[key]._path).read_text() == papers_publisher.panel_html(light[key], dark[key])
    assert {key for key, value in logged.items() if isinstance(value, wandb.Table)} == {
        "leaderboard",
        "papers",
        "molecules",
    }
    jobs = [(init["job_type"], init["name"], init.get("group")) for init in inits]
    assert jobs[:4] == [
        ("eval", "biominer/biovista", "complete-systems"),
        ("eval", "biominer/internal", "complete-systems"),
        ("eval", "openchemie/biovista", "complete-systems"),
        ("eval", "openchemie/internal", "complete-systems"),
    ]
    assert jobs[4] == ("analysis", "summary-complete-systems", None)


def test_publish_builds_everything_before_the_first_run_starts(repo, papers_publisher, monkeypatch):
    benchmark = papers_publisher.prepare_papers(two_systems(repo), repo)
    started = []
    monkeypatch.setattr(papers_publisher.wandb, "init", lambda **kwargs: started.append(kwargs))

    def refuse(*_args, **_kwargs):
        raise ResultsError("cannot build the tables")

    monkeypatch.setattr(papers_publisher, "table_rows", refuse)
    with pytest.raises(ResultsError, match="cannot build"):
        papers_publisher.publish_papers(benchmark, entity="e", project="p", repo_root=repo, echo=lambda _: None)
    assert started == []


def upload(results, repo, *args):
    spec = importlib.util.spec_from_file_location("wandb_upload", REPO / "scripts" / "wandb_upload.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = ["--results", str(results), "--repo-root", str(repo), "--entity", "test", "--project", "molscout-test", *args]
    return module.main(argv)


def test_the_uploader_defaults_to_the_structure_readers_and_names_its_choices(repo, capsys):
    spec = importlib.util.spec_from_file_location("wandb_upload", REPO / "scripts" / "wandb_upload.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.build_parser().parse_args(["--entity", "e", "--project", "p"]).benchmark == "structure-readers"
    with pytest.raises(SystemExit):
        upload(repo, repo, "--benchmark", "bogus")
    assert "complete-systems" in capsys.readouterr().err


def test_upload_refuses_inconsistent_results_before_touching_wandb(repo, papers_publisher, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb-dir"))
    results = two_systems(repo)
    make_result(repo, "decimer_ai", "biovista", OPENCHEMIE_BIOVISTA, commit="0" * 40)
    assert upload(results, repo, "--benchmark", "complete-systems", "--offline") == 1
    assert "2 commits" in capsys.readouterr().err
    assert not (tmp_path / "wandb-dir").exists()


@pytest.fixture
def fresh_wandb():
    """W&B reads its folders from the environment once per process; start and end this test with a clean slate."""
    import wandb

    wandb.teardown()
    yield
    wandb.teardown()


def test_offline_upload_logs_one_run_per_folder_and_one_summary_and_no_internal_smiles(
    repo, papers_publisher, fresh_wandb, tmp_path, monkeypatch, capsys
):
    from test_report_publish import logged_values

    roots = {name: tmp_path / name for name in ("wandb-dir", "cache", "data", "artifacts")}
    for variable, name in (
        ("WANDB_DIR", "wandb-dir"),
        ("WANDB_CACHE_DIR", "cache"),
        ("WANDB_DATA_DIR", "data"),
        ("WANDB_ARTIFACT_DIR", "artifacts"),
    ):
        roots[name].mkdir()
        monkeypatch.setenv(variable, str(roots[name]))
    monkeypatch.setenv("WANDB_MODE", "offline")
    results = two_systems(repo)
    assert upload(results, repo, "--benchmark", "complete-systems", "--offline") == 0
    assert "4 runs: biominer__biovista, biominer__internal, openchemie__biovista, openchemie__internal" in (
        capsys.readouterr().out
    )
    offline = roots["wandb-dir"] / "wandb"
    assert len(list(offline.glob("offline-run-*"))) == 5
    [eval_dir] = offline.glob("offline-run-*-biominer-biovista-*")
    logged = logged_values(eval_dir)
    assert logged["summary"]["micro/f1"] == approx(3 / 5.5) and logged["summary"]["counts/tp"] == 3
    assert {"drawn/f1", "without_submitted/f1", "speed/s_per_paper_mean", "items/papers"} <= set(logged["summary"])
    assert logged["config"]["tool"] == "biominer" and logged["config"]["gpus"] == 2
    assert logged["config"]["sources"] == [{"repo": "biominer", "commit": SOURCE_COMMIT, "dirty": False, "untracked": 2}]
    assert logged["history"] == {}
    assert logged["run"] == {"tags": ["biominer", "biovista"], "group": "complete-systems", "job_type": "eval"}
    [internal_dir] = offline.glob("offline-run-*-openchemie-internal-*")
    assert {"dev/f1", "test/f1"} <= set(logged_values(internal_dir)["summary"])
    [analysis_dir] = offline.glob("offline-run-*-summary-complete-systems-*")
    analysis = logged_values(analysis_dir)
    assert analysis["run"] == {"tags": [], "group": "", "job_type": "analysis"}
    history = analysis["history"]
    assert {key.removesuffix("/_type") for key, value in history.items() if value == "html-file"} == {
        "viz/biovista_views",
        "viz/internal_views",
        "viz/recall_by_paper",
        "viz/f1_vs_time",
    }
    assert {key.removesuffix("/_type") for key, value in history.items() if value == "table-file"} == {
        "leaderboard",
        "papers",
        "molecules",
    }
    # Nothing under any W&B folder (run logs, tables, artifact files and manifests) names an Internal structure,
    # while the BioVista structures are there, so the search does reach the files.
    contents = b"".join(path.read_bytes() for root in roots.values() for path in root.rglob("*") if path.is_file())
    assert b"CCCCl" in contents
    assert [smiles for smiles in INTERNAL_SMILES if smiles.encode() in contents] == []
