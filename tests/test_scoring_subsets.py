"""BioVista dev/heldout groups and scoring a run on a subset of a paper dataset's papers."""

import json
import shutil
from pathlib import Path

import pytest

from molscout.cli import main
from molscout.data.biovista_dev import write_dev_manifest
from molscout.runs import score_run

FIXTURES = Path(__file__).parent / "fixtures"
BIOVISTA = FIXTURES / "biovista"
HANDMADE = FIXTURES / "handmade"
HEADER = "dataset,item_id,smiles,page,bbox,confidence,tool,seconds"


def predictions(directory, dataset, rows):
    path = directory / "predictions.csv"
    lines = [HEADER] + [f"{dataset},{paper},{smiles},1,,,T 1,1" for paper, smiles in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def biovista_with_dev(tmp_path):
    """The BioVista fixture with a dev manifest beside its papers manifest: 1_aaaa and 3_cccc are dev."""
    root = tmp_path / "biovista"
    shutil.copytree(BIOVISTA, root)
    from molscout.data.biovista_dev import DevPaper

    papers = (DevPaper("1_aaaa", "aaaa", "drawn", 3, 2, 1), DevPaper("3_cccc", "cccc", "table", 1, 0, 1))
    write_dev_manifest(root / "biovista_dev.csv", papers)
    return root


BIOVISTA_ROWS = (("1_aaaa", "CCO"), ("1_aaaa", "CCCCl"), ("2_bbbb", "CCN"), ("3_cccc", "CCCl"), ("3_cccc", "CO"))


def test_biovista_reports_dev_and_heldout_with_a_dev_manifest_beside_the_papers(tmp_path, biovista_with_dev):
    report = score_run(
        predictions(tmp_path, "biovista", BIOVISTA_ROWS),
        references=biovista_with_dev,
        papers=biovista_with_dev / "manifest.csv",
    )
    groups = report["scores"]["groups"]
    assert set(groups) == {"all", "without_submitted", "dev", "heldout"}
    assert groups["dev"]["papers"] == ["1_aaaa", "3_cccc"] and groups["heldout"]["papers"] == ["2_bbbb"]
    assert groups["dev"]["counts"] == {"tp": 3, "fp": 1, "fn": 1}
    assert set(report["scores"]["drawn_only"]["groups"]) == {"all", "without_submitted", "dev", "heldout"}
    dev = report["inputs"]["references"]["dev"]
    assert dev["papers"] == 2 and len(dev["sha256"]) == 64 and dev["manifest"].endswith("biovista_dev.csv")


def test_biovista_without_a_dev_manifest_keeps_two_groups(tmp_path):
    report = score_run(
        predictions(tmp_path, "biovista", BIOVISTA_ROWS), references=BIOVISTA, papers=BIOVISTA / "manifest.csv"
    )
    assert set(report["scores"]["groups"]) == {"all", "without_submitted"}
    assert "dev" not in report["inputs"]["references"]


def test_biovista_subset_scores_only_its_papers_and_drops_empty_groups(tmp_path, biovista_with_dev):
    rows = tuple(row for row in BIOVISTA_ROWS if row[0] != "2_bbbb")
    report = score_run(
        predictions(tmp_path, "biovista", rows),
        references=biovista_with_dev,
        papers=biovista_with_dev / "manifest.csv",
        subset={"1_aaaa", "3_cccc"},
    )
    scores = report["scores"]
    assert list(scores["papers"]) == ["1_aaaa", "3_cccc"]
    assert set(scores["groups"]) == {"all", "without_submitted", "dev"}  # heldout is empty in the subset
    assert scores["groups"]["all"]["counts"] == scores["groups"]["dev"]["counts"]
    assert list(scores["drawn_only"]["papers"]) == ["1_aaaa"]
    assert report["inputs"]["subset"] == ["1_aaaa", "3_cccc"]


def test_a_prediction_outside_the_subset_is_an_error(tmp_path, biovista_with_dev):
    with pytest.raises(ValueError, match="2_bbbb"):
        score_run(
            predictions(tmp_path, "biovista", BIOVISTA_ROWS),
            references=biovista_with_dev,
            papers=biovista_with_dev / "manifest.csv",
            subset={"1_aaaa", "3_cccc"},
        )


def test_a_subset_paper_must_be_scored(tmp_path):
    with pytest.raises(ValueError, match="9_zzzz"):
        score_run(
            predictions(tmp_path, "biovista", BIOVISTA_ROWS[:1]),
            references=BIOVISTA,
            papers=BIOVISTA / "manifest.csv",
            subset={"1_aaaa", "9_zzzz"},
        )


def test_internal_subset_keeps_the_dev_split(tmp_path):
    rows = (("1", "CCO"), ("4", "CCN"))
    report = score_run(
        predictions(tmp_path, "internal", rows),
        ground_truth=HANDMADE / "papers_ground_truth.csv",
        split=HANDMADE / "papers_split.csv",
        subset={"1", "4"},
    )
    assert list(report["scores"]["papers"]) == ["1", "4"]
    assert set(report["scores"]["groups"]) == {"dev", "all"}
    assert report["scores"]["groups"]["dev"]["papers"] == ["1", "4"]
    assert report["inputs"]["subset"] == ["1", "4"]


def test_cli_subset_reads_paper_ids(tmp_path, biovista_with_dev):
    rows = tuple(row for row in BIOVISTA_ROWS if row[0] != "2_bbbb")
    subset = tmp_path / "subset.txt"
    subset.write_text("1_aaaa\n3_cccc\n")
    out = tmp_path / "scores.json"
    args = ["score", str(predictions(tmp_path, "biovista", rows)), "-o", str(out), "--references"]
    args += [str(biovista_with_dev), "--papers", str(biovista_with_dev / "manifest.csv"), "--subset", str(subset)]
    assert main(args) == 0
    assert json.loads(out.read_text())["inputs"]["subset"] == ["1_aaaa", "3_cccc"]
