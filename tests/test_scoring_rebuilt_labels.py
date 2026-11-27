"""BioVista scored against its labels as loaded, with the rows biovista_truth.repair_label rebuilds.

Paper 7_gggg (test_data_biovista_truth.LABEL_ROWS) has five labels: a naive paste (rebuilt to JOIN), a label that
is the join already, a right label over a broken backbone, one whose group does not parse, and a drawn pyridine.
Its outputs are JOIN, the consistent label and butane: TP 2, FP 1, FN 3, and in the drawn-only view TP 0, FP 1,
FN 1 with JOIN and the consistent label ignored.
"""

import json
from pathlib import Path

import pytest
from test_data_biovista_truth import CONSISTENT, JOIN, labelled_paper

from molscout.cli import main
from molscout.data import biovista_truth as bt
from molscout.predictions import read_predictions
from molscout.runs import score_run
from molscout.scoring import score_papers

FIXTURES = Path(__file__).parent / "fixtures"
BIOVISTA = FIXTURES / "biovista"
HANDMADE = FIXTURES / "handmade"
HEADER = "dataset,item_id,smiles,page,bbox,confidence,tool,seconds"
OUTPUTS = (("7_gggg", JOIN), ("7_gggg", CONSISTENT[0]), ("7_gggg", "CCCC"))


def predictions(directory, rows=OUTPUTS):
    path = directory / "predictions.csv"
    path.write_text("\n".join([HEADER, *(f"biovista,{paper},{smiles},1,,,T 1,1" for paper, smiles in rows)]) + "\n")
    return path


@pytest.fixture
def scored(tmp_path):
    root, manifest = labelled_paper(tmp_path / "biovista")
    path = predictions(tmp_path)
    return score_run(path, references=root, papers=manifest), root, manifest, path


def counts(group):
    return {**group["counts"], **({"ignored": group["ignored_outputs"]} if "ignored_outputs" in group else {})}


def test_biovista_is_scored_once_against_the_labels_with_the_rebuilt_row(scored):
    report, *_ = scored
    scores = report["scores"]
    assert list(scores) == ["kind", "papers", "groups", "drawn_only", "seconds_per_item"]
    assert counts(scores["groups"]["all"]) == {"tp": 2, "fp": 1, "fn": 3}  # JOIN matches the rebuilt label
    assert counts(scores["drawn_only"]["groups"]["all"]) == {"tp": 0, "fp": 1, "fn": 1, "ignored": 2}
    assert scores["papers"]["7_gggg"]["tp"] == 2


def test_the_scores_are_what_the_loaded_labels_give(scored):
    report, root, manifest, path = scored
    truth = bt.load_biovista_truth(root, manifest)
    assert truth.references["7_gggg"][0] == JOIN
    plain = score_papers(read_predictions(path), truth.references, truth.groups())
    assert report["scores"]["papers"] == plain["papers"] and report["scores"]["groups"] == plain["groups"]


def test_the_report_lists_the_rebuilt_rows_and_hashes_the_labels_as_scored(scored, monkeypatch):
    report, root, manifest, _ = scored
    references = report["inputs"]["references"]
    assert references["rebuilt_labels"] == {"rule": bt.REPAIR_RULE, "rows": ["7_gggg:2"]}
    assert references["sha256"] == bt.reference_set_sha256(root, manifest)
    monkeypatch.setattr(bt, "repair_label", lambda *_: None)  # the same label files with no row rebuilt
    assert bt.reference_set_sha256(root, manifest) != references["sha256"]
    assert "rebuilt_labels" not in report["scoring"]  # the scoring setup is shared with Internal and the crops


def test_a_set_with_nothing_to_rebuild_lists_no_rows(tmp_path):
    rows = (("1_aaaa", "CCO"), ("1_aaaa", "CCCCl"), ("2_bbbb", "CCN"), ("3_cccc", "CO"))
    report = score_run(predictions(tmp_path, rows), references=BIOVISTA, papers=BIOVISTA / "manifest.csv")
    assert report["inputs"]["references"]["rebuilt_labels"] == {"rule": bt.REPAIR_RULE, "rows": []}


def test_molscout_score_rescores_a_predictions_file_against_the_rebuilt_labels(tmp_path):
    root, manifest = labelled_paper(tmp_path / "biovista")
    out = tmp_path / "scores.json"
    args = ["score", str(predictions(tmp_path)), "-o", str(out), "--references", str(root), "--papers", str(manifest)]
    assert main(args) == 0
    report = json.loads(out.read_text())
    assert report["scores"]["groups"]["all"]["counts"] == {"tp": 2, "fp": 1, "fn": 3}
    assert report["inputs"]["references"]["rebuilt_labels"]["rows"] == ["7_gggg:2"]


def test_internal_lists_no_rebuilt_labels():
    report = score_run(
        HANDMADE / "papers_predictions.csv",
        ground_truth=HANDMADE / "papers_ground_truth.csv",
        split=HANDMADE / "papers_split.csv",
    )
    assert "rebuilt_labels" not in json.dumps(report["inputs"])
