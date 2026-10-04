"""End to end: `molscout score` on hand-made files matches numbers counted by hand.

Papers (fixtures/handmade/papers_*). Unique ground truth: paper 1 {CCO, benzene,
L-alanine, *c1ccccc1}; paper 2 {acetic acid, CCN, CCO}, with CCO listed twice;
paper 3 {trans-2-butene}; paper 4 {pyridine, acetonitrile}. Dev is 1 and 4; test is 2 and 3.

    paper  unique outputs                              TP FP FN | stripped TP FP FN
    1      CCO, benzene, D-alanine, *c1ccccc1, Cl,      3  3  1 |          4  2  0
           C1CC (unparsable)
    2      acetic acid, CCO, "" and not_a_smiles        2  2  1 |          2  2  1
           (both unparsable)
    3      cis-2-butene                                 0  1  1 |          1  0  0
    4      none                                         0  0  2 |          0  0  2

    group  TP FP FN  P     R     F1     macro P  macro R | stripped TP FP FN  P     R     F1
    all     5  6  5  5/11  5/10  10/21  1/4      17/48   |          7  4  3   7/11  7/10  2/3
    dev     3  3  3  3/6   3/6   1/2    1/4      3/8     |          4  2  2   4/6   4/6   2/3
    test    2  3  2  2/5   2/4   4/9    1/4      1/3     |          3  2  1   3/5   3/4   2/3

    Valid output rows: paper 1 6 of 7, paper 2 3 of 6, paper 3 1 of 1, paper 4 none;
    so all 10 of 14, dev 6 of 7, test 4 of 7. Paper 4 is the one paper without output.

Crops (fixtures/handmade/crops_*). c5's molfile has a pentavalent carbon, so c5 is
excluded; c6 has no prediction; c7's prediction is unparsable.

    crop  reference    prediction       stereo-aware  stripped  valid
    c1    CCO          OCC              right         right     yes
    c2    L-alanine    D-alanine        wrong         right     yes
    c3    benzene      Kekulé benzene   right         right     yes
    c4    *c1ccccc1    *c1ccccc1        right         right     yes
    c6    acetic acid  none             wrong         wrong     no
    c7    CCN          C1CC             wrong         wrong     no

    Accuracy 3/6, stripped 4/6, valid output 4/6.

Wilson bounds are from statsmodels proportion_confint(method="wilson"), computed separately.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from molscout.cli import main

FIXTURES = Path(__file__).parent / "fixtures" / "handmade"
PAPER_ARGS = ("--ground-truth", FIXTURES / "papers_ground_truth.csv", "--split", FIXTURES / "papers_split.csv")
CROP_ARGS = ("--references", FIXTURES / "crops_references")
HEADER = "dataset,item_id,smiles,page,bbox,confidence,tool,seconds"


def score(directory, *args):
    out = directory / "scores.json"
    assert main(["score", *map(str, args), "-o", str(out)]) == 0
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def papers(tmp_path_factory):
    return score(tmp_path_factory.mktemp("papers"), FIXTURES / "papers_predictions.csv", *PAPER_ARGS)


@pytest.fixture(scope="module")
def crops(tmp_path_factory):
    return score(tmp_path_factory.mktemp("crops"), FIXTURES / "crops_predictions.csv", *CROP_ARGS)


@pytest.mark.parametrize(
    ("group", "counts", "precision", "recall", "f1", "macro_p", "macro_r"),
    [
        ("all", (5, 6, 5), 5 / 11, 5 / 10, 10 / 21, 1 / 4, 17 / 48),
        ("dev", (3, 3, 3), 3 / 6, 3 / 6, 1 / 2, 1 / 4, 3 / 8),
        ("test", (2, 3, 2), 2 / 5, 2 / 4, 4 / 9, 1 / 4, 1 / 3),
    ],
)
def test_paper_scores_match_hand_counts(papers, group, counts, precision, recall, f1, macro_p, macro_r):
    scores = papers["scores"]["groups"][group]
    assert tuple(scores["counts"][k] for k in ("tp", "fp", "fn")) == counts
    assert scores["micro"]["precision"]["value"] == pytest.approx(precision)
    assert scores["micro"]["recall"]["value"] == pytest.approx(recall)
    assert scores["micro"]["f1"]["value"] == pytest.approx(f1)
    assert scores["macro"]["precision"]["value"] == pytest.approx(macro_p)
    assert scores["macro"]["recall"]["value"] == pytest.approx(macro_r)


@pytest.mark.parametrize(
    ("group", "counts", "precision", "recall", "f1"),
    [
        ("all", (7, 4, 3), 7 / 11, 7 / 10, 2 / 3),
        ("dev", (4, 2, 2), 4 / 6, 4 / 6, 2 / 3),
        ("test", (3, 2, 1), 3 / 5, 3 / 4, 2 / 3),
    ],
)
def test_paper_stereo_stripped_scores_match_hand_counts(papers, group, counts, precision, recall, f1):
    scores = papers["scores"]["groups"][group]["stereo_stripped"]
    assert tuple(scores["counts"][k] for k in ("tp", "fp", "fn")) == counts
    assert scores["precision"]["value"] == pytest.approx(precision)
    assert scores["recall"]["value"] == pytest.approx(recall)
    assert scores["f1"]["value"] == pytest.approx(f1)


@pytest.mark.parametrize(("group", "valid", "rows"), [("all", 10, 14), ("dev", 6, 7), ("test", 4, 7)])
def test_paper_valid_output_rate(papers, group, valid, rows):
    rate = papers["scores"]["groups"][group]["valid_output_rate"]
    assert (rate["successes"], rate["trials"]) == (valid, rows)


def test_paper_wilson_intervals_match_statsmodels(papers):
    scores = papers["scores"]["groups"]["all"]
    expected = {
        ("micro", "precision"): [0.212712716225, 0.719908462591],
        ("micro", "recall"): [0.236593090513, 0.763406909487],
        ("micro", "f1"): [0.224008446579, 0.741127596675],
        ("stereo_stripped", "precision"): [0.353801174508, 0.848335289046],
        ("stereo_stripped", "recall"): [0.396778147461, 0.892208732594],
        ("stereo_stripped", "f1"): [0.373998167257, 0.870049529965],
    }
    for (section, metric), bounds in expected.items():
        assert scores[section][metric]["ci95"] == pytest.approx(bounds, abs=1e-9), (section, metric)
    assert scores["valid_output_rate"]["ci95"] == pytest.approx([0.453509156701, 0.882786213554], abs=1e-9)


def test_paper_macro_intervals_bracket_the_mean(papers):
    macro = papers["scores"]["groups"]["all"]["macro"]
    low, high = macro["precision"]["ci95"]
    assert 0.0 <= low <= 0.25 <= high <= 0.5
    assert macro["recall"]["papers"] == 4


def test_paper_report_records_inputs(papers):
    assert papers["dataset"] == "internal"
    assert papers["tool"] == "HandMade 1.0"
    assert papers["scoring"]["rdkit_version"] == "2026.03.2"
    assert list(papers["scores"]["groups"]) == ["dev", "test", "all"]
    assert papers["scores"]["groups"]["all"]["molecules"] == 10


def test_crop_scores_match_hand_counts(crops):
    scores = crops["scores"]
    assert scores["items_scored"] == 6
    assert scores["items_excluded"] == ["c5"]
    assert scores["items_without_prediction"] == 1
    expected = {
        "accuracy": (3, [0.187616306483, 0.812383693517]),
        "accuracy_stereo_stripped": (4, [0.299993315138, 0.903228588894]),
        "valid_output_rate": (4, [0.299993315138, 0.903228588894]),
    }
    for metric, (correct, bounds) in expected.items():
        assert (scores[metric]["successes"], scores[metric]["trials"]) == (correct, 6), metric
        assert scores[metric]["ci95"] == pytest.approx(bounds, abs=1e-9), metric


def test_scores_are_reproducible(tmp_path, papers):
    assert score(tmp_path, FIXTURES / "papers_predictions.csv", *PAPER_ARGS) == papers


def test_empty_predictions_score_zero_recall(tmp_path):
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(HEADER + "\n")
    report = score(tmp_path, predictions, "--dataset", "internal", *PAPER_ARGS)
    scores = report["scores"]["groups"]["all"]
    assert scores["molecules"] == 10
    assert scores["micro"]["recall"]["value"] == 0.0
    assert report["tool"] == ""


def test_empty_predictions_need_a_dataset(tmp_path, capsys):
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(HEADER + "\n")
    assert main(["score", str(predictions), "-o", str(tmp_path / "scores.json"), *map(str, PAPER_ARGS)]) == 1
    assert "--dataset" in capsys.readouterr().err


def test_crop_dataset_needs_references(tmp_path, capsys):
    out = tmp_path / "scores.json"
    assert main(["score", str(FIXTURES / "crops_predictions.csv"), "-o", str(out)]) == 1
    assert "--references" in capsys.readouterr().err
    assert not out.exists()


def test_dataset_flag_must_agree_with_rows(tmp_path, capsys):
    args = ["score", str(FIXTURES / "crops_predictions.csv"), "--dataset", "uob", *map(str, CROP_ARGS)]
    assert main([*args, "-o", str(tmp_path / "scores.json")]) == 1
    assert "uspto" in capsys.readouterr().err


def test_biovista_has_no_ground_truth_loader_yet(tmp_path, capsys):
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(HEADER + "\nbiovista,p1,CCO,1,,,T 1,1\n")
    assert main(["score", str(predictions), "-o", str(tmp_path / "scores.json")]) == 1
    assert "no ground-truth loader for biovista" in capsys.readouterr().err


def test_format_errors_are_reported(tmp_path, capsys):
    predictions = tmp_path / "predictions.csv"
    predictions.write_text("dataset\n")
    assert main(["score", str(predictions), "-o", str(tmp_path / "scores.json")]) == 1
    assert "missing column" in capsys.readouterr().err


def test_ground_truth_must_match_the_split(tmp_path, capsys):
    split = tmp_path / "split.csv"
    split.write_text("paperID,split,molecules\n1,dev,4\n2,test,4\n3,test,1\n4,dev,3\n")
    args = ["score", str(FIXTURES / "papers_predictions.csv"), "-o", str(tmp_path / "scores.json")]
    assert main([*args, "--ground-truth", str(FIXTURES / "papers_ground_truth.csv"), "--split", str(split)]) == 1
    assert "do not match" in capsys.readouterr().err


def test_runs_as_a_module(tmp_path):
    out = tmp_path / "scores.json"
    src = str(Path(__file__).resolve().parents[1] / "src")
    command = [sys.executable, "-m", "molscout", "score", str(FIXTURES / "crops_predictions.csv"), "-o", str(out)]
    result = subprocess.run(
        [*command, *map(str, CROP_ARGS)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": src},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())["scores"]["accuracy"]["successes"] == 3


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_paper_report_pins_ground_truth_and_split(papers):
    assert papers["inputs"] == {
        "ground_truth_sha256": sha256(FIXTURES / "papers_ground_truth.csv"),
        "split_sha256": sha256(FIXTURES / "papers_split.csv"),
    }


def test_crop_report_pins_references_and_why_crops_were_excluded(crops):
    references = crops["inputs"]["references"]
    assert references["files"] == 7
    assert len(references["sha256"]) == 64
    assert list(references["unreadable"]) == ["c5"]
    assert "valence" in references["unreadable"]["c5"].lower()


def test_unwritable_output_is_reported(tmp_path, capsys):
    args = ["score", str(FIXTURES / "crops_predictions.csv"), *map(str, CROP_ARGS), "-o", str(tmp_path)]
    assert main(args) == 1
    assert "molscout score: error:" in capsys.readouterr().err


def test_unreadable_predictions_are_reported(tmp_path, capsys):
    assert main(["score", str(tmp_path), *map(str, CROP_ARGS), "-o", str(tmp_path / "scores.json")]) == 1
    assert "molscout score: error:" in capsys.readouterr().err
