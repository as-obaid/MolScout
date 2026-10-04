import pytest

from molscout.predictions import Prediction
from molscout.scoring.papers import score_papers


def row(paper, smiles):
    return Prediction("internal", paper, smiles, 1, None, None, "T 1", 5.0)


REFERENCES = {
    "1": ["CCO", "c1ccccc1", "C[C@H](N)C(=O)O", "*c1ccccc1"],
    "2": ["CC(=O)O", "CCN", "CCO", "CCO"],  # CCO listed twice, as under two names
    "3": ["C/C=C/C"],
    "4": ["c1ccncc1", "CC#N"],
}
PREDICTIONS = [
    row("1", "OCC"), row("1", "C1=CC=CC=C1"), row("1", "C[C@@H](N)C(=O)O"), row("1", "c1ccccc1*"),
    row("1", "c1ccccc1"), row("1", "C1CC"), row("1", "Cl"),
    row("2", "CC(O)=O"), row("2", "CC(=O)O"), row("2", ""), row("2", "not_a_smiles"),
    row("2", "not_a_smiles"), row("2", "CCO"),
    row("3", "C/C=C\\C"),
]
GROUPS = {"dev": {"1", "4"}, "test": {"2", "3"}, "all": {"1", "2", "3", "4"}}

# Hand counts. Paper 1: TP CCO, benzene, *c1ccccc1; FP D-alanine, Cl, C1CC; FN L-alanine.
# Stripped: alanine matches, so TP 4, FP Cl and C1CC. Paper 2: TP acetic acid, CCO; FP "" and
# not_a_smiles (duplicates collapse); FN CCN. Paper 3: cis vs trans, right only when stripped.
# Paper 4: no output.
#   paper  TP FP FN | stripped TP FP FN | valid invalid
#   1       3  3  1 |          4  2  0  |   5     1
#   2       2  2  1 |          2  2  1  |   2     2
#   3       0  1  1 |          1  0  0  |   1     0
#   4       0  0  2 |          0  0  2  |   0     0


@pytest.fixture(scope="module")
def scores():
    return score_papers(PREDICTIONS, REFERENCES, GROUPS)


def test_per_paper_counts(scores):
    aware = {p: (r["tp"], r["fp"], r["fn"]) for p, r in scores["papers"].items()}
    assert aware == {"1": (3, 3, 1), "2": (2, 2, 1), "3": (0, 1, 1), "4": (0, 0, 2)}
    stripped = {p: tuple(r["stereo_stripped"][k] for k in ("tp", "fp", "fn")) for p, r in scores["papers"].items()}
    assert stripped == {"1": (4, 2, 0), "2": (2, 2, 1), "3": (1, 0, 0), "4": (0, 0, 2)}
    assert scores["papers"]["2"]["molecules"] == 3
    rows = {p: (r["rows"], r["invalid_rows"]) for p, r in scores["papers"].items()}
    assert rows == {"1": (7, 1), "2": (6, 3), "3": (1, 0), "4": (0, 0)}


def test_micro_scores_pool_counts(scores):
    group = scores["groups"]["all"]
    assert group["molecules"] == 10
    assert group["counts"] == {"tp": 5, "fp": 6, "fn": 5}
    assert group["micro"]["precision"]["value"] == pytest.approx(5 / 11)
    assert group["micro"]["recall"]["value"] == pytest.approx(5 / 10)
    assert group["micro"]["f1"]["value"] == pytest.approx(10 / 21)
    assert group["stereo_stripped"]["counts"] == {"tp": 7, "fp": 4, "fn": 3}
    assert group["stereo_stripped"]["precision"]["value"] == pytest.approx(7 / 11)
    assert group["stereo_stripped"]["recall"]["value"] == pytest.approx(7 / 10)
    assert group["stereo_stripped"]["f1"]["value"] == pytest.approx(2 / 3)
    assert (group["valid_output_rate"]["successes"], group["valid_output_rate"]["trials"]) == (10, 14)
    assert group["papers_without_output"] == 1


def test_micro_f1_interval_is_wilson_on_its_denominator(scores):
    # F1 = TP / (TP + (FP + FN) / 2) = 5 / 10.5
    assert scores["groups"]["all"]["micro"]["f1"]["ci95"] == pytest.approx([0.224008446579, 0.741127596675], abs=1e-9)


def test_macro_averages_papers_equally(scores):
    group = scores["groups"]["all"]
    assert group["macro"]["precision"]["value"] == pytest.approx(0.25)  # (1/2 + 1/2 + 0 + 0) / 4
    assert group["macro"]["recall"]["value"] == pytest.approx(17 / 48)  # (3/4 + 2/3 + 0 + 0) / 4
    assert group["macro"]["precision"]["papers"] == 4


def test_groups_are_scored_separately(scores):
    dev, test = scores["groups"]["dev"], scores["groups"]["test"]
    assert dev["papers"] == ["1", "4"]
    assert dev["counts"] == {"tp": 3, "fp": 3, "fn": 3}
    assert test["counts"] == {"tp": 2, "fp": 3, "fn": 2}
    assert dev["macro"]["recall"]["value"] == pytest.approx(0.375)
    assert test["macro"]["recall"]["value"] == pytest.approx(1 / 3)
    assert (dev["papers_without_output"], test["papers_without_output"]) == (1, 0)


def test_default_group_is_all_papers():
    groups = score_papers(PREDICTIONS, REFERENCES)["groups"]
    assert list(groups) == ["all"]
    assert groups["all"]["papers"] == ["1", "2", "3", "4"]


def test_duplicates_collapse_before_scoring():
    once = score_papers([row("2", "CCO")], {"2": ["CCO"]})
    repeated = score_papers([row("2", "CCO"), row("2", "OCC"), row("2", "CCO")], {"2": ["CCO", "CCO"]})
    scored = ("molecules", "tp", "fp", "fn", "precision", "recall", "stereo_stripped")
    assert {k: once["papers"]["2"][k] for k in scored} == {k: repeated["papers"]["2"][k] for k in scored}
    assert (once["papers"]["2"]["rows"], repeated["papers"]["2"]["rows"]) == (1, 3)


def test_distinct_invalid_strings_are_separate_false_positives():
    scores = score_papers([row("1", "C1CC"), row("1", "C1CC "), row("1", "xx")], {"1": ["CCO"]})
    assert scores["papers"]["1"]["fp"] == 2  # "C1CC" and "C1CC " are one string once trimmed


def test_paper_without_predictions_has_zero_precision():
    paper = score_papers([], {"1": ["CCO"]})["papers"]["1"]
    assert (paper["precision"], paper["recall"]) == (0.0, 0.0)


def test_stereo_stripping_can_merge_references():
    paper = score_papers([row("1", "CC(N)C(=O)O")], {"1": ["C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"]})["papers"]["1"]
    assert (paper["tp"], paper["fn"]) == (0, 2)
    assert tuple(paper["stereo_stripped"][k] for k in ("tp", "fp", "fn")) == (1, 0, 0)


def test_unknown_paper_raises():
    with pytest.raises(ValueError, match=r"not papers in the references.*1\.0"):
        score_papers([row("1.0", "CCO")], {"1": ["CCO"]})


def test_group_with_unknown_paper_raises():
    with pytest.raises(ValueError, match="group 'dev'.*9"):
        score_papers([], {"1": ["CCO"]}, {"dev": {"9"}})


def test_empty_group_raises():
    with pytest.raises(ValueError, match="group 'dev' has no papers"):
        score_papers([], {"1": ["CCO"]}, {"dev": set()})


def test_unparsable_reference_raises_with_paper():
    with pytest.raises(ValueError, match="paper '1'.*C1CC"):
        score_papers([], {"1": ["C1CC"]})


def test_non_ascii_digits_in_paper_ids_do_not_crash():
    papers = score_papers([], {"\u00b2": ["CCO"], "10": ["CCO"], "9": ["CCO"]})["papers"]
    assert list(papers) == ["9", "10", "\u00b2"]
