import numpy as np
import pytest

from molscout.predictions import Prediction
from molscout.scoring.papers import count_paper, score_papers
from molscout.scoring.stats import wilson_interval


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


def test_micro_f1_counts_tp_out_of_tp_plus_half_the_errors(scores):
    # F1 = TP / (TP + (FP + FN) / 2) = 5 / 10.5
    f1 = scores["groups"]["all"]["micro"]["f1"]
    assert (f1["successes"], f1["trials"]) == (5, 10.5)


AWARE = {"1": (3, 3, 1), "2": (2, 2, 1), "3": (0, 1, 1), "4": (0, 0, 2)}  # tp, fp, fn from the table above
STRIPPED = {"1": (4, 2, 0), "2": (2, 2, 1), "3": (1, 0, 0), "4": (0, 0, 2)}


def paper_bootstrap(counts):
    """The spec, written out: resample the papers 10,000 times (seed 6630), pool each draw, take 2.5/97.5 percentiles."""
    table = np.array(counts, dtype=float)
    tp, fp, fn = table[np.random.default_rng(6630).integers(0, len(table), size=(10_000, len(table)))].sum(axis=1).T
    intervals = {}
    for metric, denominator in (("precision", tp + fp), ("recall", tp + fn), ("f1", tp + (fp + fn) / 2)):
        draws = np.divide(tp, denominator, out=np.zeros(len(tp)), where=denominator > 0)
        intervals[metric] = list(np.percentile(draws, [2.5, 97.5]))
    return intervals


@pytest.mark.parametrize("group", ["all", "dev", "test"])
def test_micro_intervals_are_a_bootstrap_over_the_groups_own_papers(scores, group):
    record = scores["groups"][group]
    papers = record["papers"]
    for section, counts in (("micro", AWARE), ("stereo_stripped", STRIPPED)):
        expected = paper_bootstrap([counts[p] for p in papers])
        for metric, bounds in expected.items():
            assert record[section][metric]["ci95"] == pytest.approx(bounds, rel=1e-12, abs=1e-12), (section, metric)
            assert record[section][metric]["ci_method"] == "paper bootstrap"


def test_valid_output_rate_keeps_its_wilson_interval(scores):
    rate = scores["groups"]["all"]["valid_output_rate"]
    assert rate["ci95"] == pytest.approx(list(wilson_interval(10, 14)), abs=1e-12)
    assert "ci_method" not in rate


def test_one_dominant_paper_makes_the_bootstrap_much_wider_than_wilson():
    # Paper 0 finds all of its 100 alkanes; papers 1-9 each output one wrong structure. Pooled precision 100/109.
    alkanes = ["C" * n for n in range(1, 101)]
    predictions = [row("0", s) for s in alkanes] + [row(str(p), "N") for p in range(1, 10)]
    references = {"0": alkanes, **{str(p): ["O"] for p in range(1, 10)}}
    precision = score_papers(predictions, references)["groups"]["all"]["micro"]["precision"]
    assert precision["value"] == pytest.approx(100 / 109)
    assert wilson_interval(100, 109)[0] > 0.85
    # A draw of 10 papers misses paper 0 with probability 0.9 ** 10 = 0.35 > 2.5%, and then its precision is 0.
    assert precision["ci95"][0] == 0.0


@pytest.mark.parametrize("papers", [["1"], ["1", "2", "3"]])
def test_identical_papers_or_a_single_paper_give_a_zero_width_interval(papers):
    # Each paper: TP CCO, FP CCN, FN CCC, so every draw pools to precision, recall and F1 of 1/2.
    predictions = [row(p, s) for p in papers for s in ("CCO", "CCN")]
    group = score_papers(predictions, {p: ["CCO", "CCC"] for p in papers})["groups"]["all"]
    for metric in ("precision", "recall", "f1"):
        assert group["micro"][metric]["ci95"] == [0.5, 0.5], metric


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


def test_ignored_output_is_neither_tp_nor_fp():
    result = count_paper(["CCO", "CCCl"], ["CCO"], ignored=["CCCl"])
    assert (result.stereo_aware.tp, result.stereo_aware.fp, result.stereo_aware.fn) == (1, 0, 0)
    assert result.ignored_outputs == 1


def test_ignored_that_is_also_a_reference_still_counts_as_tp():
    result = count_paper(["CCO"], ["CCO"], ignored=["CCO"])
    assert (result.stereo_aware.tp, result.stereo_aware.fp, result.stereo_aware.fn) == (1, 0, 0)
    assert result.ignored_outputs == 0


def test_ignored_applies_per_view():
    result = count_paper(["C[C@H](N)O"], ["CC"], ignored=["CC(N)O"])
    assert result.stereo_aware.fp == 1
    assert result.stereo_stripped.fp == 0
    assert result.ignored_outputs == 0


def test_unparsable_ignored_structure_is_an_error():
    with pytest.raises(ValueError, match="ignored"):
        count_paper(["CCO"], ["CCO"], ignored=["not_a_smiles"])


def test_no_ignored_leaves_records_unchanged():
    report = score_papers([row("1", "CCO")], {"1": ["CCO"]})
    assert "ignored_outputs" not in report["papers"]["1"]
    assert "ignored_outputs" not in report["groups"]["all"]


def test_score_papers_with_ignored_reports_counts_per_paper_and_group():
    report = score_papers(
        [row("1", "CCO"), row("1", "CCCl"), row("2", "CC")], {"1": ["CCO"], "2": ["CC"]}, ignored={"1": ["CCCl"], "2": []}
    )
    assert report["papers"]["1"]["ignored_outputs"] == 1
    assert report["papers"]["2"]["ignored_outputs"] == 0
    assert report["groups"]["all"]["ignored_outputs"] == 1
    assert report["groups"]["all"]["counts"] == {"tp": 2, "fp": 0, "fn": 0}
