"""Report analysis: per-crop outcomes, run metrics, agreement between tools, oracle, breakdowns and samples."""

import os
from pathlib import Path

import pytest

from molscout.predictions import Prediction
from molscout.report.analysis import (
    OUTCOMES,
    ItemResult,
    RunResult,
    Share,
    ToolResources,
    accuracy_by_group,
    agreement_accuracy,
    box_stats,
    by_item,
    check_outcomes,
    crop_seconds,
    failure_sample,
    item_results,
    load_run,
    load_truth,
    molrecbench_groups,
    oracle,
    outcome_counts,
    pairwise_agreement,
    pooled,
    run_metrics,
    tool_resources,
)

REPO = Path(__file__).resolve().parents[1]
GATE = Path(os.environ.get("MOLSCOUT_GATE_RESULT", REPO / "benchmarks" / "results" / "molscribe__uspto"))

# Six scored crops, one per outcome, and x, whose reference RDKit cannot read.
TRUTH = {
    "t1": "CCO",
    "t2": "C[C@H](N)C(=O)O",
    "t3": "c1ccccc1",
    "t4": "CCN",
    "t5": "CC(=O)O",
    "t6": "OCCO",
    "x": None,
}
ANSWERS = {
    "t1": "",  # crashed: listed in errors.json
    "t2": "C[C@@H](N)C(=O)O",  # the other enantiomer: right only without stereo
    "t3": "C1=CC=CC=C1",  # Kekulé benzene: the same canonical SMILES
    "t4": "   ",
    "t5": "C(C",
    "t6": "CCCO",
    "x": "CCC",
}
EXPECTED = {
    "t1": "crashed",
    "t2": "stereo_only",
    "t3": "correct",
    "t4": "empty",
    "t5": "invalid",
    "t6": "wrong_structure",
}


def run_result(answers, *, tool="molscribe", dataset="uspto", errors=None, scores=None, meta=None, seconds=0.5):
    predictions = tuple(
        Prediction(dataset, item, smiles, None, None, None, tool, seconds) for item, smiles in answers.items()
    )
    return RunResult(
        folder=Path(f"{tool}__{dataset}"),
        tool=tool,
        dataset=dataset,
        predictions=predictions,
        report=scores or {},
        meta=meta or {},
        errors=errors or {},
        predictions_sha256="0" * 64,
    )


def item(dataset, item_id, tool, canonical, outcome):
    return ItemResult(dataset, item_id, tool, canonical or "", canonical, outcome)


def test_every_scored_crop_gets_exactly_one_outcome():
    run = run_result(ANSWERS, errors={"t1": "RuntimeError: boom"})
    items = item_results(run, load_like(TRUTH))
    assert {i.item_id: i.outcome for i in items} == EXPECTED
    assert outcome_counts(items) == dict.fromkeys(OUTCOMES, 1)
    assert [i.item_id for i in items] == ["t1", "t2", "t3", "t4", "t5", "t6"]


def test_a_crop_without_a_prediction_row_is_empty():
    answers = {item_id: smiles for item_id, smiles in ANSWERS.items() if item_id != "t3"}
    items = item_results(run_result(answers), load_like(TRUTH))
    assert {i.item_id: i.outcome for i in items}["t3"] == "empty"


def test_crashed_comes_first_even_when_the_answer_is_right():
    run = run_result({"t3": "c1ccccc1"}, errors={"t3": "ValueError: late"})
    assert [i.outcome for i in item_results(run, {"t3": "c1ccccc1"})] == ["crashed"]


def test_outcomes_agree_with_scores_json_or_say_how_not():
    scores = {
        "scores": {
            "items_scored": 6,
            "accuracy": {"successes": 1},
            "accuracy_stereo_stripped": {"successes": 2},
            "valid_output_rate": {"successes": 3},
        }
    }
    run = run_result(ANSWERS, errors={"t1": "RuntimeError: boom"}, scores=scores)
    items = item_results(run, load_like(TRUTH))
    assert check_outcomes(run, items) == []
    wrong = run_result(ANSWERS, errors={"t1": "x"}, scores={"scores": {**scores["scores"], "items_scored": 7}})
    assert check_outcomes(wrong, items) == ["molscribe__uspto: 6 crops scored here, scores.json says 7"]


def test_run_metrics_carry_scores_speed_resources_items_and_outcomes():
    scores = {
        "scores": {
            "items_scored": 6,
            "items_excluded": ["x"],
            "accuracy": {"value": 1 / 6, "ci95": [0.03, 0.56], "successes": 1, "trials": 6},
            "accuracy_stereo_stripped": {"value": 2 / 6, "ci95": [0.1, 0.7], "successes": 2, "trials": 6},
            "valid_output_rate": {"value": 0.5, "ci95": [0.19, 0.81], "successes": 3, "trials": 6},
            "seconds_per_item": {"mean": 0.4, "median": 0.3, "items": 7, "total": 2.8},
        }
    }
    resources = {
        "tool_peak_rss_mib": 2048.0,
        "tool_cpu_seconds": 90.5,
        "gpu": {"name": "NVIDIA H200", "peak_memory_mib": 5120.0, "mean_utilization_pct": 41.5, "samples": 9},
    }
    run = run_result(ANSWERS, errors={"t1": "boom"}, scores=scores, meta={"resources": resources})
    metrics = run_metrics(run, item_results(run, load_like(TRUTH)))
    assert metrics == {
        "accuracy/stereo_aware": 1 / 6,
        "accuracy/stereo_aware_ci_low": 0.03,
        "accuracy/stereo_aware_ci_high": 0.56,
        "accuracy/stereo_stripped": 2 / 6,
        "accuracy/stereo_stripped_ci_low": 0.1,
        "accuracy/stereo_stripped_ci_high": 0.7,
        "valid_output_rate": 0.5,
        "speed/s_per_crop_mean": 0.4,
        "speed/s_per_crop_median": 0.3,
        "resources/gpu_peak_memory_gib": 5.0,
        "resources/gpu_mean_utilization_pct": 41.5,
        "resources/peak_rss_gib": 2.0,
        "resources/cpu_seconds": 90.5,
        "items/scored": 6,
        "items/excluded": 1,
        "items/crashed": 1,
        **{f"outcome/{name}": 1 for name in OUTCOMES},
    }
    without = run_result(ANSWERS, scores=scores, meta={"resources": {**resources, "gpu": None}})
    keys = set(run_metrics(without, item_results(without, load_like(TRUTH))))
    assert "resources/peak_rss_gib" in keys and "resources/gpu_peak_memory_gib" not in keys
    older = run_result(ANSWERS, scores=scores, meta={})
    assert not any(key.startswith("resources/") for key in run_metrics(older, []))


# Agreement: three tools on uspto, four on jpo. Canonical strings stand in for molecules.
AGREEMENT = [
    # i1: all three agree and are right.
    item("uspto", "i1", "molscribe", "CCO", "correct"),
    item("uspto", "i1", "molnextr", "CCO", "correct"),
    item("uspto", "i1", "decimer", "CCO", "correct"),
    # i2: two agree on a wrong answer; the third is right.
    item("uspto", "i2", "molscribe", "CCN", "wrong_structure"),
    item("uspto", "i2", "molnextr", "CCN", "wrong_structure"),
    item("uspto", "i2", "decimer", "CCC", "correct"),
    # i3: no two agree (an empty answer agrees with nothing) and none is right.
    item("uspto", "i3", "molscribe", None, "empty"),
    item("uspto", "i3", "molnextr", None, "invalid"),
    item("uspto", "i3", "decimer", "CCN", "wrong_structure"),
    # j1: two pairs tie, so there is no agreed answer.
    item("jpo", "j1", "molscribe", "CCO", "correct"),
    item("jpo", "j1", "molnextr", "CCO", "correct"),
    item("jpo", "j1", "decimer", "CCN", "wrong_structure"),
    item("jpo", "j1", "molvec", "CCN", "wrong_structure"),
    # j2: three of four agree on the right answer.
    item("jpo", "j2", "molscribe", "CCO", "correct"),
    item("jpo", "j2", "molnextr", "CCO", "correct"),
    item("jpo", "j2", "decimer", "CCO", "correct"),
    item("jpo", "j2", "molvec", "C", "wrong_structure"),
]


def test_pairwise_agreement_counts_identical_answers_over_common_crops():
    shares = pairwise_agreement(AGREEMENT)
    assert shares[("uspto", "molscribe", "molnextr")] == Share(2, 3)
    assert shares[("uspto", "molscribe", "decimer")] == Share(1, 3)
    assert shares[("jpo", "molscribe", "molnextr")] == Share(2, 2)
    assert shares[("jpo", "decimer", "molvec")] == Share(1, 2)
    assert ("uspto", "molscribe", "molvec") not in shares
    assert ("uspto", "molnextr", "molscribe") not in shares  # each pair once, in tool order
    assert pooled(shares)[("molscribe", "molnextr")] == Share(4, 5)


def test_agreement_accuracy_scores_the_answer_k_tools_share_and_skips_ties():
    shares = agreement_accuracy(AGREEMENT)
    assert shares == {("uspto", 3): Share(1, 1), ("uspto", 2): Share(0, 1), ("jpo", 3): Share(1, 1)}
    assert pooled(shares) == {(3,): Share(2, 2), (2,): Share(0, 1)}


def test_a_crashed_crop_agrees_with_nothing_even_with_a_smiles():
    items = [
        item("uspto", "i1", "molscribe", "CCO", "correct"),
        item("uspto", "i1", "molnextr", "CCO", "crashed"),
        item("uspto", "i1", "decimer", "CCN", "wrong_structure"),
    ]
    assert pairwise_agreement(items)[("uspto", "molscribe", "molnextr")] == Share(0, 1)
    assert agreement_accuracy(items) == {}


def test_oracle_counts_crops_any_tool_reads_correctly():
    assert oracle(AGREEMENT) == {"uspto": Share(2, 3), "jpo": Share(2, 2)}


def test_share_value_and_wilson_interval():
    share = Share(5255, 5704)
    assert share.value == pytest.approx(0.9212833099579243)
    assert share.ci95 == pytest.approx((0.9140078109034436, 0.9279917496437365))
    assert Share(0, 0).value == 0.0


def test_accuracy_by_group_counts_each_label_a_crop_carries():
    groups = {"i1": ("A", "Wavy Bond"), "i2": ("A",), "i3": ("B", "Wavy Bond")}
    shares = accuracy_by_group(AGREEMENT, groups)
    assert shares[("decimer", "A")] == Share(2, 2)
    assert shares[("decimer", "Wavy Bond")] == Share(1, 2)
    assert shares[("molscribe", "B")] == Share(0, 1)
    assert not any(tool == "molvec" for tool, _ in shares)  # jpo crops carry no labels here
    assert accuracy_by_group(AGREEMENT, {"i1": ("Wavy Bond", "Wavy Bond")})[("decimer", "Wavy Bond")] == Share(1, 1)


def test_molrecbench_groups_split_subsets_from_hard_cases():
    samples = {
        "a": {"evaluation_subset": "A", "hardcase_label": ("Wavy Bond", "Triple Bond")},
        "b": {"evaluation_subset": "C", "hardcase_label": ()},
    }
    subsets, hardcases = molrecbench_groups(samples)
    assert subsets == {"a": ("Subset A",), "b": ("Subset C",)}
    assert hardcases == {"a": ("Wavy Bond", "Triple Bond"), "b": ("No hard-case label",)}


def test_failure_sample_is_seeded_bounded_and_skips_crops_every_tool_reads():
    sample = failure_sample(AGREEMENT, per_dataset=5)
    assert sample == {"uspto": ["i2", "i3"], "jpo": ["j1", "j2"]}
    many = [item("uob", f"u{n:03}", "molscribe", "C", "wrong_structure") for n in range(50)]
    first = failure_sample(many, per_dataset=10)["uob"]
    assert len(first) == 10 and first == sorted(first)
    assert failure_sample(many, per_dataset=10)["uob"] == first
    assert failure_sample(many, per_dataset=0) == {}


def test_by_item_groups_each_crops_answers_by_tool():
    grouped = by_item(AGREEMENT)
    assert list(grouped[("jpo", "j2")]) == ["molscribe", "molnextr", "decimer", "molvec"]
    assert grouped[("uspto", "i2")]["decimer"].outcome == "correct"


def test_crop_seconds_pool_every_row_per_tool():
    runs = [
        run_result({"a": "C", "b": "C"}, seconds=0.5),
        run_result({"a": "C"}, dataset="jpo", seconds=2.0),
        run_result({"a": "C"}, tool="molvec", seconds=1.0),
    ]
    assert crop_seconds(runs) == {"molscribe": [0.5, 0.5, 2.0], "molvec": [1.0]}


def test_tool_resources_take_peaks_over_runs_and_pool_time_over_crops():
    def run(tool, dataset, total, items, resources):
        report = {"scores": {"seconds_per_item": {"total": total, "items": items}}}
        return run_result({}, tool=tool, dataset=dataset, scores=report, meta={"resources": resources})

    gpu = {"tool_peak_rss_mib": 1024.0, "gpu": {"peak_memory_mib": 3072.0}}
    runs = [
        run("molscribe", "uspto", 30.0, 100, gpu),
        run("molscribe", "jpo", 10.0, 100, {**gpu, "gpu": {"peak_memory_mib": 4096.0}}),
        run("molvec", "uspto", 5.0, 10, {"tool_peak_rss_mib": 512.0, "gpu": None}),
        run("decimer", "uspto", 1.0, 10, None),
    ]
    assert tool_resources(runs) == (
        ToolResources("molscribe", gpu_peak_memory_gib=4.0, peak_rss_gib=1.0, s_per_crop_mean=0.2),
        ToolResources("decimer", gpu_peak_memory_gib=None, peak_rss_gib=None, s_per_crop_mean=0.1),
        ToolResources("molvec", gpu_peak_memory_gib=None, peak_rss_gib=0.5, s_per_crop_mean=0.5),
    )


def test_box_stats_use_quartiles_and_tukey_whiskers():
    stats = box_stats([1.0, 2.0, 3.0, 4.0, 100.0])
    assert (stats.q1, stats.median, stats.q3) == (2.0, 3.0, 4.0)
    assert (stats.low, stats.high) == (1.0, 4.0)  # 100 lies beyond q3 + 1.5 IQR
    assert stats.mean == 22.0 and stats.n == 5
    with pytest.raises(ValueError):
        box_stats([])


def load_like(references):
    """Canonical references, as load_truth returns them."""
    from molscout.scoring import canonical_smiles

    return {item_id: None if smiles is None else canonical_smiles(smiles) for item_id, smiles in references.items()}


@pytest.mark.skipif(
    not (GATE / "scores.json").is_file() or not (REPO / "data" / "raw" / "uspto" / "USPTO_mol_ref").is_dir(),
    reason="needs the MolScribe-on-USPTO gate result (MOLSCOUT_GATE_RESULT) and data/raw/uspto",
)
def test_gate_result_outcomes_match_its_scores_json():
    run = load_run(GATE)
    items = item_results(run, load_truth("uspto", REPO))
    counts = outcome_counts(items)
    scores = run.report["scores"]
    assert counts["correct"] == scores["accuracy"]["successes"]
    assert counts["correct"] + counts["stereo_only"] == scores["accuracy_stereo_stripped"]["successes"]
    assert sum(counts.values()) == scores["items_scored"]
    assert check_outcomes(run, items) == []
