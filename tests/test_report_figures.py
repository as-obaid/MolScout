"""Report figures: each builder returns a styled Plotly figure with the expected traces, and no wandb."""

import json

import pytest

go = pytest.importorskip("plotly.graph_objects")

from molscout.report import figures  # noqa: E402
from molscout.report.analysis import OUTCOMES, BoxStats, Share, ToolResources  # noqa: E402

TOOLS = ("molscribe", "molnextr")
DATASETS = ("uspto", "molrecbench_wild")


def metrics(tool, dataset):
    aware = 0.9 if tool == "molscribe" else 0.6
    return {
        "accuracy/stereo_aware": aware,
        "accuracy/stereo_aware_ci_low": aware - 0.02,
        "accuracy/stereo_aware_ci_high": aware + 0.02,
        "accuracy/stereo_stripped": aware + 0.03,
        "speed/s_per_crop_mean": 0.4 if tool == "molscribe" else 2.5,
        "items/scored": 100,
        "outcome/crashed": 0,
        "outcome/empty": 2,
        "outcome/invalid": 3,
        "outcome/correct": int(aware * 100) - 5,
        "outcome/stereo_only": 5,
        "outcome/wrong_structure": 100 - int(aware * 100) - 5,
    }


METRICS = {(tool, dataset): metrics(tool, dataset) for tool in TOOLS for dataset in DATASETS}


def plain(figure):
    """The figure as W&B stores it; typed-array (bdata) encoding would not render in every Plotly.js."""
    text = figure.to_json()
    assert "bdata" not in text
    return json.loads(text)


def styled(figure, title):
    data = plain(figure)
    assert isinstance(figure, go.Figure)
    assert data["layout"]["title"]["text"] == title
    assert data["layout"]["paper_bgcolor"] == figures.SURFACE
    return data


def test_accuracy_by_dataset_has_one_bar_trace_per_tool_with_ci_error_bars():
    data = styled(figures.accuracy_bars(METRICS), "Stereo-aware exact match by dataset")
    assert [trace["name"] for trace in data["data"]] == ["MolScribe", "MolNexTR"]
    first = data["data"][0]
    assert first["type"] == "bar" and first["x"] == ["USPTO", "MolRecBench-Wild"]
    assert first["marker"]["color"] == figures.TOOL_COLORS["molscribe"]
    assert first["error_y"]["array"] == pytest.approx([0.02, 0.02])
    assert first["error_y"]["arrayminus"] == pytest.approx([0.02, 0.02])
    assert data["layout"]["yaxis"]["tickformat"] == ".0%"


def test_stereo_penalty_is_in_percentage_points():
    data = styled(figures.stereo_penalty_bars(METRICS), "Stereo penalty: stereo-stripped minus stereo-aware")
    assert data["data"][1]["y"] == pytest.approx([3.0, 3.0])


def test_accuracy_vs_speed_facets_datasets_on_a_shared_log_axis():
    data = styled(figures.accuracy_speed_scatter(METRICS), "Exact match against time per crop")
    assert len(data["data"]) == len(TOOLS) * len(DATASETS)
    assert sum(trace.get("showlegend", True) for trace in data["data"]) == len(TOOLS)
    assert data["layout"]["xaxis"]["type"] == "log" and data["layout"]["xaxis2"]["matches"] == "x"
    assert {trace["marker"]["symbol"] for trace in data["data"]} == {"circle", "square"}


def test_outcome_breakdown_stacks_outcomes_that_occur_per_dataset():
    data = styled(figures.outcome_bars(METRICS), "Outcome of every scored crop")
    names = list(dict.fromkeys(trace["name"] for trace in data["data"]))
    assert names == ["Correct", "Stereo only", "Wrong structure", "Invalid SMILES", "Empty"]  # no crashes here
    assert data["layout"]["barmode"] == "stack"
    assert len(data["data"]) == len(names) * len(DATASETS)
    correct = data["data"][0]
    assert correct["x"] == ["MolScribe", "MolNexTR"] and correct["y"] == pytest.approx([0.85, 0.55])


def test_every_outcome_has_a_name_a_colour_and_a_place_in_the_stack():
    assert set(figures.STACK_ORDER) == set(figures.OUTCOME_NAMES) == set(figures.OUTCOME_COLORS) == set(OUTCOMES)


def test_time_per_crop_draws_precomputed_boxes_on_a_log_axis():
    stats = {"molscribe": BoxStats(0.2, 0.3, 0.35, 0.4, 0.5, 0.36, 10), "molvec": BoxStats(0.0, 1, 2, 3, 6, 2.5, 4)}
    data = styled(figures.time_boxes(stats), "Time per crop, all datasets pooled")
    assert [trace["type"] for trace in data["data"]] == ["box", "box"]
    assert data["data"][0]["median"] == [0.35]
    assert data["data"][1]["lowerfence"][0] > 0  # a zero time cannot sit on a log axis
    assert data["layout"]["yaxis"]["type"] == "log"


def test_agreement_heatmap_shows_each_pair_once():
    shares = {
        ("molscribe", "molnextr"): Share(80, 100),
        ("molscribe", "decimer"): Share(60, 100),
        ("molnextr", "decimer"): Share(70, 100),
    }
    data = styled(figures.agreement_heatmap(shares), "Pairwise agreement, all datasets pooled")
    heatmap = data["data"][0]
    assert heatmap["x"] == ["MolScribe", "MolNexTR"] and heatmap["y"] == ["MolNexTR", "DECIMER"]
    assert heatmap["z"] == [[0.8, None], [0.6, 0.7]]
    assert sorted(note["text"] for note in data["layout"]["annotations"]) == ["60%", "70%", "80%"]
    gap = plain(
        figures.agreement_heatmap({("molscribe", "molnextr"): Share(8, 10), ("molnextr", "decimer"): Share(7, 10)})
    )
    assert gap["data"][0]["z"] == [[0.8, None], [None, 0.7]]  # molscribe and decimer share no crop


def test_consensus_bars_show_k_tool_agreement_and_the_oracle_per_dataset():
    agreement = {("uspto", 2): Share(5, 10), ("uspto", 3): Share(9, 10), ("jpo", 2): Share(1, 2)}
    oracle = {"uspto": Share(95, 100), "jpo": Share(9, 10)}
    data = styled(
        figures.consensus_bars(agreement, oracle), "Exact match when k tools agree, and when any tool is right"
    )
    agreed = [trace for trace in data["data"] if trace["name"] == "Agreed answer right"]
    assert [bar["x"] for bar in agreed] == [["2", "3"], ["2"]]
    assert agreed[0]["y"] == pytest.approx([0.5, 0.9])
    oracles = [trace for trace in data["data"] if trace["name"] == "Any tool right (oracle)"]
    assert [bar["x"] for bar in oracles] == [["any"], ["any"]]
    assert [bar["y"][0] for bar in oracles] == pytest.approx([0.95, 0.9])
    assert sum(trace["showlegend"] for trace in data["data"]) == 2


def test_subset_bars_group_tools_by_subset():
    shares = {(tool, subset): Share(40, 100) for tool in TOOLS for subset in ("Subset A", "Subset B")}
    data = styled(figures.subset_bars(shares), "MolRecBench-Wild exact match by evaluation subset")
    assert [trace["x"] for trace in data["data"]] == [["Subset A", "Subset B"]] * 2


def test_hardcase_heatmap_keeps_frequent_labels_hardest_first():
    shares = {
        ("molscribe", "Wavy Bond"): Share(20, 200),
        ("molnextr", "Wavy Bond"): Share(10, 200),
        ("molscribe", "No hard-case label"): Share(180, 200),
        ("molnextr", "No hard-case label"): Share(150, 200),
        ("molscribe", "Polymer"): Share(1, 5),
        ("molnextr", "Polymer"): Share(1, 5),
    }
    data = styled(figures.hardcase_heatmap(shares, min_items=100), "MolRecBench-Wild exact match by hard-case label")
    heatmap = data["data"][0]
    assert heatmap["y"] == ["Wavy Bond (200)", "No hard-case label (200)"]
    assert heatmap["z"] == [[0.1, 0.05], [0.9, 0.75]]


def test_resource_bars_show_only_measured_panels():
    tools = [ToolResources("molscribe", 5.0, 3.0, 0.4), ToolResources("molvec", None, 1.0, 1.2)]
    data = styled(figures.resource_bars(tools), "Resources per tool, all datasets")
    assert [trace["y"] for trace in data["data"]] == [[5.0, None], [3.0, 1.0], [0.4, 1.2]]
    older = plain(figures.resource_bars([ToolResources("molscribe", None, None, 0.4)]))
    assert len(older["data"]) == 1


def test_unknown_tools_keep_their_key_and_a_neutral_colour():
    data = plain(figures.accuracy_bars({("fake", "uspto"): metrics("fake", "uspto")}))
    assert data["data"][0]["name"] == "fake"
    assert data["data"][0]["marker"]["color"] == figures.OTHER_COLOR
