"""Report figures: each builder returns a styled Plotly figure with the expected traces, and no wandb."""

import json

import pytest

go = pytest.importorskip("plotly.graph_objects")

from molscout.report import figures  # noqa: E402
from molscout.report.analysis import OUTCOMES, Share, ToolResources  # noqa: E402

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
    assert data["layout"]["paper_bgcolor"] == data["layout"]["plot_bgcolor"] == figures.TRANSPARENT
    return data


def test_accuracy_by_dataset_has_one_bar_trace_per_tool_with_ci_error_bars():
    data = styled(figures.accuracy_bars(METRICS), "Stereo-aware exact match by dataset")
    assert [trace["name"] for trace in data["data"]] == ["MolScribe", "MolNexTR"]
    first = data["data"][0]
    assert first["type"] == "bar" and first["x"] == ["USPTO", "MolRecBench-Wild"]
    assert first["marker"]["color"] == figures.TOOL_COLORS["molscribe"]
    assert first["error_y"]["array"] == pytest.approx([0.02, 0.02])
    assert first["error_y"]["arrayminus"] == pytest.approx([0.02, 0.02])
    assert data["layout"]["yaxis"]["tickformat"] == ".0%" and data["layout"]["yaxis"]["range"] == [0, 1.04]
    assert data["layout"]["height"] == 430


def test_accuracy_by_dataset_labels_only_each_datasets_best_bar_above_its_whisker():
    data = plain(figures.accuracy_bars(METRICS))
    notes = data["layout"]["annotations"]
    assert [note["text"] for note in notes] == ["<b>90.0%</b>", "<b>90.0%</b>"]
    assert [note["y"] for note in notes] == pytest.approx([0.92, 0.92])
    group = 1 - data["layout"]["bargap"]
    first_bar = -group / 2 + 0.5 * group / len(TOOLS)  # MolScribe's bar centre in the first group
    assert [note["x"] for note in notes] == pytest.approx([first_bar, 1 + first_bar])
    assert notes[0]["font"] == {"size": 11, "color": figures.INK}
    ticks = data["layout"]["xaxis"]["ticktext"]
    assert [tick.split("<br>")[0] for tick in ticks] == ["USPTO", "MolRecBench-Wild"] and "100 crops" in ticks[0]


def test_accuracy_by_dataset_hover_names_the_crop_count_once():
    # A category axis with ticktext shows the tick text for %{x}, and the ticks already carry the crop count.
    data = plain(figures.accuracy_bars(METRICS))
    assert all("crops" in tick for tick in data["layout"]["xaxis"]["ticktext"])
    for trace in data["data"]:
        assert "%{x}" in trace["hovertemplate"] and "crops" not in trace["hovertemplate"]


def test_stereo_penalty_is_in_percentage_points():
    data = styled(figures.stereo_penalty_bars(METRICS), "Points gained when stereochemistry is ignored")
    assert data["data"][1]["y"] == pytest.approx([3.0, 3.0])
    assert "+%{y:.1f} points" in data["data"][1]["hovertemplate"]
    assert data["layout"]["yaxis"]["title"]["text"] == "Points (stripped minus aware)"
    assert data["layout"]["yaxis"]["dtick"] == 2 and data["layout"]["height"] == 380


SUBSETS = {
    ("molscribe", "Subset A"): Share(80, 100),
    ("molscribe", "Subset B"): Share(20, 40),
    ("molnextr", "Subset A"): Share(60, 100),
    ("molnextr", "Subset B"): Share(8, 40),
}


def test_journal_crops_draw_one_line_per_tool_across_both_panels():
    with_jpo = {**METRICS, **{(tool, "jpo"): metrics(tool, "jpo") for tool in TOOLS}}
    data = styled(figures.journal_crop_lines(with_jpo, SUBSETS), "Journal crops: the ranking changes")
    assert len(data["data"]) == 2 * len(TOOLS)
    assert [trace["name"] for trace in data["data"] if trace.get("showlegend", True)] == ["MolScribe", "MolNexTR"]
    left, right = data["data"][:2]
    assert (left["xaxis"], right["xaxis"]) == ("x", "x2")
    assert [x.split("<br>")[0] for x in left["x"]] == ["USPTO, JPO", "MolRecBench-Wild"]
    assert "200 crops" in left["x"][0] and "100 crops" in left["x"][1]
    assert left["y"] == pytest.approx([0.85, 0.85])  # pooled from outcome counts, not the run's accuracy
    assert [x.split("<br>")[0] for x in right["x"]] == ["Subset A", "Subset B"] and "40 crops" in right["x"][1]
    assert right["y"] == pytest.approx([0.8, 0.5])
    titles = [note["text"] for note in data["layout"]["annotations"]]
    assert titles == ["Standard sets vs journal crops", "MolRecBench-Wild by evaluation subset"]
    assert data["layout"]["yaxis"]["range"] == [0, 1.04] and data["layout"]["height"] == 440
    assert data["layout"]["yaxis"]["dtick"] == 0.2 and data["layout"]["yaxis"]["tickformat"] == ".0%"


def test_hardcase_heatmap_keeps_frequent_labels_hardest_first_and_labels_every_cell():
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
    assert heatmap["y"] == ["Wavy Bond  (200)", "No hard-case label  (200)"]
    assert heatmap["z"] == [[0.1, 0.05], [0.9, 0.75]]
    assert heatmap["showscale"] is False
    notes = {note["text"]: note["font"]["color"] for note in data["layout"]["annotations"]}
    assert notes == {"10%": figures.INK, "5%": figures.INK, "90%": figures.SURFACE, "75%": figures.SURFACE}
    assert data["layout"]["title"]["subtitle"]["text"].startswith("Labels on at least 100 scored crops")
    assert data["layout"]["height"] == 120 + 30 * 2


LONG_LABEL = "Built-in Arrows, Dashed/Solid Boxes, Explanatory Text, Graphics"


def test_hardcase_labels_are_shown_whole_wrapped_onto_two_lines_with_the_count_last():
    shares = {("molscribe", "x" * 34): Share(1, 100), ("molscribe", LONG_LABEL): Share(2, 491)}
    names = plain(figures.hardcase_heatmap(shares))["data"][0]["y"]
    assert names == ["Built-in Arrows, Dashed/Solid<br>Boxes, Explanatory Text, Graphics  (491)", "x" * 34 + "  (100)"]
    for name in names:
        assert "\u2026" not in name and name.count("<br>") <= 1


def test_hardcase_labels_wrap_at_a_word_boundary_and_keep_every_word():
    label = "Additional Information at Screenshot Boundary"
    shares = {("molscribe", label): Share(1, 100), ("molscribe", "Wavy Bond"): Share(1, 100)}
    names = plain(figures.hardcase_heatmap(shares))["data"][0]["y"]
    wrapped = next(name for name in names if "<br>" in name)
    assert wrapped.split("  (")[0].replace("<br>", " ") == label
    assert "Wavy Bond  (100)" in names  # a short label stays on one line


def test_a_long_label_with_no_space_is_not_cut():
    shares = {("molscribe", "y" * 80): Share(1, 100)}
    assert plain(figures.hardcase_heatmap(shares))["data"][0]["y"] == ["y" * 80 + "  (100)"]


def test_error_mix_stacks_only_the_wrong_answers_per_dataset():
    data = styled(figures.error_mix_bars(METRICS), "What the wrong answers are")
    names = list(dict.fromkeys(trace["name"] for trace in data["data"]))
    assert names == ["Stereo only", "Wrong structure", "Invalid SMILES", "No output (empty or crash)"]
    assert len(data["data"]) == len(names) * len(DATASETS)
    assert sum(trace["showlegend"] for trace in data["data"]) == len(names)
    assert data["layout"]["barmode"] == "stack" and {trace["orientation"] for trace in data["data"]} == {"h"}
    no_output = data["data"][-1]
    assert no_output["y"] == ["MolScribe", "MolNexTR"] and no_output["x"] == pytest.approx([0.02, 0.02])
    assert data["layout"]["xaxis"]["range"] == [0, 0.64]
    clean = {key: {**row, "outcome/empty": 0} for key, row in METRICS.items()}
    assert "No output (empty or crash)" not in {trace["name"] for trace in plain(figures.error_mix_bars(clean))["data"]}


def test_every_outcome_has_a_name_and_a_colour_and_every_wrong_one_a_segment():
    assert set(figures.OUTCOME_NAMES) == set(figures.OUTCOME_COLORS) == set(OUTCOMES)
    counted = [outcome for _, outcomes, _ in figures.ERROR_SEGMENTS for outcome in outcomes]
    assert sorted(counted) == sorted(set(OUTCOMES) - {"correct"})


VOTE = {"uspto": Share(88, 100), "molrecbench_wild": Share(70, 100)}
ORACLE = {"uspto": Share(95, 100), "molrecbench_wild": Share(80, 100)}


def test_headroom_marks_the_best_tool_the_vote_and_the_oracle_per_dataset_and_pooled():
    data = styled(figures.headroom_dots(METRICS, VOTE, ORACLE), "Headroom from combining readers")
    traces = {trace.get("name"): trace for trace in data["data"]}
    best = traces["Best single reader"]
    assert best["y"] == ["USPTO", "MolRecBench-Wild", "All two, pooled"]
    assert best["x"] == pytest.approx([0.85, 0.85, 0.85]) and best["text"] == ["MolScribe"] * 3
    assert best["marker"]["color"] == [figures.TOOL_COLORS["molscribe"]] * 3
    assert traces["Plurality vote of two"]["x"] == pytest.approx([0.88, 0.7, 0.79])
    assert traces["Any reader right (oracle)"]["x"] == pytest.approx([0.95, 0.8, 0.875])
    spans = [trace["x"] for trace in data["data"] if trace["mode"] == "lines"]
    assert spans == [pytest.approx([0.85, 0.95]), pytest.approx([0.7, 0.8]), pytest.approx([0.79, 0.875])]
    legend = sorted((trace for trace in data["data"] if "legendrank" in trace), key=lambda trace: trace["legendrank"])
    assert [trace["name"] for trace in legend] == [
        "Best single reader (named)",
        "Plurality vote of two",
        "Any reader right (oracle)",
    ]
    assert data["layout"]["xaxis"]["range"] == [0.5, 1.0]


def test_headroom_draws_a_smaller_vote_diamond_ringed_in_the_surface_on_top_of_the_best_dot():
    traces = plain(figures.headroom_dots(METRICS, VOTE, ORACLE))["data"]
    names = [trace.get("name") for trace in traces]
    best = names.index("Best single reader")
    assert names.index("Plurality vote of two") < best and traces[best]["marker"]["line"] == {
        "color": figures.SURFACE,
        "width": 2,
    }
    # A smaller copy of the vote sits on top, so a vote that lands on the best reader still shows, and the
    # larger dot's colour stays visible around it.
    top = traces[-1]
    assert best < len(traces) - 1
    assert top["marker"]["symbol"] == "diamond" and top["x"] == traces[names.index("Plurality vote of two")]["x"]
    assert top["showlegend"] is False and top["hoverinfo"] == "skip"
    assert top["marker"]["line"] == {"color": figures.SURFACE, "width": 2}
    assert traces[best]["marker"]["size"] - top["marker"]["size"] >= 6
    assert traces[names.index("Best single reader (named)")]["marker"]["size"] == traces[best]["marker"]["size"]


@pytest.mark.parametrize("theme", [figures.LIGHT, figures.DARK])
def test_headroom_range_bar_takes_the_themes_track_colour(theme):
    traces = plain(figures.headroom_dots(METRICS, VOTE, ORACLE, theme=theme))["data"]
    assert {trace["line"]["color"] for trace in traces if trace["mode"] == "lines"} == {theme.track}


def test_the_track_is_a_hue_free_step_behind_the_text_in_dark_and_the_old_grey_in_light():
    assert figures.LIGHT.track == "#e1e0d9"
    red, green, blue = figures._rgb(figures.DARK.track)
    assert red == green == blue
    assert 1.8 <= figures._contrast(figures.DARK.track, figures.DARK.surface) <= 2.0
    assert figures._contrast(figures.DARK.track, figures.DARK.surface) < figures._contrast(
        figures.DARK.muted, figures.DARK.surface
    )


def test_journal_crops_leave_out_a_side_a_tool_did_not_run():
    wild_only = {**METRICS, ("molvec", "molrecbench_wild"): metrics("molvec", "molrecbench_wild")}
    data = plain(figures.journal_crop_lines(wild_only, SUBSETS))
    [left] = [trace for trace in data["data"] if trace["name"] == "MolVec" and trace.get("xaxis", "x") == "x"]
    assert len(left["x"]) == 1 and left["x"][0].startswith("MolRecBench-Wild")  # no 0% point for the standard sets
    assert left["y"] == pytest.approx([0.55])  # the pooled correct share, 55 of 100 (stereo-aware accuracy says 0.6)


def test_hardcase_heatmap_keeps_a_label_on_exactly_the_cut_share():
    shares = {("molscribe", "Blurry"): Share(50, 400), ("molscribe", "Any special format"): Share(60, 401)}
    names = plain(figures.hardcase_heatmap(shares, min_items=100, crops=1000))["data"][0]["y"]
    assert names == ["Blurry  (400)"]  # 400 of 1,000 is the 40% cut itself; 401 is over it


def test_hardcase_heatmap_leaves_out_labels_on_most_crops():
    shares = {
        ("molscribe", "Blurry"): Share(50, 200),
        ("molscribe", "Any special format"): Share(300, 600),
        ("molscribe", "Wavy Bond"): Share(40, 100),
    }
    data = plain(figures.hardcase_heatmap(shares, crops=1000))
    assert data["data"][0]["y"] == ["Blurry  (200)", "Wavy Bond  (100)"]  # 600 of 1,000 crops is over the 40% cut
    assert data["layout"]["title"]["subtitle"]["text"].startswith("Labels on 100 to 400 of the 1,000 scored crops;")


def test_accuracy_against_time_has_one_labelled_point_per_tool_and_no_legend():
    resources = [ToolResources("molnextr", 1.3, 2.2, 0.25), ToolResources("molscribe", 1.3, 2.2, 0.24)]
    data = styled(figures.accuracy_speed_scatter(METRICS, resources), "Accuracy against time per crop")
    assert [trace["name"] for trace in data["data"]] == ["MolScribe", "MolNexTR"]
    assert [trace["x"] for trace in data["data"]] == [[0.24], [0.25]]
    assert [trace["y"][0] for trace in data["data"]] == pytest.approx([0.85, 0.55])
    assert [trace["textposition"] for trace in data["data"]] == ["top center", "middle right"]
    assert data["layout"]["showlegend"] is False and data["layout"]["height"] == 360
    assert data["layout"]["xaxis"]["range"] == [0, 0.5]
    assert data["layout"]["yaxis"]["range"] == pytest.approx([0.55, 0.9])  # widened from 60-90% to show MolNexTR


@pytest.mark.parametrize(
    ("build", "facets"),
    [
        (lambda: figures.accuracy_bars(METRICS), False),
        (lambda: figures.stereo_penalty_bars(METRICS), False),
        (lambda: figures.journal_crop_lines(METRICS, SUBSETS), True),
        (lambda: figures.error_mix_bars(METRICS), True),
        (lambda: figures.headroom_dots(METRICS, VOTE, ORACLE), False),
    ],
)
def test_the_legend_sits_right_under_the_title(build, facets):
    layout = plain(build())["layout"]
    assert layout["title"]["yref"] == layout["legend"]["yref"] == "container"
    assert layout["title"]["y"] == pytest.approx(1 - 8 / layout["height"])
    assert layout["legend"]["y"] == pytest.approx(1 - 40 / layout["height"])
    assert layout["margin"]["t"] == (100 if facets else 72)


@pytest.mark.parametrize(
    "build",
    [
        lambda: figures.accuracy_bars(METRICS),
        lambda: figures.journal_crop_lines(METRICS, SUBSETS),
        lambda: figures.accuracy_speed_scatter(METRICS, [ToolResources("molscribe", 1.3, 2.2, 0.24)]),
    ],
)
def test_at_height_keeps_the_title_and_legend_the_same_pixels_from_the_top(build):
    figure = build()
    before = plain(figure)["layout"]
    taller = plain(figures.at_height(figure, before["height"] + 47))["layout"]
    assert plain(figure)["layout"] == before  # the figure itself is left as it was
    assert taller["height"] == before["height"] + 47
    for part in ("title", "legend"):
        if before.get(part, {}).get("yref", "container" if part == "title" else "paper") == "container":
            pixels = (1 - before[part]["y"]) * before["height"]
            assert (1 - taller[part]["y"]) * taller["height"] == pytest.approx(pixels)
        else:
            assert taller.get(part, {}).get("y") == before.get(part, {}).get("y")


def test_unknown_tools_keep_their_key_and_a_neutral_colour():
    data = plain(figures.accuracy_bars({("fake", "uspto"): metrics("fake", "uspto")}))
    assert data["data"][0]["name"] == "fake"
    assert data["data"][0]["marker"]["color"] == figures.OTHER_COLOR


HEADROOM = lambda theme: figures.headroom_dots(METRICS, VOTE, ORACLE, theme=theme)  # noqa: E731
HARDCASES = {
    ("molscribe", "Wavy Bond"): Share(20, 200),
    ("molnextr", "Wavy Bond"): Share(10, 200),
    ("molscribe", "Polymer"): Share(180, 200),
    ("molnextr", "Polymer"): Share(150, 200),
    ("molscribe", "Blurry"): Share(110, 200),
    ("molnextr", "Blurry"): Share(130, 200),
}
BUILDERS = [
    lambda theme: figures.accuracy_bars(METRICS, theme=theme),
    lambda theme: figures.stereo_penalty_bars(METRICS, theme=theme),
    lambda theme: figures.journal_crop_lines(METRICS, SUBSETS, theme=theme),
    lambda theme: figures.hardcase_heatmap(HARDCASES, theme=theme),
    lambda theme: figures.error_mix_bars(METRICS, theme=theme),
    HEADROOM,
    lambda theme: figures.accuracy_speed_scatter(METRICS, [ToolResources("molscribe", 1.3, 2.2, 0.24)], theme=theme),
]
LIGHT_NEUTRALS = {
    figures.LIGHT.surface,
    figures.LIGHT.ink,
    figures.LIGHT.ink_secondary,
    figures.LIGHT.grid,
    figures.LIGHT.axis,
}


def colours_in(node):
    """Every #rrggbb string anywhere in a figure's JSON, lower-cased."""
    if isinstance(node, str):
        return {node.lower()} if len(node) == 7 and node.startswith("#") else set()
    children = node.values() if isinstance(node, dict) else node if isinstance(node, list) else []
    return set().union(*(colours_in(child) for child in children))


@pytest.mark.parametrize("build", BUILDERS)
def test_a_dark_figure_uses_no_light_theme_neutral(build):
    dark = plain(build(figures.DARK))
    used = colours_in(dark)
    own = {getattr(figures.DARK, role) for role in ("surface", "ink", "ink_secondary", "grid", "axis")}
    assert not (used & (LIGHT_NEUTRALS - own))  # a role DARK shares with LIGHT is not a leftover
    assert used & {figures.DARK.ink, figures.DARK.ink_secondary}


@pytest.mark.parametrize("theme", [figures.LIGHT, figures.DARK])
@pytest.mark.parametrize("build", BUILDERS)
def test_every_figure_is_transparent_so_the_panel_shows_through(build, theme):
    layout = plain(build(theme))["layout"]
    assert layout["paper_bgcolor"] == layout["plot_bgcolor"] == "rgba(0,0,0,0)"


def test_rings_and_gaps_take_the_surface_of_their_theme():
    dark = plain(figures.headroom_dots(METRICS, VOTE, ORACLE, theme=figures.DARK))
    rings = {
        trace["marker"]["line"]["color"] for trace in dark["data"] if "color" in trace.get("marker", {}).get("line", {})
    }
    assert rings == {figures.DARK.surface}
    mix = plain(figures.error_mix_bars(METRICS, theme=figures.DARK))
    assert {trace["marker"]["line"]["color"] for trace in mix["data"]} == {figures.DARK.surface}


def test_dark_tools_and_outcomes_take_their_dark_colours():
    bars = plain(figures.accuracy_bars(METRICS, theme=figures.DARK))["data"]
    assert [trace["marker"]["color"] for trace in bars] == [figures.DARK.tool_colors[tool] for tool in TOOLS]
    assert figures.tool_color("fake", figures.DARK) == figures.DARK.other
    assert set(figures.DARK.tool_colors) == set(figures.LIGHT.tool_colors)
    assert set(figures.DARK.outcome_colors) == set(figures.LIGHT.outcome_colors)


@pytest.mark.parametrize("theme", [figures.LIGHT, figures.DARK])
def test_every_heatmap_cell_label_reaches_4_5_to_1_along_the_whole_ramp(theme):
    chosen = set()
    for step in range(1001):
        cell = figures._ramp_color(theme.sequential, step / 1000)
        color = figures._label_color(cell, theme)
        chosen.add(color)
        assert figures._contrast(color, cell) >= 4.5, (step, cell, color)
    assert {theme.ink, theme.surface} <= chosen  # both ends of the ramp keep the theme's own text colours
    assert chosen <= {theme.ink, theme.surface, "#ffffff", "#000000"}


def test_a_label_keeps_ink_or_surface_whenever_one_reaches_4_5_and_otherwise_takes_white_or_black():
    dark = figures.DARK
    mid = "#2a78d6"  # neither dark text nor the dark surface reaches 4.5 here
    assert max(figures._contrast(dark.ink, mid), figures._contrast(dark.surface, mid)) < 4.5
    assert figures._label_color(mid, dark) in {"#ffffff", "#000000"}
    assert figures._label_color(dark.sequential[0][1], dark) == dark.ink


@pytest.mark.parametrize("theme", [figures.LIGHT, figures.DARK])
def test_heatmap_cells_are_labelled_with_the_colour_of_their_value(theme):
    shares = {(tool, f"Label {n}"): Share(n * 10, 200) for tool in TOOLS for n in range(1, 20)}
    notes = plain(figures.hardcase_heatmap(shares, theme=theme))["layout"]["annotations"]
    assert len(notes) == 2 * 19
    for note in notes:
        cell = figures._ramp_color(theme.sequential, int(note["text"].rstrip("%")) / 100)
        assert note["font"]["color"] == figures._label_color(cell, theme)


def test_the_ramp_colour_blends_between_stops_like_plotly_does():
    ramp = [[0.0, "#000000"], [0.5, "#ff0000"], [1.0, "#ffffff"]]
    assert figures._ramp_color(ramp, 0) == "#000000" and figures._ramp_color(ramp, 0.25) == "#800000"
    assert figures._ramp_color(ramp, 1) == "#ffffff"


def test_text_beside_a_point_is_not_clipped_at_the_plot_edge():
    traces = plain(figures.headroom_dots(METRICS, VOTE, ORACLE))["data"]
    assert [trace["cliponaxis"] for trace in traces if "text" in trace] == [False]
    resources = [ToolResources("molscribe", 1.3, 2.2, 0.24)]
    scatter = plain(figures.accuracy_speed_scatter(METRICS, resources))["data"]
    assert all(trace["cliponaxis"] is False for trace in scatter)


@pytest.mark.parametrize("theme", [figures.LIGHT, figures.DARK])
@pytest.mark.parametrize("build", BUILDERS)
def test_the_toolbar_blends_into_the_panel_in_both_themes(build, theme):
    # Plotly derives the toolbar colours from paper_bgcolor, which is transparent: a solid grey box.
    assert plain(build(theme))["layout"]["modebar"] == {
        "bgcolor": theme.surface,
        "color": theme.muted,
        "activecolor": theme.ink,
    }


@pytest.mark.parametrize("theme", [figures.LIGHT, figures.DARK])
def test_toolbar_icons_and_the_active_icon_are_legible_on_the_surface(theme):
    assert figures._contrast(theme.muted, theme.surface) >= 3
    assert figures._contrast(theme.ink, theme.surface) >= 7


@pytest.mark.parametrize(
    "build",
    [
        lambda: figures.error_mix_bars(METRICS),
        lambda: figures.headroom_dots(METRICS, VOTE, ORACLE),
        lambda: figures.accuracy_speed_scatter(METRICS, [ToolResources("molscribe", 1.3, 2.2, 0.24)]),
    ],
)
def test_the_x_axis_title_clears_the_bottom_of_the_frame(build):
    assert plain(build())["layout"]["margin"]["b"] == 62  # was 56: the title ended 1 px from the frame


def test_the_100_percent_gridline_sits_clear_of_the_subplot_titles():
    layout = plain(figures.journal_crop_lines(METRICS, SUBSETS))["layout"]
    plot = layout["height"] - layout["margin"]["t"] - layout["margin"]["b"]
    gap = (1.04 - 1) / 1.04 * plot  # px from the 100% line to the top of the plot, where the titles start
    assert gap >= 8  # more than half a 12 px tick label
    assert all(note["yanchor"] == "bottom" and note["y"] == 1 for note in layout["annotations"])
