"""Plotly figures for the benchmark summary, built from report.analysis results; no wandb here.

One style throughout: fixed tool order and colours; dataset display names; percent axes.
Every array is a plain list, so the JSON W&B stores has no typed-array encoding.
Each figure is drawn at its design height; at_height redraws it to fill whole report panel rows.
Every builder takes a Theme (LIGHT or DARK): the colours that must match the surface the figure sits on.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from molscout.report.analysis import Share, ToolResources, dataset_rank, pooled_accuracy, tool_rank
from molscout.tables import TOOL_ROWS

Metrics = Mapping[tuple[str, str], Mapping[str, float]]  # (tool, dataset) -> analysis.run_metrics

WILD = "molrecbench_wild"  # the journal crops; the other datasets are the standard sets
DATASET_NAMES = {
    "uspto": "USPTO",
    "uob": "UOB",
    "jpo": "JPO",
    "clef": "CLEF",
    WILD: "MolRecBench-Wild",
}
OUTCOME_NAMES = {
    "correct": "Correct",
    "stereo_only": "Stereo only",
    "wrong_structure": "Wrong structure",
    "invalid": "Invalid SMILES",
    "empty": "Empty",
    "crashed": "Crashed",
}
# The wrong answers in stack order, each with the outcomes it counts and the outcome whose colour it takes;
# empty and crashed are both no output.
ERROR_SEGMENTS = (
    (OUTCOME_NAMES["stereo_only"], ("stereo_only",), "stereo_only"),
    (OUTCOME_NAMES["wrong_structure"], ("wrong_structure",), "wrong_structure"),
    (OUTCOME_NAMES["invalid"], ("invalid",), "invalid"),
    ("No output (empty or crash)", ("empty", "crashed"), "empty"),
)
# Where each tool's name sits beside its point in the accuracy-against-time scatter; any other tool: right.
SCATTER_LABELS = {"molscribe": "top center", "decimer": "middle left", "molglyph": "middle left"}
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
HARDCASE_MIN_ITEMS = 100
HARDCASE_MAX_SHARE = 0.4  # labels on more of the scored crops than this are umbrella groups, not hard cases
WRAP_LABEL = 36  # hard-case labels longer than this break onto a second line of the axis label
LABEL_CONTRAST = 4.5  # WCAG AA for the text on a heatmap cell
BEST_DOT = 16  # px; the best reader is a dot a ring larger than the vote diamond it may coincide with
VOTE_DIAMOND = 7
NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


@dataclass(frozen=True)
class Theme:
    """The colours of one surface: the chart chrome and ink, and every palette a figure draws from."""

    surface: str  # what the figure sits on; rings, gaps and marker outlines match it
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    track: str  # the neutral bar behind a range: more visible than the grid, well behind the text
    axis: str
    hover_bg: str
    hover_border: str
    hover_ink: str
    other: str  # a tool with no colour of its own
    tool_colors: Mapping[str, str]
    outcome_colors: Mapping[str, str]
    sequential: Sequence[Sequence[float | str]]  # the heatmap ramp: low values near the surface, high ones strong


# Light: slots 1-6 of the validated palette (worst adjacent CVD dE 9.1, normal-vision dE 19.6; the
# custom violet is slot 6). Outcomes: right in blues, wrong in oranges, no answer in greys.
LIGHT = Theme(
    surface="#ffffff",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    track="#e1e0d9",
    axis="#c3c2b7",
    hover_bg="#ffffff",
    hover_border="#c3c2b7",
    hover_ink="#0b0b0b",
    other="#898781",
    tool_colors={
        "molscribe": "#2a78d6",
        "molnextr": "#eb6834",
        "decimer": "#1baf7a",
        "molvec": "#eda100",
        "molglyph": "#e87ba4",
        "ocsrglyph": "#7b4fd0",
    },
    outcome_colors={
        "correct": "#256abf",
        "stereo_only": "#86b6ef",
        "wrong_structure": "#d95926",
        "invalid": "#f2a77f",
        "empty": "#52514e",
        "crashed": "#c3c2b7",
    },
    sequential=[[0.0, "#e8f1fd"], [0.5, "#5598e7"], [1.0, "#0d366b"]],
)
# Dark, on W&B's #1a1d24 panel: the same hues stepped for it (worst adjacent CVD dE 8.4, normal-vision dE 19.3).
# Ink is W&B's own dark-mode text, not pure white, so no light-theme neutral is reused.
DARK = Theme(
    surface="#1a1d24",
    ink="#e8e8e9",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    track="#4a4a4a",  # 1.9:1 on the surface, hue-free
    axis="#383835",
    hover_bg="#1a1d24",
    hover_border="#383835",
    hover_ink="#e8e8e9",
    other="#898781",
    tool_colors={
        "molscribe": "#3987e5",
        "molnextr": "#d95926",
        "decimer": "#199e70",
        "molvec": "#c98500",
        "molglyph": "#d55181",
        "ocsrglyph": "#9085e9",
    },
    outcome_colors={
        "correct": "#3987e5",
        "stereo_only": "#6da7ec",
        "wrong_structure": "#d95926",
        "invalid": "#f2a77f",
        "empty": "#898781",
        "crashed": "#6b6a65",
    },
    sequential=[[0.0, "#223049"], [0.5, "#2a78d6"], [1.0, "#b7d3f6"]],
)
# The light theme's values, for callers that name one.
TOOL_COLORS, OUTCOME_COLORS, SEQUENTIAL, OTHER_COLOR = (
    LIGHT.tool_colors,
    LIGHT.outcome_colors,
    LIGHT.sequential,
    LIGHT.other,
)
SURFACE, INK, INK_SECONDARY, GRID, AXIS = LIGHT.surface, LIGHT.ink, LIGHT.ink_secondary, LIGHT.grid, LIGHT.axis
TRANSPARENT = "rgba(0,0,0,0)"  # paper and plot, so the report panel's own background shows through


def tool_name(tool: str) -> str:
    return TOOL_ROWS.get(tool, tool)


def tool_color(tool: str, theme: Theme = LIGHT) -> str:
    return theme.tool_colors.get(tool, theme.other)


def dataset_name(dataset: str) -> str:
    return DATASET_NAMES.get(dataset, dataset)


def accuracy_bars(metrics: Metrics, *, theme: Theme = LIGHT) -> go.Figure:
    """Stereo-aware exact match per dataset, one bar per tool, with Wilson 95% CIs; each dataset's best bar is labelled."""
    tools, datasets = _tools(metrics), _datasets(metrics)
    figure = go.Figure()
    for tool in tools:
        present = _datasets_of(metrics, tool)
        rows = [metrics[(tool, dataset)] for dataset in present]
        values = [row["accuracy/stereo_aware"] for row in rows]
        low = [row["accuracy/stereo_aware_ci_low"] for row in rows]
        high = [row["accuracy/stereo_aware_ci_high"] for row in rows]
        figure.add_bar(
            name=tool_name(tool),
            x=[dataset_name(dataset) for dataset in present],
            y=values,
            marker_color=tool_color(tool, theme),
            error_y=_error_bars(values, low, high, theme),
            customdata=[[lo, hi] for lo, hi in zip(low, high, strict=True)],
            # %{x} shows the tick text, which already carries the crop count.
            hovertemplate=(
                "<b>%{fullData.name}</b> on %{x}<br>%{y:.1%} exact match "
                "(95% CI %{customdata[0]:.1%} to %{customdata[1]:.1%})<extra></extra>"
            ),
        )
    _style(figure, "Stereo-aware exact match by dataset", theme, height=430)
    _header(figure)
    # The best bar's value sits just above its whisker. On a category axis each group spans 1 - bargap,
    # split evenly between the tools; bargroupgap only narrows the bars inside their slots.
    group = 1 - figure.layout.bargap
    for index, dataset in enumerate(datasets):
        best = max(
            (tool for tool in tools if (tool, dataset) in metrics),
            key=lambda tool: metrics[(tool, dataset)]["accuracy/stereo_aware"],
        )
        row = metrics[(best, dataset)]
        figure.add_annotation(
            x=index - group / 2 + (tools.index(best) + 0.5) * group / len(tools),
            y=row["accuracy/stereo_aware_ci_high"],
            yshift=10,
            text=f"<b>{row['accuracy/stereo_aware']:.1%}</b>",
            showarrow=False,
            font={"size": 11, "color": theme.ink},
        )
    names = [dataset_name(dataset) for dataset in datasets]
    figure.update_xaxes(
        categoryorder="array",
        categoryarray=names,
        tickvals=names,
        ticktext=[
            _with_crops(name, _scored(metrics, [dataset])) for name, dataset in zip(names, datasets, strict=True)
        ],
    )
    figure.update_yaxes(title_text="Exact match (95% CI)", range=[0, 1.04], dtick=0.2, tickformat=".0%")
    figure.update_layout(margin={"l": 64, "r": 16, "b": 64})
    return figure


def stereo_penalty_bars(metrics: Metrics, *, theme: Theme = LIGHT) -> go.Figure:
    """Stereo-stripped minus stereo-aware exact match, in percentage points: what ignoring stereo gains each tool."""
    figure = go.Figure()
    for tool in _tools(metrics):
        datasets = _datasets_of(metrics, tool)
        rows = [metrics[(tool, dataset)] for dataset in datasets]
        figure.add_bar(
            name=tool_name(tool),
            x=[dataset_name(dataset) for dataset in datasets],
            y=[100 * (row["accuracy/stereo_stripped"] - row["accuracy/stereo_aware"]) for row in rows],
            marker_color=tool_color(tool, theme),
            customdata=[[row["accuracy/stereo_stripped"], row["accuracy/stereo_aware"]] for row in rows],
            hovertemplate=(
                "<b>%{fullData.name}</b> on %{x}<br>+%{y:.1f} points<br>"
                "stereo-stripped %{customdata[0]:.1%}, stereo-aware %{customdata[1]:.1%}<extra></extra>"
            ),
        )
    _style(figure, "Points gained when stereochemistry is ignored", theme, height=380)
    _header(figure)
    _dataset_axis(figure, metrics)
    figure.update_yaxes(title_text="Points (stripped minus aware)", rangemode="tozero", dtick=2)
    figure.update_layout(margin={"l": 64, "r": 16, "b": 48})
    return figure


def journal_crop_lines(
    metrics: Metrics, subsets: Mapping[tuple[str, str], Share], *, theme: Theme = LIGHT
) -> go.Figure:
    """One line per tool from the standard sets to MolRecBench-Wild, then across Wild's evaluation subsets.

    `subsets` is accuracy_by_group over the Wild crops and their subsets. The standard sets are pooled,
    each weighted by its scored crops.
    """
    tools = _tools(metrics)
    standard = [dataset for dataset in _datasets(metrics) if dataset != WILD]
    sides = [group for group in (standard, [WILD]) if group]
    left_x = [_with_crops(", ".join(dataset_name(d) for d in group), _scored(metrics, group)) for group in sides]
    names = sorted({subset for _, subset in subsets})
    sizes = {name: max(share.total for (_, subset), share in subsets.items() if subset == name) for name in names}
    right_x = [_with_crops(name, sizes[name]) for name in names]
    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[0.42, 0.58],
        horizontal_spacing=0.08,
        subplot_titles=["Standard sets vs journal crops", "MolRecBench-Wild by evaluation subset"],
    )
    for tool in tools:
        # A side the tool has no run on is left out, not drawn as 0%.
        pooled = [(x, pooled_accuracy(metrics, tool, group)) for x, group in zip(left_x, sides, strict=True)]
        left_at, left = [x for x, share in pooled if share.total], [share for _, share in pooled if share.total]
        right = [subsets[(tool, name)] for name in names if (tool, name) in subsets]
        line = {"color": tool_color(tool, theme), "width": 2}
        marker = {"color": tool_color(tool, theme), "size": 8, "line": {"color": theme.surface, "width": 1.5}}
        figure.add_scatter(
            row=1,
            col=1,
            name=tool_name(tool),
            legendgroup=tool,
            x=left_at,
            y=[share.value for share in left],
            mode="lines+markers",
            line=line,
            marker=marker,
            customdata=[[share.hits, share.total] for share in left],
            hovertemplate="<b>%{fullData.name}</b><br>%{y:.1%} (%{customdata[0]:,} of %{customdata[1]:,})<extra></extra>",
        )
        figure.add_scatter(
            row=1,
            col=2,
            name=tool_name(tool),
            legendgroup=tool,
            showlegend=False,
            x=[_with_crops(name, sizes[name]) for name in names if (tool, name) in subsets],
            y=[share.value for share in right],
            mode="lines+markers",
            line=line,
            marker=marker,
            customdata=[[share.hits, share.total, share.ci95[0], share.ci95[1]] for share in right],
            hovertemplate=(
                "<b>%{fullData.name}</b><br>%{y:.1%} (%{customdata[0]:,} of %{customdata[1]:,}; "
                "95% CI %{customdata[2]:.1%} to %{customdata[3]:.1%})<extra></extra>"
            ),
        )
    _style(figure, "Journal crops: the ranking changes", theme, height=440, facets=True)
    _header(figure, facets=True)
    figure.update_xaxes(type="category", showgrid=False)
    # Inset the end points so their two-line tick labels fit inside each panel.
    figure.update_xaxes(categoryorder="array", categoryarray=left_x, range=[-0.45, len(left_x) - 0.55], row=1, col=1)
    figure.update_xaxes(categoryorder="array", categoryarray=right_x, range=[-0.35, len(right_x) - 0.65], row=1, col=2)
    figure.update_yaxes(range=[0, 1.04], dtick=0.2, tickformat=".0%")  # room for the 100% line, as in the bars
    figure.update_yaxes(title_text="Stereo-aware exact match", row=1, col=1)
    figure.update_layout(margin={"l": 64, "r": 16, "b": 72})
    return figure


def hardcase_heatmap(
    shares: Mapping[tuple[str, str], Share],
    min_items: int = HARDCASE_MIN_ITEMS,
    *,
    crops: int | None = None,
    max_share: float = HARDCASE_MAX_SHARE,
    theme: Theme = LIGHT,
) -> go.Figure:
    """MolRecBench-Wild exact match per hard-case label and tool; labels on fewer than `min_items` crops are left out.

    With `crops`, the scored crops in all, labels on more than `max_share` of them are left out too.

    Rows run from the hardest label (lowest mean exact match over tools) down. Every cell carries its
    value, so there is no colour bar.
    """
    tools = sorted({tool for tool, _ in shares}, key=tool_rank)
    counts: dict[str, int] = {}
    for (_, label), share in shares.items():
        counts[label] = max(counts.get(label, 0), share.total)
    labels = [
        label for label, count in counts.items() if count >= min_items and (crops is None or count <= max_share * crops)
    ]
    present = {label: [shares[(tool, label)] for tool in tools if (tool, label) in shares] for label in labels}
    labels.sort(key=lambda label: (sum(s.value for s in present[label]) / len(present[label]), label))
    z = [[shares[(tool, label)].value if (tool, label) in shares else None for tool in tools] for label in labels]
    names = [f"{_wrapped(label)}  ({counts[label]:,})" for label in labels]
    full = [[f"{label} ({counts[label]:,} crops)"] * len(tools) for label in labels]
    figure = go.Figure(
        go.Heatmap(
            x=[tool_name(tool) for tool in tools],
            y=names,
            z=z,
            customdata=full,
            zmin=0,
            zmax=1,
            colorscale=theme.sequential,
            showscale=False,
            xgap=2,
            ygap=2,
            hovertemplate="<b>%{x}</b><br>%{customdata}<br>%{z:.1%} exact match<extra></extra>",
            hoverongaps=False,
        )
    )
    _style(
        figure, "MolRecBench-Wild exact match by hard-case label", theme, legend=False, height=120 + 30 * len(labels)
    )
    _header(figure)
    # Plotly's own subtitle stays inside the top margin; a <br> in the title text is clipped at the top.
    subtitle = f"Labels on at least {min_items} scored crops; a crop can carry several; crop counts in brackets"
    if crops is not None:
        subtitle = (
            f"Labels on {min_items:,} to {math.floor(max_share * crops):,} of the {crops:,} scored crops; a crop can "
            "carry several; crop counts in brackets"
        )
    figure.update_layout(title_subtitle={"text": subtitle, "font": {"size": 12, "color": theme.ink_secondary}})
    _cell_labels(figure, z, [tool_name(tool) for tool in tools], names, theme)
    figure.update_xaxes(showline=False, side="top", tickfont={"size": 12, "color": theme.ink})
    figure.update_yaxes(showline=False, showgrid=False, autorange="reversed", automargin=True, tickfont={"size": 12})
    figure.update_layout(margin={"l": 16, "r": 16, "t": 84, "b": 16})
    return figure


def error_mix_bars(metrics: Metrics, *, theme: Theme = LIGHT) -> go.Figure:
    """The wrong answers as a share of scored crops, stacked by kind: one horizontal bar per tool, a panel per dataset.

    Segments no run has are left out of the stack and the legend.
    """
    datasets = _datasets(metrics)
    shown = [segment for segment in ERROR_SEGMENTS if any(_count(row, segment[1]) for row in metrics.values())]
    figure = _facets(datasets, spacing=0.035)
    for name, outcomes, key in shown:
        for column, dataset in enumerate(datasets, 1):
            tools = [tool for tool in _tools(metrics) if (tool, dataset) in metrics]
            rows = [metrics[(tool, dataset)] for tool in tools]
            counts = [_count(row, outcomes) for row in rows]
            figure.add_bar(
                row=1,
                col=column,
                orientation="h",
                name=name,
                legendgroup=name,
                showlegend=column == 1,
                y=[tool_name(tool) for tool in tools],
                x=[_fraction(count, row["items/scored"]) for count, row in zip(counts, rows, strict=True)],
                marker={"color": theme.outcome_colors[key], "line": {"color": theme.surface, "width": 1}},
                customdata=[[count, row["items/scored"]] for count, row in zip(counts, rows, strict=True)],
                hovertemplate=(
                    f"<b>%{{y}}</b> on {dataset_name(dataset)}<br>{name}: %{{x:.1%}} "
                    "(%{customdata[0]:,} of %{customdata[1]:,} crops)<extra></extra>"
                ),
            )
    wrong = [_fraction(row["items/scored"] - row["outcome/correct"], row["items/scored"]) for row in metrics.values()]
    _style(figure, "What the wrong answers are", theme, height=360, facets=True)
    _header(figure, facets=True)
    figure.update_layout(barmode="stack", bargap=0.28, margin={"l": 84, "r": 16, "b": 62})
    figure.update_xaxes(
        range=_span(wrong, 0, 0.64, 0.01),
        dtick=0.25,
        tickformat=".0%",
        showgrid=True,
        gridcolor=theme.grid,
        showline=False,
    )
    figure.update_xaxes(title_text="Share of scored crops", row=1, col=(len(datasets) + 1) // 2)
    figure.update_yaxes(autorange="reversed", showgrid=False, showline=True, linecolor=theme.axis)
    return figure


def headroom_dots(
    metrics: Metrics, vote: Mapping[str, Share], oracle: Mapping[str, Share], *, theme: Theme = LIGHT
) -> go.Figure:
    """Per dataset and pooled: the best single tool, the plurality vote of all tools, and the oracle (any tool right).

    `vote` and `oracle` come from analysis.plurality_vote and analysis.oracle. A grey bar runs from the
    lower of the best tool and the vote up to the oracle. The best tool is a larger dot under the vote.
    """
    tools, datasets = _tools(metrics), _datasets(metrics)
    rows = [
        (dataset_name(dataset), *_best(metrics, tools, [dataset]), vote[dataset], oracle[dataset])
        for dataset in datasets
    ]
    if len(datasets) > 1:
        rows.append(
            (
                f"All {_spelled(len(datasets))}, pooled",
                *_best(metrics, tools, datasets),
                _summed(vote[dataset] for dataset in datasets),
                _summed(oracle[dataset] for dataset in datasets),
            )
        )
    labels = [label for label, *_ in rows]
    voted = f"Plurality vote of {_spelled(len(tools))}"
    figure = go.Figure()
    for label, _, single, plurality, right in rows:
        figure.add_scatter(
            x=[min(single.value, plurality.value), right.value],
            y=[label, label],
            mode="lines",
            line={"color": theme.track, "width": 6},
            showlegend=False,
            hoverinfo="skip",
        )
    figure.add_scatter(  # legend entry only: the real markers take each best tool's colour
        x=[None],
        y=[None],
        mode="markers",
        name="Best single reader (named)",
        legendrank=1,
        marker={"color": theme.other, "size": BEST_DOT, "line": {"color": theme.surface, "width": 2}},
    )
    figure.add_scatter(
        x=[plurality.value for *_, plurality, _ in rows],
        y=labels,
        mode="markers",
        name=voted,
        legendrank=2,
        marker={
            "symbol": "diamond",
            "color": theme.ink_secondary,
            "size": VOTE_DIAMOND,
            "line": {"color": theme.surface, "width": 2},
        },
        customdata=[[plurality.hits, plurality.total] for *_, plurality, _ in rows],
        hovertemplate="<b>%{y}</b><br>plurality vote: %{x:.1%} (%{customdata[0]:,} of %{customdata[1]:,})<extra></extra>",
    )
    figure.add_scatter(
        x=[right.value for *_, right in rows],
        y=labels,
        mode="markers",
        name="Any reader right (oracle)",
        legendrank=3,
        marker={"symbol": "circle-open", "color": theme.ink, "size": 12, "line": {"width": 2}},
        customdata=[[right.hits, right.total] for *_, right in rows],
        hovertemplate=(
            "<b>%{y}</b><br>at least one reader right: %{x:.1%} (%{customdata[0]:,} of %{customdata[1]:,})<extra></extra>"
        ),
    )
    figure.add_scatter(
        x=[single.value for _, _, single, _, _ in rows],
        y=labels,
        mode="markers+text",
        name="Best single reader",
        showlegend=False,
        marker={
            "color": [tool_color(tool, theme) for _, tool, *_ in rows],
            "size": BEST_DOT,
            "line": {"color": theme.surface, "width": 2},
        },
        text=[tool_name(tool) for _, tool, *_ in rows],
        textposition="top center",
        textfont={"size": 11, "color": theme.ink_secondary},
        cliponaxis=False,  # the name over the top row sits above the plot edge
        customdata=[[tool_name(tool), single.hits, single.total] for _, tool, single, _, _ in rows],
        hovertemplate=(
            "<b>%{y}</b><br>best single reader: %{customdata[0]}, %{x:.1%} "
            "(%{customdata[1]:,} of %{customdata[2]:,})<extra></extra>"
        ),
    )
    figure.add_scatter(  # the vote again, on top of the larger dot: where they coincide, a ring of the tool's colour shows
        x=[plurality.value for *_, plurality, _ in rows],
        y=labels,
        mode="markers",
        showlegend=False,
        hoverinfo="skip",
        marker={
            "symbol": "diamond",
            "color": theme.ink_secondary,
            "size": VOTE_DIAMOND,
            "line": {"color": theme.surface, "width": 2},
        },
    )
    values = [share.value for _, _, *shares in rows for share in shares]
    _style(figure, "Headroom from combining readers", theme, height=380)
    _header(figure)
    figure.update_yaxes(
        autorange="reversed", showgrid=False, type="category", tickfont={"size": 12, "color": theme.ink}
    )
    figure.update_xaxes(
        range=_span(values, 0.5, 1.0, 0.1),
        dtick=0.1,
        tickformat=".0%",
        showgrid=True,
        gridcolor=theme.grid,
        showline=False,
        title_text="Stereo-aware exact match",
    )
    figure.update_layout(margin={"l": 130, "r": 24, "b": 62})
    return figure


def accuracy_speed_scatter(metrics: Metrics, resources: Sequence[ToolResources], *, theme: Theme = LIGHT) -> go.Figure:
    """Exact match pooled over every dataset against mean seconds per crop over every crop read, one point per tool.

    Each point is named beside it, so there is no legend.
    """
    datasets = _datasets(metrics)
    ordered = sorted(resources, key=lambda tool: tool_rank(tool.tool))
    shares = [pooled_accuracy(metrics, tool.tool, datasets) for tool in ordered]
    figure = go.Figure()
    for tool, share in zip(ordered, shares, strict=True):
        figure.add_scatter(
            name=tool_name(tool.tool),
            x=[tool.s_per_crop_mean],
            y=[share.value],
            mode="markers+text",
            text=[tool_name(tool.tool)],
            textposition=SCATTER_LABELS.get(tool.tool, "middle right"),
            textfont={"size": 11, "color": theme.ink_secondary},
            cliponaxis=False,
            marker={"color": tool_color(tool.tool, theme), "size": 12, "line": {"color": theme.surface, "width": 1.5}},
            hovertemplate="<b>%{fullData.name}</b><br>%{y:.1%} exact match, pooled<br>%{x:.3f} s per crop<extra></extra>",
        )
    _style(figure, "Accuracy against time per crop", theme, legend=False, height=360)
    figure.update_xaxes(
        range=_span([tool.s_per_crop_mean for tool in ordered], 0, 0.5, 0.1),
        tickvals=[0.1, 0.2, 0.3, 0.4, 0.5],
        title_text="Mean seconds per crop, all crops",
        showgrid=True,
        gridcolor=theme.grid,
    )
    figure.update_yaxes(
        range=_span([share.value for share in shares], 0.6, 0.9, 0.05),
        dtick=0.05,
        tickformat=".0%",
        title_text="Stereo-aware exact match, pooled",
    )
    figure.update_layout(margin={"l": 64, "r": 16, "t": 56, "b": 62})
    return figure


def _style(
    figure: go.Figure, title: str, theme: Theme, *, legend: bool = True, facets: bool = False, height: int = 460
) -> go.Figure:
    """The house style: left-aligned title, legend in a row beneath it, hairline grid, no chart chrome."""
    figure.update_layout(
        template="none",
        title={
            "text": title,
            "x": 0.01,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "font": {"size": 16, "color": theme.ink},
        },
        font={"family": FONT, "size": 12, "color": theme.ink_secondary},
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        # Plotly derives the toolbar from paper_bgcolor, so a transparent paper gives a solid grey box.
        modebar={"bgcolor": theme.surface, "color": theme.muted, "activecolor": theme.ink},
        height=height,
        margin={"l": 72, "r": 24, "t": 120 if facets else 96, "b": 72},
        showlegend=legend,
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": 1.13 if facets else 1.02,
            "yanchor": "bottom",
            "font": {"size": 12},
        },
        hoverlabel={
            "bgcolor": theme.hover_bg,
            "bordercolor": theme.hover_border,
            "font": {"family": FONT, "size": 12, "color": theme.hover_ink},
        },
        bargap=0.22,
        bargroupgap=0.06,
    )
    figure.update_xaxes(showgrid=False, showline=True, linecolor=theme.axis, ticks="", zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor=theme.grid, showline=False, ticks="", zeroline=False)
    figure.update_annotations(font={"size": 13, "color": theme.ink})
    return figure


def at_height(figure: go.Figure, height: int) -> go.Figure:
    """A copy `height` px tall, its title and a container-anchored legend kept the same pixels from the top.

    Container coordinates are fractions of the figure's height, so they are rescaled; paper coordinates follow
    the plot area and stay as they are.
    """
    resized = go.Figure(figure).update_layout(height=height)
    old = figure.layout.height
    title, legend = figure.layout.title, figure.layout.legend
    if title.y is not None and title.yref in (None, "container"):
        resized.layout.title.y = 1 - (1 - title.y) * old / height
    if legend.y is not None and legend.yref == "container":
        resized.layout.legend.y = 1 - (1 - legend.y) * old / height
    return resized


def _header(figure: go.Figure, *, facets: bool = False) -> go.Figure:
    """The title 8 px and the legend 40 px from the top, so the legend sits right under the title; after _style.

    _style anchors the legend to the plot area, which leaves a gap under the title that grows with the
    figure's height. Container coordinates fix both in pixels; the plot (or its facet titles) starts below.
    """
    height = figure.layout.height
    figure.update_layout(
        title={"y": 1 - 8 / height, "yref": "container", "yanchor": "top"},
        legend={"y": 1 - 40 / height, "yref": "container", "yanchor": "top"},
        margin={"t": 100 if facets else 72},
    )
    return figure


def _facets(datasets: Sequence[str], spacing: float = 0.025) -> go.Figure:
    return make_subplots(
        rows=1,
        cols=len(datasets),
        shared_yaxes=True,
        subplot_titles=[dataset_name(dataset) for dataset in datasets],
        horizontal_spacing=spacing,
    )


def _dataset_axis(figure: go.Figure, metrics: Metrics) -> None:
    figure.update_xaxes(categoryorder="array", categoryarray=[dataset_name(dataset) for dataset in _datasets(metrics)])


def _error_bars(
    values: Sequence[float], low: Sequence[float], high: Sequence[float], theme: Theme
) -> dict[str, object]:
    return {
        "type": "data",
        "symmetric": False,
        "array": [hi - value for value, hi in zip(values, high, strict=True)],
        "arrayminus": [value - lo for value, lo in zip(values, low, strict=True)],
        "color": theme.ink_secondary,
        "thickness": 1,
        "width": 3,
    }


def _cell_labels(
    figure: go.Figure, z: Sequence[Sequence[float | None]], x: Sequence[str], y: Sequence[str], theme: Theme
) -> None:
    """Each filled cell's value as a percent, in a colour that reads on the cell; see _label_color.

    After _style, which resets annotation fonts.
    """
    for row, values in zip(y, z, strict=True):
        for column, value in zip(x, values, strict=True):
            if value is not None:
                color = _label_color(_ramp_color(theme.sequential, value), theme)
                figure.add_annotation(
                    x=column, y=row, text=f"{value:.0%}", showarrow=False, font={"size": 11, "color": color}
                )


def _label_color(cell: str, theme: Theme) -> str:
    """Whichever of ink and surface contrasts more with `cell`, or pure white or black where neither reaches 4.5:1.

    A continuous ramp crosses a band of mid tones where neither theme colour does; the better of white and black
    always reaches 4.58:1.
    """
    best = max((theme.ink, theme.surface), key=lambda text: _contrast(text, cell))
    if _contrast(best, cell) >= LABEL_CONTRAST:
        return best
    return max(("#ffffff", "#000000"), key=lambda text: _contrast(text, cell))


def _rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def _ramp_color(ramp: Sequence[Sequence[float | str]], value: float) -> str:
    """The colour Plotly draws for `value` on a 0 to 1 colour scale: a straight blend in RGB between stops."""
    for (low, from_color), (high, to_color) in zip(ramp, ramp[1:], strict=False):
        if value <= high:
            t = (value - low) / (high - low)
            return "#" + "".join(f"{round(a + (b - a) * t):02x}" for a, b in zip(_rgb(from_color), _rgb(to_color)))
    return str(ramp[-1][1])


def _luminance(color: str) -> float:
    linear = [c / 255 / 12.92 if c / 255 <= 0.04045 else ((c / 255 + 0.055) / 1.055) ** 2.4 for c in _rgb(color)]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(first: str, second: str) -> float:
    """WCAG contrast ratio of two #rrggbb colours."""
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _span(values: Iterable[float], low: float, high: float, step: float) -> list[float]:
    """The axis range [low, high], widened in whole steps only where a value falls outside it."""
    values = list(values)
    if values and min(values) < low:
        low = math.floor(min(values) / step) * step
    if values and max(values) > high:
        high = math.ceil(max(values) / step) * step
    return [low, high]


def _best(metrics: Metrics, tools: Sequence[str], datasets: Sequence[str]) -> tuple[str, Share]:
    """The tool with the highest exact match pooled over `datasets`, and that exact match; ties go to tool order."""
    shares = {tool: pooled_accuracy(metrics, tool, datasets) for tool in tools}
    best = max(tools, key=lambda tool: shares[tool].value)
    return best, shares[best]


def _summed(shares: Iterable[Share]) -> Share:
    shares = list(shares)
    return Share(sum(share.hits for share in shares), sum(share.total for share in shares))


def _scored(metrics: Metrics, datasets: Sequence[str]) -> int:
    """Scored crops over `datasets`; every tool is scored on the same crops of a dataset."""
    return sum(max(int(row["items/scored"]) for (_, name), row in metrics.items() if name == d) for d in datasets)


def _with_crops(label: str, crops: int) -> str:
    """A tick label with the crop count on a smaller second line."""
    return f"{label}<br><span style='font-size:11px'>{crops:,} crops</span>"


def _count(row: Mapping[str, float], outcomes: Sequence[str]) -> int:
    return int(sum(row[f"outcome/{outcome}"] for outcome in outcomes))


def _fraction(count: float, total: float) -> float:
    return count / total if total else 0.0


def _spelled(count: int) -> str:
    return NUMBER_WORDS[count] if count < len(NUMBER_WORDS) else f"{count:,}"


def _tools(metrics: Metrics) -> list[str]:
    return sorted({tool for tool, _ in metrics}, key=tool_rank)


def _datasets(metrics: Metrics) -> list[str]:
    return sorted({dataset for _, dataset in metrics}, key=dataset_rank)


def _datasets_of(metrics: Metrics, tool: str) -> list[str]:
    return [dataset for dataset in _datasets(metrics) if (tool, dataset) in metrics]


def _wrapped(label: str) -> str:
    """A label longer than WRAP_LABEL broken in two at the space nearest its middle; one without a space is left whole."""
    spaces = [index for index, char in enumerate(label) if char == " "]
    if len(label) <= WRAP_LABEL or not spaces:
        return label
    cut = min(spaces, key=lambda index: abs(index - len(label) / 2))
    return f"{label[:cut]}<br>{label[cut + 1 :]}"
