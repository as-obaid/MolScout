"""Plotly figures for the benchmark summary, built from report.analysis results; no wandb here.

One style throughout: fixed tool order, colours and markers; dataset display names; percent axes.
Every array is a plain list, so the JSON W&B stores has no typed-array encoding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from molscout.report.analysis import BoxStats, Share, ToolResources, dataset_rank, tool_rank
from molscout.tables import TOOL_ROWS

Metrics = Mapping[tuple[str, str], Mapping[str, float]]  # (tool, dataset) -> analysis.run_metrics

DATASET_NAMES = {
    "uspto": "USPTO",
    "uob": "UOB",
    "jpo": "JPO",
    "clef": "CLEF",
    "molrecbench_wild": "MolRecBench-Wild",
}
# Categorical slots 1-6 of the validated palette, in this order (adjacent CVD separation >= 9).
TOOL_COLORS = {
    "molscribe": "#2a78d6",
    "molnextr": "#eb6834",
    "decimer": "#1baf7a",
    "molvec": "#eda100",
    "molglyph": "#e87ba4",
    "ocsrglyph": "#008300",
}
TOOL_SYMBOLS = {
    "molscribe": "circle",
    "molnextr": "square",
    "decimer": "diamond",
    "molvec": "triangle-up",
    "molglyph": "triangle-down",
    "ocsrglyph": "star",
}
OUTCOME_NAMES = {
    "correct": "Correct",
    "stereo_only": "Stereo only",
    "wrong_structure": "Wrong structure",
    "invalid": "Invalid SMILES",
    "empty": "Empty",
    "crashed": "Crashed",
}
# Right in blues, wrong in oranges, no answer in greys; the stack runs from right to no answer.
OUTCOME_COLORS = {
    "correct": "#256abf",
    "stereo_only": "#86b6ef",
    "wrong_structure": "#d95926",
    "invalid": "#f2a77f",
    "empty": "#52514e",
    "crashed": "#c3c2b7",
}
STACK_ORDER = ("correct", "stereo_only", "wrong_structure", "invalid", "empty", "crashed")
SEQUENTIAL = [[0.0, "#e8f1fd"], [0.5, "#5598e7"], [1.0, "#0d366b"]]
OTHER_COLOR = "#898781"
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MIN_SECONDS = 1e-3  # floor for a log axis; a crop timed at 0 s is drawn here
# 1-2-5 ticks for the log axes, 1 ms to 500 s; Plotly's default labels every minor tick.
LOG_TICKS = [base * 10.0**power for power in range(-3, 3) for base in (1, 2, 5)]
HARDCASE_MIN_ITEMS = 100
MAX_LABEL = 40  # characters of a hard-case label on the axis; the hover shows it whole
DARK_CELL = 0.55  # heatmap values above this get white text


def tool_name(tool: str) -> str:
    return TOOL_ROWS.get(tool, tool)


def tool_color(tool: str) -> str:
    return TOOL_COLORS.get(tool, OTHER_COLOR)


def dataset_name(dataset: str) -> str:
    return DATASET_NAMES.get(dataset, dataset)


def accuracy_bars(metrics: Metrics) -> go.Figure:
    """Stereo-aware exact match per dataset, one bar per tool, with Wilson 95% CIs."""
    figure = go.Figure()
    for tool in _tools(metrics):
        datasets = _datasets_of(metrics, tool)
        rows = [metrics[(tool, dataset)] for dataset in datasets]
        values = [row["accuracy/stereo_aware"] for row in rows]
        low = [row["accuracy/stereo_aware_ci_low"] for row in rows]
        high = [row["accuracy/stereo_aware_ci_high"] for row in rows]
        figure.add_bar(
            name=tool_name(tool),
            x=[dataset_name(dataset) for dataset in datasets],
            y=values,
            marker_color=tool_color(tool),
            error_y=_error_bars(values, low, high),
            customdata=[[lo, hi, row["items/scored"]] for lo, hi, row in zip(low, high, rows, strict=True)],
            hovertemplate=(
                "<b>%{fullData.name}</b> on %{x}<br>%{y:.1%} exact match "
                "(95% CI %{customdata[0]:.1%} to %{customdata[1]:.1%})<br>%{customdata[2]:,} crops<extra></extra>"
            ),
        )
    _style(figure, "Stereo-aware exact match by dataset")
    _dataset_axis(figure, metrics)
    figure.update_yaxes(title_text="Exact match (95% CI)", **_PERCENT)
    return figure


def stereo_penalty_bars(metrics: Metrics) -> go.Figure:
    """Stereo-stripped minus stereo-aware exact match, in percentage points: what stereo costs each tool."""
    figure = go.Figure()
    for tool in _tools(metrics):
        datasets = _datasets_of(metrics, tool)
        rows = [metrics[(tool, dataset)] for dataset in datasets]
        figure.add_bar(
            name=tool_name(tool),
            x=[dataset_name(dataset) for dataset in datasets],
            y=[100 * (row["accuracy/stereo_stripped"] - row["accuracy/stereo_aware"]) for row in rows],
            marker_color=tool_color(tool),
            customdata=[[row["accuracy/stereo_stripped"], row["accuracy/stereo_aware"]] for row in rows],
            hovertemplate=(
                "<b>%{fullData.name}</b> on %{x}<br>%{y:.1f} points<br>"
                "stereo-stripped %{customdata[0]:.1%}, stereo-aware %{customdata[1]:.1%}<extra></extra>"
            ),
        )
    _style(figure, "Stereo penalty: stereo-stripped minus stereo-aware")
    _dataset_axis(figure, metrics)
    figure.update_yaxes(title_text="Percentage points", rangemode="tozero")
    return figure


def accuracy_speed_scatter(metrics: Metrics) -> go.Figure:
    """Exact match against mean seconds per crop, one panel per dataset on a shared log axis."""
    datasets = _datasets(metrics)
    figure = _facets(datasets)
    for tool in _tools(metrics):
        for column, dataset in enumerate(datasets, 1):
            if (tool, dataset) not in metrics:
                continue
            row = metrics[(tool, dataset)]
            figure.add_scatter(
                row=1,
                col=column,
                name=tool_name(tool),
                legendgroup=tool,
                showlegend=column == 1,
                x=[max(row["speed/s_per_crop_mean"], MIN_SECONDS)],
                y=[row["accuracy/stereo_aware"]],
                mode="markers",
                marker={
                    "color": tool_color(tool),
                    "symbol": TOOL_SYMBOLS.get(tool, "circle"),
                    "size": 11,
                    "line": {"color": SURFACE, "width": 1.5},
                },
                hovertemplate=(
                    f"<b>%{{fullData.name}}</b> on {dataset_name(dataset)}<br>"
                    "%{y:.1%} exact match<br>%{x:.3f} s per crop<extra></extra>"
                ),
            )
    _style(figure, "Exact match against time per crop", facets=True)
    figure.update_xaxes(type="log", matches="x", showgrid=True, gridcolor=GRID, tickvals=LOG_TICKS, tickformat="~g")
    figure.update_xaxes(title_text="Mean seconds per crop (log scale)", row=1, col=(len(datasets) + 1) // 2)
    figure.update_yaxes(**_PERCENT)
    figure.update_yaxes(title_text="Stereo-aware exact match", row=1, col=1)
    return figure


def outcome_bars(metrics: Metrics) -> go.Figure:
    """Every scored crop's outcome as a share of the run, one 100% stacked bar per tool, a panel per dataset."""
    datasets = _datasets(metrics)
    shown = [name for name in STACK_ORDER if any(row[f"outcome/{name}"] for row in metrics.values())]
    figure = _facets(datasets)
    for name in shown:
        for column, dataset in enumerate(datasets, 1):
            tools = [tool for tool in _tools(metrics) if (tool, dataset) in metrics]
            rows = [metrics[(tool, dataset)] for tool in tools]
            figure.add_bar(
                row=1,
                col=column,
                name=OUTCOME_NAMES[name],
                legendgroup=name,
                showlegend=column == 1,
                x=[tool_name(tool) for tool in tools],
                y=[row[f"outcome/{name}"] / row["items/scored"] if row["items/scored"] else 0.0 for row in rows],
                marker={"color": OUTCOME_COLORS[name], "line": {"color": SURFACE, "width": 1}},
                customdata=[[row[f"outcome/{name}"], row["items/scored"]] for row in rows],
                hovertemplate=(
                    f"<b>%{{x}}</b> on {dataset_name(dataset)}<br>{OUTCOME_NAMES[name]}: %{{y:.1%}} "
                    "(%{customdata[0]:,} of %{customdata[1]:,} crops)<extra></extra>"
                ),
            )
    _style(figure, "Outcome of every scored crop", facets=True)
    figure.update_layout(barmode="stack", bargap=0.2)
    figure.update_xaxes(tickangle=-45)
    figure.update_yaxes(range=[0, 1], tickformat=".0%")
    figure.update_yaxes(title_text="Share of scored crops", row=1, col=1)
    return figure


def time_boxes(stats: Mapping[str, BoxStats]) -> go.Figure:
    """Seconds per crop for each tool over all its datasets: quartile box, 1.5 IQR whiskers and mean."""
    figure = go.Figure()
    for tool in sorted(stats, key=tool_rank):
        box = stats[tool]
        figure.add_box(
            name=tool_name(tool),
            x=[tool_name(tool)],
            q1=[_positive(box.q1)],
            median=[_positive(box.median)],
            q3=[_positive(box.q3)],
            lowerfence=[_positive(box.low)],
            upperfence=[_positive(box.high)],
            mean=[_positive(box.mean)],
            boxmean=True,
            marker_color=tool_color(tool),
            line={"color": tool_color(tool), "width": 1.5},
            fillcolor=_tint(tool_color(tool)),
            hoverinfo="y",  # Plotly labels each statistic: median, quartiles, fences, mean
        )
    _style(figure, "Time per crop, all datasets pooled", legend=False)
    figure.update_yaxes(type="log", tickvals=LOG_TICKS, tickformat="~g", title_text="Seconds per crop (log scale)")
    return figure


def agreement_heatmap(shares: Mapping[tuple[str, str], Share]) -> go.Figure:
    """Share of crops where two tools give the same canonical SMILES, each pair once (lower triangle)."""
    tools = sorted({tool for pair in shares for tool in pair}, key=tool_rank)
    rows, columns = tools[1:], tools[:-1]
    cells = [[_pair(shares, column, row) if _before(column, row) else None for column in columns] for row in rows]
    z = [[None if cell is None else cell.value for cell in line] for line in cells]
    totals = [[None if cell is None else cell.total for cell in line] for line in cells]
    figure = go.Figure(
        go.Heatmap(
            x=[tool_name(tool) for tool in columns],
            y=[tool_name(tool) for tool in rows],
            z=z,
            customdata=totals,
            zmin=0,
            zmax=1,
            colorscale=SEQUENTIAL,
            xgap=2,
            ygap=2,
            colorbar={"title": {"text": "Agreement"}, "tickformat": ".0%", "thickness": 12, "outlinewidth": 0},
            hovertemplate=(
                "<b>%{y}</b> and <b>%{x}</b><br>same answer on %{z:.1%} of %{customdata:,} crops<extra></extra>"
            ),
            hoverongaps=False,
        )
    )
    _style(figure, "Pairwise agreement, all datasets pooled", legend=False)
    _cell_labels(figure, z, [tool_name(tool) for tool in columns], [tool_name(tool) for tool in rows])
    figure.update_xaxes(showline=False, side="bottom")
    figure.update_yaxes(showline=False, showgrid=False, autorange="reversed")
    return figure


def consensus_bars(agreement: Mapping[tuple[str, int], Share], oracle: Mapping[str, Share]) -> go.Figure:
    """Per dataset: how often the answer k tools agree on is right, and the oracle (any tool right) as the last bar."""
    datasets = sorted({dataset for dataset, _ in agreement} | set(oracle), key=dataset_rank)
    figure = _facets(datasets)
    for column, dataset in enumerate(datasets, 1):
        ks = sorted(k for name, k in agreement if name == dataset)
        shares = [agreement[(dataset, k)] for k in ks]
        figure.add_bar(
            row=1,
            col=column,
            name="Agreed answer right",
            legendgroup="agreed",
            showlegend=column == 1,
            x=[str(k) for k in ks],
            y=[share.value for share in shares],
            marker_color=INK_SECONDARY,
            customdata=[[share.hits, share.total] for share in shares],
            hovertemplate=(
                f"<b>{dataset_name(dataset)}</b>: %{{x}} tools agree on %{{customdata[1]:,}} crops<br>"
                "their answer is right on %{y:.1%} (%{customdata[0]:,})<extra></extra>"
            ),
        )
        if dataset in oracle:
            share = oracle[dataset]
            figure.add_bar(
                row=1,
                col=column,
                name="Any tool right (oracle)",
                legendgroup="oracle",
                showlegend=column == 1,
                x=["any"],
                y=[share.value],
                marker_color=AXIS,
                customdata=[[share.hits, share.total]],
                hovertemplate=(
                    f"<b>{dataset_name(dataset)}</b>: at least one tool is right on %{{y:.1%}}<br>"
                    "(%{customdata[0]:,} of %{customdata[1]:,} crops)<extra></extra>"
                ),
            )
    _style(figure, "Exact match when k tools agree, and when any tool is right", facets=True)
    figure.update_xaxes(type="category")
    figure.update_xaxes(title_text="Tools agreeing on one answer (k)", row=1, col=(len(datasets) + 1) // 2)
    figure.update_yaxes(**_PERCENT)
    figure.update_yaxes(title_text="Exact match", row=1, col=1)
    return figure


def subset_bars(shares: Mapping[tuple[str, str], Share]) -> go.Figure:
    """MolRecBench-Wild exact match per evaluation subset, one bar per tool, with Wilson 95% CIs."""
    figure = go.Figure()
    for tool in sorted({tool for tool, _ in shares}, key=tool_rank):
        subsets = sorted(subset for name, subset in shares if name == tool)
        values = [shares[(tool, subset)] for subset in subsets]
        figure.add_bar(
            name=tool_name(tool),
            x=subsets,
            y=[share.value for share in values],
            marker_color=tool_color(tool),
            error_y=_error_bars([s.value for s in values], [s.ci95[0] for s in values], [s.ci95[1] for s in values]),
            customdata=[[share.ci95[0], share.ci95[1], share.total] for share in values],
            hovertemplate=(
                "<b>%{fullData.name}</b>, %{x}<br>%{y:.1%} exact match "
                "(95% CI %{customdata[0]:.1%} to %{customdata[1]:.1%})<br>%{customdata[2]:,} crops<extra></extra>"
            ),
        )
    _style(figure, "MolRecBench-Wild exact match by evaluation subset")
    figure.update_yaxes(title_text="Exact match (95% CI)", **_PERCENT)
    return figure


def hardcase_heatmap(shares: Mapping[tuple[str, str], Share], min_items: int = HARDCASE_MIN_ITEMS) -> go.Figure:
    """MolRecBench-Wild exact match per hard-case label and tool; labels on fewer than `min_items` crops are left out.

    Rows run from the hardest label (lowest mean exact match over tools) down.
    """
    tools = sorted({tool for tool, _ in shares}, key=tool_rank)
    counts: dict[str, int] = {}
    for (_, label), share in shares.items():
        counts[label] = max(counts.get(label, 0), share.total)
    labels = [label for label, count in counts.items() if count >= min_items]
    present = {label: [shares[(tool, label)] for tool in tools if (tool, label) in shares] for label in labels}
    labels.sort(key=lambda label: (sum(s.value for s in present[label]) / len(present[label]), label))
    z = [[shares[(tool, label)].value if (tool, label) in shares else None for tool in tools] for label in labels]
    names = [f"{_shorten(label)} ({counts[label]:,})" for label in labels]
    full = [[f"{label} ({counts[label]:,} crops)"] * len(tools) for label in labels]
    figure = go.Figure(
        go.Heatmap(
            x=[tool_name(tool) for tool in tools],
            y=names,
            z=z,
            customdata=full,
            zmin=0,
            zmax=1,
            colorscale=SEQUENTIAL,
            xgap=2,
            ygap=2,
            colorbar={"title": {"text": "Exact match"}, "tickformat": ".0%", "thickness": 12, "outlinewidth": 0},
            hovertemplate="<b>%{x}</b><br>%{customdata}<br>%{z:.1%} exact match<extra></extra>",
            hoverongaps=False,
        )
    )
    _style(figure, "MolRecBench-Wild exact match by hard-case label", legend=False, height=150 + 24 * len(labels))
    _cell_labels(figure, z, [tool_name(tool) for tool in tools], names)
    figure.update_xaxes(showline=False, side="top")
    figure.update_yaxes(showline=False, showgrid=False, autorange="reversed", automargin=True)
    figure.update_layout(margin={"t": 110})
    return figure


def resource_bars(tools: Sequence[ToolResources]) -> go.Figure:
    """Peak GPU memory and RAM over each tool's runs, and its mean seconds per crop; unmeasured panels are left out."""
    ordered = sorted(tools, key=lambda resources: tool_rank(resources.tool))
    panels = [
        ("Peak GPU memory (GiB)", [t.gpu_peak_memory_gib for t in ordered], "%{y:.1f} GiB"),
        ("Peak RAM (GiB)", [t.peak_rss_gib for t in ordered], "%{y:.1f} GiB"),
        ("Mean seconds per crop", [t.s_per_crop_mean for t in ordered], "%{y:.3f} s"),
    ]
    panels = [panel for panel in panels if any(value is not None for value in panel[1])]
    figure = make_subplots(
        rows=1, cols=len(panels), subplot_titles=[title for title, _, _ in panels], horizontal_spacing=0.08
    )
    for column, (title, values, shown) in enumerate(panels, 1):
        figure.add_bar(
            row=1,
            col=column,
            x=[tool_name(t.tool) for t in ordered],
            y=values,
            marker_color=[tool_color(t.tool) for t in ordered],
            hovertemplate=f"<b>%{{x}}</b><br>{title}: {shown}<extra></extra>",
        )
    _style(figure, "Resources per tool, all datasets", legend=False, facets=True)
    figure.update_xaxes(tickangle=-45)
    figure.update_yaxes(rangemode="tozero")
    return figure


_PERCENT = {"range": [0, 1], "tickformat": ".0%"}


def _style(figure: go.Figure, title: str, *, legend: bool = True, facets: bool = False, height: int = 460) -> go.Figure:
    """The house style: left-aligned title, legend in a row beneath it, hairline grid, no chart chrome."""
    figure.update_layout(
        template="none",
        title={
            "text": title,
            "x": 0.01,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "font": {"size": 16, "color": INK},
        },
        font={"family": FONT, "size": 12, "color": INK_SECONDARY},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
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
        hoverlabel={"bgcolor": SURFACE, "bordercolor": AXIS, "font": {"family": FONT, "size": 12, "color": INK}},
        bargap=0.22,
        bargroupgap=0.06,
    )
    figure.update_xaxes(showgrid=False, showline=True, linecolor=AXIS, ticks="", zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor=GRID, showline=False, ticks="", zeroline=False)
    figure.update_annotations(font={"size": 13, "color": INK})
    return figure


def _facets(datasets: Sequence[str]) -> go.Figure:
    return make_subplots(
        rows=1,
        cols=len(datasets),
        shared_yaxes=True,
        subplot_titles=[dataset_name(dataset) for dataset in datasets],
        horizontal_spacing=0.025,
    )


def _dataset_axis(figure: go.Figure, metrics: Metrics) -> None:
    figure.update_xaxes(categoryorder="array", categoryarray=[dataset_name(dataset) for dataset in _datasets(metrics)])


def _error_bars(values: Sequence[float], low: Sequence[float], high: Sequence[float]) -> dict[str, object]:
    return {
        "type": "data",
        "symmetric": False,
        "array": [hi - value for value, hi in zip(values, high, strict=True)],
        "arrayminus": [value - lo for value, lo in zip(values, low, strict=True)],
        "color": INK_SECONDARY,
        "thickness": 1,
        "width": 3,
    }


def _cell_labels(figure: go.Figure, z: Sequence[Sequence[float | None]], x: Sequence[str], y: Sequence[str]) -> None:
    """Each filled cell's value as a percent, white on dark cells."""
    for row, values in zip(y, z, strict=True):
        for column, value in zip(x, values, strict=True):
            if value is not None:
                figure.add_annotation(
                    x=column,
                    y=row,
                    text=f"{value:.0%}",
                    showarrow=False,
                    font={"size": 11, "color": SURFACE if value > DARK_CELL else INK},
                )


def _tools(metrics: Metrics) -> list[str]:
    return sorted({tool for tool, _ in metrics}, key=tool_rank)


def _datasets(metrics: Metrics) -> list[str]:
    return sorted({dataset for _, dataset in metrics}, key=dataset_rank)


def _datasets_of(metrics: Metrics, tool: str) -> list[str]:
    return [dataset for dataset in _datasets(metrics) if (tool, dataset) in metrics]


def _pair(shares: Mapping[tuple[str, str], Share], first: str, second: str) -> Share | None:
    """The pair's share in either order; None when the two tools share no crop."""
    return shares.get((first, second)) or shares.get((second, first))


def _before(first: str, second: str) -> bool:
    return tool_rank(first) < tool_rank(second)


def _positive(seconds: float) -> float:
    return max(seconds, MIN_SECONDS)


def _shorten(label: str) -> str:
    return label if len(label) <= MAX_LABEL else label[: MAX_LABEL - 1].rstrip(" ,/") + "\u2026"


def _tint(color: str, alpha: float = 0.22) -> str:
    red, green, blue = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"
