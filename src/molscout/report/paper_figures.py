"""Plotly figures for the complete-system benchmark, in the style of report.figures; no wandb here.

Built from report.paper_analysis results: run metrics keyed by (tool, dataset), and the per-paper rows. Every
builder takes a Theme (LIGHT or DARK) and draws at its design height, like the structure-reader figures. A system
keeps one colour in every figure. Nothing drawn names a molecule, so Internal results are safe to plot.
"""

from __future__ import annotations

import random
from collections.abc import Mapping

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from molscout.report.figures import DARK, LIGHT, Theme, error_bars, header, style
from molscout.report.paper_analysis import BIOVISTA, INTERNAL, SYSTEM_NAMES, Rows, system_rank

Metrics = Mapping[tuple[str, str], Mapping[str, float]]  # (tool, dataset) -> paper_analysis.paper_metrics

DATASET_NAMES = {BIOVISTA: "BioVista", INTERNAL: "Internal"}
# Each dataset's views: (view key in paper_analysis.view_sizes, facet title, metric key prefix). Only the first
# view has confidence intervals, as only it is recorded with them.
VIEWS = {
    BIOVISTA: (
        ("all", "All labels", "micro"),
        ("drawn", "Drawn structures only", "drawn"),
        ("without_submitted", "Without submitted versions", "without_submitted"),
    ),
    INTERNAL: (
        ("all", "All papers", "micro"),
        ("dev", "Development papers", "dev"),
        ("test", "Test papers", "test"),
    ),
}
MEASURES = (("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"))
MARKER_SYMBOLS = {BIOVISTA: "circle", INTERNAL: "diamond"}
JITTER_SEED = 6630
JITTER_WIDTH = 0.34  # half the width of a system's column of dots, in x units


def system_color(tool: str, theme: Theme = LIGHT) -> str:
    """The theme's tool palette in system order (blue, orange, green); any other system gets the neutral colour."""
    palette = list(theme.tool_colors.values())
    rank = system_rank(tool)[0]
    return palette[rank] if rank < min(len(SYSTEM_NAMES), len(palette)) else theme.other


def view_bars(
    metrics: Metrics, dataset: str, sizes: Mapping[tuple[str, str], int], *, theme: Theme = LIGHT
) -> go.Figure:
    """Precision, recall and F1 of one dataset, a bar per system in a panel per view.

    `sizes` is paper_analysis.view_sizes. The all-papers view carries Wilson 95% CIs on the pooled values; a view
    no system has numbers for is left out.
    """
    views = [view for view in VIEWS[dataset] if any(f"{view[2]}/f1" in row for (_, name), row in metrics.items() if name == dataset)]
    tools = _tools(metrics, dataset)
    figure = make_subplots(
        rows=1,
        cols=len(views),
        shared_yaxes=True,
        horizontal_spacing=0.04,
        subplot_titles=[f"{title} ({_papers(sizes[(dataset, key)])})" for key, title, _ in views],
    )
    for column, (_, _, prefix) in enumerate(views, start=1):
        for tool in tools:
            row = metrics[(tool, dataset)]
            values = [row[f"{prefix}/{measure}"] for _, measure in MEASURES]
            intervals = prefix == "micro"
            low = [row[f"{prefix}/{measure}_ci_low"] for _, measure in MEASURES] if intervals else None
            high = [row[f"{prefix}/{measure}_ci_high"] for _, measure in MEASURES] if intervals else None
            figure.add_bar(
                row=1,
                col=column,
                name=SYSTEM_NAMES.get(tool, tool),
                legendgroup=tool,
                showlegend=column == 1,
                x=[label for label, _ in MEASURES],
                y=values,
                marker_color=system_color(tool, theme),
                error_y=error_bars(values, low, high, theme) if intervals else None,
                customdata=[[a, b] for a, b in zip(low, high, strict=True)] if intervals else None,
                hovertemplate=(
                    "<b>%{fullData.name}</b> %{x}<br>%{y:.1%} (95% CI %{customdata[0]:.1%} to %{customdata[1]:.1%})"
                    "<extra></extra>"
                    if intervals
                    else "<b>%{fullData.name}</b> %{x}<br>%{y:.1%}<extra></extra>"
                ),
            )
    style(figure, f"{DATASET_NAMES.get(dataset, dataset)}: precision, recall and F1", theme, facets=True, height=440)
    header(figure, facets=True)
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(range=[0, 1.04], dtick=0.2, tickformat=".0%")
    figure.update_yaxes(title_text="Pooled over papers", row=1, col=1)
    figure.update_layout(margin={"l": 64, "r": 16, "b": 48})
    return figure


def recall_by_paper(papers: Rows, *, theme: Theme = LIGHT) -> go.Figure:
    """Each BioVista paper's recall as a dot, in a jittered column per system.

    `papers` is paper_analysis.paper_rows. The jitter is seeded, so the figure is the same every time.
    """
    columns, rows = papers
    records = [dict(zip(columns, row, strict=True)) for row in rows if row[columns.index("dataset")] == BIOVISTA]
    systems = list(dict.fromkeys(record["system"] for record in records))
    chance = random.Random(JITTER_SEED)
    figure = go.Figure()
    for index, system in enumerate(systems):
        own = [record for record in records if record["system"] == system]
        figure.add_scatter(
            name=system,
            x=[index + chance.uniform(-JITTER_WIDTH, JITTER_WIDTH) for _ in own],
            y=[record["recall"] for record in own],
            mode="markers",
            marker={"color": system_color(_tool(system), theme), "size": 7, "opacity": 0.65},
            hovertext=[
                f"{record['paper']}: {record['recall']:.0%} recall ({record['tp']} of {record['molecules']} labels)"
                for record in own
            ],
            hovertemplate="<b>%{fullData.name}</b><br>%{hovertext}<extra></extra>",
        )
    style(figure, "BioVista: recall of each paper", theme, legend=False, height=420)
    figure.update_xaxes(
        range=[-0.6, len(systems) - 0.4], tickvals=list(range(len(systems))), ticktext=systems, showgrid=False
    )
    figure.update_yaxes(range=[-0.04, 1.04], dtick=0.2, tickformat=".0%", title_text="Recall of the paper's labels")
    figure.update_layout(margin={"l": 64, "r": 16, "t": 56, "b": 48})
    return figure


def f1_vs_time(metrics: Metrics, *, theme: Theme = LIGHT) -> go.Figure:
    """Pooled F1 against mean seconds per paper, one point per run; a diamond is Internal, a circle BioVista.

    Each point is named beside it, so there is no legend. Seconds are on a log axis: the systems differ by orders.
    """
    figure = go.Figure()
    keys = sorted(metrics, key=lambda key: (system_rank(key[0]), key[1]))
    for tool, dataset in keys:
        row = metrics[(tool, dataset)]
        name = SYSTEM_NAMES.get(tool, tool)
        figure.add_scatter(
            name=name,
            legendgroup=dataset,
            x=[row["speed/s_per_paper_mean"]],
            y=[row["micro/f1"]],
            mode="markers+text",
            text=[f"{name}, {DATASET_NAMES.get(dataset, dataset)}"],
            textposition="middle right",
            textfont={"size": 11, "color": theme.ink_secondary},
            cliponaxis=False,
            marker={
                "color": system_color(tool, theme),
                "symbol": MARKER_SYMBOLS.get(dataset, "square"),
                "size": 12,
                "line": {"color": theme.surface, "width": 1.5},
            },
            hovertemplate=(
                f"<b>%{{fullData.name}}</b> on {DATASET_NAMES.get(dataset, dataset)}<br>"
                "F1 %{y:.1%}<br>%{x:.1f} s per paper<extra></extra>"
            ),
        )
    style(figure, "F1 against time per paper", theme, legend=False, height=380)
    figure.update_xaxes(type="log", title_text="Mean seconds per paper (log scale)", showgrid=True, gridcolor=theme.grid)
    figure.update_yaxes(range=[0, 1.04], dtick=0.2, tickformat=".0%", title_text="Pooled F1 over all papers")
    figure.update_layout(margin={"l": 64, "r": 120, "t": 56, "b": 62})
    return figure


def _tools(metrics: Metrics, dataset: str) -> list[str]:
    return sorted({tool for tool, name in metrics if name == dataset}, key=system_rank)


def _tool(system: str) -> str:
    """The tool ID of a system's display name; a name that is not a known system stands for itself."""
    return next((tool for tool, name in SYSTEM_NAMES.items() if name == system), system)


def _papers(count: int) -> str:
    return f"{count} paper" if count == 1 else f"{count} papers"


__all__ = ["DARK", "LIGHT", "f1_vs_time", "recall_by_paper", "system_color", "view_bars"]
