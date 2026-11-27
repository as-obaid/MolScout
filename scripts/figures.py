"""Draw the result charts in docs/figures/ from benchmarks/results/*/scores.json.

    uv run --extra figures python scripts/figures.py
    uv run --extra figures python scripts/figures.py --results <folder> --out <folder> --chart ladder

Each chart is written twice at 2x for a <picture> element: <name>-light.png on white and <name>-dark.png on
GitHub's dark background (#0d1117). Complete systems are read from the scored groups of each run: `dev`, and the
held-out group (`heldout` on BioVista, `test` on Internal). MolScout is read from molscout_both__<dataset> on the
development papers and from molscout__<dataset> on the held-out papers, each falling back to the other. A run or
group that does not exist yet is drawn as "pending" and listed on stderr. Accuracy is TP / (TP + FP + FN) from a
group's counts (extracted molecules have no true negatives); its 95% interval resamples the group's papers, as the
scorer does for precision, recall and F1. Text and time chunks are stripped from the PNGs. Run it in the MolScout
environment with the `figures` extra (matplotlib).

    systems_heldout  precision, recall, F1 and accuracy with 95% intervals on the held-out papers
    systems_dev      precision against recall on the development papers, with F1 iso-curves
    ladder           development F1 of each step of the ladder of fixes
    crop_accuracy    exact-match accuracy of each structure reader on each crop dataset
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import textwrap
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from molscout.scoring.stats import paper_proportion

REPO = Path(__file__).resolve().parents[1]

READERS = {
    "molscribe": "MolScribe",
    "molnextr": "MolNexTR",
    "decimer": "DECIMER",
    "molvec": "MolVec",
    "molglyph": "MolGlyph",
    "ocsrglyph": "OCSRGlyph",
}
VOTERS = {"MolGlyph", "MolScribe", "MolNexTR"}
CROP_DATASETS = {"uspto": "USPTO", "uob": "UOB", "jpo": "JPO", "clef": "CLEF", "molrecbench_wild": "MolRecBench-Wild"}
PAPER_DATASETS = {"internal": "Internal", "biovista": "BioVista"}
# split -> dataset -> the scores.json group that holds it
SPLITS = {"dev": {"internal": "dev", "biovista": "dev"}, "heldout": {"internal": "test", "biovista": "heldout"}}
METRICS = {"precision": "Precision", "recall": "Recall", "f1": "F1", "accuracy": "Accuracy"}

MOLSCOUT = "MolScout"
BASELINES = {"BioMiner": "biominer", "DECIMER.ai": "decimer_ai", "OpenChemIE": "openchemie"}  # name -> run prefix
SYSTEMS = (MOLSCOUT, *BASELINES)
# split -> the run prefixes (<prefix>__<dataset>) that hold MolScout; the first run that exists is read
MOLSCOUT_RUNS = {"dev": ("molscout_both", "molscout"), "heldout": ("molscout", "molscout_both")}
STEPS = ("l0_drawings", "l1_labels", "l2_tables", "l3_core", "l4_vote", "l5_stereo", "l6_junk")
LADDER = (*((step, (f"molscout_{step}",)) for step in STEPS), ("rules", ("molscout_rules",)),
          (MOLSCOUT, MOLSCOUT_RUNS["dev"]))

# name -> figure size in inches; PNGs are saved at 200 dpi, twice the 100 dpi layout size
CHARTS = {"systems_heldout": (8.0, 4.0), "systems_dev": (8.0, 3.9), "ladder": (8.0, 3.2), "crop_accuracy": (8.0, 2.4)}
DPI = 200
DROPPED_CHUNKS = {b"tEXt", b"iTXt", b"zTXt", b"eXIf", b"tIME"}
ISO_F1 = (20, 40, 60, 80)
# (dx, dy) in points, horizontal and vertical alignment: where a point's label may go, in order of preference
LABEL_OFFSETS = ((8, 0, "left", "center"), (-8, 0, "right", "center"), (7, 7, "left", "bottom"),
                 (7, -7, "left", "top"), (-7, 7, "right", "bottom"), (-7, -7, "right", "top"),
                 (0, 9, "center", "bottom"), (0, -9, "center", "top"))


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    muted: str
    grid: str  # hairline gridlines, axes and F1 iso-curves
    accent: str  # MolScout
    accent_soft: str  # MolScout's interval line
    mark: str  # baseline dots, ladder steps
    line: str  # baseline interval lines
    strong: str  # bars of the three readers MolScout votes with
    soft: str  # bars of the other readers


THEMES = (
    Theme("light", "#ffffff", "#1f2328", "#59636e", "#e3e7eb", "#2370cd", "#9dc8f7", "#8c959f", "#c9d1d9",
          "#57606a", "#afb8c1"),
    Theme("dark", "#0d1117", "#e6edf3", "#9198a1", "#262c36", "#338ced", "#1d4f8c", "#8b949e", "#3d444d",
          "#a3acb7", "#484f58"),
)


class FigureError(Exception):
    """A results folder is missing a crop run, or a scores.json lacks a value a chart needs."""


@dataclass(frozen=True)
class Estimate:
    value: float
    low: float
    high: float


@dataclass(frozen=True)
class Scores:
    precision: Estimate
    recall: Estimate
    f1: Estimate
    accuracy: Estimate


@dataclass(frozen=True)
class Panel:
    """One dataset: (system or ladder step, its scores) per row; None while the run or group does not exist."""

    dataset: str
    title: str
    rows: tuple[tuple[str, Scores | Estimate | None], ...]

    @property
    def pending(self) -> list[str]:
        return [name for name, value in self.rows if value is None]


def _scores(results: Path, run: str) -> dict | None:
    path = results / run / "scores.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["scores"]


def crop_accuracy(results: Path) -> list[tuple[str, dict[str, float]]]:
    """Stereo-aware exact-match accuracy (%) of each reader on each crop dataset, in table order."""
    runs = [f"{tool}__{dataset}" for tool in READERS for dataset in CROP_DATASETS]
    found = {run: _scores(results, run) for run in runs}
    missing = [run for run, scores in found.items() if scores is None]
    if missing:
        raise FigureError(f"missing results in {results}: {', '.join(missing)}")
    table = []
    for tool, name in READERS.items():
        row = {}
        for dataset in CROP_DATASETS:
            run = f"{tool}__{dataset}"
            try:
                row[dataset] = 100 * found[run]["accuracy"]["value"]
            except (KeyError, TypeError) as exc:
                raise FigureError(f"{run}: scores.json has no accuracy ({exc})") from exc
        table.append((name, row))
    return table


def _group(results: Path, prefixes: tuple[str, ...], dataset: str, group: str) -> tuple[str, dict, dict] | None:
    """(run, its scores, its group) for the first run that exists; None if none does, or it lacks the group."""
    for prefix in prefixes:
        run = f"{prefix}__{dataset}"
        scores = _scores(results, run)
        if scores is not None:
            groups = scores.get("groups") or {}
            return (run, scores, groups[group]) if group in groups else None
    return None


def _estimate(run: str, group: dict, metric: str) -> Estimate:
    try:
        entry = group["micro"][metric]
        low, high = entry["ci95"]
        return Estimate(100 * entry["value"], 100 * low, 100 * high)
    except (KeyError, TypeError, ValueError) as exc:
        raise FigureError(f"{run}: scores.json has no micro {metric} ({exc})") from exc


def _accuracy(run: str, scores: dict, group: dict) -> Estimate:
    """TP / (TP + FP + FN) over the group, with the paper-bootstrap 95% interval of the scorer's other metrics."""
    try:
        counts = [scores["papers"][paper] for paper in group["papers"]]
        tp = [paper["tp"] for paper in counts]
        trials = [paper["tp"] + paper["fp"] + paper["fn"] for paper in counts]
        total = group["counts"]
    except (KeyError, TypeError) as exc:
        raise FigureError(f"{run}: scores.json has no per-paper counts ({exc})") from exc
    if (sum(tp), sum(trials)) != (total["tp"], total["tp"] + total["fp"] + total["fn"]):
        raise FigureError(f"{run}: the per-paper counts do not add up to the group's counts")
    estimate = paper_proportion(tp, trials)
    low, high = estimate["ci95"]
    return Estimate(100 * estimate["value"], 100 * low, 100 * high)


def _title(dataset: str, groups: dict[str, dict]) -> str:
    """'Internal 1, 16 and 19 · 125 molecules' or 'BioVista · 20 papers, 463 molecules'; the name with no runs."""
    sets = {(frozenset(group.get("papers", ())), group.get("molecules")) for group in groups.values()}
    if len(sets) > 1:
        raise FigureError(f"{', '.join(groups)} were scored on different papers or molecules")
    name = PAPER_DATASETS[dataset]
    if not sets:
        return name
    papers, molecules = sets.pop()
    count = "" if molecules is None else f"{molecules:,} molecules"
    if dataset == "internal":
        ids = sorted(papers, key=lambda paper: (len(paper), paper))
        label = f"{name} {' and '.join(part for part in (', '.join(ids[:-1]), ''.join(ids[-1:])) if part)}".rstrip()
        return f"{label} · {count}" if count else label
    return f"{name} · {len(papers)} papers" + (f", {count}" if count else "")


def _panels(results: Path, entries: Iterable[tuple[str, tuple[str, ...]]], split: str,
            read: Callable[[str, dict, dict], Scores | Estimate]) -> list[Panel]:
    panels = []
    for dataset in PAPER_DATASETS:
        rows, groups = [], {}
        for name, prefixes in entries:
            found = _group(results, prefixes, dataset, SPLITS[split][dataset])
            if found is not None:
                groups[found[0]] = found[2]
            rows.append((name, None if found is None else read(*found)))
        panels.append(Panel(dataset, _title(dataset, groups), tuple(rows)))
    return panels


def _system_scores(run: str, scores: dict, group: dict) -> Scores:
    return Scores(*(_estimate(run, group, metric) for metric in ("precision", "recall", "f1")),
                  _accuracy(run, scores, group))


def systems(results: Path, split: str) -> list[Panel]:
    """Micro precision, recall, F1 and accuracy (%, 95% paper-bootstrap intervals) of MolScout and the baselines,
    on the "dev" or "heldout" papers."""
    entries = [(MOLSCOUT, MOLSCOUT_RUNS[split]), *((name, (prefix,)) for name, prefix in BASELINES.items())]
    return _panels(results, entries, split, _system_scores)


def ladder(results: Path) -> list[Panel]:
    """Development micro F1 (%) of each step of the ladder of fixes, ending at MolScout."""
    return _panels(results, LADDER, "dev", lambda run, scores, group: _estimate(run, group, "f1"))


def _style(plt, theme: Theme) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "text.color": theme.ink,
        "axes.edgecolor": theme.grid,
        "axes.facecolor": theme.surface,
        "axes.labelcolor": theme.muted,
        "axes.labelsize": 8,
        "xtick.color": theme.muted,
        "ytick.color": theme.muted,
        "xtick.labelcolor": theme.muted,
        "ytick.labelcolor": theme.ink,
        "xtick.labelsize": 8,
        "ytick.labelsize": 9,
        "figure.facecolor": theme.surface,
    })


def _margins(fig, left: float, right: float, top: float, bottom: float, **space) -> None:
    """Margins in inches, so the charts line up with each other."""
    width, height = fig.get_size_inches()
    fig.subplots_adjust(left=left / width, right=1 - right / width, top=1 - top / height, bottom=bottom / height,
                        **space)


def _frame(ax, grid: bool) -> None:
    """No box; with `grid`, hairline x gridlines and a baseline, otherwise no x axis at all."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(grid)
    ax.tick_params(left=False, length=0, pad=4)
    if grid:
        ax.grid(axis="x", color=ax.spines["bottom"].get_edgecolor(), linewidth=0.8)
        ax.set_axisbelow(True)
    else:
        ax.set_xticks([])


def _row_labels(ax, names: list[str], bold: set[str]) -> None:
    ax.set_yticks(range(len(names)), names)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold" if label.get_text() in bold else "normal")
    ax.set_ylim(len(names) - 0.5, -0.6)


def _panel_title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=9, fontweight="bold", pad=6)


def _pending(ax, theme: Theme, x: float, y: float) -> None:
    ax.text(x, y, "pending", va="center", fontsize=8, color=theme.muted, style="italic")


def _bars(ax, theme: Theme, rows: list[tuple[str, float | None]], colours: list[str], xmax: float) -> None:
    for y, ((_, value), colour) in enumerate(zip(rows, colours)):
        if value is None:
            _pending(ax, theme, 1.5, y)
            continue
        ax.barh(y, value, height=0.62, color=colour)
        ax.annotate(f"{value:.1f}", (value, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_xlim(0, xmax)
    _frame(ax, grid=False)


def _iso_f1(ax, theme: Theme) -> list:
    """Faint curves of equal F1 across the precision-recall plane; returns their labels."""
    labels = []
    for f1 in ISO_F1:
        f = f1 / 100
        recall = [f / (2 - f) + (1 - f / (2 - f)) * i / 200 for i in range(201)]
        ax.plot([100 * r for r in recall], [100 * f * r / (2 * r - f) for r in recall], color=theme.grid,
                linewidth=1.1, zorder=1)
        labels.append(ax.text(100, 100 * f / (2 - f) + 1.5, f"F1 {f1}", ha="right", va="bottom", fontsize=7,
                              color=theme.muted, zorder=1))
    return labels


def _place_labels(ax, labels: list[tuple[float, float, str, bool]], obstacles: list) -> None:
    """Put each (x, y, text, bold) label beside its point at the first offset that stays inside the axes and clear
    of the points, the iso-curve labels and the labels placed before it."""
    renderer = ax.figure.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    taken = list(obstacles)
    for x, y, text, bold in labels:
        for i, (dx, dy, ha, va) in enumerate((*LABEL_OFFSETS, LABEL_OFFSETS[0])):
            label = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                                fontsize=8, fontweight="bold" if bold else "normal", linespacing=1.2, zorder=4)
            box = label.get_window_extent(renderer)
            inside = frame.x0 <= box.x0 and box.x1 <= frame.x1 + 2 and frame.y0 <= box.y0 and box.y1 <= frame.y1
            if i == len(LABEL_OFFSETS) or (inside and not any(box.overlaps(other) for other in taken)):
                taken.append(box.padded(3))
                break
            label.remove()


def _dot_style(theme: Theme, name: str) -> dict:
    face = theme.accent if name == MOLSCOUT else theme.mark
    return {"marker": "o", "markersize": 8, "markerfacecolor": face, "markeredgecolor": theme.surface,
            "markeredgewidth": 1.5}


def draw_systems_dev(plt, theme: Theme, panels: list[Panel], size):
    from matplotlib.transforms import Bbox

    fig, axes = plt.subplots(1, len(panels), figsize=size, sharey=True)
    _margins(fig, left=0.55, right=0.2, top=0.3, bottom=0.5, wspace=0.12)
    for ax, panel in zip(axes, panels):
        ax.set_xlim(0, 103)
        ax.set_ylim(0, 103)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 25, 50, 75, 100])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=3, color=theme.grid)
        ax.set_xlabel("Recall (%)")
        notes = _iso_f1(ax, theme)
        if panel.pending:  # under the F1 20 curve, which never dips below a precision of 11
            notes.append(ax.text(2, 2, textwrap.fill("pending: " + ", ".join(panel.pending), 48), ha="left",
                                 va="bottom", fontsize=7.5, color=theme.muted, style="italic", linespacing=1.2))
        obstacles = [note.get_window_extent(fig.canvas.get_renderer()) for note in notes]
        points = [(name, scores) for name, scores in panel.rows if scores is not None]
        radius = 6 * fig.dpi / 72
        for name, scores in points:
            ax.plot(scores.recall.value, scores.precision.value, zorder=3, **_dot_style(theme, name))
            x, y = ax.transData.transform((scores.recall.value, scores.precision.value))
            obstacles.append(Bbox.from_extents(x - radius, y - radius, x + radius, y + radius))
        _place_labels(ax, [(s.recall.value, s.precision.value, f"{name}\nF1 {s.f1.value:.1f}", name == MOLSCOUT)
                           for name, s in points], obstacles)
        _panel_title(ax, panel.title)
    axes[0].set_ylabel("Precision (%)")
    return fig


def draw_systems_heldout(plt, theme: Theme, panels: list[Panel], size):
    fig, grid = plt.subplots(len(panels), len(METRICS), figsize=size, sharey=True, squeeze=False)
    _margins(fig, left=1.05, right=0.4, top=0.62, bottom=0.3, wspace=0.33, hspace=0.95)
    names = [name for name, _ in panels[0].rows]
    for row, panel in zip(grid, panels):
        for ax, (metric, heading) in zip(row, METRICS.items()):
            for y, (name, scores) in enumerate(panel.rows):
                if scores is None:
                    _pending(ax, theme, 2, y)
                    continue
                estimate = getattr(scores, metric)
                line = theme.accent_soft if name == MOLSCOUT else theme.line
                ax.plot([estimate.low, estimate.high], [y, y], color=line, linewidth=2.5, solid_capstyle="round",
                        zorder=2)
                ax.plot(estimate.value, y, zorder=3, **{**_dot_style(theme, name), "markersize": 7})
                ax.text(104, y, f"{estimate.value:.1f}", va="center", fontsize=8, clip_on=False,
                        fontweight="bold" if name == MOLSCOUT else "normal")
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 25, 50, 75, 100], ["0", "", "50", "", "100"])
            _frame(ax, grid=True)
            ax.set_title(heading, loc="left", fontsize=8, color=theme.muted, pad=4)
        _row_labels(row[0], names, {MOLSCOUT})
        top = row[0].get_position().y1
        fig.text(0.12 / size[0], top + 0.3 / size[1], panel.title, fontsize=9, fontweight="bold", va="bottom")
    return fig


def draw_ladder(plt, theme: Theme, panels: list[Panel], size):
    fig, axes = plt.subplots(1, len(panels), figsize=size, sharey=True)
    _margins(fig, left=1.05, right=0.1, top=0.3, bottom=0.05, wspace=0.1)
    names = [name for name, _ in panels[0].rows]
    for ax, panel in zip(axes, panels):
        rows = [(name, None if value is None else value.value) for name, value in panel.rows]
        colours = [theme.accent if name == MOLSCOUT else theme.mark for name in names]
        _bars(ax, theme, rows, colours, xmax=100)
        _panel_title(ax, panel.title)
    _row_labels(axes[0], names, {MOLSCOUT})
    return fig


def draw_crop_accuracy(plt, theme: Theme, table, size):
    fig, axes = plt.subplots(1, len(CROP_DATASETS), figsize=size, sharey=True)
    _margins(fig, left=1.05, right=0.1, top=0.3, bottom=0.05, wspace=0.12)
    names = [name for name, _ in table]
    colours = [theme.strong if name in VOTERS else theme.soft for name in names]
    for ax, (key, title) in zip(axes, CROP_DATASETS.items()):
        _bars(ax, theme, [(name, row[key]) for name, row in table], colours, xmax=128)
        _panel_title(ax, title)
    _row_labels(axes[0], names, VOTERS)
    return fig


READ = {
    "systems_heldout": lambda results: systems(results, "heldout"),
    "systems_dev": lambda results: systems(results, "dev"),
    "ladder": ladder,
    "crop_accuracy": crop_accuracy,
}
DRAW = {
    "systems_heldout": draw_systems_heldout,
    "systems_dev": draw_systems_dev,
    "ladder": draw_ladder,
    "crop_accuracy": draw_crop_accuracy,
}


def strip_metadata(path: Path) -> None:
    """Drop text, time and EXIF chunks; the image data is untouched."""
    data = path.read_bytes()
    out, i = [data[:8]], 8
    while i < len(data):
        length, name = struct.unpack(">I4s", data[i : i + 8])
        if name not in DROPPED_CHUNKS:
            out.append(data[i : i + 12 + length])
        i += 12 + length
    path.write_bytes(b"".join(out))


def _save(plt, fig, path: Path, theme: Theme) -> None:
    fig.savefig(path, dpi=DPI, facecolor=theme.surface, metadata={"Software": None})
    plt.close(fig)
    strip_metadata(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw the result charts from benchmark scores.")
    parser.add_argument("--results", type=Path, default=REPO / "benchmarks" / "results")
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "figures")
    parser.add_argument("--chart", action="append", choices=list(CHARTS), help="draw only this chart (repeatable)")
    args = parser.parse_args(argv)
    names = args.chart or list(CHARTS)
    try:
        if not args.results.is_dir():
            raise FigureError(f"no results folder at {args.results}")
        data = {name: READ[name](args.results) for name in names}
    except (OSError, ValueError, KeyError, FigureError) as exc:
        print(f"figures: error: {exc}", file=sys.stderr)
        return 1
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        _style(plt, theme)
        for name in names:
            _save(plt, DRAW[name](plt, theme, data[name], CHARTS[name]), args.out / f"{name}-{theme.name}.png", theme)
    for name in names:
        for panel in data[name] if name != "crop_accuracy" else ():
            if panel.pending:
                print(f"figures: {name}: {PAPER_DATASETS[panel.dataset]}: pending {', '.join(panel.pending)}",
                      file=sys.stderr)
    print(f"wrote {len(names) * len(THEMES)} charts to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
