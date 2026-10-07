"""Fill the Type 2 tables in docs/Benchmarking.md from benchmarks/results/<system>__<biovista|internal>/."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from molscout.tables import DASH, PAPER_DATASETS, _join, _LINK_TEXT, git_warnings as _git_warnings

SECTION = "## Type 2: Complete systems"
SETUP_ANCHOR = "### Setup"
BIOVISTA_ANCHOR = "**BioVista**"
DRAWN_ANCHOR = "**BioVista, drawn structures only**"
WITHOUT_ANCHOR = "**BioVista, without submitted versions**"
INTERNAL_ANCHOR = "**Internal**"
SPLIT_ANCHOR = "**Internal, by split**"

SYSTEM_ROWS = {"biominer": "BioMiner", "decimer_ai": "DECIMER.ai", "openchemie": "OpenChemIE"}


@dataclass(frozen=True)
class PaperRun:
    """One system on BioVista or Internal: the scores.json report plus what its meta.json records."""

    tool: str
    dataset: str
    report: Mapping[str, object]
    version: str
    hardware: str
    commit: str | None = None
    dirty: bool | None = None
    dirty_paths: tuple[str, ...] = ()
    crashed: int = 0
    papers: int = 0

    @property
    def name(self) -> str:
        return f"{self.tool}__{self.dataset}"


def load_paper_runs(results: Path) -> dict[tuple[str, str], PaperRun]:
    """Every `<system>__<biovista|internal>/` folder with a scores.json, keyed by (system, dataset)."""
    runs: dict[tuple[str, str], PaperRun] = {}
    for folder in sorted(Path(results).glob("*__*")):
        scores_path = folder / "scores.json"
        if folder.name.startswith(".") or not scores_path.is_file():
            continue
        tool, dataset = folder.name.split("__", 1)
        if dataset not in PAPER_DATASETS:
            continue
        report = json.loads(scores_path.read_text(encoding="utf-8"))
        if report["dataset"] != dataset:
            raise ValueError(f"{folder}: folder says {dataset} but scores.json says {report['dataset']}")
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        errors_path = folder / "errors.json"
        errors = json.loads(errors_path.read_text(encoding="utf-8"))["errors"] if errors_path.is_file() else {}
        runs[(tool, dataset)] = PaperRun(
            tool=tool,
            dataset=dataset,
            report=report,
            version=meta["tool"]["version"],
            hardware=_hardware(meta["hardware"]),
            commit=meta["git"]["commit"],
            dirty=meta["git"]["dirty"],
            dirty_paths=tuple(meta["git"]["dirty_paths"] or ()),
            crashed=len(errors),
            papers=len(report["scores"]["papers"]),
        )
    return runs


def missing_paper_runs(runs: Mapping[tuple[str, str], PaperRun]) -> list[str]:
    return [f"{tool}__{ds}" for tool in SYSTEM_ROWS for ds in PAPER_DATASETS if (tool, ds) not in runs]


def git_warnings(runs: Mapping[tuple[str, str], PaperRun]) -> list[str]:
    """Same checks as Type 1: uncommitted runs, and runs from several commits."""
    return _git_warnings(runs)  # type: ignore[arg-type]


def crash_warning(name: str, crashed: int, papers: int) -> str:
    return f"{name}: {crashed} of {papers} papers crashed; their labels count as missed"


def crash_warnings(runs: Mapping[tuple[str, str], PaperRun]) -> list[str]:
    """One warning per run with crashed papers: they score recall 0, so the table's numbers are deflated."""
    return [crash_warning(run.name, run.crashed, run.papers) for run in runs.values() if run.crashed]


def fill_type2(markdown: str, runs: Mapping[tuple[str, str], PaperRun]) -> str:
    """The document with the Type 2 tables filled from `runs`; nothing else changes."""
    lines = markdown.split("\n")
    start = lines.index(SECTION)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    section = lines[start:end]
    for anchor, rewrite in (
        (SETUP_ANCHOR, _setup_row),
        (BIOVISTA_ANCHOR, _metric_row("biovista", _group("all"), extra=("pdfs", "seconds"))),
        (DRAWN_ANCHOR, _metric_row("biovista", _drawn, extra=())),
        (WITHOUT_ANCHOR, _metric_row("biovista", _group("without_submitted"), extra=())),
        (INTERNAL_ANCHOR, _metric_row("internal", _group("all"), extra=("seconds",))),
        (SPLIT_ANCHOR, _split_row),
    ):
        _rewrite_table(section, anchor, rewrite, runs)
    return "\n".join(lines[:start] + section + lines[end:])


def _hardware(hardware: Mapping[str, object]) -> str:
    cluster = hardware["cluster"]
    parts = [] if cluster is None else [str(cluster).capitalize()]
    counts = Counter(gpu["name"] for gpu in hardware["gpus"])
    parts += [name if n == 1 else f"{n}× {name}" for name, n in counts.items()]
    if not counts:
        parts += ["CPU", f"{hardware['cpus_available']} cores"]
    return ", ".join(parts)


def _rewrite_table(section: list[str], anchor: str, rewrite: Callable, runs: Mapping[tuple[str, str], PaperRun]) -> None:
    if anchor not in section:
        raise ValueError(f"table anchor not found: {anchor}")
    index = section.index(anchor) + 1
    while index < len(section) and not section[index].startswith("|"):
        index += 1
    if index >= len(section):
        raise ValueError(f"no table after anchor: {anchor}")
    index += 2
    row_tools = {label: tool for tool, label in SYSTEM_ROWS.items()}
    while index < len(section) and section[index].startswith("|"):
        cells = section[index].strip().strip("|").split("|")
        match = _LINK_TEXT.search(cells[0])
        label = match.group(1) if match else cells[0].strip()
        if label in row_tools:
            tool_runs = {ds: runs[(row_tools[label], ds)] for ds in PAPER_DATASETS if (row_tools[label], ds) in runs}
            section[index] = rewrite(cells, tool_runs)
        elif not label.startswith("*"):
            raise ValueError(f"unknown row label {label!r} in table {anchor}; expected one of {', '.join(SYSTEM_ROWS.values())}")
        index += 1


def _setup_row(cells: list[str], tool_runs: Mapping[str, PaperRun]) -> str:
    if not tool_runs:
        return _join(cells)
    hardware = " / ".join(dict.fromkeys(run.hardware for run in tool_runs.values()))
    version = " / ".join(dict.fromkeys(run.version for run in tool_runs.values()))
    return _join([*cells[:4], f" {hardware} ", f" {version} "])


def _group(name: str) -> Callable[[PaperRun], Mapping[str, object]]:
    return lambda run: run.report["scores"]["groups"][name]  # type: ignore[index]


def _drawn(run: PaperRun) -> Mapping[str, object]:
    return run.report["scores"]["drawn_only"]["groups"]["all"]  # type: ignore[index]


def _pct(value: object) -> str:
    return DASH if value is None else f"{value * 100:.1f}"  # type: ignore[operator]


def _seven(group: Mapping[str, object]) -> list[str]:
    micro, macro, stripped = group["micro"], group["macro"], group["stereo_stripped"]
    return [
        _pct(m["value"])  # type: ignore[index]
        for m in (
            micro["precision"], micro["recall"], micro["f1"],  # type: ignore[index]
            macro["precision"], macro["recall"],  # type: ignore[index]
            stripped["precision"], stripped["recall"],  # type: ignore[index]
        )
    ]


def _metric_row(dataset: str, pick: Callable[[PaperRun], Mapping[str, object]], extra: Sequence[str]):
    def rewrite(cells: list[str], tool_runs: Mapping[str, PaperRun]) -> str:
        run = tool_runs.get(dataset)
        if run is None:
            values = [DASH] * (7 + len(extra))
        else:
            values = _seven(pick(run))
            for name in extra:
                if name == "pdfs":
                    values.append(str(run.report["inputs"]["references"]["papers"]))  # type: ignore[index]
                else:
                    mean = run.report["scores"]["seconds_per_item"]["mean"]  # type: ignore[index]
                    values.append(DASH if mean is None else f"{mean:.1f}")
        return _join([cells[0], *(f" {v} " for v in values)])

    return rewrite


def _split_row(cells: list[str], tool_runs: Mapping[str, PaperRun]) -> str:
    run = tool_runs.get("internal")
    if run is None:
        values = [DASH] * 6
    else:
        values = []
        for split in ("dev", "test"):
            micro = run.report["scores"]["groups"][split]["micro"]  # type: ignore[index]
            values += [_pct(micro[key]["value"]) for key in ("precision", "recall", "f1")]
    return _join([cells[0], *(f" {v} " for v in values)])
