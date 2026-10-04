"""Fill the Type 1 tables in docs/Benchmarking.md from benchmarks/results/*/scores.json."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DASH = "—"
SECTION = "## Type 1: Structure readers"
SETUP_ANCHOR = "### Setup"
STEREO_ANCHOR = "**Exact match, stereo-aware (%)**"
STRIPPED_ANCHOR = "**Exact match, stereo-stripped (%)**"
POOLED_ANCHOR = "**Valid output and speed, all datasets pooled**"

DATASETS = ("uspto", "uob", "jpo", "clef", "molrecbench_wild")
TOOL_ROWS = {
    "molscribe": "MolScribe",
    "molnextr": "MolNexTR",
    "decimer": "DECIMER",
    "molvec": "MolVec",
    "molglyph": "MolGlyph",
    "ocsrglyph": "OCSRGlyph",
}

_LINK_TEXT = re.compile(r"\[([^\]]+)\]")


@dataclass(frozen=True)
class Run:
    """One tool on one dataset, as read from its results folder."""

    tool: str
    dataset: str
    accuracy: float
    accuracy_stripped: float
    valid_successes: int
    valid_trials: int
    seconds_total: float
    items: int
    version: str
    hardware: str


def load_runs(results: Path) -> dict[tuple[str, str], Run]:
    """Every `<tool>__<dataset>/` folder under `results` that holds a scores.json, keyed by (tool, dataset)."""
    runs: dict[tuple[str, str], Run] = {}
    for folder in sorted(Path(results).glob("*__*")):
        scores_path = folder / "scores.json"
        if not scores_path.is_file():
            continue
        tool, dataset = folder.name.split("__", 1)
        report = json.loads(scores_path.read_text(encoding="utf-8"))
        if report["dataset"] != dataset:
            raise ValueError(f"{folder}: folder says {dataset} but scores.json says {report['dataset']}")
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        scores = report["scores"]
        speed = scores["seconds_per_item"]
        valid = scores["valid_output_rate"]
        runs[(tool, dataset)] = Run(
            tool=tool,
            dataset=dataset,
            accuracy=scores["accuracy"]["value"],
            accuracy_stripped=scores["accuracy_stereo_stripped"]["value"],
            valid_successes=valid["successes"],
            valid_trials=valid["trials"],
            seconds_total=speed["total"],
            items=speed["items"],
            version=meta["tool"]["version"],
            hardware=_hardware(meta["hardware"]),
        )
    return runs


def fill_type1(markdown: str, runs: Mapping[tuple[str, str], Run]) -> str:
    """The document with the Type 1 tables filled from `runs`; nothing outside them changes."""
    lines = markdown.split("\n")
    start = lines.index(SECTION)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    section = lines[start:end]
    for anchor, rewrite in (
        (SETUP_ANCHOR, _setup_row),
        (STEREO_ANCHOR, _accuracy_row("accuracy")),
        (STRIPPED_ANCHOR, _accuracy_row("accuracy_stripped")),
        (POOLED_ANCHOR, _pooled_row),
    ):
        _rewrite_table(section, anchor, rewrite, runs)
    return "\n".join(lines[:start] + section + lines[end:])


def missing_runs(runs: Mapping[tuple[str, str], Run]) -> list[str]:
    """The `<tool>__<dataset>` names of Type 1 runs not in `runs`."""
    return [f"{tool}__{dataset}" for tool in TOOL_ROWS for dataset in DATASETS if (tool, dataset) not in runs]


def _hardware(hardware: Mapping[str, object]) -> str:
    parts = [str(hardware["cluster"]).capitalize()]
    names = list(dict.fromkeys(gpu["name"] for gpu in hardware["gpus"]))
    parts += names or ["CPU", f"{hardware['cpus']} cores"]
    return ", ".join(parts)


def _rewrite_table(section: list[str], anchor: str, rewrite, runs: Mapping[tuple[str, str], Run]) -> None:
    index = section.index(anchor) + 1
    while not section[index].startswith("|"):
        index += 1
    index += 2
    row_tools = {label: tool for tool, label in TOOL_ROWS.items()}
    while index < len(section) and section[index].startswith("|"):
        cells = section[index].strip().strip("|").split("|")
        match = _LINK_TEXT.search(cells[0])
        label = match.group(1) if match else cells[0].strip()
        if label in row_tools:
            tool_runs = {d: runs[(row_tools[label], d)] for d in DATASETS if (row_tools[label], d) in runs}
            section[index] = rewrite(cells, tool_runs)
        index += 1


def _join(cells: Sequence[str]) -> str:
    return "|" + "|".join(cells) + "|"


def _setup_row(cells: list[str], tool_runs: Mapping[str, Run]) -> str:
    if not tool_runs:
        return _join(cells)
    hardware = " / ".join(dict.fromkeys(run.hardware for run in tool_runs.values()))
    version = " / ".join(dict.fromkeys(run.version for run in tool_runs.values()))
    return _join([*cells[:3], f" {hardware} ", f" {version} "])


def _accuracy_row(field: str):
    def rewrite(cells: list[str], tool_runs: Mapping[str, Run]) -> str:
        values = [f"{getattr(tool_runs[d], field) * 100:.1f}" if d in tool_runs else DASH for d in DATASETS]
        return _join([cells[0], *(f" {value} " for value in values)])

    return rewrite


def _pooled_row(cells: list[str], tool_runs: Mapping[str, Run]) -> str:
    if len(tool_runs) < len(DATASETS):
        values = [DASH, DASH]
    else:
        valid = sum(run.valid_successes for run in tool_runs.values()) / sum(run.valid_trials for run in tool_runs.values())
        speed = sum(run.seconds_total for run in tool_runs.values()) / sum(run.items for run in tool_runs.values())
        values = [f"{valid * 100:.1f}", f"{speed:.3f}"]
    return _join([cells[0], *(f" {value} " for value in values)])
