"""Complete-system (Type 2) results as numbers: per-paper rows, run metrics, molecule outcomes and the consistency checks.

The paper twin of report.analysis, over the `<tool>__biovista` and `<tool>__internal` folders the harness writes;
nothing here draws or talks to W&B. Counts come from the scorer's scores.json, and the molecule outcomes use the
scorer's own canonical SMILES, so both agree with it. Internal results leave this module as metrics and counts
only: `molecule_rows` is BioVista-only.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from molscout.data.biovista_truth import BIOVISTA_PAPERS_PATH, reference_set_sha256
from molscout.data.internal import GROUND_TRUTH_PATH, SPLIT_PATH
from molscout.datasets import Kind, dataset_kind
from molscout.hashing import sha256_file
from molscout.report.analysis import (
    COMMIT_SHOWN,
    ResultsError,
    RunResult,
    _listed,
    _resources,
    _short,
    load_run,
)
from molscout.runs import read_paper_seconds
from molscout.scoring import canonical_smiles

BIOVISTA = "biovista"
INTERNAL = "internal"
BIOVISTA_ROOT = Path("data/raw/biovista")
SYSTEMS = ("biominer", "decimer_ai", "openchemie")  # the complete systems, in table order
SYSTEM_NAMES = {"biominer": "BioMiner", "decimer_ai": "DECIMER.ai", "openchemie": "OpenChemIE"}
PAPER_DATASETS = (BIOVISTA, INTERNAL)
PRF = ("precision", "recall", "f1")
OUTCOMES = ("tp", "fp", "invalid", "fn")
PAPER_COLUMNS = (
    "system",
    "dataset",
    "paper",
    "molecules",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "seconds",
    "drawn_tp",
    "drawn_fp",
    "drawn_fn",
    "drawn_ignored",
)
MOLECULE_COLUMNS = ("system", "paper", "smiles", "canonical", "outcome")


def _prf_keys(prefix: str, *, ci: bool = False) -> list[str]:
    keys = []
    for metric in PRF:
        keys.append(f"{prefix}/{metric}")
        if ci:
            keys += [f"{prefix}/{metric}_ci_low", f"{prefix}/{metric}_ci_high"]
    return keys


PAPER_METRIC_KEYS = (
    *_prf_keys("micro", ci=True),
    "macro/precision",
    "macro/precision_ci_low",
    "macro/precision_ci_high",
    "macro/recall",
    "macro/recall_ci_low",
    "macro/recall_ci_high",
    *_prf_keys("stripped"),
    "counts/tp",
    "counts/fp",
    "counts/fn",
    "valid_output_rate",
    "speed/s_per_paper_mean",
    "speed/s_per_paper_median",
    "resources/gpu_peak_memory_gib",
    "resources/gpu_mean_utilization_pct",
    "resources/peak_rss_gib",
    "resources/cpu_seconds",
    "items/papers",
    "items/molecules",
    "items/papers_without_output",
    "items/crashed",
    *_prf_keys("drawn"),  # BioVista: the drawn structures only
    "drawn/macro_precision",
    "drawn/macro_recall",
    *_prf_keys("without_submitted"),  # BioVista: without the submitted versions
    *_prf_keys("dev"),  # Internal: the development papers
    *_prf_keys("test"),
)

Rows = tuple[list[str], list[list[Any]]]


@dataclass(frozen=True, slots=True)
class PaperRun(RunResult):
    """A `<tool>__<paper dataset>/` result folder, read and checked: a RunResult plus seconds per paper."""

    timing: Mapping[str, float]  # paper ID -> seconds, from timing.json


def system_name(tool: str) -> str:
    return SYSTEM_NAMES.get(tool, tool)


def system_rank(tool: str) -> tuple[int, str]:
    """Sort key: the three complete systems in table order, then any other tool by name."""
    return (SYSTEMS.index(tool), "") if tool in SYSTEMS else (len(SYSTEMS), tool)


def paper_dataset_rank(dataset: str) -> tuple[int, str]:
    return (PAPER_DATASETS.index(dataset), "") if dataset in PAPER_DATASETS else (len(PAPER_DATASETS), dataset)


# Loading and checks


def paper_result_folders(root: str | Path) -> list[Path]:
    """The `<tool>__<dataset>/` folders under `root` holding a scores.json for a paper dataset.

    Hidden folders (what a killed harness leaves) and crop datasets are left out.
    """
    folders = []
    for folder in sorted(Path(root).glob("*__*")):
        if folder.name.startswith(".") or not (folder / "scores.json").is_file():
            continue
        if dataset_kind(folder.name.split("__", 1)[1]) is Kind.PAPER:
            folders.append(folder)
    return folders


def load_paper_run(folder: str | Path) -> PaperRun:
    """Read one paper result folder; ResultsError if a file is missing or the files disagree on what they hold."""
    folder = Path(folder)
    base = load_run(folder)
    try:
        paper_dataset = dataset_kind(base.dataset) is Kind.PAPER
    except ValueError:
        paper_dataset = False
    if not paper_dataset:
        raise ResultsError(f"{folder}: {base.dataset} is not a paper dataset")
    path = folder / "timing.json"
    if not path.is_file():
        raise ResultsError(f"{folder}: missing timing.json")
    try:
        timing = read_paper_seconds(path)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ResultsError(f"{folder}: timing.json is invalid: {exc}") from None
    scored = set(base.report["scores"]["papers"])
    if scored != set(timing):
        differ = sorted(scored ^ set(timing))
        raise ResultsError(
            f"{folder}: timing.json and scores.json disagree on {len(differ)} paper(s): {_listed(differ)}"
        )
    return PaperRun(**{field.name: getattr(base, field.name) for field in fields(RunResult)}, timing=timing)


def load_paper_runs(root: str | Path) -> tuple[PaperRun, ...]:
    """Every paper-dataset result folder under `root`, in system then dataset order."""
    runs = [load_paper_run(folder) for folder in paper_result_folders(root)]
    return tuple(sorted(runs, key=lambda run: (system_rank(run.tool), paper_dataset_rank(run.dataset))))


def check_paper_consistency(runs: Sequence[PaperRun]) -> list[str]:
    """Why the runs are not one benchmark: several or unknown commits, uncommitted code (ours or an upstream
    clone's), or different references within a dataset."""
    problems = []
    by_commit: dict[str | None, list[str]] = defaultdict(list)
    for run in runs:
        git = run.meta.get("git") or {}
        by_commit[git.get("commit")].append(run.name)
        if git.get("dirty") is None:
            problems.append(f"{run.name} does not record whether its code was committed")
        elif git["dirty"]:
            problems.append(f"{run.name} ran with uncommitted changes: {_listed(git.get('dirty_paths') or [])}")
        for source in run.meta.get("sources") or []:
            if source.get("dirty"):
                problems.append(
                    f"{run.name} ran with uncommitted upstream code in {Path(source['path']).name}: "
                    f"{_listed(source.get('dirty_paths') or [])}"
                )
    if len(by_commit) > 1:
        groups = "; ".join(f"{_short(commit)} ({', '.join(names)})" for commit, names in by_commit.items())
        problems.append(f"the runs come from {len(by_commit)} commits: {groups}")
    elif None in by_commit:
        problems.append(f"no git commit recorded for {', '.join(by_commit[None])}")
    by_references: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        by_references[run.dataset][_references_key(run)].append(run.tool)
    for dataset, keys in by_references.items():
        if len(keys) > 1:
            groups = "; ".join(f"{_short(key)} ({', '.join(tools)})" for key, tools in keys.items())
            problems.append(f"{dataset}: the tools were scored against {len(keys)} reference sets: {groups}")
    return problems


def check_paper_references(runs: Sequence[PaperRun], repo_root: str | Path) -> list[str]:
    """Datasets whose references under `repo_root` are not the ones a run was scored against.

    BioVista is checked by biovista_truth.reference_set_sha256; Internal by the sha256 of its ground-truth
    CSV and of its split manifest. A reference file that cannot be read is a problem, not an exception.
    """
    root = Path(repo_root)
    problems = []
    for dataset in sorted({run.dataset for run in runs}, key=paper_dataset_rank):
        here = [run for run in runs if run.dataset == dataset]
        if dataset == BIOVISTA:
            directory = root / BIOVISTA_ROOT
            try:
                local = reference_set_sha256(directory, root / BIOVISTA_PAPERS_PATH)
            except (OSError, ValueError) as exc:
                problems.append(f"{dataset}: cannot read the references under {directory}: {exc}")
                continue
            stale = [run.name for run in here if _references_key(run) != local]
            if stale:
                problems.append(
                    f"{dataset}: {directory} (sha256 {_short(local)}) holds other references than "
                    f"{', '.join(stale)} was scored against"
                )
        elif dataset == INTERNAL:
            problems += _internal_problems(here, root)
    return problems


# Metrics and tables


def paper_metrics(run: PaperRun) -> dict[str, float | int]:
    """The run's numbers under PAPER_METRIC_KEYS; a key is left out when the run does not record its value."""
    scores = run.report["scores"]
    group = scores["groups"]["all"]
    metrics: dict[str, Any] = {
        **_prf("micro", group["micro"], ci=True),
        **{
            f"macro/{name}{suffix}": value
            for name in ("precision", "recall")
            for suffix, value in _with_ci(group["macro"][name]).items()
        },
        **{f"stripped/{name}": group["stereo_stripped"][name]["value"] for name in PRF},
        **{f"counts/{name}": count for name, count in group["counts"].items()},
        "valid_output_rate": group["valid_output_rate"]["value"],
        "speed/s_per_paper_mean": scores["seconds_per_item"]["mean"],
        "speed/s_per_paper_median": scores["seconds_per_item"]["median"],
        **_resources(run.meta.get("resources")),
        "items/papers": len(scores["papers"]),
        "items/molecules": group["molecules"],
        "items/papers_without_output": group["papers_without_output"],
        "items/crashed": len(run.errors),
    }
    if run.dataset == BIOVISTA:
        drawn = scores["drawn_only"]["groups"]["all"]
        metrics.update(_prf("drawn", drawn["micro"]))
        metrics["drawn/macro_precision"] = drawn["macro"]["precision"]["value"]
        metrics["drawn/macro_recall"] = drawn["macro"]["recall"]["value"]
        metrics.update(_prf("without_submitted", scores["groups"]["without_submitted"]["micro"]))
    else:
        for split in ("dev", "test"):
            metrics.update(_prf(split, scores["groups"][split]["micro"]))
    return {key: value for key, value in metrics.items() if value is not None}


def view_sizes(runs: Sequence[PaperRun]) -> dict[tuple[str, str], int]:
    """(dataset, view) -> the papers a view scores: BioVista all, drawn and without_submitted; Internal all, dev, test.

    Every system is scored on the same papers of a dataset, so the first run of each dataset stands for it.
    """
    sizes: dict[tuple[str, str], int] = {}
    for run in runs:
        if any(dataset == run.dataset for dataset, _ in sizes):
            continue
        scores = run.report["scores"]
        groups = scores["groups"]
        if run.dataset == BIOVISTA:
            views = {
                "all": groups["all"],
                "drawn": scores["drawn_only"]["groups"]["all"],
                "without_submitted": groups["without_submitted"],
            }
        else:
            views = {name: groups[name] for name in ("all", "dev", "test")}
        sizes.update({(run.dataset, view): len(group["papers"]) for view, group in views.items()})
    return sizes


def paper_rows(runs: Sequence[PaperRun]) -> Rows:
    """One row per run and paper: counts, precision, recall and seconds; BioVista rows add the drawn-only counts."""
    rows = []
    for run in runs:
        scores = run.report["scores"]
        drawn = (scores.get("drawn_only") or {}).get("papers", {})
        for paper, record in scores["papers"].items():
            counts = drawn.get(paper)
            rows.append(
                [
                    system_name(run.tool),
                    run.dataset,
                    paper,
                    record["molecules"],
                    record["tp"],
                    record["fp"],
                    record["fn"],
                    record["precision"],
                    record["recall"],
                    run.timing[paper],
                    *((counts[key] for key in ("tp", "fp", "fn", "ignored_outputs")) if counts else (None,) * 4),
                ]
            )
    return list(PAPER_COLUMNS), rows


def molecule_rows(runs: Sequence[PaperRun], references: Mapping[str, Sequence[str]]) -> Rows:
    """BioVista's molecules: each distinct output of each paper with its outcome, then each missed label.

    `references` maps a paper to its readable label SMILES (biovista_truth.references). An output is `tp` when its
    canonical SMILES is a label, `fp` when it is not, `invalid` when RDKit cannot read it; a label no output
    matches is an `fn` row holding the label. Internal runs are never listed: their molecules stay private.
    """
    rows = []
    for run in runs:
        if run.dataset != BIOVISTA:
            continue
        outputs: dict[str, list[str]] = defaultdict(list)
        for prediction in run.predictions:
            outputs[prediction.item_id].append(prediction.smiles)
        for paper in run.report["scores"]["papers"]:
            rows += _paper_molecules(system_name(run.tool), paper, outputs[paper], references[paper])
    return list(MOLECULE_COLUMNS), rows


def check_molecule_rows(runs: Sequence[PaperRun], rows: Sequence[Sequence[Any]]) -> list[str]:
    """Where the molecule rows' counts disagree with the runs' scores.json; empty when they match."""
    counted: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: dict.fromkeys(OUTCOMES, 0))
    for system, paper, _, _, outcome in rows:
        counted[(system, paper)][outcome] += 1
    problems = []
    for run in runs:
        if run.dataset != BIOVISTA:
            continue
        for paper, record in run.report["scores"]["papers"].items():
            found = counted[(system_name(run.tool), paper)]
            here = (found["tp"], found["fp"] + found["invalid"], found["fn"])
            there = (record["tp"], record["fp"], record["fn"])
            if here != there:
                problems.append(f"{run.name}: paper {paper} has tp, fp, fn {here} here, scores.json says {there}")
    return problems


def _prf(prefix: str, micro: Mapping[str, Any], *, ci: bool = False) -> dict[str, float]:
    metrics = {}
    for name in PRF:
        for suffix, value in _with_ci(micro[name], ci=ci).items():
            metrics[f"{prefix}/{name}{suffix}"] = value
    return metrics


def _with_ci(proportion: Mapping[str, Any], *, ci: bool = True) -> dict[str, float]:
    """A scores.json proportion's value, and with `ci` its 95% interval, keyed by suffix."""
    values = {"": proportion["value"]}
    if ci:
        values["_ci_low"], values["_ci_high"] = proportion["ci95"]
    return values


def _paper_molecules(system: str, paper: str, outputs: Sequence[str], labels: Sequence[str]) -> list[list[Any]]:
    truth = {canonical_smiles(label) for label in labels}
    rows: list[list[Any]] = []
    seen: set[str | None] = set()
    for raw in outputs:
        canonical = canonical_smiles(raw)
        key = raw.strip() if canonical is None else canonical
        if key in seen:
            continue
        seen.add(key)
        outcome = "invalid" if canonical is None else "tp" if canonical in truth else "fp"
        rows.append([system, paper, raw, canonical, outcome])
    missed: set[str | None] = set()
    for label in labels:
        canonical = canonical_smiles(label)
        if canonical not in seen and canonical not in missed:
            missed.add(canonical)
            rows.append([system, paper, label, canonical, "fn"])
    return rows


def _references_key(run: PaperRun) -> str:
    """What the run was scored against: BioVista's reference-set sha256, or Internal's ground truth plus split."""
    inputs = run.report["inputs"]
    if run.dataset == BIOVISTA:
        return inputs["references"]["sha256"]
    return f"{inputs['ground_truth_sha256']}+{inputs['split_sha256']}"


def _internal_problems(runs: Sequence[PaperRun], root: Path) -> list[str]:
    problems = []
    for label, path, key in (
        ("ground truth", root / GROUND_TRUTH_PATH, "ground_truth_sha256"),
        ("split", root / SPLIT_PATH, "split_sha256"),
    ):
        try:
            local = sha256_file(path)
        except OSError as exc:
            problems.append(f"{INTERNAL}: cannot read {path}: {exc.strerror or exc}")
            continue
        stale = [run.name for run in runs if run.report["inputs"][key] != local]
        if stale:
            problems.append(
                f"{INTERNAL}: the {label} {path} (sha256 {local[:COMMIT_SHOWN]}) is not the one "
                f"{', '.join(stale)} was scored against"
            )
    return problems


__all__ = [
    "BIOVISTA",
    "INTERNAL",
    "MOLECULE_COLUMNS",
    "PAPER_COLUMNS",
    "PAPER_METRIC_KEYS",
    "SYSTEM_NAMES",
    "PaperRun",
    "check_molecule_rows",
    "check_paper_consistency",
    "check_paper_references",
    "load_paper_run",
    "load_paper_runs",
    "molecule_rows",
    "paper_metrics",
    "paper_result_folders",
    "paper_rows",
    "system_name",
    "system_rank",
    "view_sizes",
]
