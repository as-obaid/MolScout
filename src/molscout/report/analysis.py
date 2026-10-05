"""Benchmark results as numbers: per-crop outcomes, run metrics, agreement between tools and breakdowns.

Pure functions over the result folders `molscout bench` writes; nothing here draws or talks to W&B.
Outcomes use the scorer's own canonical SMILES, so their counts equal scores.json.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from molscout.bench.configs import TYPE1_DATASETS
from molscout.data import molfiles, molrecbench
from molscout.datasets import Kind, dataset_kind
from molscout.hashing import sha256_file
from molscout.predictions import Prediction, read_predictions
from molscout.scoring import canonical_smiles, stereo_stripped_smiles
from molscout.scoring.stats import ratio, wilson_interval
from molscout.tables import DATASETS, TOOL_ROWS

# Mutually exclusive; a crop gets the first that applies.
OUTCOMES = ("crashed", "empty", "invalid", "correct", "stereo_only", "wrong_structure")
RESULT_FILES = ("predictions.csv", "scores.json", "meta.json", "config.yaml", "errors.json")
OPTIONAL_FILES = frozenset({"errors.json"})  # the harness writes it only when run.py does
METRIC_KEYS = (
    "accuracy/stereo_aware",
    "accuracy/stereo_aware_ci_low",
    "accuracy/stereo_aware_ci_high",
    "accuracy/stereo_stripped",
    "accuracy/stereo_stripped_ci_low",
    "accuracy/stereo_stripped_ci_high",
    "valid_output_rate",
    "speed/s_per_crop_mean",
    "speed/s_per_crop_median",
    "resources/gpu_peak_memory_gib",
    "resources/gpu_mean_utilization_pct",
    "resources/peak_rss_gib",
    "resources/cpu_seconds",
    "items/scored",
    "items/excluded",
    "items/crashed",  # every image whose predict call raised; outcome/crashed counts the scored ones
    *(f"outcome/{name}" for name in OUTCOMES),
)
FAILURE_SEED = 6630
NO_HARDCASE = "No hard-case label"
MIB_PER_GIB = 1024
COMMIT_SHOWN = 12
PATHS_SHOWN = 5


class ResultsError(ValueError):
    """A result folder is incomplete, or its files do not belong together."""


class InconsistentResults(ResultsError):
    """The runs cannot be published as one benchmark; `problems` says why."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        listed = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"the results are not one consistent benchmark:\n{listed}")


@dataclass(frozen=True, slots=True)
class Share:
    """`hits` out of `total`, such as correct crops out of scored crops."""

    hits: int
    total: int

    @property
    def value(self) -> float:
        return ratio(self.hits, self.total)

    @property
    def ci95(self) -> tuple[float, float]:
        """Wilson 95% interval, as scores.json computes it."""
        return wilson_interval(self.hits, self.total)


@dataclass(frozen=True, slots=True)
class RunResult:
    """One `<tool>__<dataset>/` result folder, read and checked."""

    folder: Path
    tool: str
    dataset: str
    predictions: tuple[Prediction, ...]
    report: Mapping[str, Any]  # scores.json
    meta: Mapping[str, Any]
    errors: Mapping[str, str]  # crop ID -> the error its predict call raised
    predictions_sha256: str

    @property
    def name(self) -> str:
        return f"{self.tool}__{self.dataset}"


@dataclass(frozen=True, slots=True)
class ItemResult:
    """One tool's answer for one scored crop: the raw SMILES, its canonical form and its outcome."""

    dataset: str
    item_id: str
    tool: str
    smiles: str
    canonical: str | None
    outcome: str


@dataclass(frozen=True, slots=True)
class BoxStats:
    """Quartiles, Tukey whiskers (the furthest values within 1.5 IQR of the box) and the mean."""

    low: float
    q1: float
    median: float
    q3: float
    high: float
    mean: float
    n: int


@dataclass(frozen=True, slots=True)
class ToolResources:
    """A tool's peak memory over its runs and its mean seconds per crop over every crop it read."""

    tool: str
    gpu_peak_memory_gib: float | None
    peak_rss_gib: float | None
    s_per_crop_mean: float


def tool_rank(tool: str) -> tuple[int, str]:
    """Sort key: the six structure readers in table order, then any other tool by name."""
    order = list(TOOL_ROWS)
    return (order.index(tool), "") if tool in TOOL_ROWS else (len(order), tool)


def dataset_rank(dataset: str) -> tuple[int, str]:
    """Sort key: the Type 1 datasets in table order, then any other by name."""
    return (DATASETS.index(dataset), "") if dataset in DATASETS else (len(DATASETS), dataset)


# Loading and checks


def result_folders(root: str | Path) -> list[Path]:
    """The `<tool>__<dataset>/` folders under `root` holding a scores.json for a crop dataset.

    Hidden folders (what a killed harness leaves) and paper datasets (BioVista, Internal) are left out.
    """
    folders = []
    for folder in sorted(Path(root).glob("*__*")):
        if folder.name.startswith(".") or not (folder / "scores.json").is_file():
            continue
        if dataset_kind(folder.name.split("__", 1)[1]) is Kind.CROP:
            folders.append(folder)
    return folders


def load_run(folder: str | Path) -> RunResult:
    """Read one result folder; ResultsError if a file is missing or the files disagree on what they hold."""
    folder = Path(folder)
    missing = [name for name in RESULT_FILES if name not in OPTIONAL_FILES and not (folder / name).is_file()]
    if missing:
        raise ResultsError(f"{folder}: missing {', '.join(missing)}")
    tool, _, dataset = folder.name.partition("__")
    report = _json(folder / "scores.json")
    meta = _json(folder / "meta.json")
    try:
        named = (meta["tool"]["tool"], meta["dataset"], report["dataset"])
        scored_sha256 = report["predictions"]["sha256"]
    except (KeyError, TypeError) as exc:
        raise ResultsError(f"{folder}: meta.json or scores.json lacks {exc}") from None
    if named != (tool, dataset, dataset):
        raise ResultsError(
            f"{folder}: meta.json says {named[0]} on {named[1]} and scores.json says {named[2]}, "
            f"but the folder is {tool} on {dataset}"
        )
    digest = sha256_file(folder / "predictions.csv")
    if digest != scored_sha256:
        raise ResultsError(
            f"{folder}: scores.json was computed from another predictions.csv "
            f"(sha256 {scored_sha256[:COMMIT_SHOWN]}, the file has {digest[:COMMIT_SHOWN]})"
        )
    errors_path = folder / "errors.json"
    errors = _json(errors_path)["errors"] if errors_path.is_file() else {}
    predictions = read_predictions(folder / "predictions.csv")
    return RunResult(folder, tool, dataset, predictions, report, meta, dict(errors), digest)


def load_runs(root: str | Path) -> tuple[RunResult, ...]:
    """Every crop-dataset result folder under `root`, in tool then dataset order."""
    runs = [load_run(folder) for folder in result_folders(root)]
    return tuple(sorted(runs, key=lambda run: (tool_rank(run.tool), dataset_rank(run.dataset))))


def check_consistency(runs: Sequence[RunResult]) -> list[str]:
    """Why the runs are not one benchmark: several or unknown commits, uncommitted code, or different references."""
    problems = []
    by_commit: dict[str | None, list[str]] = defaultdict(list)
    for run in runs:
        git = run.meta.get("git") or {}
        by_commit[git.get("commit")].append(run.name)
        if git.get("dirty") is None:
            problems.append(f"{run.name} does not record whether its code was committed")
        elif git["dirty"]:
            problems.append(f"{run.name} ran with uncommitted changes: {_listed(git.get('dirty_paths') or [])}")
    if len(by_commit) > 1:
        groups = "; ".join(f"{_short(commit)} ({', '.join(names)})" for commit, names in by_commit.items())
        problems.append(f"the runs come from {len(by_commit)} commits: {groups}")
    elif None in by_commit:
        problems.append(f"no git commit recorded for {', '.join(by_commit[None])}")
    by_references: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        by_references[run.dataset][_references_sha256(run)].append(run.tool)
    for dataset, shas in by_references.items():
        if len(shas) > 1:
            groups = "; ".join(f"{_short(sha)} ({', '.join(tools)})" for sha, tools in shas.items())
            problems.append(f"{dataset}: the tools were scored against {len(shas)} reference sets: {groups}")
    return problems


def check_references(runs: Sequence[RunResult], repo_root: str | Path) -> list[str]:
    """Datasets whose references under `repo_root` are not the ones a run was scored against."""
    problems = []
    for dataset in sorted({run.dataset for run in runs}, key=dataset_rank):
        directory = reference_path(dataset, repo_root)
        local = _loader(dataset).reference_set_sha256(directory)
        stale = [run.name for run in runs if run.dataset == dataset and _references_sha256(run) != local]
        if stale:
            problems.append(
                f"{dataset}: {directory} (sha256 {_short(local)}) holds other references than "
                f"{', '.join(stale)} was scored against"
            )
    return problems


def reference_path(dataset: str, repo_root: str | Path) -> Path:
    """A crop dataset's references: its molfile folder, or the MolRecBench-Wild root."""
    return Path(repo_root) / TYPE1_DATASETS[dataset].references


def image_path(dataset: str, repo_root: str | Path) -> Path:
    """A crop dataset's images folder."""
    return Path(repo_root) / TYPE1_DATASETS[dataset].images


def load_truth(dataset: str, repo_root: str | Path) -> dict[str, str | None]:
    """Each crop's canonical reference SMILES; None where RDKit cannot read it, so the crop is not scored."""
    loaded = _loader(dataset).load_references(reference_path(dataset, repo_root))
    return {item: None if ref.smiles is None else canonical_smiles(ref.smiles) for item, ref in loaded.items()}


# Outcomes and metrics


def item_results(run: RunResult, truth: Mapping[str, str | None]) -> tuple[ItemResult, ...]:
    """One result per scored crop, in crop ID order; a crop without a prediction row has an empty answer."""
    answers = {prediction.item_id: prediction.smiles for prediction in run.predictions}
    results = []
    for item_id in sorted(item for item, reference in truth.items() if reference is not None):
        smiles = answers.get(item_id, "")
        canonical = _canonical(smiles)
        outcome = _outcome(item_id in run.errors, smiles, canonical, truth[item_id])
        results.append(ItemResult(run.dataset, item_id, run.tool, smiles, canonical, outcome))
    return tuple(results)


def outcome_counts(items: Iterable[ItemResult]) -> dict[str, int]:
    """Crops per outcome, every outcome present."""
    counts = Counter(result.outcome for result in items)
    return {name: counts[name] for name in OUTCOMES}


def check_outcomes(run: RunResult, items: Sequence[ItemResult]) -> list[str]:
    """Where the run's outcome counts disagree with its scores.json; empty when they match."""
    scores = run.report["scores"]
    counts = outcome_counts(items)
    checks = (
        ("crops scored", len(items), scores["items_scored"]),
        ("correct", counts["correct"], scores["accuracy"]["successes"]),
        (
            "correct without stereo",
            counts["correct"] + counts["stereo_only"],
            scores["accuracy_stereo_stripped"]["successes"],
        ),
        (
            "valid outputs",
            sum(result.canonical is not None for result in items),
            scores["valid_output_rate"]["successes"],
        ),
    )
    return [
        f"{run.name}: {here} {label} here, scores.json says {there}" for label, here, there in checks if here != there
    ]


def run_metrics(run: RunResult, items: Iterable[ItemResult]) -> dict[str, float | int]:
    """The run's numbers under METRIC_KEYS; a key is left out when the run does not record its value."""
    scores = run.report["scores"]
    aware, stripped = scores["accuracy"], scores["accuracy_stereo_stripped"]
    metrics = {
        "accuracy/stereo_aware": aware["value"],
        "accuracy/stereo_aware_ci_low": aware["ci95"][0],
        "accuracy/stereo_aware_ci_high": aware["ci95"][1],
        "accuracy/stereo_stripped": stripped["value"],
        "accuracy/stereo_stripped_ci_low": stripped["ci95"][0],
        "accuracy/stereo_stripped_ci_high": stripped["ci95"][1],
        "valid_output_rate": scores["valid_output_rate"]["value"],
        "speed/s_per_crop_mean": scores["seconds_per_item"]["mean"],
        "speed/s_per_crop_median": scores["seconds_per_item"]["median"],
        **_resources(run.meta.get("resources")),
        "items/scored": scores["items_scored"],
        "items/excluded": len(scores["items_excluded"]),
        "items/crashed": len(run.errors),
        **{f"outcome/{name}": count for name, count in outcome_counts(items).items()},
    }
    return {key: value for key, value in metrics.items() if value is not None}


def tool_resources(runs: Sequence[RunResult]) -> tuple[ToolResources, ...]:
    """Per tool, in tool order: peak GPU memory and RAM over its runs, and seconds per crop pooled over them."""
    by_tool: dict[str, list[RunResult]] = defaultdict(list)
    for run in runs:
        by_tool[run.tool].append(run)
    resources = []
    for tool in sorted(by_tool, key=tool_rank):
        blocks = [run.meta.get("resources") or {} for run in by_tool[tool]]
        speeds = [run.report["scores"]["seconds_per_item"] for run in by_tool[tool]]
        gpu = [_gib((block.get("gpu") or {}).get("peak_memory_mib")) for block in blocks]
        rss = [_gib(block.get("tool_peak_rss_mib")) for block in blocks]
        seconds = ratio(sum(speed["total"] for speed in speeds), sum(speed["items"] for speed in speeds))
        resources.append(ToolResources(tool, _peak(gpu), _peak(rss), seconds))
    return tuple(resources)


# Comparisons between tools


def by_item(items: Iterable[ItemResult]) -> dict[tuple[str, str], dict[str, ItemResult]]:
    """Each (dataset, crop ID)'s results keyed by tool, in tool order."""
    grouped: dict[tuple[str, str], dict[str, ItemResult]] = defaultdict(dict)
    for result in sorted(items, key=lambda result: tool_rank(result.tool)):
        grouped[(result.dataset, result.item_id)][result.tool] = result
    return dict(grouped)


def pairwise_agreement(items: Iterable[ItemResult]) -> dict[tuple[str, str, str], Share]:
    """(dataset, tool_a, tool_b) -> crops where both give the same canonical SMILES, out of crops both were scored on.

    Each pair appears once, tool_a first in tool order. An empty, invalid or crashed answer agrees with nothing.
    """
    hits: Counter[tuple[str, str, str]] = Counter()
    totals: Counter[tuple[str, str, str]] = Counter()
    for (dataset, _), answers in by_item(items).items():
        tools = list(answers)
        for index, first in enumerate(tools):
            for second in tools[index + 1 :]:
                key = (dataset, first, second)
                totals[key] += 1
                answer = _answer(answers[first])
                hits[key] += answer is not None and answer == _answer(answers[second])
    return {key: Share(hits[key], totals[key]) for key in totals}


def agreement_accuracy(items: Iterable[ItemResult]) -> dict[tuple[str, int], Share]:
    """(dataset, k) -> how often the answer that k tools share is right, for k >= 2.

    k is the size of the largest group of tools giving one canonical SMILES. A crop where two
    answers tie for largest has no agreed answer and is left out.
    """
    hits: Counter[tuple[str, int]] = Counter()
    totals: Counter[tuple[str, int]] = Counter()
    for (dataset, _), answers in by_item(items).items():
        groups: dict[str, list[ItemResult]] = defaultdict(list)
        for result in answers.values():
            if _answer(result) is not None:
                groups[result.canonical].append(result)
        sizes = sorted((len(group) for group in groups.values()), reverse=True)
        if not sizes or sizes[0] < 2 or sizes[1:2] == sizes[:1]:
            continue
        agreed = max(groups.values(), key=len)
        key = (dataset, len(agreed))
        totals[key] += 1
        hits[key] += any(result.outcome == "correct" for result in agreed)
    order = sorted(totals, key=lambda key: (dataset_rank(key[0]), key[1]))
    return {key: Share(hits[key], totals[key]) for key in order}


def oracle(items: Iterable[ItemResult]) -> dict[str, Share]:
    """Per dataset: crops at least one tool reads correctly, out of all its scored crops."""
    hits: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for (dataset, _), answers in by_item(items).items():
        totals[dataset] += 1
        hits[dataset] += any(result.outcome == "correct" for result in answers.values())
    return {dataset: Share(hits[dataset], totals[dataset]) for dataset in sorted(totals, key=dataset_rank)}


def pooled(shares: Mapping[tuple[Any, ...], Share]) -> dict[tuple[Any, ...], Share]:
    """Shares summed over datasets: each key loses its first element, the dataset."""
    hits: Counter[tuple[Any, ...]] = Counter()
    totals: Counter[tuple[Any, ...]] = Counter()
    for key, share in shares.items():
        hits[key[1:]] += share.hits
        totals[key[1:]] += share.total
    return {key: Share(hits[key], totals[key]) for key in totals}


def accuracy_by_group(items: Iterable[ItemResult], groups: Mapping[str, Iterable[str]]) -> dict[tuple[str, str], Share]:
    """(tool, group) -> accuracy over the crops in a group; a crop counts in every group it carries.

    `groups` maps crop ID to its groups (a repeated group counts once); crops it does not list are left out.
    """
    hits: Counter[tuple[str, str]] = Counter()
    totals: Counter[tuple[str, str]] = Counter()
    for result in items:
        for group in dict.fromkeys(groups.get(result.item_id, ())):
            key = (result.tool, group)
            totals[key] += 1
            hits[key] += result.outcome == "correct"
    order = sorted(totals, key=lambda key: (tool_rank(key[0]), key[1]))
    return {key: Share(hits[key], totals[key]) for key in order}


def molrecbench_groups(
    samples: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """MolRecBench-Wild crops grouped two ways: by evaluation subset, and by hard-case label."""
    subsets = {
        item: (f"Subset {sample['evaluation_subset']}",)
        for item, sample in samples.items()
        if sample["evaluation_subset"]
    }
    hardcases = {item: tuple(sample["hardcase_label"]) or (NO_HARDCASE,) for item, sample in samples.items()}
    return subsets, hardcases


# Speed and samples


def crop_seconds(runs: Iterable[RunResult]) -> dict[str, list[float]]:
    """Every prediction row's seconds, pooled per tool over its runs."""
    seconds: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        seconds[run.tool].extend(prediction.seconds for prediction in run.predictions)
    return dict(seconds)


def box_stats(values: Sequence[float]) -> BoxStats:
    """Quartiles (linear interpolation), Tukey whiskers and the mean of at least one value."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("box_stats needs at least one value")
    q1, median, q3 = (float(value) for value in np.percentile(data, [25, 50, 75]))
    reach = 1.5 * (q3 - q1)
    low = float(data[data >= q1 - reach].min())
    high = float(data[data <= q3 + reach].max())
    return BoxStats(low, q1, median, q3, high, float(data.mean()), int(data.size))


def failure_sample(items: Iterable[ItemResult], per_dataset: int, seed: int = FAILURE_SEED) -> dict[str, list[str]]:
    """Up to `per_dataset` crop IDs per dataset where some tool is not correct, drawn with random.Random(seed).

    Each dataset draws from its own Random(seed), so its sample does not depend on the others.
    """
    if per_dataset <= 0:
        return {}
    candidates: dict[str, set[str]] = defaultdict(set)
    for result in items:
        if result.outcome != "correct":
            candidates[result.dataset].add(result.item_id)
    sample = {}
    for dataset in sorted(candidates, key=dataset_rank):
        pool = sorted(candidates[dataset])
        sample[dataset] = sorted(random.Random(seed).sample(pool, min(per_dataset, len(pool))))
    return sample


def _answer(result: ItemResult) -> str | None:
    """The canonical SMILES a tool answered with; None for no usable answer, a crash included."""
    return None if result.outcome == "crashed" else result.canonical


def _outcome(crashed: bool, smiles: str, canonical: str | None, reference: str) -> str:
    # The stripped comparison is the scorer's own (scoring.crops): both sides stripped from canonical form.
    if crashed:
        return "crashed"
    if not smiles.strip():
        return "empty"
    if canonical is None:
        return "invalid"
    if canonical == reference:
        return "correct"
    if _stripped(canonical) == _stripped(reference):
        return "stereo_only"
    return "wrong_structure"


@cache
def _canonical(smiles: str) -> str | None:
    return canonical_smiles(smiles)


@cache
def _stripped(canonical: str) -> str | None:
    return stereo_stripped_smiles(canonical)


def _resources(block: Mapping[str, Any] | None) -> dict[str, float | None]:
    """meta.json's optional `resources` block; runs recorded before it existed have none."""
    if not block:
        return {}
    gpu = block.get("gpu") or {}
    return {
        "resources/gpu_peak_memory_gib": _gib(gpu.get("peak_memory_mib")),
        "resources/gpu_mean_utilization_pct": gpu.get("mean_utilization_pct"),
        "resources/peak_rss_gib": _gib(block.get("tool_peak_rss_mib")),
        "resources/cpu_seconds": block.get("tool_cpu_seconds"),
    }


def _gib(mib: float | None) -> float | None:
    return None if mib is None else mib / MIB_PER_GIB


def _peak(values: Iterable[float | None]) -> float | None:
    known = [value for value in values if value is not None]
    return max(known) if known else None


def _loader(dataset: str) -> ModuleType:
    """The module whose load_references and reference_set_sha256 read this dataset."""
    return molrecbench if dataset == "molrecbench_wild" else molfiles


def _references_sha256(run: RunResult) -> str:
    return run.report["inputs"]["references"]["sha256"]


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResultsError(f"{path}: not valid JSON: {exc}") from None


def _short(value: str | None) -> str:
    return "unknown" if value is None else value[:COMMIT_SHOWN]


def _listed(paths: Sequence[str]) -> str:
    shown = ", ".join(paths[:PATHS_SHOWN])
    hidden = len(paths) - PATHS_SHOWN
    return f"{shown} and {hidden} more" if hidden > 0 else shown
