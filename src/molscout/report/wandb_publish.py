"""Publish the structure-reader benchmark to Weights & Biases.

One `eval` run per result folder (config, namespaced summary, the folder as an artifact) and one
`analysis` run named summary (figures and tables). Run IDs come from the predictions' sha256, so
uploading the same results again updates those runs instead of adding new ones. Everything is
computed and checked before the first run starts.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import wandb

from molscout.bench.harness import IMAGE_SUFFIXES
from molscout.data.molrecbench import read_sample_labels
from molscout.report import figures
from molscout.report.analysis import (
    FAILURE_SEED,
    METRIC_KEYS,
    RESULT_FILES,
    InconsistentResults,
    ItemResult,
    ResultsError,
    RunResult,
    accuracy_by_group,
    agreement_accuracy,
    box_stats,
    by_item,
    check_consistency,
    check_outcomes,
    check_references,
    crop_seconds,
    dataset_rank,
    failure_sample,
    image_path,
    item_results,
    load_runs,
    load_truth,
    molrecbench_groups,
    oracle,
    pairwise_agreement,
    pooled,
    reference_path,
    run_metrics,
    tool_rank,
    tool_resources,
)

GROUP = "structure-readers"
ARTIFACT_TYPE = "benchmark-run"
SUMMARY_NAME = "summary"
HASH_SHOWN = 10
WILD = "molrecbench_wild"

Rows = tuple[list[str], list[list[Any]]]


@dataclass(frozen=True, slots=True)
class Benchmark:
    """The runs with every crop's outcome and each run's metrics, checked against scores.json."""

    runs: tuple[RunResult, ...]
    items: Mapping[str, tuple[ItemResult, ...]]  # run name -> one result per scored crop
    truth: Mapping[str, Mapping[str, str | None]]  # dataset -> crop ID -> canonical reference
    metrics: Mapping[tuple[str, str], Mapping[str, float | int]]  # (tool, dataset) -> run_metrics
    configs: Mapping[str, Mapping[str, Any]]  # run name -> run_config

    def all_items(self) -> list[ItemResult]:
        return list(chain.from_iterable(self.items.values()))


def prepare(
    results: str | Path,
    repo_root: str | Path,
    *,
    allow_inconsistent: bool = False,
    warn: Callable[[str], None] = print,
) -> Benchmark:
    """Load and check every result folder; InconsistentResults unless `allow_inconsistent` (development only)."""
    runs = load_runs(results)
    if not runs:
        raise ResultsError(f"no crop-dataset result folders with a scores.json under {results}")
    problems = check_consistency(runs) + check_references(runs, repo_root)
    if problems and not allow_inconsistent:
        raise InconsistentResults(problems)
    for problem in problems:
        warn(f"allowed for development: {problem}")
    truth = {
        dataset: load_truth(dataset, repo_root) for dataset in sorted({run.dataset for run in runs}, key=dataset_rank)
    }
    items = {run.name: item_results(run, truth[run.dataset]) for run in runs}
    mismatches = [problem for run in runs for problem in check_outcomes(run, items[run.name])]
    if mismatches:
        raise ResultsError("outcome counts disagree with scores.json:\n" + "\n".join(f"  - {m}" for m in mismatches))
    metrics = {(run.tool, run.dataset): run_metrics(run, items[run.name]) for run in runs}
    return Benchmark(runs, items, truth, metrics, {run.name: _checked_config(run) for run in runs})


def run_id(run: RunResult) -> str:
    return f"{run.tool}-{run.dataset}-{run.predictions_sha256[:HASH_SHOWN]}"


def summary_run_id(runs: Sequence[RunResult]) -> str:
    digest = hashlib.sha256("\n".join(sorted(run.predictions_sha256 for run in runs)).encode()).hexdigest()
    return f"{SUMMARY_NAME}-{digest[:HASH_SHOWN]}"


def run_config(run: RunResult) -> dict[str, Any]:
    """What produced the run: tool and version, dataset, commit, checkpoints, environment lock and device."""
    tool = run.meta["tool"]
    return {
        "tool": run.tool,
        "name": tool["name"],
        "version": tool["version"],
        "dataset": run.dataset,
        "git_commit": run.meta["git"]["commit"],
        "checkpoints": [{"file": Path(c["path"]).name, "sha256": c["sha256"]} for c in tool["checkpoints"]],
        "environment_sha256": run.meta["environment"]["sha256"],
        "device": _device(run.meta),
        "slurm_partition": run.meta["slurm"]["partition"],
    }


def _checked_config(run: RunResult) -> dict[str, Any]:
    """run_config, or ResultsError naming the meta.json field the run lacks."""
    try:
        return run_config(run)
    except (KeyError, IndexError, TypeError) as exc:
        raise ResultsError(f"{run.name}: meta.json lacks {exc}") from None


def figure_panels(benchmark: Benchmark, repo_root: str | Path) -> dict[str, go.Figure]:
    """The summary run's figures by W&B key; comparisons between tools need two tools, breakdowns MolRecBench-Wild."""
    runs, metrics, items = benchmark.runs, benchmark.metrics, benchmark.all_items()
    panels = {
        "accuracy/by_dataset": figures.accuracy_bars(metrics),
        "accuracy/stereo_penalty": figures.stereo_penalty_bars(metrics),
        "accuracy/vs_speed": figures.accuracy_speed_scatter(metrics),
        "outcomes/by_tool": figures.outcome_bars(metrics),
        "speed/time_per_crop": figures.time_boxes({tool: box_stats(s) for tool, s in crop_seconds(runs).items() if s}),
        "resources/by_tool": figures.resource_bars(tool_resources(runs)),
    }
    if len({run.tool for run in runs}) > 1:
        panels["agreement/pairwise"] = figures.agreement_heatmap(pooled(pairwise_agreement(items)))
        panels["agreement/consensus"] = figures.consensus_bars(agreement_accuracy(items), oracle(items))
    if any(run.dataset == WILD for run in runs):
        subsets, hardcases = molrecbench_groups(read_sample_labels(reference_path(WILD, repo_root)))
        wild = [result for result in items if result.dataset == WILD]
        panels[f"{WILD}/by_subset"] = figures.subset_bars(accuracy_by_group(wild, subsets))
        panels[f"{WILD}/by_hardcase"] = figures.hardcase_heatmap(accuracy_by_group(wild, hardcases))
    return panels


def leaderboard_rows(benchmark: Benchmark) -> Rows:
    """One row per run with every summary metric; None where a run does not record one."""
    keys = sorted(benchmark.metrics, key=lambda key: (tool_rank(key[0]), dataset_rank(key[1])))
    rows = [
        [tool, dataset, *(benchmark.metrics[(tool, dataset)].get(name) for name in METRIC_KEYS)]
        for tool, dataset in keys
    ]
    return ["tool", "dataset", *METRIC_KEYS], rows


def prediction_rows(benchmark: Benchmark) -> Rows:
    """Every scored crop: its canonical reference, and each tool's raw answer and outcome."""
    tools = _tools(benchmark)
    rows = []
    for (dataset, item_id), answers in _ordered(by_item(benchmark.all_items())):
        rows.append([dataset, item_id, benchmark.truth[dataset][item_id], *_answers(answers, tools)])
    return ["dataset", "item_id", "reference", *_answer_columns(tools)], rows


def failure_rows(benchmark: Benchmark, per_dataset: int, repo_root: str | Path) -> Rows:
    """Up to `per_dataset` crops per dataset where some tool is wrong, seeded; the image column holds its path."""
    tools = _tools(benchmark)
    grouped = by_item(benchmark.all_items())
    rows = []
    for dataset, item_ids in failure_sample(benchmark.all_items(), per_dataset, FAILURE_SEED).items():
        images = image_path(dataset, repo_root)
        for item_id in item_ids:
            image = _find_image(images, item_id)
            rows.append(
                [
                    dataset,
                    item_id,
                    image,
                    benchmark.truth[dataset][item_id],
                    *_answers(grouped[(dataset, item_id)], tools),
                ]
            )
    return ["dataset", "item_id", "image", "reference", *_answer_columns(tools)], rows


def publish(
    benchmark: Benchmark,
    *,
    entity: str,
    project: str,
    repo_root: str | Path,
    offline: bool = False,
    images_per_dataset: int = 200,
    echo: Callable[[str], None] = print,
) -> None:
    """Log one eval run per result folder, then the summary run; echo each run's URL (its ID offline).

    Everything logged is built before the first run starts, so a failure leaves W&B untouched.
    """
    panels = figure_panels(benchmark, repo_root)
    tables = _tables(benchmark, images_per_dataset, repo_root)
    artifacts = {run.name: _artifact(run) for run in benchmark.runs}
    summary_config = _summary_config(benchmark, images_per_dataset)
    # Offline runs cannot resume; `wandb sync` updates the run with the same ID instead.
    common = {"entity": entity, "project": project, "save_code": False, "settings": _settings()}
    common.update({"mode": "offline"} if offline else {"resume": "allow"})
    for run in benchmark.runs:
        name = f"{run.tool}/{run.dataset}"
        tags = [run.tool, run.dataset]
        with wandb.init(id=run_id(run), name=name, group=GROUP, job_type="eval", tags=tags, **common) as logged:
            logged.config.update(dict(benchmark.configs[run.name]), allow_val_change=True)
            logged.summary.update(dict(benchmark.metrics[(run.tool, run.dataset)]))
            logged.log_artifact(artifacts[run.name])
            echo(f"{name}: {logged.id if offline else logged.url}")
    with wandb.init(id=summary_run_id(benchmark.runs), name=SUMMARY_NAME, job_type="analysis", **common) as logged:
        logged.config.update(summary_config, allow_val_change=True)
        logged.log({**{key: wandb.Plotly(figure) for key, figure in panels.items()}, **tables})
        echo(f"{SUMMARY_NAME}: {logged.id if offline else logged.url}")


def _tables(benchmark: Benchmark, images_per_dataset: int, repo_root: str | Path) -> dict[str, wandb.Table]:
    tables = {
        "leaderboard": wandb.Table(*leaderboard_rows(benchmark)),
        "predictions": wandb.Table(*prediction_rows(benchmark)),
    }
    if images_per_dataset > 0:
        columns, rows = failure_rows(benchmark, images_per_dataset, repo_root)
        tables["failures"] = wandb.Table(columns, [[*row[:2], wandb.Image(str(row[2])), *row[3:]] for row in rows])
    return tables


def _artifact(run: RunResult) -> wandb.Artifact:
    """The result folder's files, as the harness wrote them."""
    artifact = wandb.Artifact(
        run.name, type=ARTIFACT_TYPE, description=f"{run.tool} on {run.dataset}: the harness's result folder"
    )
    for name in RESULT_FILES:
        if (run.folder / name).is_file():
            artifact.add_file(str(run.folder / name), name=name)
    return artifact


def _summary_config(benchmark: Benchmark, images_per_dataset: int) -> dict[str, Any]:
    commits = sorted({config["git_commit"] for config in benchmark.configs.values()}, key=lambda commit: commit or "")
    return {
        "git_commit": commits[0] if len(commits) == 1 else commits,
        "runs": {f"{run.tool}/{run.dataset}": run.predictions_sha256 for run in benchmark.runs},
        "failures_per_dataset": images_per_dataset,
        "failures_seed": FAILURE_SEED,
    }


def _settings() -> wandb.Settings:
    """No system metrics, host metadata, git state, code or console capture: the uploader's machine is not the run's."""
    return wandb.Settings(
        console="off",
        disable_code=True,
        disable_git=True,
        silent=True,
        x_disable_machine_info=True,
        x_disable_meta=True,
        x_disable_stats=True,
        x_save_requirements=False,
    )


def _device(meta: Mapping[str, Any]) -> str | None:
    """The GPU the tool ran on, or the CPU model when it used none; without a resources block, the job's first GPU."""
    hardware = meta["hardware"]
    resources = meta.get("resources")
    if resources is not None:
        gpu = resources.get("gpu")
        return gpu["name"] if gpu else hardware["cpu_model"]
    return hardware["gpus"][0]["name"] if hardware["gpus"] else hardware["cpu_model"]


def _find_image(folder: Path, item_id: str) -> Path:
    for suffix in sorted(IMAGE_SUFFIXES):
        path = folder / f"{item_id}{suffix}"
        if path.is_file():
            return path
    raise ResultsError(f"no image for crop {item_id!r} in {folder}; fetch the dataset or pass --images-per-dataset 0")


def _tools(benchmark: Benchmark) -> list[str]:
    return sorted({run.tool for run in benchmark.runs}, key=tool_rank)


def _ordered(
    grouped: Mapping[tuple[str, str], Mapping[str, ItemResult]],
) -> list[tuple[tuple[str, str], Mapping[str, ItemResult]]]:
    return sorted(grouped.items(), key=lambda entry: (dataset_rank(entry[0][0]), entry[0][1]))


def _answer_columns(tools: Sequence[str]) -> list[str]:
    return [column for tool in tools for column in (f"{tool}/smiles", f"{tool}/outcome")]


def _answers(answers: Mapping[str, ItemResult], tools: Sequence[str]) -> list[str | None]:
    return [
        value
        for tool in tools
        for value in ((answers[tool].smiles, answers[tool].outcome) if tool in answers else (None, None))
    ]
