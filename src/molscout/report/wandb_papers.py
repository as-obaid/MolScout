"""Publish the complete-system (Type 2) benchmark to Weights & Biases.

The paper twin of report.wandb_publish: one `eval` run per `<tool>__<dataset>` result folder (config, namespaced
summary, the folder as an artifact) in group `complete-systems`, and one `analysis` run (figures and tables). Run
IDs come from the predictions' sha256, so uploading the same results again updates those runs.

The Internal set is private: its runs publish metrics and per-paper counts only. No Internal SMILES reaches a table,
a config or an artifact (the artifact leaves out predictions.csv, and its errors.json keeps only the error types).
BioVista is public data and uploads everything. Everything is built before the first run starts.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import wandb

from molscout.data.biovista_truth import BIOVISTA_PAPERS_PATH, load_biovista_truth
from molscout.report import paper_figures
from molscout.report.analysis import RESULT_FILES, InconsistentResults, ResultsError
from molscout.report.figures import LIGHT, Theme
from molscout.report.paper_analysis import (
    BIOVISTA,
    BIOVISTA_ROOT,
    INTERNAL,
    PAPER_METRIC_KEYS,
    PaperRun,
    check_molecule_rows,
    check_paper_consistency,
    check_paper_references,
    load_paper_runs,
    molecule_rows,
    paper_dataset_rank,
    paper_metrics,
    paper_rows,
    system_name,
    system_rank,
    view_sizes,
)
from molscout.report.wandb_publish import (
    ARTIFACT_TYPE,
    FIGURE_PREFIX,
    HASH_SHOWN,
    _settings,
    panel_html,
    run_config,
    run_id,
)
from molscout.tables_papers import crash_warning

GROUP = "complete-systems"
SUMMARY_NAME = "summary-complete-systems"
# Python exception names, which never hold a structure; anything else is reported as plain "error".
ERROR_TYPE = re.compile(r"[A-Za-z_][\w.]*(Error|Exception|Exit|Warning|Interrupt)")

Rows = tuple[list[str], list[list[Any]]]


@dataclass(frozen=True, slots=True)
class PaperBenchmark:
    """The paper runs with each run's metrics and W&B config, checked."""

    runs: tuple[PaperRun, ...]
    metrics: Mapping[tuple[str, str], Mapping[str, float | int]]  # (tool, dataset) -> paper_metrics
    configs: Mapping[str, Mapping[str, Any]]  # run name -> paper_config


def prepare_papers(
    results: str | Path,
    repo_root: str | Path,
    *,
    allow_inconsistent: bool = False,
    warn: Callable[[str], None] = print,
) -> PaperBenchmark:
    """Load and check every paper result folder; InconsistentResults unless `allow_inconsistent` (development only)."""
    runs = load_paper_runs(results)
    if not runs:
        raise ResultsError(f"no paper-dataset result folders with a scores.json under {results}")
    problems = check_paper_consistency(runs) + check_paper_references(runs, repo_root)
    if problems and not allow_inconsistent:
        raise InconsistentResults(problems)
    for problem in problems:
        warn(f"allowed for development: {problem}")
    for run in runs:
        if run.errors:
            warn(crash_warning(run.name, len(run.errors), len(run.report["scores"]["papers"])))
    metrics = {(run.tool, run.dataset): paper_metrics(run) for run in runs}
    return PaperBenchmark(runs, metrics, {run.name: _checked_config(run) for run in runs})


def paper_config(run: PaperRun) -> dict[str, Any]:
    """What produced the run: run_config's tool, commit, checkpoints and device, plus the GPU count and source clones.

    A source is named by its folder, with its commit and whether it had uncommitted changes.
    """
    return {
        **run_config(run),
        "gpus": len(run.meta["hardware"]["gpus"]),
        "sources": [
            {
                "repo": Path(source["path"]).name,
                "commit": source["commit"],
                "dirty": source["dirty"],
                "untracked": source["untracked"],
            }
            for source in run.meta.get("sources") or []
        ],
    }


def summary_run_id(runs: tuple[PaperRun, ...] | list[PaperRun]) -> str:
    digest = hashlib.sha256("\n".join(sorted(run.predictions_sha256 for run in runs)).encode()).hexdigest()
    return f"{SUMMARY_NAME}-{digest[:HASH_SHOWN]}"


def figure_panels(benchmark: PaperBenchmark, theme: Theme = LIGHT) -> dict[str, go.Figure]:
    """The summary run's figures by W&B key, in report order; a dataset with no run has no figure."""
    runs, metrics = benchmark.runs, benchmark.metrics
    datasets = {run.dataset for run in runs}
    sizes = view_sizes(runs)
    panels = {}
    for dataset in (BIOVISTA, INTERNAL):
        if dataset in datasets:
            panels[f"{FIGURE_PREFIX}{dataset}_views"] = paper_figures.view_bars(metrics, dataset, sizes, theme=theme)
    if BIOVISTA in datasets:
        panels[f"{FIGURE_PREFIX}recall_by_paper"] = paper_figures.recall_by_paper(paper_rows(runs), theme=theme)
    panels[f"{FIGURE_PREFIX}f1_vs_time"] = paper_figures.f1_vs_time(metrics, theme=theme)
    return panels


def leaderboard_rows(benchmark: PaperBenchmark) -> Rows:
    """One row per run with every summary metric; None where a run does not record one."""
    keys = sorted(benchmark.metrics, key=lambda key: (system_rank(key[0]), paper_dataset_rank(key[1])))
    rows = [
        [system_name(tool), dataset, *(benchmark.metrics[(tool, dataset)].get(name) for name in PAPER_METRIC_KEYS)]
        for tool, dataset in keys
    ]
    return ["system", "dataset", *PAPER_METRIC_KEYS], rows


def table_rows(benchmark: PaperBenchmark, repo_root: str | Path) -> dict[str, Rows]:
    """The summary run's tables: leaderboard and per-paper counts for every run, molecules for BioVista only.

    Internal runs appear in the first two as counts and metrics; their molecules are never listed.
    """
    tables = {"leaderboard": leaderboard_rows(benchmark), "papers": paper_rows(benchmark.runs)}
    if any(run.dataset == BIOVISTA for run in benchmark.runs):
        root = Path(repo_root)
        truth = load_biovista_truth(root / BIOVISTA_ROOT, root / BIOVISTA_PAPERS_PATH)
        columns, rows = molecule_rows(benchmark.runs, truth.references)
        problems = check_molecule_rows(benchmark.runs, rows)
        if problems:
            raise ResultsError(
                "molecule rows disagree with scores.json:\n" + "\n".join(f"  - {problem}" for problem in problems)
            )
        tables["molecules"] = (columns, rows)
    return tables


def artifact_for(run: PaperRun) -> wandb.Artifact:
    """The result folder's files, as the harness wrote them; for Internal, without predictions.csv.

    An Internal run's errors.json keeps each paper's error type and drops the message, which a tool may
    word with a structure.
    """
    private = run.dataset == INTERNAL
    artifact = wandb.Artifact(
        run.name,
        type=ARTIFACT_TYPE,
        description=f"{run.tool} on {run.dataset}: the harness's result folder"
        + (" without predictions.csv (private data)" if private else ""),
    )
    for name in (*RESULT_FILES, "timing.json"):
        path = run.folder / name
        if not path.is_file() or (private and name == "predictions.csv"):
            continue
        if private and name == "errors.json":
            with artifact.new_file(name) as handle:
                handle.write(_error_types(path))
        else:
            artifact.add_file(str(path), name=name)
    return artifact


def publish_papers(
    benchmark: PaperBenchmark,
    *,
    entity: str,
    project: str,
    repo_root: str | Path,
    offline: bool = False,
    echo: Callable[[str], None] = print,
) -> None:
    """Log one eval run per result folder, then the summary run; echo each run's URL (its ID offline).

    Everything logged is built before the first run starts, so a failure leaves W&B untouched.
    """
    light = figure_panels(benchmark)
    dark = figure_panels(benchmark, theme=paper_figures.DARK)
    plots = {key: wandb.Html(panel_html(light[key], dark[key]), inject=False) for key in light}
    tables = {name: wandb.Table(*rows) for name, rows in table_rows(benchmark, repo_root).items()}
    artifacts = {run.name: artifact_for(run) for run in benchmark.runs}
    summary_config = _summary_config(benchmark)
    # Offline runs cannot resume; `wandb sync` updates the run with the same ID instead.
    common = {"entity": entity, "project": project, "save_code": False}
    common.update({"mode": "offline"} if offline else {"resume": "allow"})
    for run in benchmark.runs:
        name = f"{run.tool}/{run.dataset}"
        # A fresh Settings per run: wandb.init writes each run's tags and group into the one it is given.
        with wandb.init(
            id=run_id(run),
            name=name,
            group=GROUP,
            job_type="eval",
            tags=[run.tool, run.dataset],
            settings=_settings(),
            **common,
        ) as logged:
            logged.config.update(dict(benchmark.configs[run.name]), allow_val_change=True)
            logged.summary.update(dict(benchmark.metrics[(run.tool, run.dataset)]))
            logged.log_artifact(artifacts[run.name])
            echo(f"{name}: {logged.id if offline else logged.url}")
    with wandb.init(
        id=summary_run_id(benchmark.runs), name=SUMMARY_NAME, job_type="analysis", settings=_settings(), **common
    ) as logged:
        logged.config.update(summary_config, allow_val_change=True)
        logged.log({**plots, **tables})
        echo(f"{SUMMARY_NAME}: {logged.id if offline else logged.url}")


def _checked_config(run: PaperRun) -> dict[str, Any]:
    """paper_config, or ResultsError naming the meta.json field the run lacks."""
    try:
        return paper_config(run)
    except (KeyError, IndexError, TypeError) as exc:
        raise ResultsError(f"{run.name}: meta.json lacks {exc}") from None


def _summary_config(benchmark: PaperBenchmark) -> dict[str, Any]:
    commits = sorted({config["git_commit"] for config in benchmark.configs.values()}, key=lambda commit: commit or "")
    return {
        "git_commit": commits[0] if len(commits) == 1 else commits,
        "runs": {f"{run.tool}/{run.dataset}": run.predictions_sha256 for run in benchmark.runs},
    }


def _error_types(path: Path) -> str:
    """errors.json with each message cut to its error type (`ValueError: ...` becomes `ValueError`)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    kept = {}
    for paper, message in data["errors"].items():
        kind = message.split(":", 1)[0].strip()
        kept[paper] = kind if ERROR_TYPE.fullmatch(kind) else "error"
    return json.dumps({**data, "errors": kept})
