"""Publish MolScout's pipeline runs to Weights & Biases, for monitoring and for the ladder and joiner comparison.

Each `molscout*__<dataset>/` result folder the harness wrote becomes one `eval` run in group `pipeline`: its config
(variant, joiner, fixes, export, thresholds, readers, stage environments, commit, subset), the summary metrics every
complete system has (micro/, macro/, dev/, heldout/, test/, speed/, resources/, items/) plus pipe/ keys (each stage's
records in and out, seconds, peak GPU memory and errors, and the funnel summed over papers), a per-paper table
(pipe/papers) and the folder as an artifact. One `analysis` run holds the ladder and funnel figures under pipe/ and
the pipe/ladder and pipe/stages tables. Run IDs come from each predictions.csv's sha256, so publishing the same
results again updates the same runs.

Internal is private: its runs log metrics and counts only. Tables hold paper IDs, statuses, stage names and numbers,
never an error message; the artifact leaves out predictions.csv and keeps stages.json with errors cut to their
types. check_private refuses an Internal payload in which any SMILES of the run's predictions.csv appears.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import wandb

from molscout.report.analysis import ResultsError
from molscout.report.figures import DARK, LIGHT, Theme
from molscout.report.paper_analysis import (
    INTERNAL,
    PaperRun,
    check_paper_consistency,
    load_paper_run,
    paper_dataset_rank,
    paper_metrics,
    paper_result_folders,
)
from molscout.report.wandb_papers import ERROR_TYPE, _error_types, paper_config
from molscout.report.wandb_publish import ARTIFACT_TYPE, HASH_SHOWN, _settings, panel_html, run_id

GROUP = "pipeline"
PREFIX = "pipe/"  # figure, table and stage-metric keys; complete-system figures use sys/
SUMMARY_NAME = "summary-pipeline"
TOOL = "molscout"
VARIANT_ORDER = (
    "l0_drawings", "l1_labels", "l2_tables", "l3_core", "l4_vote", "l5_stereo", "l6_junk",
    "rules", "rules_review", "32b", "both", "default",
)
FUNNEL_STEPS = ("pages", "regions", "structures", "voted", "molecules", "exported")
PAPER_COLUMNS = (
    "variant", "dataset", "paper", "status", "stopped_at", "errors", "seconds", "truth", "tp", "fp", "fn",
    "precision", "recall", "pages", "tables", "regions", "structures", "voted", "molecules", "accepted", "review",
    "abstained", "exported",
)
FUNNEL_COLUMNS = {
    "pages": "pages", "tables": "tables", "regions": "regions", "structures": "structures", "voted": "voted",
    "molecules": "molecules", "accepted": "status_accepted", "review": "status_review",
    "abstained": "status_abstained", "exported": "exported",
}
LADDER_COLUMNS = ("variant", "dataset", "fixes", "joiner", "export", "papers", "precision", "recall", "f1", "dev_f1",
                  "seconds_per_paper", "failed_papers")
STAGE_COLUMNS = ("variant", "dataset", "stage", "records_in", "records_out", "seconds", "peak_gpu_mb", "errors")
MIN_SMILES_CHECKED = 3  # shorter outputs (C, CO) could match ordinary text by chance

Rows = tuple[list[str], list[list[Any]]]


@dataclass(frozen=True)
class PipelineRun:
    """A MolScout result folder: the harness's files read as a PaperRun, and the pipeline's stages.json."""

    run: PaperRun
    stages: Mapping[str, Any]

    @property
    def name(self) -> str:
        return self.run.name

    @property
    def dataset(self) -> str:
        return self.run.dataset

    @property
    def variant(self) -> str:
        return variant_of(self.run.tool)


@dataclass(frozen=True)
class PipelineBenchmark:
    runs: tuple[PipelineRun, ...]
    metrics: Mapping[str, Mapping[str, float | int]]  # run name -> pipeline_metrics
    configs: Mapping[str, Mapping[str, Any]]  # run name -> pipeline_config
    papers: Mapping[str, Rows]  # run name -> paper_table


def variant_of(tool: str) -> str:
    return "default" if tool == TOOL else tool.removeprefix(f"{TOOL}_")


def load_pipeline_runs(results: str | Path) -> tuple[PipelineRun, ...]:
    """Every MolScout result folder under `results`, dev subsets included, in variant then dataset order."""
    runs = []
    for folder in paper_result_folders(results):
        tool = folder.name.split("__", 1)[0]
        if tool != TOOL and not tool.startswith(f"{TOOL}_"):
            continue
        path = folder / "stages.json"
        if not path.is_file():
            raise ResultsError(f"{folder}: missing stages.json; MolScout's run.py writes it")
        runs.append(PipelineRun(load_paper_run(folder), json.loads(path.read_text(encoding="utf-8"))))
    return tuple(sorted(runs, key=lambda r: (_variant_rank(r.variant), paper_dataset_rank(r.dataset))))


def pipeline_metrics(run: PipelineRun) -> dict[str, float | int]:
    """paper_metrics (scores, speed, resources, items) plus pipe/stage/<stage>/... and pipe/funnel/... counts."""
    metrics: dict[str, float | int] = dict(paper_metrics(run.run))
    for stage, record in run.stages["stages"].items():
        for name in ("records_in", "records_out", "seconds", "errors", "peak_gpu_mb"):
            if record.get(name) is not None:
                metrics[f"{PREFIX}stage/{stage}/{name}"] = record[name]
        for kind, count in (record.get("diagnostics") or {}).items():
            metrics[f"{PREFIX}stage/{stage}/diagnostics/{kind}"] = count
    for name, count in run.stages["funnel"].items():
        metrics[f"{PREFIX}funnel/{name}"] = count
    statuses = [record["status"] for record in run.stages["per_paper"].values()]
    metrics[f"{PREFIX}papers_failed"] = sum(status != "ok" for status in statuses)
    metrics[f"{PREFIX}papers_stopped"] = sum(status == "stopped" for status in statuses)
    return metrics


def pipeline_config(run: PipelineRun) -> dict[str, Any]:
    """paper_config (tool, version, commit, checkpoints, device, GPUs) plus the variant and the pipeline settings."""
    return {
        **paper_config(run.run),
        "variant": run.variant,
        "pipeline": run.stages["config"],
        "config_sha256": run.stages["config_sha256"],
        "subset": run.run.report["inputs"].get("subset"),
    }


def paper_table(run: PipelineRun) -> Rows:
    """One row per paper: status, error count, seconds, scores and funnel counts. No message and no SMILES."""
    scores = run.run.report["scores"]["papers"]
    rows = []
    for paper, record in run.stages["per_paper"].items():
        score = scores.get(paper, {})
        rows.append([
            run.variant, run.dataset, paper, record["status"], record["stopped_at"], len(record["errors"]),
            record["seconds"], score.get("molecules"), score.get("tp"), score.get("fp"), score.get("fn"),
            score.get("precision"), score.get("recall"),
            *(record["funnel"].get(key) for key in FUNNEL_COLUMNS.values()),
        ])
    return list(PAPER_COLUMNS), rows


def ladder_rows(benchmark: PipelineBenchmark) -> Rows:
    """One row per run: the variant's settings and its scores, for the ladder and joiner comparison."""
    rows = []
    for run in benchmark.runs:
        metrics, config = benchmark.metrics[run.name], run.stages["config"]
        fixes = ",".join(name for name, on in config["fixes"].items() if on) or "none"
        rows.append([
            run.variant, run.dataset, fixes, config["joiner"], config["export"], metrics.get("items/papers"),
            metrics.get("micro/precision"), metrics.get("micro/recall"), metrics.get("micro/f1"),
            metrics.get("dev/f1"), metrics.get("speed/s_per_paper_mean"), metrics[f"{PREFIX}papers_failed"],
        ])
    return list(LADDER_COLUMNS), rows


def stage_rows(benchmark: PipelineBenchmark) -> Rows:
    rows = [
        [run.variant, run.dataset, stage, *(record.get(name) for name in STAGE_COLUMNS[3:])]
        for run in benchmark.runs
        for stage, record in run.stages["stages"].items()
    ]
    return list(STAGE_COLUMNS), rows


def figure_panels(benchmark: PipelineBenchmark, theme: Theme = LIGHT) -> dict[str, go.Figure]:
    """pipe/ladder_<dataset> (F1, precision and recall per variant) and pipe/funnel_<dataset> (the funnel of the
    first run in variant order with exported molecules)."""
    panels = {}
    for dataset in sorted({run.dataset for run in benchmark.runs}, key=paper_dataset_rank):
        runs = [run for run in benchmark.runs if run.dataset == dataset]
        panels[f"{PREFIX}ladder_{dataset}"] = _ladder_figure(runs, benchmark.metrics, dataset, theme)
        panels[f"{PREFIX}funnel_{dataset}"] = _funnel_figure(runs, dataset, theme)
    return panels


def check_private(run: PipelineRun, payload: object) -> None:
    """ResultsError when an Internal run's payload holds any SMILES its predictions.csv holds."""
    if run.dataset != INTERNAL:
        return
    text = json.dumps(payload, default=str)
    leaked = sorted({p.smiles for p in run.run.predictions if len(p.smiles) >= MIN_SMILES_CHECKED and p.smiles in text})
    if leaked:
        raise ResultsError(f"{run.name}: {len(leaked)} SMILES from the private predictions would be logged; refusing")


def prepare_pipeline(
    results: str | Path,
    repo_root: str | Path,
    *,
    allow_inconsistent: bool = False,
    warn: Callable[[str], None] = print,
) -> PipelineBenchmark:
    """Load every MolScout run and build what each logs; ResultsError on problems check_paper_consistency finds
    (uncommitted code, different scoring, PDF or reference sets; runs from several commits pass when none of those
    differ), which `allow_inconsistent` turns into warnings for development."""
    runs = load_pipeline_runs(results)
    if not runs:
        raise ResultsError(f"no MolScout result folders with a scores.json under {results}")
    problems = check_paper_consistency([run.run for run in runs])
    if problems and not allow_inconsistent:
        raise ResultsError("; ".join(problems))
    for problem in problems:
        warn(problem)
    try:
        configs = {run.name: pipeline_config(run) for run in runs}
    except (KeyError, IndexError, TypeError) as exc:
        raise ResultsError(f"a MolScout run's meta.json or stages.json lacks {exc}") from None
    benchmark = PipelineBenchmark(
        runs=runs,
        metrics={run.name: pipeline_metrics(run) for run in runs},
        configs=configs,
        papers={run.name: paper_table(run) for run in runs},
    )
    for run in runs:
        check_private(run, [benchmark.configs[run.name], benchmark.metrics[run.name], benchmark.papers[run.name]])
    return benchmark


def artifact_for(run: PipelineRun) -> wandb.Artifact:
    """The result folder; for Internal without predictions.csv, and errors.json and stages.json cut to error types."""
    private = run.dataset == INTERNAL
    artifact = wandb.Artifact(run.name, type=ARTIFACT_TYPE, description=f"MolScout {run.variant} on {run.dataset}")
    for name in ("predictions.csv", "scores.json", "config.yaml", "meta.json", "errors.json", "timing.json", "stages.json"):
        path = run.run.folder / name
        if not path.is_file() or (private and name == "predictions.csv"):
            continue
        if private and name in ("errors.json", "stages.json"):
            with artifact.new_file(name) as handle:
                handle.write(_error_types(path) if name == "errors.json" else json.dumps(_typed_stages(run.stages)))
        else:
            artifact.add_file(str(path), name=name)
    return artifact


def publish_pipeline(
    benchmark: PipelineBenchmark,
    *,
    entity: str,
    project: str,
    repo_root: str | Path,
    offline: bool = False,
    echo: Callable[[str], None] = print,
) -> None:
    """Log one eval run per MolScout folder, then the pipeline summary run; everything is built before the first run."""
    light, dark = figure_panels(benchmark), figure_panels(benchmark, theme=DARK)
    plots = {key: wandb.Html(panel_html(light[key], dark[key]), inject=False) for key in light}
    tables = {f"{PREFIX}ladder": wandb.Table(*ladder_rows(benchmark)), f"{PREFIX}stages": wandb.Table(*stage_rows(benchmark))}
    papers = {name: wandb.Table(*rows) for name, rows in benchmark.papers.items()}
    artifacts = {run.name: artifact_for(run) for run in benchmark.runs}
    common: dict[str, Any] = {"entity": entity, "project": project, "save_code": False}
    common.update({"mode": "offline"} if offline else {"resume": "allow"})
    for run in benchmark.runs:
        name = f"{run.run.tool}/{run.dataset}"
        with wandb.init(
            id=run_id(run.run), name=name, group=GROUP, job_type="eval",
            tags=[TOOL, run.variant, run.dataset, GROUP], settings=_settings(), **common,
        ) as logged:
            logged.config.update(dict(benchmark.configs[run.name]), allow_val_change=True)
            logged.summary.update(dict(benchmark.metrics[run.name]))
            logged.log({f"{PREFIX}papers": papers[run.name]})
            logged.log_artifact(artifacts[run.name])
            echo(f"{name}: {logged.id if offline else logged.url}")
    with wandb.init(
        id=summary_run_id(benchmark), name=SUMMARY_NAME, group=GROUP, job_type="analysis", settings=_settings(),
        **common,
    ) as logged:
        logged.config.update({"runs": {run.name: run.run.predictions_sha256 for run in benchmark.runs}}, allow_val_change=True)
        logged.log({**plots, **tables})
        echo(f"{SUMMARY_NAME}: {logged.id if offline else logged.url}")


def summary_run_id(benchmark: PipelineBenchmark) -> str:
    digest = hashlib.sha256("\n".join(sorted(r.run.predictions_sha256 for r in benchmark.runs)).encode()).hexdigest()
    return f"{SUMMARY_NAME}-{digest[:HASH_SHOWN]}"


def _typed_stages(stages: Mapping[str, Any]) -> dict[str, Any]:
    """stages.json with every error message cut to its type, as run.py writes it for Internal (again, to be sure)."""
    per_paper = {
        paper: {
            **record,
            "errors": [{**error, "error": _error_type(error["error"])} for error in record["errors"]],
            "stages": {s: {**c, "errors": [_error_type(m) for m in c["errors"]]} for s, c in record["stages"].items()},
        }
        for paper, record in stages["per_paper"].items()
    }
    return {**stages, "per_paper": per_paper}


def _error_type(message: str) -> str:
    """`ValueError: ...` -> `ValueError`; a message that does not start with an exception name -> `error`."""
    kind = str(message).split(":", 1)[0].strip()
    return kind if ERROR_TYPE.fullmatch(kind) else "error"


def _variant_rank(variant: str) -> tuple[int, str]:
    return (VARIANT_ORDER.index(variant), "") if variant in VARIANT_ORDER else (len(VARIANT_ORDER), variant)


def _layout(figure: go.Figure, title: str, theme: Theme, *, y_title: str) -> go.Figure:
    figure.update_layout(
        title={"text": title, "x": 0, "xanchor": "left"}, height=360, margin={"l": 60, "r": 20, "t": 60, "b": 80},
        paper_bgcolor=theme.surface, plot_bgcolor=theme.surface, font={"color": theme.ink, "size": 13},
        legend={"orientation": "h", "y": -0.25}, barmode="group",
    )
    figure.update_xaxes(color=theme.ink_secondary, linecolor=theme.axis)
    figure.update_yaxes(title=y_title, color=theme.ink_secondary, gridcolor=theme.grid, zeroline=False)
    return figure


def _ladder_figure(runs: Sequence[PipelineRun], metrics: Mapping[str, Mapping[str, Any]], dataset: str, theme: Theme) -> go.Figure:
    variants = [run.variant for run in runs]
    colors = list(theme.tool_colors.values()) or [theme.ink]
    figure = go.Figure()
    for i, (label, key) in enumerate((("F1", "micro/f1"), ("Precision", "micro/precision"), ("Recall", "micro/recall"))):
        values = [metrics[run.name].get(key) for run in runs]
        figure.add_bar(x=variants, y=values, name=label, marker_color=colors[i % len(colors)],
                       hovertemplate="%{x}: %{y:.3f}<extra>" + label + "</extra>")
    figure.update_yaxes(range=[0, 1])
    return _layout(figure, f"MolScout variants on {dataset}", theme, y_title="micro score")


def _funnel_figure(runs: Sequence[PipelineRun], dataset: str, theme: Theme) -> go.Figure:
    shown = next((run for run in runs if run.stages["funnel"].get("exported")), runs[0])
    counts = [shown.stages["funnel"].get(step, 0) for step in FUNNEL_STEPS]
    figure = go.Figure(go.Bar(x=list(FUNNEL_STEPS), y=counts, marker_color=theme.other, text=counts,
                              textposition="outside", hovertemplate="%{x}: %{y}<extra></extra>"))
    return _layout(figure, f"Funnel of MolScout {shown.variant} on {dataset}, summed over papers", theme, y_title="count")
