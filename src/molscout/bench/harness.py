"""Run one benchmark config: the tool's run.py in its own environment, then check, score and record its output."""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter

from molscout.bench import BenchError
from molscout.bench.config import RunConfig, load_config
from molscout.bench.meta import build_meta, environment_lock, git_state, utc_now
from molscout.hashing import sha256_tree
from molscout.predictions import Prediction, PredictionsFormatError, read_predictions
from molscout.runs import score_run
from molscout.scoring import check_rdkit_version, write_scores

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp"})
DROPPED_VARIABLES = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
GIT_PATHS = ("src", "pyproject.toml")
EXAMPLES_SHOWN = 10


def run_benchmark(config_path: str | Path, *, repo_root: str | Path, results_root: str | Path) -> Path:
    """Run a config's tool, check and score its predictions.csv, and return results/<tool>__<dataset>/.

    The folder holds predictions.csv, scores.json, config.yaml and meta.json. Any failure raises
    and leaves an earlier results folder for the same run untouched.
    """
    started, clock = utc_now(), perf_counter()
    repo_root = Path(repo_root).absolute()
    results_root = Path(results_root).absolute()
    try:
        check_rdkit_version()
    except RuntimeError as exc:
        raise BenchError(str(exc)) from None
    config = load_config(config_path, repo_root)
    image_ids = check_inputs(config)
    env = tool_environment(config)
    environment = environment_lock(config.python, config.lock_commands, env, config.run_dir)
    git = git_state(repo_root, [config.run_dir, config.path, *(repo_root / path for path in GIT_PATHS)])

    target = results_root / config.run_name
    results_root.mkdir(parents=True, exist_ok=True)
    staging = results_root / f".staging-{config.run_name}-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        predictions_path = staging / "predictions.csv"
        command = [
            str(config.python),
            "run.py",
            "--images",
            str(config.images),
            "--dataset",
            config.dataset,
            "--tool",
            config.tool_label,
            "--output",
            str(predictions_path),
            *config.args,
        ]
        tool_clock = perf_counter()
        run_tool(command, config, env)
        tool_seconds = perf_counter() - tool_clock
        predictions = checked_predictions(predictions_path, config, image_ids)
        scoring_clock = perf_counter()
        report = score_run(predictions_path, dataset=config.dataset, references=config.references)
        scoring_seconds = perf_counter() - scoring_clock
        write_scores(staging / "scores.json", _as_from_repo_root(report, target, config.references, repo_root))
        (staging / "config.yaml").write_bytes(config.text.encode("utf-8"))
        timing = {
            "started_utc": started,
            "finished_utc": utc_now(),
            "wall_seconds": round(perf_counter() - clock, 3),
            "tool_seconds": round(tool_seconds, 3),
            "scoring_seconds": round(scoring_seconds, 3),
            "item_seconds_total": math.fsum(p.seconds for p in predictions),
        }
        meta = build_meta(
            config, items=len(image_ids), git=git, environment=environment, timing=timing, command=command
        )
        (staging / "meta.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _move_into_place(staging, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def check_inputs(config: RunConfig) -> frozenset[str]:
    """Check what the run needs before the tool starts; return the crop IDs (image stems)."""
    run = config.run_name
    if not config.references.exists():
        raise BenchError(f"{run}: references not found: {config.references}")
    run_py = config.run_dir / "run.py"
    if not run_py.is_file():
        raise BenchError(f"{run}: run.py not found: {run_py}")
    if not config.python.is_file() or not os.access(config.python, os.X_OK):
        raise BenchError(f"{run}: tool Python not found or not executable: {config.python}; run the tool's setup.sh")
    for checkpoint in config.checkpoints:
        if not checkpoint.path.exists():
            raise BenchError(f"{run}: checkpoint not found: {checkpoint.path}")
        actual = sha256_tree(checkpoint.path)
        if actual != checkpoint.sha256:
            raise BenchError(
                f"{run}: checkpoint {checkpoint.path} has sha256 {actual}, but the config pins {checkpoint.sha256}; "
                "restore the pinned weights, or update tool.yaml and regenerate the configs"
            )
    return image_stems(config.images)


def image_stems(directory: Path) -> frozenset[str]:
    """The stems of the image files in a folder, hidden files skipped; each stem is one crop ID."""
    if not directory.is_dir():
        raise BenchError(f"images folder not found: {directory}")
    stems: dict[str, str] = {}
    for path in sorted(directory.iterdir()):
        if path.name.startswith(".") or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in stems:
            raise BenchError(f"{directory}: two images for crop {path.stem!r}: {stems[path.stem]} and {path.name}")
        stems[path.stem] = path.name
    if not stems:
        raise BenchError(f"no images in {directory}; expected files ending in {', '.join(sorted(IMAGE_SUFFIXES))}")
    return frozenset(stems)


def tool_environment(config: RunConfig) -> dict[str, str]:
    """The harness's environment without its Python settings, plus the config's `env`."""
    inherited = {name: value for name, value in os.environ.items() if name not in DROPPED_VARIABLES}
    return {**inherited, "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1", **config.env}


def run_tool(command: Sequence[str], config: RunConfig, env: Mapping[str, str]) -> None:
    """Run run.py in its folder; its output goes straight to the job log."""
    print(f"{config.run_name}: running {shlex.join(command)} in {config.run_dir}", flush=True)
    try:
        result = subprocess.run(list(command), cwd=config.run_dir, env=dict(env), check=False)
    except OSError as exc:
        raise BenchError(f"{config.run_name}: cannot start run.py: {exc}") from None
    if result.returncode < 0:
        raise BenchError(f"{config.run_name}: run.py was killed by signal {-result.returncode}")
    if result.returncode != 0:
        raise BenchError(f"{config.run_name}: run.py exited with status {result.returncode}; see its output above")


def checked_predictions(path: Path, config: RunConfig, image_ids: frozenset[str]) -> tuple[Prediction, ...]:
    """Read run.py's predictions.csv and check it covers exactly the images, for this dataset and tool."""
    run = config.run_name
    if not path.is_file():
        raise BenchError(f"{run}: run.py exited with status 0 but wrote no predictions.csv")
    try:
        predictions = read_predictions(path)
    except PredictionsFormatError as exc:
        raise BenchError(f"{run}: run.py wrote an invalid predictions.csv: {exc}") from None
    datasets = sorted({p.dataset for p in predictions} - {config.dataset})
    if datasets:
        raise BenchError(f"{run}: rows name dataset {datasets[0]!r}, but the config's dataset is {config.dataset!r}")
    tools = sorted({p.tool for p in predictions} - {config.tool_label})
    if tools:
        raise BenchError(f"{run}: rows name tool {tools[0]!r}, but the config's tool is {config.tool_label!r}")
    found = {p.item_id for p in predictions}
    missing = sorted(image_ids - found)
    if missing:
        raise BenchError(
            f"{run}: predictions.csv misses {len(missing)} of {len(image_ids)} images, e.g. {_examples(missing)}; "
            "a run that stopped early is not scored"
        )
    extra = sorted(found - image_ids)
    if extra:
        raise BenchError(
            f"{run}: predictions.csv has {len(extra)} item(s) with no image in {config.images}, e.g. {_examples(extra)}"
        )
    return predictions


def _examples(items: Sequence[str]) -> str:
    return ", ".join(items[:EXAMPLES_SHOWN])


def _as_from_repo_root(
    report: Mapping[str, object], target: Path, references: Path, repo_root: Path
) -> dict[str, object]:
    """The report with the paths `molscout score` writes when run from the repository root on the final folder."""
    predictions = {**report["predictions"], "path": _shown(target / "predictions.csv", repo_root)}  # type: ignore[dict-item]
    inputs = dict(report["inputs"])  # type: ignore[call-overload]
    if isinstance(inputs.get("references"), Mapping):
        inputs["references"] = {**inputs["references"], "directory": _shown(references, repo_root)}
    return {**report, "predictions": predictions, "inputs": inputs}


def _shown(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _move_into_place(staging: Path, target: Path) -> None:
    """Rename staging to target; an earlier target is moved aside first and deleted only after."""
    if not target.exists():
        staging.rename(target)
        return
    old = target.with_name(f".old-{target.name}-{os.getpid()}")
    shutil.rmtree(old, ignore_errors=True)
    target.rename(old)
    try:
        staging.rename(target)
    except OSError:
        old.rename(target)
        raise
    shutil.rmtree(old, ignore_errors=True)
