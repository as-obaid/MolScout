"""Run one benchmark config: the tool's run.py in its own environment, then check, score and record its output."""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

from molscout.bench import BenchError, Terminated
from molscout.bench.config import RunConfig, load_config
from molscout.bench.meta import build_meta, environment_lock, environment_variables, git_state, utc_now
from molscout.data import molfiles, molrecbench
from molscout.hashing import sha256_tree
from molscout.predictions import Prediction, PredictionsFormatError, read_predictions
from molscout.runs import score_run
from molscout.scoring import check_rdkit_version, write_scores

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp"})
DROPPED_VARIABLES = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
GIT_PATHS = ("src", "pyproject.toml", "benchmarks/tools/crop_runner.py", "benchmarks/slurm/run.sbatch")
EXAMPLES_SHOWN = 10
ERRORS_FILE = "predictions.errors.json"  # what benchmarks/tools/crop_runner.py writes beside predictions.csv
ERRORS_KEYS = frozenset({"images", "failed", "errors"})
STOP_GRACE_SECONDS = 10


def run_benchmark(config_path: str | Path, *, repo_root: str | Path, results_root: str | Path) -> Path:
    """Run a config's tool, check and score its predictions.csv, and return results/<tool>__<dataset>/.

    The folder holds predictions.csv, scores.json, config.yaml, meta.json and, when run.py writes
    one, errors.json. Any failure raises and leaves an earlier results folder for the same run
    untouched. So does SIGTERM (a SLURM time limit or scancel): while the run is in progress it
    raises Terminated, which stops the tool and removes the staging folder.
    """
    with _sigterm_raises():
        return _run(config_path, Path(repo_root).absolute(), Path(results_root).absolute())


def _run(config_path: str | Path, repo_root: Path, results_root: Path) -> Path:
    started, clock = utc_now(), perf_counter()
    check_repo_root(repo_root)
    try:
        check_rdkit_version()
    except RuntimeError as exc:
        raise BenchError(str(exc)) from None
    config = load_config(config_path, repo_root)
    image_ids = check_inputs(config)
    inputs = {"images": str(config.images), "images_sha256": sha256_tree(config.images)}
    env = tool_environment(config)
    environment = {
        **environment_lock(config.python, config.lock_commands, env, config.run_dir),
        "variables": environment_variables(env, config.env),
    }
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
        tool_errors = read_tool_errors(staging / ERRORS_FILE, config, image_ids)
        scoring_clock = perf_counter()
        report = score_run(predictions_path, dataset=config.dataset, references=config.references)
        scoring_seconds = perf_counter() - scoring_clock
        check_references_have_images(report, config, image_ids)
        write_scores(staging / "scores.json", _as_from_repo_root(report, target, config.references, repo_root))
        (staging / "config.yaml").write_bytes(config.text.encode("utf-8"))
        if tool_errors is not None:
            (staging / ERRORS_FILE).rename(staging / "errors.json")
        timing = {
            "started_utc": started,
            "finished_utc": utc_now(),
            "wall_seconds": round(perf_counter() - clock, 3),
            "tool_seconds": round(tool_seconds, 3),
            "scoring_seconds": round(scoring_seconds, 3),
            "item_seconds_total": math.fsum(p.seconds for p in predictions),
        }
        meta = build_meta(
            config,
            items=len(image_ids),
            tool_errors=tool_errors,
            inputs=inputs,
            git=git,
            environment=environment,
            timing=timing,
            command=command,
        )
        (staging / "meta.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _move_into_place(staging, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def check_repo_root(repo_root: Path) -> None:
    """Relative config paths and the git state start at the repository root, so it must be the right folder."""
    if not (repo_root / "pyproject.toml").is_file():
        raise BenchError(
            f"{repo_root} is not the MolScout repository root (it has no pyproject.toml); "
            "run from the repository root or pass --repo-root"
        )


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
    """Run run.py in its folder; its output goes straight to the job log. An interrupted wait stops run.py first."""
    print(f"{config.run_name}: running {shlex.join(command)} in {config.run_dir}", flush=True)
    try:
        process = subprocess.Popen(list(command), cwd=config.run_dir, env=dict(env))
    except OSError as exc:
        raise BenchError(f"{config.run_name}: cannot start run.py: {exc}") from None
    try:
        returncode = process.wait()
    except BaseException:
        _stop(process)
        raise
    if returncode < 0:
        raise BenchError(f"{config.run_name}: run.py was killed by signal {-returncode}")
    if returncode != 0:
        raise BenchError(f"{config.run_name}: run.py exited with status {returncode}; see its output above")


def _stop(process: subprocess.Popen[bytes]) -> None:
    """Terminate run.py, and kill it if it is still running after STOP_GRACE_SECONDS."""
    process.terminate()
    try:
        process.wait(timeout=STOP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


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


def read_tool_errors(path: Path, config: RunConfig, image_ids: frozenset[str]) -> int | None:
    """The number of images whose predict call raised, from crop_runner's errors file; None without one.

    The file is `{"images": n, "failed": m, "errors": {crop ID: "<Error>: <message>"}}`; its crop
    IDs must be images of this run.
    """
    run = config.run_name
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchError(f"{run}: run.py's {path.name} is not valid JSON: {exc}") from None
    if not (
        isinstance(data, dict)
        and set(data) == ERRORS_KEYS
        and all(isinstance(data[key], int) and not isinstance(data[key], bool) for key in ("images", "failed"))
        and isinstance(data["errors"], dict)
        and all(isinstance(message, str) for message in data["errors"].values())
    ):
        raise BenchError(
            f"{run}: {path.name} must be a JSON object with images, failed and errors "
            '({"images": n, "failed": m, "errors": {crop ID: message}})'
        )
    errors = data["errors"]
    if data["failed"] != len(errors):
        raise BenchError(f"{run}: in {path.name}, failed is {data['failed']}, but errors lists {len(errors)} crop(s)")
    if data["images"] != len(image_ids):
        raise BenchError(
            f"{run}: in {path.name}, images is {data['images']}, but the images folder holds {len(image_ids)}"
        )
    unknown = sorted(set(errors) - image_ids)
    if unknown:
        raise BenchError(
            f"{run}: {path.name} lists {len(unknown)} crop(s) with no image in {config.images}, "
            f"e.g. {_examples(unknown)}"
        )
    return data["failed"]


def check_references_have_images(report: Mapping[str, object], config: RunConfig, image_ids: frozenset[str]) -> None:
    """Every scored reference must have an image: a truncated images folder fails instead of scoring as wrong."""
    missing = report["scores"]["items_without_prediction"]  # type: ignore[index]
    if not missing:
        return
    unreadable = report["inputs"]["references"]["unreadable"]  # type: ignore[index]
    if config.dataset == "molrecbench_wild":
        references = set(molrecbench.read_labels(config.references))
    else:
        references = {path.stem for path in molfiles.reference_files(config.references)}
    without_image = sorted(references - image_ids - set(unreadable))
    raise BenchError(
        f"{config.run_name}: {missing} scored reference(s) have no image in {config.images}, "
        f"e.g. {_examples(without_image)}; restore the images (or export them again) and rerun"
    )


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
    """Rename staging to target; an earlier target is moved aside first and deleted only after.

    SIGINT and SIGTERM wait until this is done, so a stop never leaves the run with neither folder.
    """
    with _signals_deferred():
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


@contextmanager
def _sigterm_raises() -> Iterator[None]:
    """While the block runs, SIGTERM raises Terminated, so `finally` clauses stop the tool and remove staging."""
    if threading.current_thread() is not threading.main_thread():
        yield  # only the main thread can set a signal handler
        return

    def handler(signum: int, frame: object) -> None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # a second SIGTERM must not cut the cleanup short
        raise Terminated("stopped by SIGTERM (a SLURM time limit or scancel)")

    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)


@contextmanager
def _signals_deferred() -> Iterator[None]:
    """Hold SIGINT and SIGTERM until the block ends; a held signal is handled (and raises) right after.

    A signal mask would not do: it covers one thread, and Linux may deliver the signal to another,
    after which Python still runs the handler in the main thread inside the block.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    pending: list[int] = []
    names = (signal.SIGINT, signal.SIGTERM)
    previous = {signum: signal.signal(signum, lambda signum, frame: pending.append(signum)) for signum in names}
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, signal.SIG_DFL if handler is None else handler)
        if pending:
            signal.raise_signal(pending[0])
