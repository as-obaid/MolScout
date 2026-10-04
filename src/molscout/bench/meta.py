"""What a benchmark run records in meta.json: commit, environment lock, hardware, SLURM job and timing."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from molscout.bench import BenchError
from molscout.bench.config import RunConfig

LOCK_TIMEOUT_SECONDS = 600
PROBE_TIMEOUT_SECONDS = 60
OUTPUT_SHOWN = 2000
PYTHON_VERSION = "import platform; print(platform.python_version())"
# Variables that change what a tool computes or where it reads from; meta.json records their values.
RECORDED_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "PYSTOW_HOME",
    "MOLSCOUT_STORE",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_PARTITION",
)
SECRET_NAME = re.compile("TOKEN|SECRET|PASSWORD|KEY", re.IGNORECASE)
SLURM_FIELDS = (
    ("job", "SLURM_JOB_ID"),
    ("array_job", "SLURM_ARRAY_JOB_ID"),
    ("array_task", "SLURM_ARRAY_TASK_ID"),
    ("partition", "SLURM_JOB_PARTITION"),
    ("nodes", "SLURM_JOB_NODELIST"),
)


def build_meta(
    config: RunConfig,
    *,
    items: int,
    tool_errors: int | None,
    inputs: Mapping[str, object],
    git: Mapping[str, object],
    environment: Mapping[str, object],
    timing: Mapping[str, object],
    command: Sequence[str],
) -> dict[str, object]:
    """The meta.json record for one run; hardware and SLURM details are read here.

    `tool_errors` counts the images whose predict call raised (crop_runner's errors file), or is None
    when run.py wrote no errors file.
    """
    checkpoints = [{"path": str(c.path), "sha256": c.sha256} for c in config.checkpoints]
    return {
        "run": config.run_name,
        "tool": {"tool": config.tool, "name": config.name, "version": config.version, "checkpoints": checkpoints},
        "dataset": config.dataset,
        "items": items,
        "tool_errors": tool_errors,
        "inputs": dict(inputs),
        "git": dict(git),
        "environment": dict(environment),
        "hardware": hardware(),
        "slurm": slurm(),
        "timing": dict(timing),
        "command": list(command),
    }


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def git_state(repo_root: Path, paths: Sequence[Path]) -> dict[str, object]:
    """HEAD and the uncommitted changes under `paths`; all None outside a git repository."""
    unknown: dict[str, object] = {"commit": None, "dirty": None, "dirty_paths": None}
    git = shutil.which("git")
    if git is None:
        return unknown
    commit = _quiet([git, "-C", str(repo_root), "rev-parse", "HEAD"])
    if commit is None:
        return unknown
    inside = [relative for relative in (_relative(path, repo_root) for path in paths) if relative is not None]
    if not inside:
        return {"commit": commit.strip(), "dirty": False, "dirty_paths": []}
    status = _quiet([git, "-C", str(repo_root), "status", "--porcelain", "-z", "--untracked-files=all", "--", *inside])
    if status is None:
        return {**unknown, "commit": commit.strip()}
    dirty = _porcelain_paths(status)
    return {"commit": commit.strip(), "dirty": bool(dirty), "dirty_paths": dirty}


def environment_lock(
    python: Path, lock_commands: Sequence[Sequence[str]], env: Mapping[str, str], cwd: Path
) -> dict[str, object]:
    """The tool environment's Python version and packages, plus any lock commands' output, with one sha256.

    Packages come from `pip freeze --all`, or from `uv pip freeze` when the environment has no pip.
    """
    version = _output([str(python), "-c", PYTHON_VERSION], env, cwd).strip()
    freeze_command, packages = _freeze(python, env, cwd)
    lock = {
        "python_version": version,
        "packages": packages,
        "conda_meta": _conda_meta(python),
        "lock_commands": [
            {"command": list(command), "output": _output(command, env, cwd, merge_stderr=True)}
            for command in lock_commands
        ],
    }
    digest = hashlib.sha256(json.dumps(lock, sort_keys=True).encode("utf-8")).hexdigest()
    return {"python": str(python), "freeze_command": freeze_command, **lock, "sha256": digest}


def environment_variables(env: Mapping[str, str], extra: Iterable[str]) -> dict[str, str | None]:
    """The RECORDED_VARIABLES and `extra` names as the tool saw them (None when unset).

    A name that looks like a credential (TOKEN, SECRET, PASSWORD, KEY) is never recorded.
    """
    names = dict.fromkeys([*RECORDED_VARIABLES, *extra])
    return {name: env.get(name) for name in names if not SECRET_NAME.search(name)}


def hardware() -> dict[str, object]:
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
        "cpu_model": _cpu_model(),
        "cpus_available": _cpus_available(),
        "slurm_mem_per_node_mb": _int_or_none(os.environ.get("SLURM_MEM_PER_NODE")),
        "slurm_mem_per_cpu_mb": _int_or_none(os.environ.get("SLURM_MEM_PER_CPU")),
        "gpus": _gpus(),
    }


def slurm() -> dict[str, str | None]:
    return {field: os.environ.get(variable) for field, variable in SLURM_FIELDS}


def _freeze(python: Path, env: Mapping[str, str], cwd: Path) -> tuple[list[str], list[str]]:
    pip = [str(python), "-m", "pip", "freeze", "--all"]
    found = _quiet(pip, env=env, cwd=cwd, timeout=LOCK_TIMEOUT_SECONDS)
    if found is not None:
        return pip, _lines(found)
    uv = shutil.which("uv", path=env.get("PATH"))
    if uv is None:
        raise BenchError(
            f"cannot read the environment lock for {python}: `pip freeze --all` failed and uv is not on PATH"
        )
    command = [uv, "pip", "freeze", "--python", str(python)]
    return command, _lines(_output(command, env, cwd))


def _output(command: Sequence[str], env: Mapping[str, str], cwd: Path, *, merge_stderr: bool = False) -> str:
    """A lock command's output; any failure is a BenchError, since the run could not be reproduced."""
    shown = shlex.join(command)
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
            text=True,
            timeout=LOCK_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BenchError(f"cannot read the environment lock: `{shown}` failed: {exc}") from None
    if result.returncode != 0:
        detail = (result.stdout or "") + (result.stderr or "")
        raise BenchError(
            f"cannot read the environment lock: `{shown}` exited with status {result.returncode}\n"
            f"{detail[-OUTPUT_SHOWN:]}".rstrip()
        )
    return result.stdout


def _quiet(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float = PROBE_TIMEOUT_SECONDS,
) -> str | None:
    """A command's stdout, or None if it cannot run or fails."""
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=None if env is None else dict(env),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _conda_meta(python: Path) -> list[str] | None:
    """The package records of a conda environment (`<env>/conda-meta/*.json`), or None for other environments."""
    folder = python.parent.parent / "conda-meta"
    if not folder.is_dir():
        return None
    return sorted(path.stem for path in folder.glob("*.json"))


def _relative(path: Path, root: Path) -> str | None:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _porcelain_paths(status: str) -> list[str]:
    """Paths from `git status --porcelain -z`; a rename's original path is the field after it."""
    fields = iter(status.split("\0"))
    paths = []
    for field in fields:
        if not field:
            continue
        paths.append(field[3:])
        if field[0] in "RC":
            next(fields, None)
    return sorted(paths)


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    if platform.system() == "Darwin":
        brand = _quiet(["sysctl", "-n", "machdep.cpu.brand_string"])
        if brand:
            return brand.strip()
    return platform.processor() or None


def _cpus_available() -> int | None:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


def _int_or_none(text: str | None) -> int | None:
    return int(text) if text and text.isdigit() else None


def _gpus() -> list[dict[str, str]]:
    """GPUs visible to this job, from nvidia-smi; empty when there is none."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return []
    found = _quiet([smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    gpus = []
    for line in (found or "").splitlines():
        parts = [part.strip() for part in line.rsplit(",", 2)]
        if len(parts) == 3:
            gpus.append({"name": parts[0], "memory": parts[1], "driver": parts[2]})
    return gpus
