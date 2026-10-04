"""Benchmark configs: one tool on one crop dataset, as benchmarks/configs/<tool>__<dataset>.yaml."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

from molscout.datasets import Kind, dataset_kind

REQUIRED_KEYS = (
    "tool",
    "name",
    "version",
    "dataset",
    "images",
    "references",
    "run_dir",
    "python",
    "args",
    "checkpoints",
)
OPTIONAL_KEYS = ("env", "lock_commands")
CHECKPOINT_KEYS = ("path", "sha256")
TEXT_KEYS = ("tool", "name", "version", "dataset", "images", "references", "run_dir", "python")
TOOL_NAME = re.compile(r"[a-z0-9_]+")
SHA256 = re.compile(r"[0-9a-f]{64}")
VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A weight file or folder, and the sha256_tree it must have."""

    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class RunConfig:
    """One loaded config: `${VAR}` expanded and relative paths resolved from the repository root."""

    path: Path
    text: str
    tool: str
    name: str
    version: str
    dataset: str
    images: Path
    references: Path
    run_dir: Path
    python: Path
    args: tuple[str, ...]
    checkpoints: tuple[Checkpoint, ...]
    env: Mapping[str, str]
    lock_commands: tuple[tuple[str, ...], ...]

    @property
    def run_name(self) -> str:
        return f"{self.tool}__{self.dataset}"

    @property
    def tool_label(self) -> str:
        """The tool column every predictions.csv row must carry."""
        return f"{self.name} {self.version}"


def load_config(path: str | Path, repo_root: str | Path) -> RunConfig:
    """Read and validate a config; raise ValueError naming the file and the problem."""
    path = Path(path).absolute()
    root = Path(repo_root).absolute()
    text = path.read_bytes().decode("utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: not valid YAML: {exc}") from None
    where = str(path)
    validate_config(data, where)
    expected = f"{data['tool']}__{data['dataset']}.yaml"
    if path.name != expected:
        raise ValueError(f"{path}: file name must be {expected}, from its tool and dataset")

    def expand(value: str, field: str) -> str:
        return _expand(value, f"{where}: {field}")

    def resolve(value: str, field: str) -> Path:
        expanded = Path(expand(value, field))
        return expanded if expanded.is_absolute() else root / expanded

    checkpoints = tuple(
        Checkpoint(resolve(entry["path"], f"checkpoints[{i}].path"), entry["sha256"])
        for i, entry in enumerate(data["checkpoints"])
    )
    env = {name: expand(str(value), f"env.{name}") for name, value in (data.get("env") or {}).items()}
    lock_commands = tuple(
        tuple(expand(str(part), f"lock_commands[{i}]") for part in command)
        for i, command in enumerate(data.get("lock_commands") or [])
    )
    return RunConfig(
        path=path,
        text=text,
        tool=data["tool"],
        name=data["name"],
        version=data["version"],
        dataset=data["dataset"],
        images=resolve(data["images"], "images"),
        references=resolve(data["references"], "references"),
        run_dir=resolve(data["run_dir"], "run_dir"),
        python=resolve(data["python"], "python"),
        args=tuple(expand(str(arg), f"args[{i}]") for i, arg in enumerate(data["args"])),
        checkpoints=checkpoints,
        env=MappingProxyType(env),
        lock_commands=lock_commands,
    )


def validate_config(data: object, where: str) -> None:
    """Check a config's keys and value types without expanding variables."""
    check_keys(data, REQUIRED_KEYS, OPTIONAL_KEYS, where)
    assert isinstance(data, dict)
    for key in TEXT_KEYS:
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"{where}: {key} must be a string; quote it if YAML reads it as something else")
    check_tool_name(data["tool"], where)
    try:
        kind = dataset_kind(data["dataset"])
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from None
    if kind is not Kind.CROP:
        raise ValueError(f"{where}: {data['dataset']} is a paper dataset; the harness runs crop datasets")
    _check_args(data["args"], where)
    _check_checkpoints(data["checkpoints"], where)
    _check_env(data.get("env"), where)
    _check_lock_commands(data.get("lock_commands"), where)


def check_keys(data: object, required: tuple[str, ...], optional: tuple[str, ...], where: str) -> None:
    """Raise ValueError unless `data` is a mapping with every required key and no unknown ones."""
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a YAML mapping")
    missing = [key for key in required if key not in data]
    unknown = sorted(str(key) for key in set(data) - set(required) - set(optional))
    if missing:
        raise ValueError(f"{where}: missing key(s): {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{where}: unknown key(s): {', '.join(unknown)}")


def check_tool_name(tool: object, where: str) -> None:
    if not isinstance(tool, str) or not TOOL_NAME.fullmatch(tool):
        raise ValueError(f"{where}: tool must match [a-z0-9_]+, got {tool!r}")


def _expand(value: str, where: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        found = os.environ.get(name)
        if not found:
            raise ValueError(f"{where} uses ${{{name}}}, which is not set (or empty)")
        return found

    return VARIABLE.sub(replace, value)


def _is_scalar(value: object) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)


def _check_args(args: object, where: str) -> None:
    if not isinstance(args, list):
        raise ValueError(f"{where}: args must be a list")
    for i, arg in enumerate(args):
        if not _is_scalar(arg):
            raise ValueError(f"{where}: args[{i}] must be a string or number, got {arg!r}")


def _check_checkpoints(checkpoints: object, where: str) -> None:
    if not isinstance(checkpoints, list):
        raise ValueError(f"{where}: checkpoints must be a list of {{path, sha256}}")
    for i, entry in enumerate(checkpoints):
        entry_where = f"{where}: checkpoints[{i}]"
        check_keys(entry, CHECKPOINT_KEYS, (), entry_where)
        assert isinstance(entry, dict)
        if not isinstance(entry["path"], str) or not entry["path"].strip():
            raise ValueError(f"{entry_where}: path must be a string")
        if not isinstance(entry["sha256"], str) or not SHA256.fullmatch(entry["sha256"]):
            raise ValueError(f"{entry_where}: sha256 must be 64 lowercase hex characters")


def _check_env(env: object, where: str) -> None:
    if env is None:
        return
    if not isinstance(env, dict):
        raise ValueError(f"{where}: env must be a mapping of variable name to value")
    for name, value in env.items():
        if not isinstance(name, str) or not VARIABLE_NAME.fullmatch(name):
            raise ValueError(f"{where}: env has a bad variable name {name!r}")
        if not _is_scalar(value):
            raise ValueError(f"{where}: env.{name} must be a string or number")


def _check_lock_commands(commands: object, where: str) -> None:
    if commands is None:
        return
    if not isinstance(commands, list):
        raise ValueError(f"{where}: lock_commands must be a list of commands, each a list of strings")
    for i, command in enumerate(commands):
        if not isinstance(command, list) or not command or not all(_is_scalar(part) for part in command):
            raise ValueError(f"{where}: lock_commands[{i}] must be a non-empty list of strings")
