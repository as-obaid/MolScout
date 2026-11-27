"""The pipeline's settings: which environment runs each worker, the joiner, the fixes, thresholds and export.

A YAML file holds them (the packaged default is `pipeline.yaml` beside this module). `${VAR}` is read from the
environment when a worker the run uses needs it, so a rules-only run does not need the 32B environment set.
A relative worker path names a script in `molscout/pipeline/workers/`; other relative paths start at the YAML
file's folder. Worker `args` (and `device_args.<device>`, added for the configured `device`, gpu or cpu) may
hold `${VAR}` and `{vllm_url}`, `{opsin_jar}`, `{java}`, `{dpi}`, `{device}` (gpu or cpu), `{torch_device}`
(cuda or cpu) and `{thresholds.<name>}`, filled in when the worker runs. `opsin_jar` names the OPSIN
command-line jar or the folder scripts/fetch_opsin.py fills; in a folder the newest
`opsin-cli-<version>-jar-with-dependencies.jar` is used. The OPTIONAL stages run after segment when
`workers` names their worker, as the packaged configs do: ocr_labels (OCR of the compound labels printed in
the figures) and cells (drawn R groups cut from table cells for the readers).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
WORKERS_DIR = PACKAGE_DIR / "workers"
DEFAULT_CONFIG = PACKAGE_DIR / "pipeline.yaml"
FIX_NAMES = (
    "labels", "relevance", "tables", "core_check", "vote", "stereo", "junk", "names", "formulas",
    "attachments", "slot_labels", "legends", "partial_stereo",
)
JOINERS = ("rules", "32b", "both")
DEVICES = ("gpu", "cpu")
OPSIN_JAR = re.compile(r"opsin-cli-(\d+(?:\.\d+)*)-jar-with-dependencies\.jar")  # as molscout.pipeline.opsin
EXPORTS = {"accepted": frozenset({"accepted"}), "accepted+review": frozenset({"accepted", "review"})}
READERS = ("molglyph", "molscribe", "molnextr")
UPSTREAM = ("render", "parse", "segment")  # plus the OPTIONAL stages configured and one read_<reader> per reader
CELLS = "cells"  # the OPTIONAL stage that cuts drawn R groups from table cells (workers/cells.py)
OPTIONAL = ("ocr_labels", CELLS)  # upstream worker stages that run when the config names their worker
DOWNSTREAM = ("vote", "link", "join_32b", "merge", "finalize", "export")
OUTPUTS = MappingProxyType(
    {
        "render": "pages.jsonl",
        "parse": "layout.jsonl",
        "segment": "regions.jsonl",
        "ocr_labels": "ocr.jsonl",
        "cells": "cells.jsonl",  # and the cell regions it adds to regions.jsonl
        "read": "reads.jsonl",  # every reader stage, unless its worker names another file
        "vote": "voted.jsonl",
        "link": "molecules_rules.jsonl",
        "join_32b": "molecules_32b.jsonl",
        "merge": "molecules_merged.jsonl",
        "finalize": "molecules.jsonl",
        "export": "molecules.csv",
    }
)
TOP_KEYS = frozenset(
    {
        "workers", "readers", "primary_reader", "joiner", "vllm_url", "opsin_jar", "java", "dpi", "fixes",
        "thresholds", "export", "allow_empty", "shared", "check_endpoint", "device",
    }
)
WORKER_KEYS = frozenset({"python", "worker", "args", "device_args", "output", "env"})
VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
PLACEHOLDER = re.compile(r"\{([a-z_]+(?:\.[A-Za-z0-9_]+)?)\}")


@dataclass(frozen=True, slots=True)
class Fixes:
    """The fixes of the design, as switches; each can be turned off for the ladder.

    Reads like an object (`fixes.vote`), a mapping (`fixes["vote"]`, `fixes.get("vote")`) and a set of the
    fixes that are on (`"vote" in fixes`).
    """

    labels: bool = True
    relevance: bool = True
    tables: bool = True
    core_check: bool = True
    vote: bool = True
    stereo: bool = True
    junk: bool = True
    names: bool = True
    formulas: bool = True
    attachments: bool = True
    slot_labels: bool = True
    legends: bool = True
    partial_stereo: bool = True

    def __getitem__(self, name: str) -> bool:
        if name not in FIX_NAMES:
            raise KeyError(name)
        return bool(getattr(self, name))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in FIX_NAMES and bool(getattr(self, name))

    def get(self, name: str, default: bool | None = None) -> bool | None:
        return self[name] if name in FIX_NAMES else default

    def enabled(self) -> tuple[str, ...]:
        return tuple(name for name in FIX_NAMES if getattr(self, name))

    def as_dict(self) -> dict[str, bool]:
        return {name: bool(getattr(self, name)) for name in FIX_NAMES}


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """One worker stage: the environment's Python, the script, extra arguments, its output file and env."""

    python: Path
    worker: Path
    args: tuple[str, ...] = ()
    output: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    device_args: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineConfig:
    """A loaded, checked pipeline configuration. Build it with load_pipeline_config; change it with with_overrides."""

    raw_workers: Mapping[str, Mapping[str, Any]]
    base: Path
    readers: tuple[str, ...] = READERS
    primary_reader: str = "molglyph"
    joiner: str = "rules"
    vllm_url: str | None = None
    opsin_jar: Path | None = None
    java: str = "java"
    dpi: int = 300
    fixes: Fixes = Fixes()
    thresholds: Mapping[str, float] = field(default_factory=dict)
    export: str = "accepted"
    allow_empty: frozenset[str] = frozenset()
    shared: Path | None = None
    check_endpoint: bool = True
    device: str = "gpu"
    workers: Mapping[str, WorkerSpec] = field(init=False)

    def __post_init__(self) -> None:
        _check(self)
        object.__setattr__(self, "workers", MappingProxyType(_worker_specs(self)))

    @property
    def export_statuses(self) -> frozenset[str]:
        return EXPORTS[self.export]

    @property
    def torch_device(self) -> str:
        return "cuda" if self.device == "gpu" else "cpu"

    def opsin_jar_file(self) -> Path | None:
        """The OPSIN jar: opsin_jar itself, or the newest OPSIN CLI jar in it when it is a folder (None if none)."""
        if self.opsin_jar is None or not self.opsin_jar.is_dir():
            return self.opsin_jar
        jars = [
            (tuple(int(part) for part in match.group(1).split(".")), path)
            for path in self.opsin_jar.iterdir()
            if (match := OPSIN_JAR.fullmatch(path.name))
        ]
        return max(jars)[1] if jars else None

    def template_args(self, stage: str) -> list[str]:
        """A worker's arguments for the configured device, placeholders not yet filled."""
        spec = self.workers[stage]
        return [*spec.args, *spec.device_args.get(self.device, ())]

    def reader_stages(self) -> list[str]:
        return [f"read_{reader}" for reader in self.readers]

    def optional_stages(self) -> list[str]:
        """The OPTIONAL stages whose worker this configuration names (ocr_labels: labels read in the figures; cells:
        drawn R groups cut from table cells)."""
        return [stage for stage in OPTIONAL if stage in self.raw_workers]

    @property
    def cells(self) -> bool:
        """Whether the cells stage runs (the config has a workers.cells entry)."""
        return CELLS in self.optional_stages()

    def stage_names(self) -> list[str]:
        """The stages this configuration runs, in order."""
        joining = {
            "rules": ["link"],
            "32b": ["join_32b"],
            "both": ["link", "join_32b", "merge"],
        }[self.joiner]
        return [*UPSTREAM, *self.optional_stages(), *self.reader_stages(), "vote", *joining, "finalize", "export"]

    def worker_stages(self) -> list[str]:
        workers = ("parse", "segment", *OPTIONAL, "join_32b")
        return [stage for stage in self.stage_names() if stage in workers or stage.startswith("read_")]

    def output(self, stage: str) -> str:
        """The file a stage writes in each paper folder."""
        spec = self.workers.get(stage)
        if spec is not None and spec.output:
            return spec.output
        return OUTPUTS["read" if stage.startswith("read_") else stage]

    def worker_args(self, stage: str) -> list[str]:
        """The stage worker's extra arguments with every placeholder filled in."""
        return [_fill(arg, self, stage) for arg in self.template_args(stage)]

    def with_overrides(
        self,
        *,
        joiner: str | None = None,
        vllm_url: str | None = None,
        fixes_off: Iterable[str] = (),
        fixes_only: Iterable[str] | None = None,
        export: str | None = None,
        shared: Path | None = None,
        device: str | None = None,
    ) -> PipelineConfig:
        """A new config with these settings changed; checked like a loaded one."""
        fixes = self.fixes
        if fixes_only is not None:
            only = set(_fix_names(fixes_only, "fixes_only"))
            fixes = Fixes(**{name: name in only for name in FIX_NAMES})
        off = _fix_names(fixes_off, "fixes_off")
        if off:
            fixes = replace(fixes, **dict.fromkeys(off, False))
        changes: dict[str, Any] = {"fixes": fixes}
        overrides = (("joiner", joiner), ("vllm_url", vllm_url), ("export", export), ("shared", shared), ("device", device))
        for name, value in overrides:
            if value is not None:
                changes[name] = value
        settings = {f.name: getattr(self, f.name) for f in fields(self) if f.init}
        return PipelineConfig(**{**settings, **changes})

    def describe(self) -> dict[str, Any]:
        """Every setting that can change a result, as plain JSON values (the vLLM URL and shared folder excluded)."""
        return {
            "readers": list(self.readers),
            "primary_reader": self.primary_reader,
            "joiner": self.joiner,
            "opsin_jar": None if self.opsin_jar is None else str(self.opsin_jar),
            "java": self.java,
            "dpi": self.dpi,
            "fixes": self.fixes.as_dict(),
            "thresholds": dict(sorted(self.thresholds.items())),
            "export": self.export,
            "allow_empty": sorted(self.allow_empty),
            "device": self.device,
            "workers": {
                stage: {
                    "python": str(spec.python),
                    "worker": str(spec.worker),
                    "args": self.template_args(stage),
                    "output": self.output(stage),
                    "env": dict(sorted(spec.env.items())),
                }
                for stage, spec in sorted(self.workers.items())
            },
        }

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.describe(), sort_keys=True).encode()).hexdigest()

    def public(self) -> dict[str, Any]:
        """describe() without paths, for results and W&B: each worker's environment name (the folder holding
        bin/python), its script name and the names of its environment variables; the OPSIN jar's file name."""
        described = self.describe()
        described["opsin_jar"] = None if self.opsin_jar is None else self.opsin_jar.name
        described["workers"] = {
            stage: {
                "env": _environment_name(spec.python),
                "worker": spec.worker.name,
                "args": [_without_paths(arg) for arg in self.template_args(stage)],
                "output": self.output(stage),
                "variables": sorted(spec.env),
            }
            for stage, spec in sorted(self.workers.items())
        }
        return described


def load_pipeline_config(
    path: str | Path = DEFAULT_CONFIG,
    *,
    joiner: str | None = None,
    vllm_url: str | None = None,
    fixes_off: Iterable[str] = (),
    fixes_only: Iterable[str] | None = None,
    export: str | None = None,
    shared: Path | None = None,
    device: str | None = None,
) -> PipelineConfig:
    """Read a pipeline YAML file and apply the overrides; ValueError naming the file and the problem."""
    path = Path(path).absolute()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: not valid YAML: {exc}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    unknown = sorted(str(key) for key in set(data) - TOP_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown key(s): {', '.join(unknown)}")
    base = path.parent
    settings: dict[str, Any] = {"raw_workers": _raw_workers(data.get("workers"), path), "base": base}
    for key in ("primary_reader", "joiner", "export", "device"):
        if key in data:
            settings[key] = _text(data[key], f"{path}: {key}")
    if "java" in data:
        settings["java"] = _expand(_text(data["java"], f"{path}: java"), f"{path}: java")
    if data.get("vllm_url") is not None:
        settings["vllm_url"] = _expand(_text(data["vllm_url"], f"{path}: vllm_url"), f"{path}: vllm_url")
    if "readers" in data:
        settings["readers"] = tuple(_text(r, f"{path}: readers") for r in _list(data["readers"], f"{path}: readers"))
    if "dpi" in data:
        settings["dpi"] = data["dpi"]
    if "check_endpoint" in data:
        settings["check_endpoint"] = data["check_endpoint"]
    for key in ("opsin_jar", "shared"):
        if data.get(key) is not None:
            settings[key] = _path(_expand(_text(data[key], f"{path}: {key}"), f"{path}: {key}"), base)
    settings["fixes"] = _fixes(data.get("fixes") or {}, path)
    settings["thresholds"] = _thresholds(data.get("thresholds") or {}, path)
    settings["allow_empty"] = frozenset(_list(data.get("allow_empty") or [], f"{path}: allow_empty"))
    try:
        config = PipelineConfig(**settings)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None
    return config.with_overrides(
        joiner=joiner, vllm_url=vllm_url, fixes_off=fixes_off, fixes_only=fixes_only, export=export, shared=shared,
        device=device,
    )


def _check(config: PipelineConfig) -> None:
    if config.joiner not in JOINERS:
        raise ValueError(f"joiner must be one of {', '.join(JOINERS)}, got {config.joiner!r}")
    if config.device not in DEVICES:
        raise ValueError(f"device must be one of {', '.join(DEVICES)}, got {config.device!r}")
    if config.export not in EXPORTS:
        raise ValueError(f"export must be one of {', '.join(EXPORTS)}, got {config.export!r}")
    if not config.readers or len(set(config.readers)) != len(config.readers):
        raise ValueError(f"readers must be distinct names, got {list(config.readers)}")
    if config.primary_reader not in config.readers:
        raise ValueError(f"primary_reader {config.primary_reader!r} is not one of the readers {list(config.readers)}")
    if isinstance(config.dpi, bool) or not isinstance(config.dpi, int) or config.dpi < 1:
        raise ValueError(f"dpi must be a positive integer, got {config.dpi!r}")
    if not isinstance(config.check_endpoint, bool):
        raise ValueError(f"check_endpoint must be true or false, got {config.check_endpoint!r}")
    if config.joiner in ("32b", "both") and not config.vllm_url:
        raise ValueError(f"joiner {config.joiner} needs vllm_url, the vLLM server's OpenAI endpoint")
    if config.joiner in ("rules", "both") and config.opsin_jar is None:
        raise ValueError(f"joiner {config.joiner} needs opsin_jar, the OPSIN command-line jar")
    stages = set(config.stage_names())
    unknown = sorted(config.allow_empty - stages - set(DOWNSTREAM) - set(UPSTREAM) - set(OPTIONAL))
    if unknown:
        raise ValueError(f"allow_empty names unknown stage(s): {', '.join(unknown)}")


def _worker_specs(config: PipelineConfig) -> dict[str, WorkerSpec]:
    specs = {}
    for stage in config.worker_stages():
        raw = config.raw_workers.get(stage)
        if raw is None:
            raise ValueError(f"workers.{stage} is missing; the {config.joiner} joiner with these readers needs it")
        where = f"workers.{stage}"
        worker = Path(_expand(raw["worker"], f"{where}.worker"))
        specs[stage] = WorkerSpec(
            python=_path(_expand(raw["python"], f"{where}.python"), config.base),
            worker=worker if worker.is_absolute() else WORKERS_DIR / worker,
            args=tuple(_expand(arg, f"{where}.args") for arg in raw.get("args", ())),
            output=raw.get("output"),
            env={name: _expand(value, f"{where}.env.{name}") for name, value in raw.get("env", {}).items()},
            device_args={
                device: tuple(_expand(arg, f"{where}.device_args.{device}") for arg in args)
                for device, args in raw.get("device_args", {}).items()
            },
        )
    return specs


def _raw_workers(workers: object, path: Path) -> dict[str, dict[str, Any]]:
    if workers is None:
        return {}
    if not isinstance(workers, dict):
        raise ValueError(f"{path}: workers must map stage names to {{python, worker}}")
    raw = {}
    for stage, spec in workers.items():
        where = f"{path}: workers.{stage}"
        if not isinstance(spec, dict):
            raise ValueError(f"{where} must be a mapping with python and worker")
        unknown = sorted(str(key) for key in set(spec) - WORKER_KEYS)
        if unknown or "python" not in spec or "worker" not in spec:
            raise ValueError(f"{where} needs python and worker, and may have args, output and env; got {sorted(spec)}")
        entry: dict[str, Any] = {"python": _text(spec["python"], f"{where}.python"), "worker": _text(spec["worker"], f"{where}.worker")}
        entry["args"] = tuple(str(arg) for arg in _list(spec.get("args") or [], f"{where}.args"))
        device_args = spec.get("device_args") or {}
        if not isinstance(device_args, dict) or set(device_args) - set(DEVICES):
            raise ValueError(f"{where}.device_args must map {' and '.join(DEVICES)} to lists of args")
        entry["device_args"] = {
            str(device): tuple(str(arg) for arg in _list(args or [], f"{where}.device_args.{device}"))
            for device, args in device_args.items()
        }
        if spec.get("output") is not None:
            entry["output"] = _text(spec["output"], f"{where}.output")
        env = spec.get("env") or {}
        if not isinstance(env, dict):
            raise ValueError(f"{where}.env must be a mapping")
        entry["env"] = {str(name): str(value) for name, value in env.items()}
        raw[str(stage)] = entry
    return raw


def _fixes(data: object, path: Path) -> Fixes:
    if not isinstance(data, dict):
        raise ValueError(f"{path}: fixes must map fix names to true or false")
    unknown = sorted(str(key) for key in set(data) - set(FIX_NAMES))
    if unknown:
        raise ValueError(f"{path}: unknown fix(es): {', '.join(unknown)}; known: {', '.join(FIX_NAMES)}")
    for name, value in data.items():
        if not isinstance(value, bool):
            raise ValueError(f"{path}: fixes.{name} must be true or false, got {value!r}")
    return Fixes(**data)


def _fix_names(names: Iterable[str], where: str) -> tuple[str, ...]:
    names = tuple(names)
    unknown = sorted(set(names) - set(FIX_NAMES))
    if unknown:
        raise ValueError(f"{where}: unknown fix(es): {', '.join(unknown)}; known: {', '.join(FIX_NAMES)}")
    return names


def _thresholds(data: object, path: Path) -> dict[str, float]:
    if not isinstance(data, dict):
        raise ValueError(f"{path}: thresholds must map names to numbers")
    out = {}
    for name, value in data.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{path}: thresholds.{name} must be a number, got {value!r}")
        out[str(name)] = float(value)
    return out


def _fill(arg: str, config: PipelineConfig, stage: str) -> str:
    def value(match: re.Match[str]) -> str:
        name = match.group(1)
        if name.startswith("thresholds."):
            key = name.split(".", 1)[1]
            if key not in config.thresholds:
                raise ValueError(f"workers.{stage}.args uses {{{name}}}, but thresholds has no {key}")
            return repr(config.thresholds[key])
        known = {
            "vllm_url": config.vllm_url, "opsin_jar": config.opsin_jar_file(), "java": config.java, "dpi": config.dpi,
            "device": config.device, "torch_device": config.torch_device,
        }
        if name not in known:
            raise ValueError(f"workers.{stage}.args uses an unknown placeholder {{{name}}}")
        if known[name] is None:
            raise ValueError(f"workers.{stage}.args uses {{{name}}}, which this config does not set")
        return str(known[name])

    return PLACEHOLDER.sub(value, arg)


def _expand(value: str, where: str) -> str:
    def replace_variable(match: re.Match[str]) -> str:
        found = os.environ.get(match.group(1))
        if not found:
            raise ValueError(f"{where} uses ${{{match.group(1)}}}, which is not set (or empty)")
        return found

    return VARIABLE.sub(replace_variable, value)


def _without_paths(arg: str) -> str:
    """An argument with any absolute path cut to its file name, for results and W&B."""
    return Path(arg).name if arg.startswith("/") else arg


def _environment_name(python: Path) -> str:
    """`.../envs/mineru/bin/python` -> `mineru`; a Python not inside a bin folder keeps only its file name."""
    return python.parent.parent.name if python.parent.name == "bin" else python.name


def _path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string, got {value!r}")
    return value


def _list(value: object, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be a list, got {value!r}")
    return value
