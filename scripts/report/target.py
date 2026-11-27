"""The W&B entity and project the report lives in: --entity and --project, else WANDB_ENTITY and WANDB_PROJECT.

There is no default entity; the project defaults to MolScout. build_report.py and the scripts that import it read
their command line with this, beside their own flags (--stage, --dry-run).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

DEFAULT_PROJECT = "MolScout"


def wandb_target(argv: Sequence[str], environ: Mapping[str, str]) -> tuple[str | None, str]:
    """(entity or None, project) from `argv` (`--entity X` or `--entity=X`), else the environment."""
    entity = _option(argv, "--entity") or environ.get("WANDB_ENTITY") or None
    project = _option(argv, "--project") or environ.get("WANDB_PROJECT") or DEFAULT_PROJECT
    return entity, project


def _option(argv: Sequence[str], name: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg == name or arg.startswith(f"{name}="):
            value = arg.partition("=")[2] if "=" in arg else (argv[index + 1] if index + 1 < len(argv) else "")
            if not value or value.startswith("--"):
                raise SystemExit(f"{name} needs a value")
            return value
    return None
