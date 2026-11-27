"""Keep the machine out of meta.json: paths written from their roots, and no host names or SLURM job IDs.

meta.json is committed with a run's results, so it must not name the account, the nodes or the jobs that made it.
A path under MOLSCOUT_STORE is written `$MOLSCOUT_STORE/...`, one under the repository relative to its root, one
under the home folder `~/...`, and the account name as the second folder of any other path (`/scratch/<user>/...`)
`$USER`. The host and the job, array and node fields are left out; the partition stays, since reports show it.

The harness scrubs every meta.json it writes (meta.build_meta); scripts/scrub_meta.py scrubs older ones in place.
The environment lock's sha256 is taken over the scrubbed lock (lock_digest), so it does not change with the store.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STORE = "$MOLSCOUT_STORE"
HOME = "~"
USER = "$USER"
LOCK_FIELDS = ("python_version", "packages", "conda_meta", "lock_commands")
JOB_FIELDS = frozenset({"job", "array_job", "array_task", "nodes"})
HOST_FIELDS = frozenset({"host"})
_BEFORE = r"(?<![\w.~$-])"  # a root starts a path: not the tail of a longer folder name
_AFTER = r"(?=/|$|[^\w.~-])"  # and ends at a folder boundary: /a/molscout is not /a/molscout-store


@dataclass(frozen=True, slots=True)
class Places:
    """The folders a run's paths are written from, and the account name; None leaves that rule out."""

    repo: str | None = None
    store: str | None = None
    home: str | None = None
    user: str | None = None

    @classmethod
    def here(cls, repo_root: Path, environ: Mapping[str, str] = os.environ) -> Places:
        """This machine's places: the repository root, MOLSCOUT_STORE, HOME and USER (or LOGNAME)."""
        return cls(
            repo=str(repo_root),
            store=environ.get("MOLSCOUT_STORE") or None,
            home=environ.get("HOME") or None,
            user=environ.get("USER") or environ.get("LOGNAME") or None,
        )


def scrub_text(text: str, places: Places) -> str:
    """`text` with every path under a root written from it, longest root first, then the account name as $USER."""
    roots = [(_root(places.store), STORE), (_root(places.repo), "."), (_root(places.home), HOME)]
    for root, placeholder in sorted(((r, p) for r, p in roots if r), key=lambda pair: -len(pair[0])):
        text = _from_root(text, root, placeholder)
    if places.user:
        user = re.compile(rf"{_BEFORE}(/[^/\s\"'=:,;]+/){re.escape(places.user)}{_AFTER}")
        text = user.sub(lambda match: f"{match.group(1)}{USER}", text)
    return text


def scrub_meta(meta: Mapping[str, Any], places: Places) -> dict[str, Any]:
    """A new meta.json record with every string scrubbed (scrub_text) and the host and job fields left out.

    The environment's sha256 is taken again when the scrub changed the lock, so it stays the digest of what is
    recorded. Everything else, the git record that `molscout is-current` reads included, is kept as it is.
    """
    scrubbed: dict[str, Any] = _map_strings(meta, lambda text: scrub_text(text, places))
    if isinstance(scrubbed.get("hardware"), Mapping):
        scrubbed["hardware"] = _without(scrubbed["hardware"], HOST_FIELDS)
    if isinstance(scrubbed.get("slurm"), Mapping):
        scrubbed["slurm"] = _without(scrubbed["slurm"], JOB_FIELDS)
    if isinstance(scrubbed.get("segments"), list):
        scrubbed["segments"] = [_segment(record) for record in scrubbed["segments"]]
    environment = scrubbed.get("environment")
    if isinstance(environment, Mapping) and "sha256" in environment:
        if _lock(environment) != _lock(meta.get("environment") or {}):
            scrubbed["environment"] = {**environment, "sha256": lock_digest(environment)}
    return scrubbed


def lock_digest(lock: Mapping[str, Any]) -> str:
    """The sha256 of an environment lock: its LOCK_FIELDS as sorted JSON."""
    return hashlib.sha256(json.dumps(_lock(lock), sort_keys=True).encode("utf-8")).hexdigest()


def _lock(environment: Mapping[str, Any]) -> dict[str, Any]:
    return {field: environment[field] for field in LOCK_FIELDS if field in environment}


def _root(path: str | None) -> str | None:
    """An absolute folder without its trailing slash; None for no folder, a relative one or `/` itself."""
    if not path or not path.startswith("/"):
        return None
    return path.rstrip("/") or None


def _from_root(text: str, root: str, placeholder: str) -> str:
    """Every `root` path written from `placeholder`; the repository's (`.`) is plain relative when it starts the text."""
    pattern = re.compile(f"{_BEFORE}{re.escape(root)}{_AFTER}")
    starts = pattern.match(text) is not None
    text = pattern.sub(lambda _: placeholder, text)
    if placeholder == "." and starts and text.startswith("./"):
        return text[2:]
    return text


def _map_strings(value: Any, change: Callable[[str], str]) -> Any:
    """A copy of a JSON value with `change` applied to every string in it; keys are left alone."""
    if isinstance(value, str):
        return change(value)
    if isinstance(value, Mapping):
        return {key: _map_strings(item, change) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_map_strings(item, change) for item in value]
    return value


def _without(record: Mapping[str, Any], fields: frozenset[str]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in fields}


def _segment(record: Any) -> Any:
    if not isinstance(record, Mapping):
        return record
    segment = _without(record, HOST_FIELDS)
    if isinstance(segment.get("slurm"), Mapping):
        segment["slurm"] = _without(segment["slurm"], JOB_FIELDS)
    return segment
