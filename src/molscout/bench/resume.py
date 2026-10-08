"""Checkpoints that let an interrupted benchmark run carry on where it stopped.

An unfinished run keeps <results root>/.checkpoints/<run>/: the tool's partial predictions.csv and
predictions.errors.jsonl (crop_runner's --resume), and state.json, {"config_sha256", "code_fingerprint",
"segments"}, with one record for each time the tool ran. The code fingerprint (current.code_fingerprint)
covers only the run's own code, so a commit elsewhere in the repository keeps the checkpoint.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

CHECKPOINTS = ".checkpoints"
STATE_FILE = "state.json"


class Checkpoint:
    """One run's checkpoint folder and its state.json."""

    def __init__(self, folder: Path, state: Mapping[str, object]) -> None:
        self.folder = folder
        self._state = dict(state)

    @classmethod
    def open(
        cls,
        results_root: Path,
        run: str,
        *,
        config_text: str,
        code_fingerprint: str | None,
        sources: Sequence[Mapping[str, object]] = (),
        extra_key: Mapping[str, object] | None = None,
    ) -> Checkpoint:
        """The checkpoint an interrupted run of this config with this code left, or a new, empty one.

        `code_fingerprint` is the run's code at HEAD (current.code_fingerprint). `sources` are the
        fingerprints of the upstream clones (meta.source_fingerprint); a run whose clones changed starts
        over, since its finished rows came from other code. Without sources the key is the config and
        the code fingerprint alone.

        `extra_key` adds more entries to the key (paper runs pass the environment lock and the tracked
        source changes); without it the key is unchanged.

        Any other checkpoint of the run is deleted, and the log says so.
        """
        folder = results_root / CHECKPOINTS / run
        key: dict[str, object] = {
            "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
            "code_fingerprint": code_fingerprint,
        }
        if sources:
            key["sources"] = [dict(source) for source in sources]
        if extra_key:
            key.update(extra_key)
        state = _read_state(folder / STATE_FILE)
        if state is not None and {name: state.get(name) for name in key} == key:
            checkpoint = cls(folder, state)
            print(f"{run}: resuming with {checkpoint.rows()} rows done, from {folder}", flush=True)
            return checkpoint
        if folder.exists():
            print(f"{run}: discarding the checkpoint in {folder}, which is not from this config and code", flush=True)
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        checkpoint = cls(folder, {**key, "segments": []})
        checkpoint._save()
        return checkpoint

    @property
    def predictions(self) -> Path:
        """The partial predictions.csv that run.py's --resume reads and appends to."""
        return self.folder / "predictions.csv"

    @property
    def segments(self) -> list[dict[str, object]]:
        return list(self._state["segments"])  # type: ignore[call-overload]

    def rows(self) -> int:
        return count_rows(self.predictions)

    def add_segment(self, segment: Mapping[str, object]) -> None:
        self._state = {**self._state, "segments": [*self.segments, dict(segment)]}
        self._save()

    def remove(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)

    def _save(self) -> None:
        part = self.folder / f"{STATE_FILE}.part"
        part.write_text(json.dumps(self._state, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(part, self.folder / STATE_FILE)


def count_rows(path: Path) -> int:
    """The rows of a checkpoint predictions.csv, without its header and a last row that a kill cut short.

    crop_runner's _read_rows reads checkpoints the same way; it cannot import molscout, hence the copy.
    """
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except FileNotFoundError:
        return 0
    records: list[list[str]] = []
    try:
        records.extend(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error:  # the file ends inside a quoted field, so its last row is not in records
        pass
    else:
        if not text.endswith("\n"):  # every finished row ends with a newline
            records = records[:-1]
    return max(len(records) - 1, 0)


def _read_state(path: Path) -> dict[str, object] | None:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) and isinstance(state.get("segments"), list) else None
