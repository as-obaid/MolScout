"""Assemble and write scores.json with what is needed to reproduce it."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from rdkit import rdBase

import molscout
from molscout.scoring.stats import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, Z_95


def build_report(
    scores: Mapping[str, object], *, dataset: str, tool: str, predictions_path: Path
) -> dict[str, object]:
    """Wrap scores with the dataset, tool, input checksum and scoring settings."""
    return {
        "dataset": dataset,
        "tool": tool,
        "predictions": {"path": str(predictions_path), "sha256": _sha256(predictions_path)},
        "scoring": {
            "molscout_version": molscout.__version__,
            "rdkit_version": rdBase.rdkitVersion,
            "match": "exact RDKit canonical SMILES, stereo-aware and stereo-stripped",
            "ci95_proportions": {"method": "wilson", "z": Z_95},
            "ci95_macro": {
                "method": "percentile bootstrap over papers",
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
            },
        },
        "scores": dict(scores),
    }


def write_scores(path: str | Path, report: Mapping[str, object]) -> None:
    """Write the report as indented JSON; NaN or infinity is an error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
