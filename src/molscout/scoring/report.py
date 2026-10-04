"""Assemble and write scores.json with what is needed to reproduce it."""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from rdkit import rdBase

import molscout
from molscout.hashing import sha256_file
from molscout.scoring.stats import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, Z_95


def build_report(
    scores: Mapping[str, object],
    *,
    dataset: str,
    tool: str,
    predictions_path: Path,
    inputs: Mapping[str, object],
) -> dict[str, object]:
    """Wrap scores with the dataset, tool, input checksums and scoring settings.

    `inputs` pins the reference side, such as ground-truth and split checksums.
    """
    return {
        "dataset": dataset,
        "tool": tool,
        "predictions": {"path": str(predictions_path), "sha256": sha256_file(predictions_path)},
        "inputs": dict(inputs),
        "scoring": {
            "molscout_version": molscout.__version__,
            "rdkit_version": rdBase.rdkitVersion,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
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

