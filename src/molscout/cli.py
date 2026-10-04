"""Command line: `molscout score` turns a predictions.csv into scores.json."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from molscout.data.internal import (
    GROUND_TRUTH_PATH,
    SPLIT_PATH,
    check_split_counts,
    load_internal_ground_truth,
    load_internal_split,
    references_by_paper,
)
from molscout.data.molfiles import load_references, reference_set_sha256
from molscout.datasets import Kind, dataset_kind
from molscout.hashing import sha256_file
from molscout.predictions import Prediction, read_predictions
from molscout.scoring import (
    build_report,
    canonical_smiles,
    check_rdkit_version,
    score_crops,
    score_papers,
    write_scores,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="molscout")
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="score a predictions.csv and write scores.json")
    score.add_argument("predictions", type=Path, help="predictions.csv from one tool on one dataset")
    score.add_argument("-o", "--output", type=Path, required=True, help="scores.json to write")
    score.add_argument("--dataset", help="dataset name; needed only when predictions.csv has no rows")
    score.add_argument("--references", type=Path, help="crop datasets: directory of reference .mol/.sdf files")
    score.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_PATH, help="internal: ground-truth CSV")
    score.add_argument("--split", type=Path, default=SPLIT_PATH, help="internal: dev/test split manifest")
    args = parser.parse_args(argv)
    try:
        write_scores(args.output, _score(args))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"molscout score: error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}")
    return 0


def _score(args: argparse.Namespace) -> dict[str, object]:
    check_rdkit_version()
    predictions = read_predictions(args.predictions)
    dataset = _dataset(predictions, args.dataset)
    if dataset_kind(dataset) is Kind.CROP:
        scores, inputs = _score_crops(predictions, dataset, args.references)
    elif dataset == "internal":
        scores, inputs = _score_internal(predictions, args.ground_truth, args.split)
    else:
        raise ValueError(f"no ground-truth loader for {dataset} yet")
    tool = predictions[0].tool if predictions else ""
    return build_report(scores, dataset=dataset, tool=tool, predictions_path=args.predictions, inputs=inputs)


def _score_crops(
    predictions: Sequence[Prediction], dataset: str, directory: Path | None
) -> tuple[dict[str, object], dict[str, object]]:
    if directory is None:
        raise ValueError(f"{dataset} is a crop dataset; pass --references DIR")
    loaded = load_references(directory)
    references = {item: ref.smiles for item, ref in loaded.items()}
    unreadable = {item: ref.error or "" for item, ref in loaded.items() if ref.smiles is None}
    for item, smiles in references.items():
        if smiles is not None and canonical_smiles(smiles) is None:
            unreadable[item] = "RDKit cannot re-parse the SMILES it wrote for this file"
    inputs = {
        "references": {
            "directory": str(directory),
            "files": len(loaded),
            "sha256": reference_set_sha256(directory),
            "unreadable": dict(sorted(unreadable.items())),
        }
    }
    return score_crops(predictions, references), inputs


def _score_internal(
    predictions: Sequence[Prediction], ground_truth_path: Path, split_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    ground_truth = load_internal_ground_truth(ground_truth_path)
    split = load_internal_split(split_path)
    check_split_counts(ground_truth, split)
    inputs = {"ground_truth_sha256": sha256_file(ground_truth_path), "split_sha256": sha256_file(split_path)}
    return score_papers(predictions, references_by_paper(ground_truth), split.groups()), inputs


def _dataset(predictions: Sequence[Prediction], requested: str | None) -> str:
    """The run's dataset: from the rows, or from --dataset when there are none."""
    found = {p.dataset for p in predictions}
    if requested is None:
        if not found:
            raise ValueError("predictions.csv has no rows; pass --dataset")
        return found.pop()
    dataset_kind(requested)
    if found and found != {requested}:
        raise ValueError(f"--dataset {requested} but the predictions are for {found.pop()}")
    return requested
