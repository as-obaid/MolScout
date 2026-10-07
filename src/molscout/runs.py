"""Score one run: a predictions.csv from one tool on one dataset, into the scores.json report."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from molscout.data import biovista_truth, molfiles, molrecbench
from molscout.data.biovista_truth import BIOVISTA_PAPERS_PATH
from molscout.data.internal import (
    GROUND_TRUTH_PATH,
    SPLIT_PATH,
    check_split_counts,
    load_internal_ground_truth,
    load_internal_split,
    references_by_paper,
)
from molscout.datasets import Kind, dataset_kind
from molscout.hashing import sha256_file
from molscout.predictions import Prediction, read_predictions
from molscout.scoring import (
    build_report,
    canonical_smiles,
    check_rdkit_version,
    score_crops,
    score_papers,
    seconds_per_item,
)


def score_run(
    predictions_path: Path,
    *,
    dataset: str | None = None,
    references: Path | None = None,
    ground_truth: Path = GROUND_TRUTH_PATH,
    split: Path = SPLIT_PATH,
    papers: Path = BIOVISTA_PAPERS_PATH,
    paper_seconds: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """The scores.json report for one predictions.csv."""
    check_rdkit_version()
    predictions = read_predictions(predictions_path)
    dataset = _dataset(predictions, dataset)
    if dataset_kind(dataset) is Kind.CROP:
        scores, inputs = _score_crops(predictions, dataset, references)
    elif dataset == "internal":
        scores, inputs = _score_internal(predictions, ground_truth, split)
    elif dataset == "biovista":
        if references is None:
            raise ValueError("no ground-truth loader for biovista without --references DIR (the dataset root)")
        scores, inputs = _score_biovista(predictions, references, papers)
    else:
        raise ValueError(f"no ground-truth loader for {dataset} yet")
    if paper_seconds is not None and isinstance(scores.get("papers"), dict):
        _check_paper_seconds(paper_seconds, scores["papers"])
    scores = {**scores, "seconds_per_item": seconds_per_item(predictions, paper_seconds)}
    tool = predictions[0].tool if predictions else ""
    return build_report(scores, dataset=dataset, tool=tool, predictions_path=predictions_path, inputs=inputs)


def _score_crops(
    predictions: Sequence[Prediction], dataset: str, directory: Path | None
) -> tuple[dict[str, object], dict[str, object]]:
    if directory is None:
        raise ValueError(f"{dataset} is a crop dataset; pass --references DIR")
    labels = dataset == "molrecbench_wild"
    loader = molrecbench if labels else molfiles
    loaded = loader.load_references(directory)
    references = {item: ref.smiles for item, ref in loaded.items()}
    unreadable = {item: ref.error or "" for item, ref in loaded.items() if ref.smiles is None}
    for item, smiles in references.items():
        if smiles is not None and canonical_smiles(smiles) is None:
            unreadable[item] = "RDKit cannot re-parse the SMILES it wrote for this file"
    inputs = {
        "references": {
            "directory": str(directory),
            "labels" if labels else "files": len(loaded),
            "sha256": loader.reference_set_sha256(directory),
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


def _score_biovista(
    predictions: Sequence[Prediction], root: Path, manifest: Path
) -> tuple[dict[str, object], dict[str, object]]:
    truth = biovista_truth.load_biovista_truth(root, manifest)
    scores = score_papers(predictions, truth.references, truth.groups())
    in_drawn = [p for p in predictions if p.item_id in truth.drawn]
    drawn = score_papers(in_drawn, truth.drawn, truth.drawn_groups(), ignored=truth.enumerated)
    scores["drawn_only"] = {"papers": drawn["papers"], "groups": drawn["groups"]}
    inputs = {
        "references": {
            "directory": str(root),
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "sha256": biovista_truth.reference_set_sha256(root, manifest),
            "papers": len(truth.references),
            "papers_without_submitted": len(truth.groups()["without_submitted"]),
            "labels": truth.labels,
            "unreadable": dict(truth.unreadable),
            # Papers with an unreadable label are not run or scored; `papers` and `labels` count the rest.
            "dropped_papers": sorted(truth.dropped),
            "dropped_reason": biovista_truth.DROPPED,
            "drawn_only": {
                "papers": len(truth.drawn),
                "labels": sum(len(smiles) for smiles in truth.drawn.values()),
                "papers_without_drawn": [p for p in truth.references if p not in truth.drawn],
            },
        }
    }
    return scores, inputs


def _check_paper_seconds(paper_seconds: Mapping[str, float], scored: Mapping[str, object]) -> None:
    missing, extra = sorted(set(scored) - set(paper_seconds)), sorted(set(paper_seconds) - set(scored))
    if missing or extra:
        parts = [f"{label} {len(ids)}: {', '.join(ids[:5])}" for label, ids in (("missing", missing), ("extra", extra)) if ids]
        raise ValueError("the timing file's papers differ from the scored papers (" + "; ".join(parts) + ")")


def read_paper_seconds(path: str | Path) -> dict[str, float]:
    """A run's timing.json: `{"papers": n, "seconds": {paper: seconds}}`."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        papers, seconds = data["papers"], data["seconds"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"{path}: expected {{\"papers\": n, \"seconds\": {{...}}}}") from exc
    if not isinstance(seconds, dict) or papers != len(seconds):
        raise ValueError(f"{path}: papers is {papers!r} but seconds has {len(seconds) if isinstance(seconds, dict) else 'no'} entries")
    for paper, value in seconds.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{path}: seconds for {paper} must be a finite number of at least 0, got {value!r}")
    return {paper: float(value) for paper, value in seconds.items()}


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
