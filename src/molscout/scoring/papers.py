"""Whole-PDF scoring: the set of molecules a tool reports for each paper."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass

from molscout.predictions import Prediction
from molscout.scoring.smiles import canonical_smiles, stereo_stripped_smiles
from molscout.scoring.stats import macro, proportion, ratio


@dataclass(frozen=True, slots=True)
class PaperCounts:
    """True positives, false positives and false negatives over unique structures."""

    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return ratio(self.tp, self.tp + self.fn)


@dataclass(frozen=True, slots=True)
class PaperResult:
    stereo_aware: PaperCounts
    stereo_stripped: PaperCounts
    valid_outputs: int
    invalid_outputs: int
    rows: int
    invalid_rows: int


def count_paper(predicted: Sequence[str], reference: Sequence[str]) -> PaperResult:
    """Match one paper's predicted SMILES against its reference SMILES.

    Both sides collapse to unique canonical SMILES. Each distinct unparsable
    output, compared after trimming whitespace, is one false positive.
    """
    truth = set()
    for smiles in reference:
        canonical = canonical_smiles(smiles)
        if canonical is None:
            raise ValueError(f"reference SMILES {smiles!r} cannot be parsed by RDKit")
        truth.add(canonical)
    parsed = [(smiles, canonical_smiles(smiles)) for smiles in predicted]
    found = {canonical for _, canonical in parsed if canonical is not None}
    invalid = {smiles.strip() for smiles, canonical in parsed if canonical is None}
    return PaperResult(
        stereo_aware=_counts(found, truth, len(invalid)),
        stereo_stripped=_counts(_stripped(found), _stripped(truth), len(invalid)),
        valid_outputs=len(found),
        invalid_outputs=len(invalid),
        rows=len(parsed),
        invalid_rows=sum(canonical is None for _, canonical in parsed),
    )


def score_papers(
    predictions: Sequence[Prediction],
    references: Mapping[str, Sequence[str]],
    groups: Mapping[str, Collection[str]] | None = None,
) -> dict[str, object]:
    """Score whole-PDF predictions per paper, then pool them for each group of papers.

    `references` maps paper ID to its reference SMILES. `groups` names the sets of
    papers to report, such as dev, test and all; the default is one group, "all".
    """
    predicted: dict[str, list[str]] = defaultdict(list)
    for prediction in predictions:
        predicted[prediction.item_id].append(prediction.smiles)
    unknown = sorted(set(predicted) - set(references))
    if unknown:
        raise ValueError(
            f"{len(unknown)} prediction item(s) are not papers in the references, e.g. {', '.join(unknown[:5])}"
        )
    named = {"all": tuple(references)} if groups is None else groups
    for name, papers in named.items():
        if not papers:
            raise ValueError(f"group {name!r} has no papers")
        missing = sorted(set(papers) - set(references))
        if missing:
            raise ValueError(f"group {name!r} names papers not in the references: {', '.join(missing)}")
    results: dict[str, PaperResult] = {}
    for paper in _ordered(references):
        try:
            results[paper] = count_paper(predicted.get(paper, []), references[paper])
        except ValueError as exc:
            raise ValueError(f"paper {paper!r}: {exc}") from exc
    return {
        "kind": "paper",
        "papers": {paper: _paper_record(result) for paper, result in results.items()},
        "groups": {name: _group_record(_ordered(papers), results) for name, papers in named.items()},
    }


def _counts(found: set[str], truth: set[str], invalid: int) -> PaperCounts:
    return PaperCounts(tp=len(found & truth), fp=len(found - truth) + invalid, fn=len(truth - found))


def _stripped(smiles: Iterable[str]) -> set[str]:
    # Inputs are canonical SMILES, so stripping never fails.
    return {stereo_stripped_smiles(s) for s in smiles}  # type: ignore[misc]


def _ordered(papers: Iterable[str]) -> list[str]:
    return sorted(papers, key=lambda p: (0, int(p), p) if re.fullmatch(r"[0-9]+", p) else (1, 0, p))


def _count_record(counts: PaperCounts) -> dict[str, object]:
    return {"tp": counts.tp, "fp": counts.fp, "fn": counts.fn, "precision": counts.precision, "recall": counts.recall}


def _paper_record(result: PaperResult) -> dict[str, object]:
    aware = result.stereo_aware
    return {
        "molecules": aware.tp + aware.fn,
        **_count_record(aware),
        "stereo_stripped": _count_record(result.stereo_stripped),
        "valid_outputs": result.valid_outputs,
        "invalid_outputs": result.invalid_outputs,
        "rows": result.rows,
        "invalid_rows": result.invalid_rows,
    }


def _micro(counts: PaperCounts) -> dict[str, object]:
    tp, fp, fn = counts.tp, counts.fp, counts.fn
    return {
        "precision": proportion(tp, tp + fp),
        "recall": proportion(tp, tp + fn),
        # F1 = TP / (TP + (FP + FN) / 2), so its Wilson interval uses that denominator.
        "f1": proportion(tp, tp + (fp + fn) / 2),
    }


def _pooled(counts: Iterable[PaperCounts]) -> PaperCounts:
    items = list(counts)
    return PaperCounts(sum(c.tp for c in items), sum(c.fp for c in items), sum(c.fn for c in items))


def _group_record(papers: list[str], results: Mapping[str, PaperResult]) -> dict[str, object]:
    members = [results[p] for p in papers]
    aware = _pooled(r.stereo_aware for r in members)
    stripped = _pooled(r.stereo_stripped for r in members)
    rows = sum(r.rows for r in members)
    invalid_rows = sum(r.invalid_rows for r in members)
    return {
        "papers": papers,
        "molecules": aware.tp + aware.fn,
        "counts": {"tp": aware.tp, "fp": aware.fp, "fn": aware.fn},
        "micro": _micro(aware),
        "macro": {
            "precision": macro([r.stereo_aware.precision for r in members]),
            "recall": macro([r.stereo_aware.recall for r in members]),
        },
        "stereo_stripped": {"counts": {"tp": stripped.tp, "fp": stripped.fp, "fn": stripped.fn}, **_micro(stripped)},
        "valid_output_rate": proportion(rows - invalid_rows, rows),
        "papers_without_output": sum(r.rows == 0 for r in members),
    }
