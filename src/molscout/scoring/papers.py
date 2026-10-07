"""Whole-PDF scoring: the set of molecules a tool reports for each paper."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass

from molscout.predictions import Prediction
from molscout.scoring.smiles import canonical_smiles, stereo_stripped_smiles
from molscout.scoring.stats import macro, paper_proportion, proportion, ratio


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
    ignored_outputs: int | None = None


def count_paper(
    predicted: Sequence[str], reference: Sequence[str], ignored: Collection[str] | None = None
) -> PaperResult:
    """Match one paper's predicted SMILES against its reference SMILES.

    Both sides collapse to unique canonical SMILES. Each distinct unparsable
    output, compared after trimming whitespace, is one false positive.

    `ignored` lists structures that are neither right nor wrong to output. In each view an
    output that matches one of them and no reference is dropped before counting;
    `ignored_outputs` is the stereo-aware number dropped. Without `ignored` the result has no
    such count.
    """
    truth = _canonical_set(reference, "reference")
    skip = None if ignored is None else _canonical_set(ignored, "ignored")
    parsed = [(smiles, canonical_smiles(smiles)) for smiles in predicted]
    found = {canonical for _, canonical in parsed if canonical is not None}
    invalid = {smiles.strip() for smiles, canonical in parsed if canonical is None}
    found_stripped, truth_stripped = _stripped(found), _stripped(truth)
    dropped = 0
    if skip is not None:
        skipped = skip - truth
        dropped = len(found & skipped)
        found = found - skipped
        found_stripped = found_stripped - (_stripped(skip) - truth_stripped)
    return PaperResult(
        stereo_aware=_counts(found, truth, len(invalid)),
        stereo_stripped=_counts(found_stripped, truth_stripped, len(invalid)),
        valid_outputs=len(found),
        invalid_outputs=len(invalid),
        rows=len(parsed),
        invalid_rows=sum(canonical is None for _, canonical in parsed),
        ignored_outputs=None if skip is None else dropped,
    )


def score_papers(
    predictions: Sequence[Prediction],
    references: Mapping[str, Sequence[str]],
    groups: Mapping[str, Collection[str]] | None = None,
    ignored: Mapping[str, Collection[str]] | None = None,
) -> dict[str, object]:
    """Score whole-PDF predictions per paper, then pool them for each group of papers.

    `references` maps paper ID to its reference SMILES. `groups` names the sets of
    papers to report, such as dev, test and all; the default is one group, "all". `ignored` maps
    paper ID to structures that are not counted when output (see `count_paper`); with it every
    paper and group record also carries `ignored_outputs`.

    A group's micro precision, recall and F1 (stereo-aware and stereo-stripped) get their 95% CI from a
    bootstrap over that group's papers (`ci_method` "paper bootstrap"), since the molecules of one paper are
    not independent; the valid-output rate keeps its Wilson interval.
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
            results[paper] = count_paper(
                predicted.get(paper, []), references[paper], None if ignored is None else ignored.get(paper, ())
            )
        except ValueError as exc:
            raise ValueError(f"paper {paper!r}: {exc}") from exc
    return {
        "kind": "paper",
        "papers": {paper: _paper_record(result) for paper, result in results.items()},
        "groups": {name: _group_record(_ordered(papers), results) for name, papers in named.items()},
    }


def _canonical_set(smiles_list: Iterable[str], role: str) -> set[str]:
    canonicals = set()
    for smiles in smiles_list:
        canonical = canonical_smiles(smiles)
        if canonical is None:
            raise ValueError(f"{role} SMILES {smiles!r} cannot be parsed by RDKit")
        canonicals.add(canonical)
    return canonicals


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
        **({} if result.ignored_outputs is None else {"ignored_outputs": result.ignored_outputs}),
    }


def _micro(per_paper: Sequence[PaperCounts]) -> dict[str, object]:
    """Pooled precision, recall and F1 over these papers, each with a 95% CI from resampling the papers."""
    tp = [c.tp for c in per_paper]
    return {
        "precision": paper_proportion(tp, [c.tp + c.fp for c in per_paper]),
        "recall": paper_proportion(tp, [c.tp + c.fn for c in per_paper]),
        # F1 = TP / (TP + (FP + FN) / 2): the pooled ratio of these sums, resampled like precision and recall.
        "f1": paper_proportion(tp, [c.tp + (c.fp + c.fn) / 2 for c in per_paper]),
    }


def _pooled(counts: Iterable[PaperCounts]) -> PaperCounts:
    items = list(counts)
    return PaperCounts(sum(c.tp for c in items), sum(c.fp for c in items), sum(c.fn for c in items))


def _group_record(papers: list[str], results: Mapping[str, PaperResult]) -> dict[str, object]:
    members = [results[p] for p in papers]
    per_paper_aware = [r.stereo_aware for r in members]
    per_paper_stripped = [r.stereo_stripped for r in members]
    aware, stripped = _pooled(per_paper_aware), _pooled(per_paper_stripped)
    rows = sum(r.rows for r in members)
    invalid_rows = sum(r.invalid_rows for r in members)
    return {
        "papers": papers,
        "molecules": aware.tp + aware.fn,
        "counts": {"tp": aware.tp, "fp": aware.fp, "fn": aware.fn},
        "micro": _micro(per_paper_aware),
        "macro": {
            "precision": macro([r.stereo_aware.precision for r in members]),
            "recall": macro([r.stereo_aware.recall for r in members]),
        },
        "stereo_stripped": {
            "counts": {"tp": stripped.tp, "fp": stripped.fp, "fn": stripped.fn},
            **_micro(per_paper_stripped),
        },
        "valid_output_rate": proportion(rows - invalid_rows, rows),
        "papers_without_output": sum(r.rows == 0 for r in members),
        **_ignored_record(members),
    }


def _ignored_record(members: Sequence[PaperResult]) -> dict[str, object]:
    if all(r.ignored_outputs is None for r in members):
        return {}
    return {"ignored_outputs": sum(r.ignored_outputs or 0 for r in members)}
