"""BioVista ground truth: the structure labels of the papers in the benchmark.

A scored paper (status ok, at least one structure) with any unreadable label is dropped: it is not run and
not scored, because a system that reads that structure correctly would be charged a false positive for it.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from molscout.hashing import sha256_file
from molscout.scoring.smiles import canonical_smiles

BIOVISTA_PAPERS_PATH = Path("data/manifests/biovista_papers.csv")
LABELS = Path("bioactivity_extraction/labels")
GROUPS = ("all", "without_submitted")
SUBMITTED = "submittedVersion"
MANIFEST_COLUMNS = ("paper_id", "pdb_id", "structures", "status", "oa_version", "sha256")
LABEL_COLUMNS = ("smiles", "backbone")
NO_SMILES = "NA: the label has no SMILES"
WHITESPACE = "RDKit cannot parse it; text after whitespace is part of the SMILES"
UNPARSABLE = "RDKit cannot parse it"
DROPPED = "has an unreadable label"


@dataclass(frozen=True, slots=True)
class ScoredPaper:
    paper_id: str
    pdb_id: str
    oa_version: str
    sha256: str
    structures: int


@dataclass(frozen=True)
class BioVistaTruth:
    """Labels per benchmark paper. A drawn label has backbone `NA`; the others are enumerated.

    `papers`, `references`, `drawn`, `enumerated` and the groups hold only the benchmark papers, and `labels`
    counts their rows. `unreadable` lists every unreadable row of the scored papers (`<paper>:<CSV line>` to
    reason), and `dropped` maps each paper left out because of them to why.
    """

    papers: tuple[ScoredPaper, ...]
    references: Mapping[str, tuple[str, ...]]
    drawn: Mapping[str, tuple[str, ...]]
    enumerated: Mapping[str, tuple[str, ...]]
    unreadable: Mapping[str, str]
    dropped: Mapping[str, str]
    labels: int

    def groups(self) -> dict[str, frozenset[str]]:
        return _groups(self.papers, self.references)

    def drawn_groups(self) -> dict[str, frozenset[str]]:
        return _groups(self.papers, self.drawn)


def scored_papers(manifest: str | Path = BIOVISTA_PAPERS_PATH) -> tuple[ScoredPaper, ...]:
    """Papers with status ok and at least one structure, in manifest (index) order."""
    path = Path(manifest)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in MANIFEST_COLUMNS if c not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{path}: missing columns {', '.join(missing)}")
        papers = []
        for row in reader:
            count = (row["structures"] or "").strip()
            if not count.isdecimal():
                raise ValueError(f"{path}: line {reader.line_num}: paper {row['paper_id']}: structures must be an integer, got {count!r}")
            if row["status"] == "ok" and int(count) > 0:
                papers.append(
                    ScoredPaper(row["paper_id"], row["pdb_id"], row["oa_version"], row["sha256"], int(count))
                )
    return tuple(papers)


def benchmark_papers(root: str | Path, manifest: str | Path = BIOVISTA_PAPERS_PATH) -> tuple[ScoredPaper, ...]:
    """The papers the benchmark runs and scores: the scored papers whose every label is readable, in manifest order."""
    return load_biovista_truth(root, manifest).papers


def load_biovista_truth(root: str | Path, manifest: str | Path = BIOVISTA_PAPERS_PATH) -> BioVistaTruth:
    """Read every scored paper's label file; ValueError if the labels differ from the frozen manifest.

    A paper with an unreadable label (`NA`, or text RDKit cannot parse) is dropped. Drawn means `backbone`
    is exactly `NA`; any other value, blank included, is enumerated.
    """
    kept: list[ScoredPaper] = []
    references: dict[str, tuple[str, ...]] = {}
    drawn: dict[str, tuple[str, ...]] = {}
    enumerated: dict[str, tuple[str, ...]] = {}
    unreadable: dict[str, str] = {}
    dropped: dict[str, str] = {}
    labels = 0
    for paper in scored_papers(manifest):
        rows = _read_labels(_label_path(root, paper), paper)
        reasons = {f"{paper.paper_id}:{line}": _unreadable_reason(smiles) for line, smiles, _ in rows}
        bad = {row: reason for row, reason in reasons.items() if reason is not None}
        if bad:
            unreadable.update(bad)
            dropped[paper.paper_id] = DROPPED
            continue
        kept.append(paper)
        labels += len(rows)
        readable = [(smiles.strip(), backbone.strip()) for _, smiles, backbone in rows]
        references[paper.paper_id] = tuple(s for s, _ in readable)
        enumerated[paper.paper_id] = tuple(s for s, b in readable if b != "NA")
        if any(b == "NA" for _, b in readable):
            drawn[paper.paper_id] = tuple(s for s, b in readable if b == "NA")
    return BioVistaTruth(
        papers=tuple(kept),
        references=MappingProxyType(references),
        drawn=MappingProxyType(drawn),
        enumerated=MappingProxyType(enumerated),
        unreadable=MappingProxyType(unreadable),
        dropped=MappingProxyType(dropped),
        labels=labels,
    )


def reference_set_sha256(root: str | Path, manifest: str | Path = BIOVISTA_PAPERS_PATH) -> str:
    """sha256 over the manifest and every scored label file, as sorted `<sha256>  <relative path>` lines.

    Dropped papers' label files are included: a fix that makes a dropped paper readable changes the paper set.
    """
    manifest = Path(manifest)
    entries = {manifest.name: sha256_file(manifest)}
    for paper in scored_papers(manifest):
        path = _label_path(root, paper)
        if not path.is_file():
            raise ValueError(f"paper {paper.paper_id}: no label file at {path}")
        entries[(LABELS / path.name).as_posix()] = sha256_file(path)
    lines = "".join(f"{digest}  {name}\n" for name, digest in sorted(entries.items()))
    return hashlib.sha256(lines.encode()).hexdigest()


def _groups(papers: tuple[ScoredPaper, ...], present: Mapping[str, object]) -> dict[str, frozenset[str]]:
    kept = [p for p in papers if p.paper_id in present]
    return {
        "all": frozenset(p.paper_id for p in kept),
        "without_submitted": frozenset(p.paper_id for p in kept if p.oa_version != SUBMITTED),
    }


def _label_path(root: str | Path, paper: ScoredPaper) -> Path:
    return Path(root) / LABELS / f"{paper.paper_id}_structure.csv"


def _read_labels(path: Path, paper: ScoredPaper) -> list[tuple[int, str, str]]:
    if not path.is_file():
        raise ValueError(f"paper {paper.paper_id}: no label file at {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in LABEL_COLUMNS if c not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{path}: missing columns {', '.join(missing)}")
        rows = [(reader.line_num, row["smiles"] or "", row["backbone"] or "") for row in reader]
    if len(rows) != paper.structures:
        raise ValueError(
            f"paper {paper.paper_id}: {len(rows)} label rows but the manifest froze {paper.structures}; "
            "the labels changed since the manifest was written"
        )
    return rows


def _unreadable_reason(smiles: str) -> str | None:
    text = smiles.strip()
    if text in ("", "NA"):
        return NO_SMILES
    if canonical_smiles(text) is not None and not any(c.isspace() for c in text):
        return None
    return WHITESPACE if any(c.isspace() for c in text) else UNPARSABLE
