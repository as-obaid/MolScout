"""BioVista ground truth: the structure labels of the papers in the benchmark.

A scored paper (status ok, at least one structure) with any unreadable label is dropped: it is not run and
not scored, because a system that reads that structure correctly would be charged a false positive for it.

Some enumerated labels are rebuilt when they are loaded. Their SMILES was made by pasting each R group's text over
its `[*:n]` in the backbone text, and a group's ring-closure digit collides with one still open in the backbone, so
the paste is another molecule. Such a label takes the RDKit molzip join of its backbone and groups instead
(repair_label), and only where that is provable; every other label is scored as published.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from rdkit import Chem, rdBase

from molscout.hashing import sha256_file
from molscout.scoring.smiles import canonical_smiles

BIOVISTA_PAPERS_PATH = Path("data/manifests/biovista_papers.csv")
LABELS = Path("bioactivity_extraction/labels")
GROUPS = ("all", "without_submitted", "dev", "heldout")
SUBMITTED = "submittedVersion"
MANIFEST_COLUMNS = ("paper_id", "pdb_id", "structures", "status", "oa_version", "sha256")
LABEL_COLUMNS = ("smiles", "backbone", "groups")  # repair_label reads the backbone and groups
NO_SMILES = "NA: the label has no SMILES"
WHITESPACE = "RDKit cannot parse it; text after whitespace is part of the SMILES"
UNPARSABLE = "RDKit cannot parse it"
DROPPED = "has an unreadable label"
REPAIR_RULE = (
    "an enumerated label takes the RDKit molzip join of its backbone and groups when its SMILES is the naive paste "
    "(each group's text, minus its leading [*:n], written over that [*:n] in the backbone text), the join parses, "
    "the join is another molecule, the paste with the groups' ring-closure digits renumbered to digits the "
    "backbone does not use is the join, no [*:n] starts the backbone, and no ring of 4 or fewer atoms in the "
    "backbone has 3 or more atoms written aromatic"
)
ATTACHMENT = re.compile(r"\[\*:\d+\]")
RING_LABEL = re.compile(r"%(\d\d)|%\((\d+)\)")  # a two-digit ring-closure label, or RDKit's %(n) form
SMALL_RING = 4  # a backbone ring this small with 3 or more atoms written aromatic is a ring digit reused
SMALL_RING_AROMATIC = 3


@dataclass(frozen=True, slots=True)
class Repair:
    """A rebuilt label: its SMILES as labelled, and the canonical molzip join that replaces it."""

    original: str
    repaired: str


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
    reason), and `dropped` maps each paper left out because of them to why. With `dev` (the frozen development
    papers, data/manifests/biovista_dev.csv) the groups add `dev` and `heldout`, the other benchmark papers.

    `repairs` maps each rebuilt label (`<paper>:<CSV line>`) to its SMILES as labelled and the join that replaced it
    in `references` and `enumerated` (repair_label).
    """

    papers: tuple[ScoredPaper, ...]
    references: Mapping[str, tuple[str, ...]]
    drawn: Mapping[str, tuple[str, ...]]
    enumerated: Mapping[str, tuple[str, ...]]
    unreadable: Mapping[str, str]
    dropped: Mapping[str, str]
    labels: int
    dev: frozenset[str] | None = None
    repairs: Mapping[str, Repair] = field(default_factory=lambda: MappingProxyType({}))

    def groups(self) -> dict[str, frozenset[str]]:
        return _groups(self.papers, self.references, self.dev)

    def drawn_groups(self) -> dict[str, frozenset[str]]:
        return _groups(self.papers, self.drawn, self.dev)


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


def load_biovista_truth(
    root: str | Path, manifest: str | Path = BIOVISTA_PAPERS_PATH, *, dev: Collection[str] | None = None
) -> BioVistaTruth:
    """Read every scored paper's label file; ValueError if the labels differ from the frozen manifest.

    A paper with an unreadable label (`NA`, or text RDKit cannot parse) is dropped. Drawn means `backbone`
    is exactly `NA`; any other value, blank included, is enumerated. `dev` names the development papers
    (biovista_dev.dev_paper_ids); each must be a benchmark paper.

    Each label repair_label rebuilds takes the join's SMILES and is listed in `repairs`; rebuilding never changes
    which papers are kept or dropped.
    """
    kept: list[ScoredPaper] = []
    references: dict[str, tuple[str, ...]] = {}
    drawn: dict[str, tuple[str, ...]] = {}
    enumerated: dict[str, tuple[str, ...]] = {}
    unreadable: dict[str, str] = {}
    dropped: dict[str, str] = {}
    repairs: dict[str, Repair] = {}
    labels = 0
    for paper in scored_papers(manifest):
        rows = _read_labels(_label_path(root, paper), paper)
        reasons = {f"{paper.paper_id}:{line}": _unreadable_reason(smiles) for line, smiles, _, _ in rows}
        bad = {row: reason for row, reason in reasons.items() if reason is not None}
        if bad:
            unreadable.update(bad)
            dropped[paper.paper_id] = DROPPED
            continue
        kept.append(paper)
        labels += len(rows)
        readable = []
        for line, smiles, backbone, groups in rows:
            smiles, backbone = smiles.strip(), backbone.strip()
            fixed = repair_label(smiles, backbone, groups)
            if fixed is not None:
                repairs[f"{paper.paper_id}:{line}"] = Repair(original=smiles, repaired=fixed)
            readable.append((smiles if fixed is None else fixed, backbone))
        references[paper.paper_id] = tuple(s for s, _ in readable)
        enumerated[paper.paper_id] = tuple(s for s, b in readable if b != "NA")
        if any(b == "NA" for _, b in readable):
            drawn[paper.paper_id] = tuple(s for s, b in readable if b == "NA")
    if dev is not None:
        unknown = sorted(set(dev) - set(references))
        if unknown:
            raise ValueError(f"development paper(s) {', '.join(unknown)} are not benchmark papers")
    return BioVistaTruth(
        papers=tuple(kept),
        references=MappingProxyType(references),
        drawn=MappingProxyType(drawn),
        enumerated=MappingProxyType(enumerated),
        unreadable=MappingProxyType(unreadable),
        dropped=MappingProxyType(dropped),
        labels=labels,
        dev=None if dev is None else frozenset(dev),
        repairs=MappingProxyType(repairs),
    )


def repair_label(smiles: str, backbone: str, groups: str) -> str | None:
    """The rebuilt SMILES for one label, or None when the label stays as it is.

    The rebuilt SMILES is the canonical SMILES of `Chem.molzip` over the backbone and its `;`-separated groups,
    each group led by the `[*:n]` it attaches at. It replaces an enumerated label only when the label is provably
    wrong and the join provably right:

    (a) the label's SMILES is the naive paste: each group's text, minus its `[*:n]`, written over that `[*:n]`
        in the backbone text;
    (b) the join parses;
    (c) the join is another molecule than the label;
    (d) the paste with each group's ring-closure digits renumbered to digits the backbone does not use is the
        join, so a digit collision is the paste's only defect;
    (e) no `[*:n]` starts the backbone: a paste there bonds through the group's last-written atom, so which atom
        the label meant to attach is unclear;
    (f) the backbone does not garble itself: parsed unsanitized, none of its smallest rings of 4 or fewer atoms
        has 3 or more atoms written aromatic, which only a reused ring digit writes.

    Groups that do not pair one to one with the backbone's attachment points, or that hold whitespace, leave the
    label alone.
    """
    parts = _label_parts(backbone, groups)
    key = canonical_smiles(smiles)
    if parts is None or key is None or canonical_smiles(_naive_paste(*parts)) != key:  # (a)
        return None
    if ATTACHMENT.match(parts[0]) or _garbles_itself(parts[0]):  # (e), (f)
        return None
    joined = _molzip_join(*parts)
    if joined is None or joined == key:  # (b), (c)
        return None
    renumbered = _renumbered_paste(*parts)
    return joined if renumbered is not None and canonical_smiles(renumbered) == joined else None  # (d)


def reference_set_sha256(root: str | Path, manifest: str | Path = BIOVISTA_PAPERS_PATH) -> str:
    """sha256 of the labels as scored: sorted `<sha256>  <relative path>` lines over the manifest and every scored
    label file, then `<paper>:<CSV line>  <label SMILES>  <rebuilt SMILES>` per rebuilt label, in manifest and line
    order.

    Dropped papers' label files are included: a fix that makes a dropped paper readable changes the paper set. The
    rebuilt lines make it change with what repair_label rebuilds from the same files, as under another RDKit; a set
    with no rebuilt label hashes its files alone.
    """
    manifest = Path(manifest)
    entries = {manifest.name: sha256_file(manifest)}
    for paper in scored_papers(manifest):
        path = _label_path(root, paper)
        if not path.is_file():
            raise ValueError(f"paper {paper.paper_id}: no label file at {path}")
        entries[(LABELS / path.name).as_posix()] = sha256_file(path)
    lines = [f"{digest}  {name}\n" for name, digest in sorted(entries.items())]
    repairs = load_biovista_truth(root, manifest).repairs
    lines += [f"{row}  {repair.original}  {repair.repaired}\n" for row, repair in repairs.items()]
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def _groups(
    papers: tuple[ScoredPaper, ...], present: Mapping[str, object], dev: frozenset[str] | None
) -> dict[str, frozenset[str]]:
    kept = [p for p in papers if p.paper_id in present]
    groups = {
        "all": frozenset(p.paper_id for p in kept),
        "without_submitted": frozenset(p.paper_id for p in kept if p.oa_version != SUBMITTED),
    }
    if dev is not None:
        groups["dev"] = frozenset(p.paper_id for p in kept if p.paper_id in dev)
        groups["heldout"] = frozenset(p.paper_id for p in kept if p.paper_id not in dev)
    return groups


def _label_path(root: str | Path, paper: ScoredPaper) -> Path:
    return Path(root) / LABELS / f"{paper.paper_id}_structure.csv"


def _read_labels(path: Path, paper: ScoredPaper) -> list[tuple[int, str, str, str]]:
    """(CSV line, smiles, backbone, groups) per label row."""
    if not path.is_file():
        raise ValueError(f"paper {paper.paper_id}: no label file at {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in LABEL_COLUMNS if c not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{path}: missing columns {', '.join(missing)}")
        rows = [(reader.line_num, row["smiles"] or "", row["backbone"] or "", row["groups"] or "") for row in reader]
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


def _label_parts(backbone: str, groups: str) -> tuple[str, tuple[tuple[str, str], ...]] | None:
    """The backbone and each group as (its `[*:n]`, its text), or None unless the groups pair one to one with the
    backbone's `[*:n]`: each group holds one, at its start, and no text holds whitespace."""
    backbone = backbone.strip()
    texts = [text.strip() for text in groups.split(";")]
    if backbone in ("", "NA") or any(c.isspace() for c in backbone + "".join(texts)):
        return None
    parts = []
    for text in texts:
        found = ATTACHMENT.findall(text)
        if len(found) != 1 or not text.startswith(found[0]) or text == found[0]:
            return None
        parts.append((found[0], text))
    tags = [tag for tag, _ in parts]
    if len(set(tags)) != len(tags) or sorted(tags) != sorted(ATTACHMENT.findall(backbone)):
        return None
    return backbone, tuple(parts)


def _naive_paste(backbone: str, parts: tuple[tuple[str, str], ...]) -> str:
    """Each group's text after its `[*:n]`, written over that `[*:n]` in the backbone text."""
    pasted = backbone
    for tag, text in parts:
        pasted = pasted.replace(tag, text[len(tag):])
    return pasted


def _renumbered_paste(backbone: str, parts: tuple[tuple[str, str], ...]) -> str | None:
    """The naive paste with each group's ring-closure labels renumbered to labels above every one the backbone
    uses, a fresh set per group; None when a text has a part this does not read (CXSMILES, an open bracket)."""
    used = _ring_labels(backbone)
    if used is None:
        return None
    label = max((number for _, _, number in used), default=0)
    pasted = backbone
    for tag, text in parts:
        group = text[len(tag):]
        tokens = _ring_labels(group)
        if tokens is None:
            return None
        fresh: dict[int, int] = {}
        pieces, last = [], 0
        for start, end, number in tokens:
            if number not in fresh:
                label += 1
                fresh[number] = label
            pieces += [group[last:start], _ring_label_text(fresh[number])]
            last = end
        pasted = pasted.replace(tag, "".join([*pieces, group[last:]]))
    return pasted


def _ring_labels(text: str) -> list[tuple[int, int, int]] | None:
    """(start, end, label) of each ring-closure label outside bracket atoms; None when the text holds a part this
    does not read: CXSMILES after `|`, an unclosed bracket, or a `%` without a label."""
    labels, i = [], 0
    while i < len(text):
        char = text[i]
        if char == "[":
            close = text.find("]", i)
            if close < 0:
                return None
            i = close + 1
        elif char == "|":
            return None
        elif char == "%":
            match = RING_LABEL.match(text, i)
            if match is None:
                return None
            labels.append((i, match.end(), int(match.group(1) or match.group(2))))
            i = match.end()
        else:
            if char.isdigit():
                labels.append((i, i + 1, int(char)))
            i += 1
    return labels


def _ring_label_text(number: int) -> str:
    return str(number) if number < 10 else f"%{number}" if number < 100 else f"%({number})"


def _garbles_itself(backbone: str) -> bool:
    """Whether the backbone text, parsed unsanitized so atoms keep the aromatic flag they were written with, has a
    smallest ring of SMALL_RING or fewer atoms with SMALL_RING_AROMATIC or more of them aromatic. A reused ring
    digit writes that (221_6oac: `c1nc(N1CCOCC1)` closes a c-n-c-N 4-ring); a drawn molecule hardly ever does,
    and benzocyclobutene's 4-ring has 2. An unparsable backbone counts as garbled."""
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(backbone, sanitize=False)
    if mol is None:
        return True
    # The smallest set of smallest rings, not FastFindRings: that one can return a larger ring around a small fused
    # one (biphenylene's 8-ring for its 4-ring).
    rings = Chem.GetSymmSSSR(mol)
    return any(
        len(ring) <= SMALL_RING and sum(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring) >= SMALL_RING_AROMATIC
        for ring in rings
    )


def _molzip_join(backbone: str, parts: tuple[tuple[str, str], ...]) -> str | None:
    """Canonical SMILES of Chem.molzip over `backbone.group1.group2...`; None when RDKit cannot build it."""
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(".".join([backbone, *(text for _, text in parts)]))
        if mol is None:
            return None
        try:
            joined = Chem.molzip(mol)
        except (RuntimeError, ValueError):
            return None
    return canonical_smiles(Chem.MolToSmiles(joined))
