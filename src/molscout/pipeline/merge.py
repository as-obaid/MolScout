"""The `both` joiner mode: rules molecules first; BioMiner-Instruct's (32b) molecules check and fill them.

merge() combines link_and_build's molecules (source drawn, table or name) with join_32b's
(molecules_32b.jsonl, source 32b), paper by paper, and accounts for every input: each one is in the
output or counted under a reason. A 32b molecule is usable when it is not abstained and RDKit reads
its SMILES. Structures match by InChIKey connectivity (the key's first
block); labels match through their aliases (32b joins aliases with ";") after dropping case, generic
words ("compound 5" is "5"), brackets, trailing dots and dash variants.

- Agreement. A rules molecule whose connectivity a usable 32b molecule shares gains
  policy.agree_boost confidence (at most 1) and the reason "32b agrees", plus "32b differs in stereo
  only" when the full keys differ. A 32b molecule with the same full key speaks first. Without a label
  the rules molecule takes 32b's (adopt_labels); with promote_agreed a review becomes accepted when
  that 32b molecule is accepted.
- Conflict. A usable 32b molecule with a rules molecule's label but another structure is not added;
  the rules molecule keeps its status and gets a reason.
- Tables (fill_tables, which needs fill_check; junk_free by default). A rules row that did not build
  (abstained) is replaced by the 32b molecule with its label, a row of the same table first; the fill
  takes the rules row's
  table_id, row, page and bbox. A rules table with fewer built rows than it has (table_rows, else the
  rows rules saw) takes the 32b rows join_32b placed in it (same table_id) whose labels rules never
  gave, in row order, up to the gap; they keep their table_id and row. A fill is accepted only when
  fill_check passes and 32b itself accepted the molecule; otherwise it is review.
- Any other usable 32b molecule joins as review ("32b only") when add_unmatched, once per InChIKey.

merge() returns (molecules, counts); counts has "in", "out" and one "dropped_<why>" per reason a record
left (DROPPED), and in == out + the dropped always holds (AssertionError otherwise). Relevance, the junk
and formula checks and dedupe across sources belong to finalize, which runs after the last step of every
mode. Inputs are never modified: changed molecules are new objects.
"""

from __future__ import annotations

import dataclasses
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from rdkit import Chem, RDLogger

if TYPE_CHECKING:
    from molscout.pipeline.schema import Molecule

RDLogger.DisableLog("rdApp.*")

GENERIC = re.compile(r"^(?:compounds?|compd|cpds?|ligands?|inhibitors?|analogues?|analogs?)\b\.?\s*")
DASHES = re.compile(r"[‐-―−]")
CONNECTIVITY = 14  # characters in an InChIKey's first block

DROP_ABSTAINED = "dropped_32b_abstained"
DROP_UNREADABLE = "dropped_32b_unreadable"  # no SMILES RDKit reads
DROP_AGREES = "dropped_32b_agrees"  # the agreeing rules molecule stands for it
DROP_CONFLICT = "dropped_32b_conflict"  # a rules molecule or a fill has its label and another structure
DROP_DUPLICATE = "dropped_32b_duplicate"
DROP_UNMATCHED = "dropped_32b_unmatched"  # add_unmatched off
DROP_REPLACED = "dropped_rules_row_filled"  # an abstained rules row a 32b fill replaced
DROPPED = (DROP_ABSTAINED, DROP_UNREADABLE, DROP_AGREES, DROP_CONFLICT, DROP_DUPLICATE, DROP_UNMATCHED,
           DROP_REPLACED)


def junk_free(molecule: Molecule) -> bool:
    """The default fill check: checks.junk_reason finds nothing wrong with the SMILES."""
    from molscout.pipeline.checks import junk_reason

    return bool(molecule.smiles) and junk_reason(molecule.smiles) is None


@dataclass(frozen=True)
class MergePolicy:
    agree_boost: float = 0.2  # confidence a rules molecule gains when 32b builds the same structure
    adopt_labels: bool = True  # an unlabelled rules molecule takes the agreeing 32b molecule's label
    promote_agreed: bool = False  # review -> accepted when an accepted 32b molecule agrees
    fill_tables: bool = True  # 32b rows fill rules rows that did not build and short tables
    fill_check: Callable[[Molecule], bool] | None = junk_free  # required with fill_tables
    add_unmatched: bool = True  # other 32b molecules join as review

    def __post_init__(self) -> None:
        if self.fill_tables and self.fill_check is None:
            raise ValueError("fill_tables needs a fill_check (junk_free by default); "
                             "pass fill_tables=False to merge without filling tables")


class MergeResult(NamedTuple):
    """The merged molecules and their counts: {"in", "out", "dropped_<why>", ...}."""

    molecules: list[Molecule]
    counts: dict[str, int]


def counts_for(records_in: int, molecules: Sequence[Molecule], dropped: Mapping[str, int]) -> dict[str, int]:
    """{"in", "out", "dropped_<why>"...} (reasons that dropped nothing left out); AssertionError unless they add up."""
    counts = {"in": records_in, "out": len(molecules), **{why: n for why, n in dropped.items() if n}}
    if not all(why in DROPPED for why in dropped) or counts["in"] != counts["out"] + sum(dropped.values()):
        raise AssertionError(f"merge lost track of records: {counts}")
    return counts


def aliases(label: str | None) -> set[str]:
    """A label's normalized aliases: '5a', 'compound 5a' and 'Cpd. 5a' are all '5a'."""
    found = set()
    for part in str(label or "").split(";"):
        text = GENERIC.sub("", DASHES.sub("-", part.strip().casefold()))
        text = re.sub(r"\s+", "", text.strip(" .,:;()[]{}"))
        if text:
            found.add(text)
    return found


def _with(molecule: Molecule, *reasons: str, **changes) -> Molecule:
    """A new molecule with reasons appended (list or tuple, as given) and fields changed."""
    joined = [*molecule.reasons, *reasons]
    return dataclasses.replace(molecule, reasons=tuple(joined) if isinstance(molecule.reasons, tuple) else joined,
                               **changes)


class _Keys:
    """InChIKeys of the molecules' SMILES, cached; None when RDKit cannot read the SMILES."""

    def __init__(self) -> None:
        self.cache: dict[str, str | None] = {}

    def full(self, molecule: Molecule) -> str | None:
        smiles = molecule.smiles
        if not smiles:
            return None
        if smiles not in self.cache:
            mol = Chem.MolFromSmiles(smiles)
            self.cache[smiles] = (Chem.MolToInchiKey(mol) if mol is not None else "") or None
        return self.cache[smiles]

    def connectivity(self, molecule: Molecule) -> str | None:
        key = self.full(molecule)
        return key[:CONNECTIVITY] if key else None


def merge(rules: Sequence[Molecule], m32b: Sequence[Molecule], policy: MergePolicy | None = None,
          table_rows: Mapping | None = None) -> MergeResult:
    """Rules molecules, checked and filled with the 32b molecules, then the 32b-only ones; per paper.

    table_rows: rows per table, keyed by (paper_id, table_id) or table_id; by default a table has the
    rows rules saw (built or abstained).
    """
    policy = policy or MergePolicy()
    molecules, dropped = [], Counter()
    for paper in dict.fromkeys([m.paper_id for m in rules] + [m.paper_id for m in m32b]):
        out, gone = _merge_paper(paper, [m for m in rules if m.paper_id == paper],
                                 [m for m in m32b if m.paper_id == paper], policy, table_rows or {})
        molecules += out
        dropped.update(gone)
    return MergeResult(molecules, counts_for(len(rules) + len(m32b), molecules, dropped))


def _merge_paper(paper, rules, found, policy, table_rows):
    keys, dropped = _Keys(), Counter()
    usable = []
    for molecule in found:
        if molecule.status == "abstained":
            dropped[DROP_ABSTAINED] += 1
        elif keys.full(molecule) is None:
            dropped[DROP_UNREADABLE] += 1
        else:
            usable.append(molecule)
    usable.sort(key=lambda m: (m.status != "accepted", not m.label))  # they speak for their structure first
    built = [r for r in rules if r.status != "abstained"]
    shared = {keys.connectivity(r) for r in built} - {None}
    taken = {id(m) for m in usable if keys.connectivity(m) in shared}  # agreements
    dropped[DROP_AGREES] += len(taken)
    out, row_fills = [], []
    for rule in rules:
        if rule.status == "abstained":
            fill = _fill_row(rule, usable, taken, policy)
            row_fills += [fill] if fill else []
            out.append(fill or rule)
        else:
            partner = _partner(rule, usable, keys)
            out.append(_agree(rule, partner, keys, policy) if partner else _judge(rule, usable))
    dropped[DROP_REPLACED] += len(row_fills)
    table_fills = _fill_tables(paper, out, rules, usable, taken, policy, table_rows) if policy.fill_tables else []
    extras, gone = _extras(usable, taken, built + row_fills + table_fills, row_fills + table_fills, keys, policy)
    dropped.update(gone)
    return out + table_fills + extras, dropped


def _partner(rule, usable, keys):
    """The usable 32b molecule with the rule's connectivity, one with the same full InChIKey first."""
    if keys.full(rule) is None:
        return None
    same = [m for m in usable if keys.connectivity(m) == keys.connectivity(rule)]
    return next((m for m in same if keys.full(m) == keys.full(rule)), same[0] if same else None)


def _agree(rule, other, keys, policy):
    changes = {"confidence": min(1.0, (rule.confidence or 0.0) + policy.agree_boost)}
    reasons = ["32b agrees"]
    if keys.full(other) != keys.full(rule):
        reasons.append("32b differs in stereo only")
    if policy.adopt_labels and not rule.label and other.label:
        changes["label"] = other.label
        reasons.append("label from 32b")
    if policy.promote_agreed and rule.status == "review" and other.status == "accepted":
        changes["status"] = "accepted"
    return _with(rule, *reasons, **changes)


def _judge(rule, usable):
    """A rules molecule 32b did not reproduce: a 32b molecule with its label means another structure."""
    names = aliases(rule.label)
    if names and any(names & aliases(m.label) for m in usable):
        return _with(rule, "32b reads a different structure for this label")
    return rule


def _filled(fill, policy, reason, **changes):
    """A 32b molecule as a fill: accepted only when fill_check passes and 32b accepted it."""
    ok = policy.fill_check(fill)
    reasons = [reason] if ok else [reason, "fill check failed"]
    return _with(fill, *reasons, status="accepted" if ok and fill.status == "accepted" else "review", **changes)


def _fill_row(rule, usable, taken, policy):
    """The unused 32b molecule with the label of a rules row that did not build, as that row; or None."""
    names = aliases(rule.label)
    if not (policy.fill_tables and names):
        return None
    candidates = [m for m in usable if id(m) not in taken and names & aliases(m.label)]
    if not candidates:
        return None
    fill = next((m for m in candidates if m.table_id == rule.table_id), candidates[0])
    taken.add(id(fill))
    return _filled(fill, policy, "fills a rules row that did not build", table_id=rule.table_id, row=rule.row,
                   page=rule.page, bbox=rule.bbox)


def _fill_tables(paper, out, rules, usable, taken, policy, table_rows):
    """32b rows of rules tables with fewer built rows than they have, up to the gap, in row order."""
    tables = {}
    for molecule in out:
        if molecule.table_id is not None:
            tables.setdefault(molecule.table_id, []).append(molecule)
    named = set().union(*(aliases(r.label) for r in rules))
    fills = []
    for table_id, members in tables.items():
        size = table_rows.get((paper, table_id), table_rows.get(table_id))
        if size is None:
            size = len({m.row if m.row is not None else id(m) for m in members})
        gap = size - sum(m.status != "abstained" for m in members)
        rows = [m for m in usable if id(m) not in taken and m.table_id == table_id and not aliases(m.label) & named]
        for molecule in sorted(rows, key=lambda m: (m.row is None, m.row or 0))[:max(gap, 0)]:
            taken.add(id(molecule))
            fills.append(_filled(molecule, policy, "fills a short rules table"))
    return fills


def _extras(usable, taken, labelled, fills, keys, policy):
    """The unused 32b molecules as review ("32b only"), and the count of those dropped, by reason."""
    labels = set().union(*(aliases(m.label) for m in labelled))
    seen = {keys.full(m) for m in fills}
    extras, dropped = [], Counter()
    for molecule in usable:
        if id(molecule) in taken:
            continue
        if aliases(molecule.label) & labels:
            dropped[DROP_CONFLICT] += 1
        elif not policy.add_unmatched:
            dropped[DROP_UNMATCHED] += 1
        elif keys.full(molecule) in seen:
            dropped[DROP_DUPLICATE] += 1
        else:
            seen.add(keys.full(molecule))
            extras.append(_with(molecule, "32b only", status="review"))
    return extras, dropped
