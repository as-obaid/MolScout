"""Finalize: the checks every joiner mode shares, applied once per paper after its last joining step.

The runner calls `finalize` on the rules, 32b or merged molecules, so every mode passes the same
rules, in this order:

1. A molecule that is not abstained but has no SMILES is abstained ("no SMILES").
2. Junk (`junk` fix): a structure `checks.junk_reason` rejects is abstained with "junk: <why>"; a
   salt or solvent beside the parent is dropped from the SMILES ("salt or solvent removed").
3. Printed formulas (`formulas` fix): when the paper prints a formula for the molecule's label, a
   match adds the reason "printed formula F matches" and FORMULA_BOOST confidence (at most 1); a
   mismatch sends an accepted molecule to review.
4. Relevance (`relevance` fix): a drawn or table molecule stays accepted when it has a compound
   label, or when it has no label but at least MIN_UNLABELLED_ATOMS heavy atoms (drug-sized, not a
   reagent or building block); a smaller unlabelled molecule goes to review ("no compound label"). A
   molecule read only from a chemical name in the text (source "name") also needs its label in an
   activity table or sentence when the paper has any. A label the paper reports activity for adds
   that reason. On the BioVista development papers, drawn and table molecules whose labels were in
   no activity table or sentence the rules found were nine in ten in the answer key; named ones
   were one in fifty.
5. Dedupe: molecules with the same label and InChIKey connectivity block become one record, the
   one with the highest status (accepted > review > abstained, then the first), with the highest
   confidence of the group and a reason naming the merged sources. Records without a SMILES never
   merge.
6. An abstained molecule's SMILES is None.

Statuses only ever move down, except through dedupe, which keeps the best record. Labels compare
by their names: aliases joined with ";" (32b) each count, and a leading "compound", brackets,
spaces and case are ignored, so "Compd. 5a" is "5a".

Counts: `in` = `out` + `dropped_duplicate`; `accepted`, `review` and `abstained` after finalize;
and, when they happen, `no_smiles`, `junk`, `salt_stripped`, `formula_match`, `formula_mismatch`,
`not_relevant` and `unlabelled_kept`.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from molscout.pipeline.activity import reports_activity, tested_labels
from molscout.pipeline.checks import formula_status, junk_reason, printed_formulas, strip_salts
from molscout.pipeline.schema import STATUSES, LayoutBlock, Molecule, fix_enabled, need, read_jsonl
from molscout.pipeline.tables import find_rgroup_tables

ACCEPTED, REVIEW, ABSTAINED = STATUSES
RANK = {status: rank for rank, status in enumerate(STATUSES)}
FORMULA_BOOST = 0.1
MIN_UNLABELLED_ATOMS = 20  # an unlabelled molecule this large is a compound of the paper, not a reagent
NAME_SOURCE = "name"  # read only from a chemical name in the text (names.py)
CONNECTIVITY_CHARS = 14
_GENERIC = re.compile(r"^(?i:compounds?|compd|cmpd|cpds?|ligands?|inhibitors?|analogues?|analogs?)\b\.?\s*")
_SPACES = re.compile(r"\s+")


def activity_names(blocks: Sequence[LayoutBlock]) -> frozenset[str]:
    """The names (see `_aliases`) of the labels the paper reports activity for, in its tables and sentences."""
    if not reports_activity(blocks):
        return frozenset()
    return frozenset(name for label in tested_labels(blocks, find_rgroup_tables(blocks)) for name in _aliases(label))


def finalize(
    paper_dir: Path | str, molecules: Sequence[Molecule], *, fixes: Any = None
) -> tuple[list[Molecule], dict[str, int]]:
    """The paper's final molecules and the counts; see the module docstring for the steps.

    Raises FileNotFoundError without layout.jsonl and ValueError for molecules of another paper
    (the folder name is the paper id).
    """
    paper_dir = Path(paper_dir)
    _check_paper(paper_dir.name, molecules)
    blocks = read_jsonl(need(paper_dir / "layout.jsonl"), LayoutBlock)
    counts: Counter[str] = Counter({"in": len(molecules)})
    out = [_with_smiles(m, counts) for m in molecules]
    if fix_enabled(fixes, "junk"):
        out = [_junk(m, counts) for m in out]
    if fix_enabled(fixes, "formulas"):
        formulas = {label.casefold(): formula for label, formula in printed_formulas(blocks).items()}
        out = [_formula(m, formulas, counts) for m in out]
    if fix_enabled(fixes, "relevance"):
        tested = activity_names(blocks)
        out = [_relevant(m, tested, counts) for m in out]
    out = [_without_abstained_smiles(m) for m in _dedupe(out, counts)]
    counts["out"] = len(out)
    for status in STATUSES:
        counts[status] = sum(m.status == status for m in out)
    if counts["in"] != counts["out"] + counts["dropped_duplicate"]:
        raise RuntimeError(f"finalize lost records: {dict(counts)}")
    return out, dict(counts)


def _check_paper(paper_id: str, molecules: Sequence[Molecule]) -> None:
    others = sorted({m.paper_id for m in molecules} - {paper_id})
    if others:
        raise ValueError(f"molecules of paper(s) {others} given to finalize for paper {paper_id!r}")


def _reasons(molecule: Molecule, *more: str) -> list[str]:
    return [*molecule.reasons, *more]


def _with_smiles(molecule: Molecule, counts: Counter[str]) -> Molecule:
    if molecule.status == ABSTAINED or molecule.smiles:
        return molecule
    counts["no_smiles"] += 1
    return replace(molecule, status=ABSTAINED, reasons=_reasons(molecule, "no SMILES"))


def _junk(molecule: Molecule, counts: Counter[str]) -> Molecule:
    if molecule.status == ABSTAINED:
        return molecule
    reason = junk_reason(molecule.smiles or "")
    if reason is not None:
        counts["junk"] += 1
        return replace(molecule, status=ABSTAINED, reasons=_reasons(molecule, f"junk: {reason}"))
    if "." not in molecule.smiles:
        return molecule
    counts["salt_stripped"] += 1
    return replace(molecule, smiles=strip_salts(molecule.smiles), reasons=_reasons(molecule, "salt or solvent removed"))


def _formula(molecule: Molecule, formulas: dict[str, str], counts: Counter[str]) -> Molecule:
    formula = next((formulas[name] for name in sorted(_aliases(molecule.label)) if name in formulas), None)
    if molecule.status == ABSTAINED or formula is None:
        return molecule
    verdict = formula_status(molecule.smiles or "", formula)
    if verdict == "match":
        counts["formula_match"] += 1
        confidence = min(1.0, (molecule.confidence or 0.0) + FORMULA_BOOST)
        return replace(molecule, confidence=confidence, reasons=_reasons(molecule, f"printed formula {formula} matches"))
    if verdict == "mismatch":
        counts["formula_mismatch"] += 1
        status = REVIEW if molecule.status == ACCEPTED else molecule.status
        return replace(molecule, status=status, reasons=_reasons(molecule, f"printed formula {formula} does not match"))
    return molecule


def _relevant(molecule: Molecule, tested: frozenset[str], counts: Counter[str]) -> Molecule:
    if molecule.status == ABSTAINED:
        return molecule
    names = _aliases(molecule.label)
    if names & tested:
        return replace(molecule, reasons=_reasons(molecule, f"label {molecule.label} is in an activity table or sentence"))
    named = molecule.source == NAME_SOURCE
    if names and not (named and tested):
        return molecule
    if not names and not named and (atoms := _heavy_atoms(molecule.smiles)) >= MIN_UNLABELLED_ATOMS:
        counts["unlabelled_kept"] += 1
        return replace(molecule, reasons=_reasons(molecule, f"no compound label, but drug-sized ({atoms} heavy atoms)"))
    problem = f"label {molecule.label} is not in an activity table or sentence" if names else "no compound label"
    counts["not_relevant"] += 1
    status = REVIEW if molecule.status == ACCEPTED else molecule.status
    return replace(molecule, status=status, reasons=_reasons(molecule, problem))


def _heavy_atoms(smiles: str | None) -> int:
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles or "")
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def _dedupe(molecules: Sequence[Molecule], counts: Counter[str]) -> list[Molecule]:
    groups: list[list[Molecule]] = []
    where: dict[tuple[str | None, str], int] = {}
    for molecule in molecules:
        key = _key(molecule)
        if key is not None and key in where:
            groups[where[key]].append(molecule)
            continue
        if key is not None:
            where[key] = len(groups)
        groups.append([molecule])
    counts["dropped_duplicate"] += sum(len(group) - 1 for group in groups)
    return [_merged(group) for group in groups]


def _merged(group: list[Molecule]) -> Molecule:
    if len(group) == 1:
        return group[0]
    best = min(range(len(group)), key=lambda i: (RANK[group[i].status], i))
    merged = ", ".join(_describe(m) for i, m in enumerate(group) if i != best)
    confidence = max((m.confidence or 0.0) for m in group)
    return replace(group[best], confidence=confidence, reasons=_reasons(group[best], f"duplicate merged: {merged}"))


def _describe(molecule: Molecule) -> str:
    where = f" {molecule.table_id} row {molecule.row}" if molecule.table_id is not None else ""
    return f"{molecule.source}{where} ({molecule.status})"


def _key(molecule: Molecule) -> tuple[str | None, str] | None:
    if not molecule.smiles:
        return None
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(molecule.smiles)
        inchikey = Chem.MolToInchiKey(mol) if mol is not None else ""
    if not inchikey:
        return None
    names = _aliases(molecule.label)
    return (min(names) if names else None), inchikey[:CONNECTIVITY_CHARS]


def _without_abstained_smiles(molecule: Molecule) -> Molecule:
    return replace(molecule, smiles=None) if molecule.status == ABSTAINED and molecule.smiles is not None else molecule


def _aliases(label: str | None) -> set[str]:
    """A label's names, compared case-folded: 32b joins aliases with ";" and may write "compound 5a"."""
    names = set()
    for part in str(label or "").split(";"):
        name = _SPACES.sub("", _GENERIC.sub("", part.strip()).strip(" .,:;()[]{}"))
        if name:
            names.add(name.casefold())
    return names
