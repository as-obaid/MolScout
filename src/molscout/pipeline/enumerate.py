"""Check a Markush core against an R-group table and build each row's molecule.

Slots are dummy atoms with atom-map numbers, `[*:n]`. Slot n stands for `variables[n - 1]`,
the table's n-th variable column. A reader labels its slots in `Read.slots` ("7" -> "R1");
`align_slots` renumbers a core to that convention by matching labels to column headers, so
a label drawn at two places becomes two slots with one number.

`build` fills every slot with its fragment:

- substituent: joined with RDKit molzip on a slot with one neighbour;
- linker: on a slot with two neighbours; its first `*` (in SMILES order) bonds to the
  neighbour that comes first in the core's atom order, its second to the other;
- atom: the slot atom itself becomes the fragment's first atom (aromatic if the slot was),
  keeping all its bonds; the fragment's other atoms hang off it.

A slot drawn twice gets its own copy of the fragment at each place.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from rdkit import Chem, rdBase

from molscout.pipeline.rgroups import Fragment
from molscout.scoring.smiles import canonical_smiles

_PRIMES = dict.fromkeys(map(ord, "′’‘`´"), "'")
_LABEL_MARKUP = re.compile(r"</?su[bp]>|[\s${}^_]")


@dataclass(frozen=True)
class CoreCheck:
    """Slots found in a core versus the table's variables (numbered 1..len(variables))."""

    ok: bool
    slots: set[int]
    missing: set[int]
    extra: set[int]


def check_core(core_smiles: str, variables: list[str], slots: dict[str, str] | None = None) -> CoreCheck:
    """Compare the core's slots with the table's variable columns.

    Without `slots` the core's numbers are taken as aligned already (slot n is
    `variables[n - 1]`). With the reader's `slots` map the core is aligned first, so a
    label drawn twice counts once. `missing` are variables with no slot in the core,
    `extra` are slots no variable fills (an unlabelled `*` without a map counts as slot 0).
    An unparseable core misses every variable.
    """
    expected = set(range(1, len(variables) + 1))
    aligned = core_smiles if slots is None else align_slots(core_smiles, slots, variables)
    core = _parse(aligned) if aligned is not None else None
    if core is None:
        return CoreCheck(False, set(), expected, set())
    found = {atom.GetAtomMapNum() for atom in core.GetAtoms() if atom.GetAtomicNum() == 0}
    missing, extra = expected - found, found - expected
    return CoreCheck(not missing and not extra, found, missing, extra)


def align_slots(core_smiles: str, slots: dict[str, str], variables: list[str]) -> str | None:
    """Renumber a reader's core so every slot labelled `variables[i]` becomes `[*:i+1]`.

    Labels match column headers by `label_key` ("R¹" = "R<sup>1</sup>" = "r1"); a label drawn
    twice gives two slots with the same number, which `build` fills with one copy each. A
    label that is not a variable gets one number above len(variables), shared by every slot
    with that label, so `check_core` reports it as extra once; each unlabelled slot ("*" or
    no label) gets its own.
    """
    core = _parse(core_smiles)
    if core is None:
        return None
    columns = {label_key(name): number for number, name in enumerate(variables, start=1)}
    unknown: dict[str, int] = {}
    for atom in core.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        old = atom.GetAtomMapNum()
        key = label_key(slots.get(str(old), ""))
        number = columns.get(key)
        if number is None:
            own = key if key not in ("", "*") else f"*{old}:{atom.GetIdx()}"
            number = unknown.setdefault(own, len(variables) + len(unknown) + 1)
        atom.SetAtomMapNum(number)
    return Chem.MolToSmiles(core)


def build(core_smiles: str, assignment: dict[int, Fragment]) -> str | None:
    """Canonical SMILES of the core with every slot filled, or None.

    None when the core does not parse, a slot has no fragment or a fragment no slot, a
    fragment does not fit its slot's neighbour count, or the result fails RDKit
    sanitization. The result never contains dummy atoms or atom maps.
    """
    core = _parse(core_smiles)
    if core is None:
        return None
    dummies = [atom.GetIdx() for atom in core.GetAtoms() if atom.GetAtomicNum() == 0]
    if {core.GetAtomWithIdx(index).GetAtomMapNum() for index in dummies} != set(assignment):
        return None
    editable = Chem.RWMol(core)
    for atom in editable.GetAtoms():
        if atom.GetAtomicNum() != 0:
            atom.SetAtomMapNum(0)
    pieces: list[Chem.Mol] = []
    for index in dummies:
        fragment = assignment[editable.GetAtomWithIdx(index).GetAtomMapNum()]
        if not _fill(editable, index, fragment, pieces):
            return None
    return _finish(editable, pieces)


def _fill(editable: Chem.RWMol, index: int, fragment: Fragment, pieces: list[Chem.Mol]) -> bool:
    """Put one fragment at one slot; molzip pieces are collected for the final join."""
    degree = editable.GetAtomWithIdx(index).GetDegree()
    if fragment.kind == "atom":
        _replace_atom(editable, index, fragment.smiles)
        return True
    if fragment.kind == "substituent" and degree == 1:
        label = _next_label(editable, pieces)
        editable.GetAtomWithIdx(index).SetAtomMapNum(label)
        pieces.append(_labelled(fragment.smiles, [label]))
        return True
    if fragment.kind == "linker" and degree == 2:
        near, far = sorted(neighbour.GetIdx() for neighbour in editable.GetAtomWithIdx(index).GetNeighbors())
        second = editable.AddAtom(Chem.Atom(0))
        _move_bond(editable, far, index, second)
        first_label = _next_label(editable, pieces)
        editable.GetAtomWithIdx(index).SetAtomMapNum(first_label)
        editable.GetAtomWithIdx(second).SetAtomMapNum(first_label + 1)
        pieces.append(_labelled(fragment.smiles, [first_label, first_label + 1]))
        return True
    return False


def _replace_atom(editable: Chem.RWMol, index: int, smiles: str) -> None:
    fragment = _parse(smiles)
    first = fragment.GetAtomWithIdx(0)
    atom = Chem.Atom(first.GetAtomicNum())
    atom.SetFormalCharge(first.GetFormalCharge())
    if first.GetNoImplicit():
        atom.SetNumExplicitHs(first.GetNumExplicitHs())
        atom.SetNoImplicit(True)
    atom.SetIsAromatic(editable.GetAtomWithIdx(index).GetIsAromatic())
    editable.ReplaceAtom(index, atom)
    placed = {0: index}
    for other in list(fragment.GetAtoms())[1:]:
        placed[other.GetIdx()] = editable.AddAtom(Chem.Atom(other))
    for bond in fragment.GetBonds():
        editable.AddBond(placed[bond.GetBeginAtomIdx()], placed[bond.GetEndAtomIdx()], bond.GetBondType())


def _move_bond(editable: Chem.RWMol, atom_index: int, old: int, new: int) -> None:
    """Bond `atom_index` to `new` instead of `old`, keeping its tetrahedral configuration."""
    before = _neighbour_order(editable, atom_index)
    bond_type = editable.GetBondBetweenAtoms(atom_index, old).GetBondType()
    editable.RemoveBond(atom_index, old)
    editable.AddBond(atom_index, new, bond_type)
    expected = [new if neighbour == old else neighbour for neighbour in before]
    if _is_odd_permutation(expected, _neighbour_order(editable, atom_index)):
        editable.GetAtomWithIdx(atom_index).InvertChirality()


def _neighbour_order(mol: Chem.Mol, atom_index: int) -> list[int]:
    return [bond.GetOtherAtomIdx(atom_index) for bond in mol.GetAtomWithIdx(atom_index).GetBonds()]


def _is_odd_permutation(reference: list[int], order: list[int]) -> bool:
    positions = [reference.index(item) for item in order]
    inversions = sum(1 for i, left in enumerate(positions) for right in positions[i + 1 :] if left > right)
    return inversions % 2 == 1


def _next_label(editable: Chem.RWMol, pieces: list[Chem.Mol]) -> int:
    used = [atom.GetAtomMapNum() for atom in editable.GetAtoms()]
    used += [atom.GetAtomMapNum() for piece in pieces for atom in piece.GetAtoms()]
    return max(used, default=0) + 1


def _labelled(smiles: str, labels: list[int]) -> Chem.Mol:
    """The fragment with its dummies numbered `labels` in SMILES order."""
    fragment = _parse(smiles)
    dummies = [atom for atom in fragment.GetAtoms() if atom.GetAtomicNum() == 0]
    for atom in fragment.GetAtoms():
        atom.SetAtomMapNum(0)
    for atom, label in zip(dummies, labels, strict=True):
        atom.SetAtomMapNum(label)
    return fragment


def _finish(editable: Chem.RWMol, pieces: list[Chem.Mol]) -> str | None:
    mol = editable.GetMol()
    for piece in pieces:
        mol = Chem.CombineMols(mol, piece)
    with rdBase.BlockLogs():
        if pieces:
            mol = Chem.molzip(mol)
        if Chem.SanitizeMol(mol, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
            return None
        mol = Chem.RemoveHs(mol)
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return canonical_smiles(Chem.MolToSmiles(mol))


def label_key(label: str) -> str:
    """A drawn label or column header in one spelling: NFKC, primes as `'`, no spaces,
    `<sup>`/`<sub>` tags or TeX marks, case folded ("R<sup>1</sup>" and "R¹" give "r1")."""
    text = unicodedata.normalize("NFKC", label).translate(_PRIMES)
    return _LABEL_MARKUP.sub("", text).casefold()


def _parse(smiles: str) -> Chem.Mol | None:
    with rdBase.BlockLogs():
        return Chem.MolFromSmiles(smiles)
