"""MolRecBench-Wild reference SMILES, built from its CARBON graph labels.

The conversion follows the official evaluator's SMILES track (github.com/opendatalab/
MolRecBench-Wild, commit 500da87d767ba8ea48b1fec5f765a01e8c9a2394, `evaluate/utils.py` and
`evaluate/smiles_metric.py`), adapted under the Apache License 2.0; a copy of the license is
in `carbon_templates.LICENSE`. `carbon_templates.json.gz` is that commit's `ABBR2MOLBLOCKS`
table of drawn abbreviations.

1. `exclusion_reasons`: graphs SMILES cannot express are left out.
2. `graph_smiles`: the graph becomes a molblock with its 2D coordinates and wedge bonds,
   with labels carried as technetium isotopes, and then SMILES with the labels in brackets.
3. `expand_abbreviations`: labels with a template (`[Me]`, `[CO2Et]`) are replaced by their
   atoms. A label with no template excludes the graph unless the evaluator keeps it as a
   token (`R1`, `X`, `Ar` and others).

MolScout adds one step, `label_atoms`: R-group and variable labels become `*` atoms, numbered
for `R1`, `R2`, so the reference parses as SMILES; a graph with any other label left is left out.
"""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Mapping, Sequence
from functools import cache
from importlib import resources
from typing import Any

from rdkit import Chem, rdBase
from rdkit.Geometry import Point3D

from molscout.data.molfiles import Reference, sanitize_error

TEMPLATES_FILE = "carbon_templates.json.gz"

GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")
# Element symbols that MolRecBench-Wild drawings use as labels (Ar is aryl, Ts tosyl, Ac acetyl).
LOOKALIKE_LABELS = frozenset({"Ar", "Ts", "Ac", "D", "Np", "Sn", "Mo", "W"})
# Labels the evaluator keeps as tokens instead of excluding the graph (EXP_SKIP_SYMBOLS).
KEPT_LABELS = frozenset(
    {
        "Rα", "Rβ", "Rδ", "Rγ", "Rε", "R'", "*C", "Alkyl", "R2X", "Ar", "X", "TBETOf", "CEB",
        "OPA", "E", "*", "F5", "_AP1", "?", "_", "R_", "DG", "R1", "R2", "M", "Ha", "Hb", "Hc",
        "Hd", "AF", "FA", "Y", "Y1", "L", "BO", "Z", "PEG",
    }
)  # fmt: skip
NUMBERED_R_GROUP = re.compile(r"R([1-9][0-9]*)")
VARIABLE_LABEL = re.compile(r"R|R[0-9]*'+|R[a-z_]|[XYZLMEQAG][0-9]*'*|Ar|Alkyl|\*")
BRACKET_TOKEN = re.compile(r"\[([^\[\]]+)\]")

# CARBON bond types: 1-3 single to triple, 4 aromatic, 5 solid wedge, 6 dashed wedge, and
# 7-23 drawing variants of single, double and triple bonds.
_BOND_ORDER = {
    **dict.fromkeys((1, 7, 8, 11, 12, 13, 15, 16, 17, 21, 23), 1),
    **dict.fromkeys((2, 9, 10, 14, 18, 19), 2),
    **dict.fromkeys((3, 20, 22), 3),
    4: 4,
    5: 5,
    6: 6,
}
_RDKIT_BOND = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
    5: Chem.BondType.SINGLE,
    6: Chem.BondType.SINGLE,
}
_CONVERSION_ERRORS = (ValueError, RuntimeError, IndexError, KeyError, TypeError)


@cache
def load_templates() -> dict[str, dict[str, str]]:
    """The evaluator's abbreviation table: label -> {number of attachment bonds: molblock}."""
    data = resources.files("molscout.data").joinpath(TEMPLATES_FILE).read_bytes()
    return json.loads(gzip.decompress(data))


def exclusion_reasons(record: Mapping[str, Any]) -> list[str]:
    """Why a graph cannot be scored as SMILES, in the evaluator's order; empty if it can."""
    symbols = [str(symbol) for symbol in record.get("symbols") or []]
    reasons = []
    if any(GREEK.search(symbol) for symbol in symbols):
        reasons.append("Greek letter in a label")
    if any("?" in symbol or "？" in symbol for symbol in symbols):
        reasons.append("question mark in a label")
    if _is_nonempty(record.get("brackets")):
        reasons.append("repeat-unit brackets")
    if _has_drawing_bond(record.get("bonds")):
        reasons.append("drawing-specific bond type above 6")
    if _is_nonempty(record.get("attach_points")):
        reasons.append("attachment points")
    return reasons


def graph_smiles(record: Mapping[str, Any]) -> str:
    """SMILES for the graph as drawn, with labels kept in brackets (`CC[Me]`)."""
    symbols = record.get("symbols")
    bonds = record.get("bonds")
    if not isinstance(symbols, list) or not isinstance(bonds, list):
        raise ValueError("a CARBON record needs list fields 'symbols' and 'bonds'")
    coords = record.get("coords")
    if not isinstance(coords, list):
        coords = [[0.0, 0.0] for _ in symbols]
    with rdBase.BlockLogs():
        molblock, label_isotopes = _molblock(record, symbols, coords, _simplify_bonds(bonds))
        mol = Chem.MolFromMolBlock(molblock, sanitize=False)
    if mol is None:
        raise ValueError("RDKit cannot read the molblock built from the graph")
    smiles = _to_smiles(mol)
    for label, isotope in label_isotopes.items():
        smiles = smiles.replace(f"{isotope}Tc", label)
    return smiles


def expand_abbreviations(smiles: str, templates: Mapping[str, Mapping[str, str]]) -> tuple[str, list[str]]:
    """Replace each label that has a template by its atoms.

    Returns the expanded SMILES and the labels that excluded the graph: labels with no
    template for their number of bonds that the evaluator does not keep as tokens.
    """
    marked, placeholders = _mark_labels(smiles)
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(marked, sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit cannot parse {smiles!r}")
    rw = Chem.RWMol(mol)
    missing = []
    labelled = [
        (atom.GetIdx(), placeholders[key])
        for atom in rw.GetAtoms()
        if (key := f"{atom.GetIsotope()}{atom.GetSymbol()}") in placeholders
    ]
    for index, label in labelled:
        variants = templates.get(label)
        if variants is None:
            if label not in KEPT_LABELS:
                missing.append(label)
            continue
        if not _expand_one(rw, index, variants):
            missing.append(label)
    mol = rw.GetMol()
    with rdBase.BlockLogs():
        Chem.SanitizeMol(mol, catchErrors=True)
    expanded = _to_smiles(mol)
    for key, label in placeholders.items():
        expanded = expanded.replace(key, label)
    return expanded, missing


def label_atoms(smiles: str) -> tuple[str, list[str]]:
    """Turn R-group and variable labels into `*` atoms; also return the labels left unknown.

    `[R1]` becomes `[1*]`, and `[X]`, `[R']`, `[Ar]` become `*`. Bracket atoms stay as they are.
    """
    unknown: list[str] = []

    def replace(match: re.Match[str]) -> str:
        inner = match.group(1)
        if inner not in LOOKALIKE_LABELS and _is_atom(inner):
            return match.group(0)
        if number := NUMBERED_R_GROUP.fullmatch(inner):
            return f"[{number.group(1)}*]"
        if VARIABLE_LABEL.fullmatch(inner):
            return "*"
        unknown.append(inner)
        return match.group(0)

    return BRACKET_TOKEN.sub(replace, smiles), unknown


def carbon_reference(
    item_id: str, record: Mapping[str, Any], templates: Mapping[str, Mapping[str, str]] | None = None
) -> Reference:
    """The crop's reference SMILES, or the reason the crop is left out."""
    reasons = exclusion_reasons(record)
    if reasons:
        return Reference(item_id, None, "; ".join(reasons))
    try:
        raw = graph_smiles(record)
        if not raw.strip():
            raise ValueError("the graph gave an empty SMILES")
        expanded, missing = expand_abbreviations(raw, load_templates() if templates is None else templates)
    except _CONVERSION_ERRORS as exc:
        return Reference(item_id, None, f"graph could not be converted: {exc}")
    if missing:
        return Reference(item_id, None, f"abbreviation(s) with no template: {', '.join(missing)}")
    smiles, unknown = label_atoms(expanded)
    if unknown:
        return Reference(item_id, None, f"label(s) with no SMILES form: {', '.join(unknown)}")
    with rdBase.BlockLogs():
        if Chem.MolFromSmiles(smiles) is None:
            return Reference(item_id, None, sanitize_error(Chem.MolFromSmiles(smiles, sanitize=False)))
    # Kept as written, not canonicalized: the scorer canonicalizes references and predictions
    # once each, and a second pass can change the text (an aromatic ring with `*` atoms).
    return Reference(item_id, smiles)


def _molblock(
    record: Mapping[str, Any], symbols: Sequence[Any], coords: Sequence[Sequence[float]], bonds: list[list[int]]
) -> tuple[str, dict[str, int]]:
    """The graph as a V2000 molblock; labels become Tc atoms with isotopes from 41 up."""
    rw = Chem.RWMol()
    label_isotopes: dict[str, int] = {}
    charges, radicals, isotopes = record.get("charges"), record.get("radicals"), record.get("isotopes")
    for index, original in enumerate(symbols):
        symbol = str(original).replace("[", "").replace("]", "")
        atom = None
        if symbol not in LOOKALIKE_LABELS:
            atom = Chem.AtomFromSmiles(f"[{symbol}]" if symbol in {"Fe", "H"} else symbol)
        if atom is None:
            atom = Chem.Atom("Tc")
            atom.SetIsotope(label_isotopes.setdefault(symbol, 41 + len(label_isotopes)))
        if charge := int(_value(charges, index, 0)):
            atom.SetFormalCharge(charge)
        if (isotope := _value(isotopes, index, None)) is not None:
            atom.SetIsotope(int(isotope))
        radical = _value(radicals, index, None)
        if radical in {1, 3}:
            atom.SetNumRadicalElectrons(2)
        elif radical == 2:
            atom.SetNumRadicalElectrons(1)
        atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        rw.AddAtom(atom)
    for begin, end, kind in bonds:
        if kind not in _RDKIT_BOND:
            raise ValueError(f"unsupported bond type {kind}")
        rw.AddBond(begin, end, _RDKIT_BOND[kind])
        if kind == 5:
            rw.GetBondBetweenAtoms(begin, end).SetBondDir(Chem.BondDir.BEGINWEDGE)
        elif kind == 6:
            rw.GetBondBetweenAtoms(begin, end).SetBondDir(Chem.BondDir.BEGINDASH)
    mol = rw.GetMol()
    if coords:
        conformer = Chem.Conformer(len(symbols))
        conformer.Set3D(False)
        for index, point in enumerate(coords):
            x = float(point[0]) if len(point) > 0 else 0.0
            y = float(point[1]) if len(point) > 1 else 0.0
            conformer.SetAtomPosition(index, Point3D(x, y, 0.0))
        mol.AddConformer(conformer)
    return Chem.MolToMolBlock(mol, kekulize=False), label_isotopes


def _expand_one(rw: Chem.RWMol, index: int, variants: Mapping[str, str]) -> bool:
    """Put a template's atoms in place of the label atom at `index`; False if none fits.

    The template is chosen by the sum of the distinct bond types to the label, as the
    evaluator does. The first attachment point takes the label's place; further neighbours
    move to further attachment points.
    """
    neighbours = [neighbour.GetIdx() for neighbour in rw.GetAtomWithIdx(index).GetNeighbors()]
    bond_types = {n: rw.GetBondBetweenAtoms(index, n).GetBondType() for n in neighbours}
    key = str(int(sum(set(bond_types.values()))))
    if key not in variants:
        return False
    molblock = variants[key]
    with rdBase.BlockLogs():
        fragment = Chem.MolFromMolBlock(molblock, removeHs=False)
    attach = _attachment_points(molblock)
    if fragment is None or not attach:
        return True  # the evaluator keeps the label as it is
    source = fragment.GetAtomWithIdx(attach[0])
    replacement = Chem.Atom(source.GetAtomicNum())
    replacement.SetFormalCharge(source.GetFormalCharge())
    rw.ReplaceAtom(index, replacement)
    atom_map = {attach[0]: index}
    for source_index in range(fragment.GetNumAtoms()):
        if source_index != attach[0]:
            atom_map[source_index] = rw.AddAtom(fragment.GetAtomWithIdx(source_index))
    for bond in fragment.GetBonds():
        begin, end = atom_map[bond.GetBeginAtomIdx()], atom_map[bond.GetEndAtomIdx()]
        if rw.GetBondBetweenAtoms(begin, end) is None:
            rw.AddBond(begin, end, bond.GetBondType())
    for position in range(1, min(len(neighbours), len(attach))):
        neighbour = neighbours[position]
        if rw.GetBondBetweenAtoms(index, neighbour) is not None:
            rw.RemoveBond(index, neighbour)
            target = atom_map[attach[position]]
            if rw.GetBondBetweenAtoms(target, neighbour) is None:
                rw.AddBond(target, neighbour, bond_types[neighbour])
    return True


def _attachment_points(molblock: str) -> list[int]:
    """Atom indices from the molblock's `M  APO` lines; an atom with both points appears twice."""
    points: list[int] = []
    for line in molblock.splitlines():
        if not line.startswith("M  APO"):
            continue
        parts = line.split()
        offset = 3
        for _ in range(int(parts[2])):
            atom = int(parts[offset]) - 1
            points.append(atom)
            if int(parts[offset + 1]) == 3:
                points.append(atom)
            offset += 2
    return points


def _mark_labels(smiles: str) -> tuple[str, dict[str, str]]:
    """Replace each bracketed label with a Tc isotope from 20 up; return the placeholders."""
    placeholders: dict[str, str] = {}
    isotope = 20
    for label in set(_outer_brackets(smiles)):
        if label not in LOOKALIKE_LABELS and _is_atom(label):
            continue
        placeholders[f"{isotope}Tc"] = label
        smiles = smiles.replace(f"[{label}]", f"[{isotope}Tc]")
        isotope += 1
    return smiles, placeholders


def _outer_brackets(smiles: str) -> list[str]:
    stack: list[int] = []
    found = []
    for index, char in enumerate(smiles):
        if char == "[":
            stack.append(index)
        elif char == "]" and stack:
            start = stack.pop()
            if not stack:
                found.append(smiles[start + 1 : index])
    return found


def _is_atom(text: str) -> bool:
    with rdBase.BlockLogs():
        return Chem.AtomFromSmiles(f"[{text}]") is not None


def _to_smiles(mol: Chem.Mol) -> str:
    with rdBase.BlockLogs():
        try:
            return Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=True)
        except (RuntimeError, ValueError):  # ValueError covers RDKit's KekulizeException
            return Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=False)


def _simplify_bonds(bonds: Sequence[Sequence[Any]]) -> list[list[int]]:
    simplified = []
    for bond in bonds:
        if len(bond) < 3:
            raise ValueError(f"bond entry needs three values: {bond!r}")
        begin, end, kind = (int(value) for value in bond[:3])
        if kind in _BOND_ORDER:
            simplified.append([begin, end, _BOND_ORDER[kind]])
    return simplified


def _has_drawing_bond(bonds: Any) -> bool:
    if not isinstance(bonds, (list, tuple)):
        return False
    for bond in bonds:
        if isinstance(bond, (list, tuple)) and len(bond) >= 3:
            try:
                if float(bond[2]) > 6:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def _is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return any(_is_nonempty(item) for item in value)
    if isinstance(value, dict):
        return bool(value)
    return value not in ("", 0)


def _value(values: Sequence[Any] | None, index: int, default: Any) -> Any:
    if values is None or index >= len(values) or values[index] is None:
        return default
    return values[index]
