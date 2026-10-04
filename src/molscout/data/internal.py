"""The private internal set: ground-truth SMILES for six papers and the frozen dev/test split."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from rdkit import Chem, rdBase

GROUND_TRUTH_PATH = Path("data/internal/ground_truth_SMILES_confirmed_1,2,4,6,16,19.csv")
SPLIT_PATH = Path("data/manifests/internal_split.csv")
GROUND_TRUTH_COLUMNS = ("paperID", "name of molecule", "canonical_SMILES")
SPLIT_COLUMNS = ("paperID", "split", "molecules")
SPLITS = ("dev", "test")


@dataclass(frozen=True, slots=True)
class GroundTruthMolecule:
    paper_id: str
    name: str
    smiles: str


@dataclass(frozen=True)
class InternalSplit:
    """Each paper's split (dev or test) and its ground-truth row count."""

    split_of: Mapping[str, str]
    molecules: Mapping[str, int]

    def papers(self, split: str) -> frozenset[str]:
        """Papers in "dev", "test" or "all"."""
        if split == "all":
            return frozenset(self.split_of)
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected dev, test or all")
        return frozenset(p for p, s in self.split_of.items() if s == split)

    def groups(self) -> dict[str, frozenset[str]]:
        """The three groups internal scores are reported for."""
        return {name: self.papers(name) for name in (*SPLITS, "all")}


def load_internal_ground_truth(path: str | Path = GROUND_TRUTH_PATH) -> tuple[GroundTruthMolecule, ...]:
    """Read the ground-truth CSV; every SMILES must parse, `*` atoms included."""
    errors: list[str] = []
    molecules: list[GroundTruthMolecule] = []
    for line, row in _read_csv(path, GROUND_TRUTH_COLUMNS):
        paper_id, name, smiles = (row[column].strip() for column in GROUND_TRUTH_COLUMNS)
        if not (paper_id and name and smiles):
            errors.append(f"line {line}: empty field")
        elif not _parses(smiles):
            errors.append(f"line {line}: RDKit cannot parse {smiles!r}")
        else:
            molecules.append(GroundTruthMolecule(paper_id, name, smiles))
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))
    return tuple(molecules)


def references_by_paper(molecules: Iterable[GroundTruthMolecule]) -> dict[str, tuple[str, ...]]:
    """Paper ID to its ground-truth SMILES, in file order."""
    grouped: dict[str, list[str]] = {}
    for molecule in molecules:
        grouped.setdefault(molecule.paper_id, []).append(molecule.smiles)
    return {paper: tuple(smiles) for paper, smiles in grouped.items()}


def load_internal_split(path: str | Path = SPLIT_PATH) -> InternalSplit:
    """Read internal_split.csv: one row per paper with its split and molecule count."""
    errors: list[str] = []
    split_of: dict[str, str] = {}
    molecules: dict[str, int] = {}
    for line, row in _read_csv(path, SPLIT_COLUMNS):
        paper, split, count = (row[column].strip() for column in SPLIT_COLUMNS)
        row_errors = []
        if not paper:
            row_errors.append(f"line {line}: paperID is empty")
        if paper in split_of:
            row_errors.append(f"line {line}: paper {paper} listed twice")
        if split not in SPLITS:
            row_errors.append(f"line {line}: split must be dev or test, got {split!r}")
        if not count.isdigit() or int(count) < 1:
            row_errors.append(f"line {line}: molecules must be a positive integer, got {count!r}")
        if row_errors:
            errors.extend(row_errors)
            continue
        split_of[paper] = split
        molecules[paper] = int(count)
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))
    if not split_of:
        raise ValueError(f"{path}: no papers")
    return InternalSplit(MappingProxyType(split_of), MappingProxyType(molecules))


def check_split_counts(molecules: Iterable[GroundTruthMolecule], split: InternalSplit) -> None:
    """Raise ValueError unless the ground truth has exactly the split's papers and row counts."""
    found = dict(Counter(m.paper_id for m in molecules))
    expected = dict(split.molecules)
    if found != expected:
        raise ValueError(f"ground-truth rows per paper {found} do not match the split manifest {expected}")


def _read_csv(path: str | Path, columns: tuple[str, ...]) -> list[tuple[int, dict[str, str]]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != columns:
            raise ValueError(f"{path}: expected columns {', '.join(columns)}; found {reader.fieldnames}")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}: line {reader.line_num}: expected {len(columns)} fields")
            rows.append((reader.line_num, row))
    return rows


def _parses(smiles: str) -> bool:
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    return mol is not None and mol.GetNumAtoms() > 0
