"""Reference molecules for crop datasets, read from molfiles and SDfiles.

USPTO, UOB and CLEF ship one molfile per crop; JPO ships one SDfile per crop.
The crop ID is the file name without its extension, which the image shares.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem, rdBase

from molscout.hashing import sha256_file

MOLFILE_SUFFIXES = frozenset({".mol"})
SDF_SUFFIXES = frozenset({".sdf", ".sd"})


@dataclass(frozen=True, slots=True)
class Reference:
    """A crop's reference molecule as SMILES, or the reason RDKit could not read it."""

    item_id: str
    smiles: str | None
    error: str | None = None


def read_molfile(path: str | Path) -> Reference:
    """Read one molfile with RDKit defaults (sanitized, hydrogens removed, stereo from wedges)."""
    path = Path(path)
    with rdBase.BlockLogs():
        mol = Chem.MolFromMolFile(str(path))
    return _reference(path, mol, _read_molfile_unsanitized)


def read_sdfile(path: str | Path) -> Reference:
    """Read an SDfile holding exactly one record."""
    path = Path(path)
    try:
        with rdBase.BlockLogs():
            records = list(Chem.SDMolSupplier(str(path)))
    except OSError:  # RDKit refuses an empty file outright
        return Reference(path.stem, None, "RDKit cannot parse the file")
    if len(records) != 1:
        return Reference(path.stem, None, f"expected 1 record, found {len(records)}")
    return _reference(path, records[0], _read_sdfile_unsanitized)


def load_references(directory: str | Path) -> dict[str, Reference]:
    """Read every .mol and .sdf file in a directory, keyed by crop ID."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"reference directory not found: {directory}")
    references: dict[str, Reference] = {}
    for path in reference_files(directory):
        reference = read_molfile(path) if path.suffix.lower() in MOLFILE_SUFFIXES else read_sdfile(path)
        if reference.item_id in references:
            raise ValueError(f"two reference files for crop {reference.item_id!r} in {directory}")
        references[reference.item_id] = reference
    if not references:
        raise ValueError(f"no .mol or .sdf files in {directory}")
    return references


def reference_files(directory: Path) -> list[Path]:
    """The .mol and .sdf files in a directory, sorted by name; hidden files are skipped."""
    return [
        path
        for path in sorted(directory.iterdir())
        if not path.name.startswith(".") and path.suffix.lower() in MOLFILE_SUFFIXES | SDF_SUFFIXES
    ]


def reference_set_sha256(directory: str | Path) -> str:
    """One checksum for a reference set: sha256 of `<sha256>  <name>` lines, as sha256sum prints them."""
    lines = "".join(f"{sha256_file(path)}  {path.name}\n" for path in reference_files(Path(directory)))
    return hashlib.sha256(lines.encode()).hexdigest()


def _reference(path: Path, mol: Chem.Mol | None, unsanitized: Callable[[Path], Chem.Mol | None]) -> Reference:
    if mol is None:
        return Reference(path.stem, None, sanitize_error(unsanitized(path)))
    if mol.GetNumAtoms() == 0:
        return Reference(path.stem, None, "file has no atoms")
    return Reference(path.stem, Chem.MolToSmiles(mol))


def _read_molfile_unsanitized(path: Path) -> Chem.Mol | None:
    with rdBase.BlockLogs():
        return Chem.MolFromMolFile(str(path), sanitize=False)


def _read_sdfile_unsanitized(path: Path) -> Chem.Mol | None:
    with rdBase.BlockLogs():
        return next(iter(Chem.SDMolSupplier(str(path), sanitize=False)), None)


def sanitize_error(mol: Chem.Mol | None) -> str:
    """RDKit's reason for rejecting a molecule that failed the default read."""
    if mol is None:
        return "RDKit cannot parse the file"
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(mol)
    except Chem.rdchem.MolSanitizeException as exc:
        return str(exc)
    return "RDKit rejected the molecule"
