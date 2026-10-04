"""Canonical SMILES and exact match under the pinned RDKit.

Every SMILES, predicted or reference, goes through the same functions, so a
prediction matches exactly when RDKit's default canonical strings are identical.
"""

from __future__ import annotations

from rdkit import Chem, rdBase

RDKIT_VERSION = "2026.03.2"

# By default RDKit reads text after whitespace as a molecule name, so "CCO CCN" would parse
# as CCO. Treat that text as part of the output instead; CXSMILES extensions still parse.
_PARSER = Chem.SmilesParserParams()
_PARSER.parseName = False


def check_rdkit_version() -> None:
    """Raise RuntimeError unless RDKit is the version every score is computed with."""
    if rdBase.rdkitVersion != RDKIT_VERSION:
        raise RuntimeError(
            f"scoring needs RDKit {RDKIT_VERSION}, found {rdBase.rdkitVersion}; "
            "canonical SMILES can differ between versions"
        )


def canonical_smiles(smiles: str) -> str | None:
    """RDKit default canonical SMILES; None if empty, whitespace or unparsable. `*` is valid."""
    mol = _parse(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def stereo_stripped_smiles(smiles: str) -> str | None:
    """Canonical SMILES without tetrahedral or double-bond stereo; isotopes and labels kept."""
    mol = _parse(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def is_exact_match(predicted: str, reference: str, *, stereo: bool = True) -> bool:
    """True when both parse and their canonical forms are identical."""
    key = canonical_smiles if stereo else stereo_stripped_smiles
    left = key(predicted)
    return left is not None and left == key(reference)


def _parse(smiles: str) -> Chem.Mol | None:
    text = smiles.strip()
    if not text:
        return None
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(text, _PARSER)
    # MolFromSmiles can return a zero-atom molecule; that is no output, not a molecule.
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return mol
