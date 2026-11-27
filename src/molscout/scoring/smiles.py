"""Canonical SMILES and exact match under the pinned RDKit.

Every SMILES, predicted or reference, goes through the same functions, so a
prediction matches exactly when RDKit's default canonical strings are identical.

RDKit's canonical SMILES is not always stable: reading it back and writing it again can
change the text, as with aromatic rings that contain `*` atoms or an explicit [H] left by
stripping stereo. The canonical form is therefore repeated until the string stops changing,
so a prediction already in RDKit's canonical form and a raw reference reach the same string.
"""

from __future__ import annotations

from rdkit import Chem, rdBase

RDKIT_VERSION = "2026.03.2"

# Read-and-write passes at most; every public and internal reference is stable within two.
MAX_PASSES = 5

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
    """Stable RDKit default canonical SMILES; None if empty, whitespace or unparsable. `*` is valid."""
    return _stable(smiles, stereo=True)


def stereo_stripped_smiles(smiles: str) -> str | None:
    """Stable canonical SMILES without tetrahedral or double-bond stereo; isotopes and labels kept."""
    return _stable(smiles, stereo=False)


def is_exact_match(predicted: str, reference: str, *, stereo: bool = True) -> bool:
    """True when both parse and their canonical forms are identical."""
    key = canonical_smiles if stereo else stereo_stripped_smiles
    left = key(predicted)
    return left is not None and left == key(reference)


def _stable(smiles: str, *, stereo: bool) -> str | None:
    # Whether the input is valid is decided by the first pass alone. Later passes stop when a pass
    # returns the string it read; a string that cannot be read back, or MAX_PASSES, ends them and
    # the last string written is kept.
    current = _write(smiles, stereo=stereo)
    if current is None:
        return None
    for _ in range(MAX_PASSES - 1):
        following = _write(current, stereo=stereo)
        if following is None or following == current:
            break
        current = following
    return current


def _write(smiles: str, *, stereo: bool) -> str | None:
    mol = _parse(smiles)
    if mol is None:
        return None
    if not stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


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
