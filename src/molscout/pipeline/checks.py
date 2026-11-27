"""Reject junk structures and compare molecules with the formulas a paper prints.

`junk_reason` names the first problem it finds, in this order: invalid SMILES, a leftover
dummy or R token, two or more fragments that are not common salts or solvents, a radical,
fewer than 5 heavy atoms, an acyclic all-carbon chain of 40 or more atoms. Salts and
solvents (HCl, TFA, Na+, water, DMSO and the others in SALTS_AND_SOLVENTS) are not junk:
`strip_salts` gives the parent fragment to keep.

`printed_formulas` reads formulas stated for a compound ("HRMS (ESI) calcd for C19H21N6
[M + H]+ 333.1822") and turns adduct ion formulas back into neutral ones.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from molscout.pipeline.schema import LayoutBlock
from molscout.scoring.smiles import canonical_smiles, stereo_stripped_smiles

MIN_HEAVY_ATOMS = 5
MAX_CARBON_CHAIN = 40

# Counter-ions, acids, bases and solvents dropped from multi-fragment SMILES (compared without stereo).
SALTS_AND_SOLVENTS = (
    "F", "Cl", "Br", "I", "[F-]", "[Cl-]", "[Br-]", "[I-]",
    "[Li+]", "[Na+]", "[K+]", "[Mg+2]", "[Ca+2]", "[Zn+2]", "[NH4+]", "N", "O", "[OH-]",
    "OC(=O)C(F)(F)F", "[O-]C(=O)C(F)(F)F", "OC=O", "[O-]C=O", "CC(=O)O", "CC(=O)[O-]",
    "CS(=O)(=O)O", "CS(=O)(=O)[O-]", "OS(=O)(=O)c1ccccc1", "[O-]S(=O)(=O)c1ccccc1",
    "Cc1ccc(S(=O)(=O)O)cc1", "Cc1ccc(S(=O)(=O)[O-])cc1", "OS(=O)(=O)O", "OS(=O)(=O)[O-]",
    "[O-]S(=O)(=O)[O-]", "OP(=O)(O)O", "O=[N+]([O-])O", "O=[N+]([O-])[O-]",
    "OC(=O)C=CC(=O)O", "OC(=O)C(=O)O", "OC(=O)CCC(=O)O", "OC(=O)C(O)C(O)C(=O)O",
    "OC(=O)CC(O)(CC(=O)O)C(=O)O", "CCN(CC)CC", "CC[NH+](CC)CC",
    "CO", "CCO", "CC(C)O", "CC#N", "CS(C)=O", "CN(C)C=O", "CC(C)=O", "ClCCl", "ClC(Cl)Cl",
    "CCOC(C)=O", "CCOCC", "C1CCOC1", "C1COCCO1", "Cc1ccccc1",
)  # fmt: skip

_R_TOKEN = re.compile(r"\*|\[[RXZQAG][\d']*\]")
_LONG_CHAIN = Chem.MolFromSmarts("~".join(["[#6;!R]"] * MAX_CARBON_CHAIN))
_SALT_KEYS = frozenset(stereo_stripped_smiles(smiles) for smiles in SALTS_AND_SOLVENTS)

# "calcd for C19H21N6", "Calcd. for", "calculated for", "calc'd for"; then the formula and an optional charge.
_STATEMENT = re.compile(r"\b[Cc]alc(?:ulated|d\.?|\.|'d)?\s+for\s*:?\s*(?P<formula>C\d*H\d*(?:[A-Z][a-z]?\d*)*)(?P<charge>[+-])?")
_ADDUCT = re.compile(r"[\[(]M(?P<sign>[+-])(?P<species>NH4|Na|Li|K|H)[\])][+-]?|M(?P<sign2>\+)(?P<species2>H)\b")
_MZ = re.compile(r"\b\d{2,4}\.\d{3,5}\b")
_FOUND = re.compile(r"[Ff]ound")
_LABEL = re.compile(r"\((\d{1,3}[a-z]{0,2}'?)\)|\b[Cc]ompound\s+(\d{1,3}[a-z]{0,2}'?)\b|\*\*(\d{1,3}[a-z]{0,2}'?)\*\*")
_ELEMENT_COUNT = re.compile(r"([A-Z][a-z]?)(\d*)")
_SPECIES = {"H": {"H": 1}, "Na": {"Na": 1}, "K": {"K": 1}, "Li": {"Li": 1}, "NH4": {"N": 1, "H": 4}}
_ADDUCT_WINDOW_BEFORE = 30
_AFTER_WINDOW = 60
_MASS_TOLERANCE = 0.02

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−﹣－"), "-")
_LATEX_COMMAND = re.compile(r"\\(?:mathrm|text|rm)")
_LATEX_MARKUP = re.compile(r"[${}_^]")
_PERIODIC_TABLE = Chem.GetPeriodicTable()


def junk_reason(smiles: str) -> str | None:
    """Why a structure is junk, or None: invalid, leftover dummy, multiple fragments,
    radical, too few heavy atoms, long carbon chain."""
    mol = _parse(smiles)
    if mol is None:
        return "leftover dummy" if _R_TOKEN.search(smiles or "") else "invalid"
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        return "leftover dummy"
    parent_smiles = strip_salts(smiles)
    if parent_smiles is None:
        return "multiple fragments"
    parent = _parse(parent_smiles)
    if any(atom.GetNumRadicalElectrons() for atom in parent.GetAtoms()):
        return "radical"
    if parent.GetNumHeavyAtoms() < MIN_HEAVY_ATOMS:
        return "too few heavy atoms"
    if parent.HasSubstructMatch(_LONG_CHAIN):
        return "long carbon chain"
    return None


def strip_salts(smiles: str) -> str | None:
    """Canonical SMILES of the parent fragment once salts and solvents are dropped.

    One fragment is returned as it is. With several, the one fragment not in
    SALTS_AND_SOLVENTS is kept; if all are listed, the largest. None when the SMILES does not
    parse or two or more fragments are not listed.
    """
    mol = _parse(smiles)
    if mol is None:
        return None
    fragments = Chem.GetMolFrags(mol, asMols=True)
    if len(fragments) == 1:
        return canonical_smiles(smiles)
    unlisted = [part for part in fragments if stereo_stripped_smiles(Chem.MolToSmiles(part)) not in _SALT_KEYS]
    if len(unlisted) > 1:
        return None
    parent = unlisted[0] if unlisted else max(fragments, key=lambda part: part.GetNumHeavyAtoms())
    return canonical_smiles(Chem.MolToSmiles(parent))


def printed_formulas(blocks: Iterable[LayoutBlock]) -> dict[str, str]:
    """Compound label -> neutral molecular formula (Hill order) stated in the text.

    A formula follows "calcd for"; its label is the last one before it in the same block
    ("(12a)", "compound 12a", "**12a**"), or the last label of the heading block just before.
    An adduct ([M+H]+, [M+Na]+, [M+K]+, [M+NH4]+, [M-H]-) next to it marks an ion formula,
    which is turned back into the neutral one; when the calculated m/z is printed it decides
    whether the formula is the ion's or already neutral, and a formula that fits neither is
    skipped. A charged formula with no adduct is skipped. A label given two different
    formulas is dropped.
    """
    found: dict[str, set[str]] = {}
    previous: LayoutBlock | None = None
    for block in sorted(blocks, key=lambda item: (item.page, item.index)):
        if not block.text:  # image and table blocks carry no text
            previous = block
            continue
        heading = previous if previous and previous.type == "title" and previous.text else None
        heading_label = _last_label(_normalize(heading.text)) if heading else None
        for label, formula in _block_formulas(_normalize(block.text), heading_label):
            found.setdefault(label, set()).add(formula)
        previous = block
    return {label: formulas.pop() for label, formulas in found.items() if len(formulas) == 1}


def formula_status(smiles: str, formula: str | None) -> str:
    """"match" when the molecule, or its parent without salts, has the formula's element
    counts (charge ignored); "mismatch" when neither does; "none" when either side cannot
    be read."""
    expected = _counts(formula) if formula else None
    mol = _parse(smiles)
    if expected is None or mol is None:
        return "none"
    candidates = {smiles, strip_salts(smiles)} - {None}
    return "match" if any(_mol_counts(candidate) == expected for candidate in candidates) else "mismatch"


def _block_formulas(text: str, heading_label: str | None) -> list[tuple[str, str]]:
    statements = list(_STATEMENT.finditer(text))
    labels = [(match.start(), next(group for group in match.groups() if group)) for match in _LABEL.finditer(text)]
    results = []
    for number, statement in enumerate(statements):
        label = next((name for start, name in reversed(labels) if start < statement.start()), heading_label)
        floor = statements[number - 1].end() if number else 0
        before = text[max(floor, statement.start() - _ADDUCT_WINDOW_BEFORE) : statement.start()]
        ceiling = statements[number + 1].start() if number + 1 < len(statements) else len(text)
        after = text[statement.end() : min(ceiling, statement.end() + _AFTER_WINDOW)]
        after = after[: found.start()] if (found := _FOUND.search(after)) else after
        neutral = _neutral_formula(statement, before, after)
        if label is not None and neutral is not None:
            results.append((label, neutral))
    return results


def _neutral_formula(statement: re.Match[str], before: str, after: str) -> str | None:
    counts = _counts(statement["formula"])
    if counts is None:
        return None
    adduct = _adduct("".join(after.split())) or _adduct("".join(before.split()))
    if adduct is None:
        return None if statement["charge"] else _hill(counts)
    sign, species = adduct
    shift = Counter(_SPECIES[species])
    if (mz := _MZ.search(after)) is not None:
        printed = float(mz.group())
        shifted = _mass(counts) + (_mass(shift) if sign == "+" else -_mass(shift))
        if abs(_mass(counts) - printed) > _MASS_TOLERANCE:
            # Not the ion's formula: keep it only if it is the neutral one the m/z implies.
            neutral = abs(shifted - printed) <= _MASS_TOLERANCE and not statement["charge"]
            return _hill(counts) if neutral else None
    if sign == "-":
        return _hill(counts + shift)
    if any(counts[element] < count for element, count in shift.items()):
        return None
    return _hill(counts - shift)


def _adduct(window: str) -> tuple[str, str] | None:
    match = _ADDUCT.search(window)
    if match is None:
        return None
    return (match["sign"] or match["sign2"], match["species"] or match["species2"])


def _last_label(text: str) -> str | None:
    labels = [next(group for group in match.groups() if group) for match in _LABEL.finditer(text)]
    return labels[-1] if labels else None


def _counts(formula: str) -> Counter[str] | None:
    """Element counts of a formula such as "C19H20N6"; None for unknown elements or stray text."""
    pairs = _ELEMENT_COUNT.findall(formula)
    if not pairs or "".join(element + count for element, count in pairs) != formula:
        return None
    counts: Counter[str] = Counter()
    for element, count in pairs:
        if not _is_element(element):
            return None
        counts[element] += int(count or 1)
    return counts


def _mol_counts(smiles: str) -> Counter[str] | None:
    mol = _parse(smiles)
    if mol is None:
        return None
    return _counts(re.sub(r"[+-]\d*$", "", rdMolDescriptors.CalcMolFormula(mol)))


def _hill(counts: Counter[str]) -> str:
    """Hill order: C, then H, then the rest alphabetically (alphabetical throughout without carbon)."""
    order = sorted(counts, key=lambda element: ({"C": 0, "H": 1}.get(element, 2), element)) if counts["C"] else sorted(counts)
    return "".join(element + (str(counts[element]) if counts[element] > 1 else "") for element in order if counts[element])


def _mass(counts: Counter[str]) -> float:
    return sum(_PERIODIC_TABLE.GetMostCommonIsotopeMass(element) * count for element, count in counts.items())


def _is_element(symbol: str) -> bool:
    try:
        with rdBase.BlockLogs():
            return _PERIODIC_TABLE.GetAtomicNumber(symbol) > 0
    except RuntimeError:
        return False


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_DASHES)
    return _LATEX_MARKUP.sub("", _LATEX_COMMAND.sub("", text))


def _parse(smiles: str | None) -> Chem.Mol | None:
    if not smiles or not smiles.strip():
        return None
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles.strip())
    return mol if mol is not None and mol.GetNumAtoms() else None
