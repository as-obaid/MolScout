"""R-group table cells to SMILES fragments.

A cell becomes a `Fragment` of one of three kinds:

- substituent: one `*` marks the attachment, e.g. "OMe" -> `*OC`;
- linker: two `*`, left end first as written, e.g. "-NHCO-" -> `*NC(=O)*`;
- atom: no `*`; the first atom replaces a variable atom of the core (a ring atom such as
  A = N or CH, or a one-atom bridge such as X = O), and any other atoms hang off it,
  e.g. "CF" -> `CF`. Hydrogens written in the cell are kept exactly: "CH" -> `[CH]`.

Rules, in order: abbreviations, ring-atom cells, condensed formulas ("CH2OH", "-CH2CH2-"),
substituted phenyl shorthand ("4-F-Ph", "3,4-diCl-C6H4"), positional groups ("(p)-OH",
"4-OMe": a substituent whose `position` records where the cell places it), SMILES with `*`,
and last, names through OPSIN as radicals ("2-pyridyl", "morpholin-4-yl"). Text that no
rule reads gives None rather than a guess, as do groups whose place is unclear ("F-Ph",
"2,6-diF"). A cell wrapped in dashes is only ever read as a linker.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

from rdkit import Chem, rdBase

from molscout.scoring.smiles import canonical_smiles

# Dummy atoms per fragment kind.
_DUMMIES = {"substituent": 1, "linker": 2, "atom": 0}

# Cell text -> substituent SMILES. Each value starts with `*` followed by the attachment atom,
# so value[1:] continues a chain or fills a branch.
ABBREVIATIONS: dict[str, str] = {
    "H": "*[H]",
    "F": "*F",
    "Cl": "*Cl",
    "Br": "*Br",
    "I": "*I",
    "Me": "*C",
    "Et": "*CC",
    "Pr": "*CCC",
    "nPr": "*CCC",
    "n-Pr": "*CCC",
    "iPr": "*C(C)C",
    "i-Pr": "*C(C)C",
    "Bu": "*CCCC",
    "nBu": "*CCCC",
    "n-Bu": "*CCCC",
    "iBu": "*CC(C)C",
    "i-Bu": "*CC(C)C",
    "sBu": "*C(C)CC",
    "s-Bu": "*C(C)CC",
    "tBu": "*C(C)(C)C",
    "t-Bu": "*C(C)(C)C",
    "cPr": "*C1CC1",
    "c-Pr": "*C1CC1",
    "cBu": "*C1CCC1",
    "c-Bu": "*C1CCC1",
    "cPent": "*C1CCCC1",
    "cHex": "*C1CCCCC1",
    "c-Hex": "*C1CCCCC1",
    "Cy": "*C1CCCCC1",
    "OMe": "*OC",
    "MeO": "*OC",
    "OEt": "*OCC",
    "EtO": "*OCC",
    "OiPr": "*OC(C)C",
    "OPh": "*Oc1ccccc1",
    "PhO": "*Oc1ccccc1",
    "OBn": "*OCc1ccccc1",
    "BnO": "*OCc1ccccc1",
    "OAc": "*OC(C)=O",
    "AcO": "*OC(C)=O",
    "OH": "*O",
    "HO": "*O",
    "SH": "*S",
    "SMe": "*SC",
    "MeS": "*SC",
    "SCF3": "*SC(F)(F)F",
    "NH2": "*N",
    "H2N": "*N",
    "NHMe": "*NC",
    "MeHN": "*NC",
    "NMe2": "*N(C)C",
    "Me2N": "*N(C)C",
    "NHEt": "*NCC",
    "NEt2": "*N(CC)CC",
    "NHAc": "*NC(C)=O",
    "AcHN": "*NC(C)=O",
    "AcNH": "*NC(C)=O",
    "NHBoc": "*NC(=O)OC(C)(C)C",
    "NHPh": "*Nc1ccccc1",
    "CN": "*C#N",
    "NO2": "*[N+](=O)[O-]",
    "O2N": "*[N+](=O)[O-]",
    "CF3": "*C(F)(F)F",
    "F3C": "*C(F)(F)F",
    "CHF2": "*C(F)F",
    "OCF3": "*OC(F)(F)F",
    "F3CO": "*OC(F)(F)F",
    "OCHF2": "*OC(F)F",
    "SO2Me": "*S(C)(=O)=O",
    "MeSO2": "*S(C)(=O)=O",
    "Ms": "*S(C)(=O)=O",
    "SO2NH2": "*S(N)(=O)=O",
    "H2NSO2": "*S(N)(=O)=O",
    "Ts": "*S(=O)(=O)c1ccc(C)cc1",
    "COOH": "*C(=O)O",
    "CO2H": "*C(=O)O",
    "HOOC": "*C(=O)O",
    "HO2C": "*C(=O)O",
    "COOMe": "*C(=O)OC",
    "CO2Me": "*C(=O)OC",
    "MeOOC": "*C(=O)OC",
    "MeO2C": "*C(=O)OC",
    "COOEt": "*C(=O)OCC",
    "CO2Et": "*C(=O)OCC",
    "EtOOC": "*C(=O)OCC",
    "EtO2C": "*C(=O)OCC",
    "CONH2": "*C(N)=O",
    "H2NOC": "*C(N)=O",
    "CONHMe": "*C(=O)NC",
    "CONMe2": "*C(=O)N(C)C",
    "CHO": "*C=O",
    "Ac": "*C(C)=O",
    "Bz": "*C(=O)c1ccccc1",
    "Ph": "*c1ccccc1",
    "Bn": "*Cc1ccccc1",
    "Boc": "*C(=O)OC(C)(C)C",
    "Cbz": "*C(=O)OCc1ccccc1",
    "c-pr": "*C1CC1",
    "c-bu": "*C1CCC1",
    "c-pent": "*C1CCCC1",
    "c-hex": "*C1CCCCC1",
    "Piv": "*C(=O)C(C)(C)C",
    "Tf": "*S(=O)(=O)C(F)(F)F",
    "TMS": "*[Si](C)(C)C",
    "Et2N": "*N(CC)CC",
    "N3": "*N=[N+]=[N-]",
    # Condensed formulas of whole groups.
    "C2H5": "*CC",
    "C3H7": "*CCC",
    "n-C3H7": "*CCC",
    "i-C3H7": "*C(C)C",
    "C4H9": "*CCCC",
    "n-C4H9": "*CCCC",
    "t-C4H9": "*C(C)(C)C",
    "C6H5": "*c1ccccc1",
    "C6H11": "*C1CCCCC1",
    "Bu-t": "*C(C)(C)C",
    "Pr-i": "*C(C)C",
    "Pent": "*CCCCC",
    "nPent": "*CCCCC",
    "n-Pent": "*CCCCC",
    "Hex": "*CCCCCC",
    "nHex": "*CCCCCC",
    "n-Hex": "*CCCCCC",
    "allyl": "*CC=C",
    "vinyl": "*C=C",
    "propargyl": "*CC#C",
    "Ad": "*C12CC3CC(CC(C3)C1)C2",
    "1-Ad": "*C12CC3CC(CC(C3)C1)C2",
    "Mes": "*c1c(C)cc(C)cc1C",
    "Mor": "*N1CCOCC1",
    # Rings whose abbreviation names the attachment ("3-Py"); a bare "Py" or "Th" does not, and gives None.
    "2-Py": "*c1ccccn1",
    "3-Py": "*c1cccnc1",
    "4-Py": "*c1ccncc1",
    "2-Th": "*c1cccs1",
    "3-Th": "*c1ccsc1",
    "2-Fu": "*c1ccco1",
    "3-Fu": "*c1ccoc1",
    "1-Naph": "*c1cccc2ccccc12",
    "2-Naph": "*c1ccc2ccccc2c1",
}

# Condensed-formula backbone units: SMILES piece and the number of single bonds it leaves open.
_UNITS: dict[str, tuple[str, int]] = {
    "CH3": ("C", 1),
    "CH2": ("C", 2),
    "CF2": ("C(F)(F)", 2),
    "NH2": ("N", 1),
    "NH": ("N", 2),
    "NMe": ("N(C)", 2),
    "N(Me)": ("N(C)", 2),
    "N(CH3)": ("N(C)", 2),
    "OH": ("O", 1),
    "O": ("O", 2),
    "SH": ("S", 1),
    "S": ("S", 2),
    "SO2": ("S(=O)(=O)", 2),
    "SO": ("S(=O)", 2),
    "CO": ("C(=O)", 2),
    "C(O)": ("C(=O)", 2),
    "C(=O)": ("C(=O)", 2),
    "C=O": ("C(=O)", 2),
    "COO": ("C(=O)O", 2),
    "CO2": ("C(=O)O", 2),
    "CONH": ("C(=O)N", 2),
    "NHCO": ("NC(=O)", 2),
    "CH=CH": ("C=C", 2),
    "C≡C": ("C#C", 2),
}
# Units with two backbone atoms: alone in a cell they are linkers, not single atoms.
_TWO_ATOM_UNITS = {"COO", "CO2", "CONH", "NHCO", "CH=CH", "C≡C"}
# Longest first, so "CH2" wins over "C" and "SO2Me" over "SO2". "H" alone never joins a chain.
_CHAIN_KEYS = sorted((set(_UNITS) | set(ABBREVIATIONS)) - {"H"}, key=len, reverse=True)
_REPEATED_UNIT = re.compile(r"\((CH2|CF2)\)(\d)")

# Groups that may follow C or N with no separator in a ring-atom cell ("CF", "NMe").
_BARE_RING_GROUPS = {"F", "Cl", "Br", "I", "Me", "Et", "iPr", "cPr", "CF3", "CN", "Ph", "Bn", "Ac", "Boc"}
_RING_ELEMENT = re.compile(r"C|N|O|S|Se|Si|P|B")
_RING_HYDRIDE = re.compile(r"(?P<el>C|N|Si)H(?P<n>\d?)")
_MAX_RING_H = {"C": 2, "N": 1, "Si": 2}
_RING_WITH_GROUP = re.compile(r"(?P<el>C|N)(?:-(?P<dash>.+)|\((?P<paren>.+)\)|(?P<bare>[A-Z].*))")

# Substituted phenyl shorthand: "<locants>-<group>[-<locants>-<group>...]" then a phenyl or benzyl base.
_ARYL = re.compile(r"(?P<prefix>.+?)-?(?P<base>Ph|C6H4|phenyl|Bn|benzyl)")
_ARYL_BASES = {"Ph": "*", "C6H4": "*", "phenyl": "*", "Bn": "*C", "benzyl": "*C"}
_LOCANTS = re.compile(r"[omp]|[2-6](?:,[2-6])*")
_LETTER_POSITIONS = {"o": 2, "m": 3, "p": 4}
_COUNT_PREFIXES = {"di": 2, "bis": 2, "tri": 3, "tris": 3}
_PREFIXED_COUNT = re.compile(r"(di|bis|tri|tris)(.+)")
_PAREN_COUNT = re.compile(r"\((.+)\)(\d)")
_SUFFIX_COUNT = re.compile(r"(.+?)(\d)")

_POSITIONAL = re.compile(r"\(?(?P<loc>[omp]|\d'*)\)?-(?P<group>.+)")
_POSITION_NAMES = {"o": "ortho", "m": "meta", "p": "para"}

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−⁻﹣－"), "-")
_PRIMES = dict.fromkeys(map(ord, "′’‘`´"), "'")
_LATEX_COMMAND = re.compile(r"\\(?:mathrm|text|rm)")
_MARKUP = re.compile(r"[\s${}_^]")
_NAME_WORD = re.compile(r"[a-z]{3,}")


class NameResolver(Protocol):
    """Anything with Opsin's `to_smiles`; Opsin itself in the pipeline."""

    def to_smiles(self, names: list[str], radicals: bool = False) -> list[str | None]: ...


@dataclass(frozen=True)
class Fragment:
    """A piece that fills one variable of a core; see the module docstring for the kinds."""

    smiles: str
    kind: str
    position: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _DUMMIES:
            raise ValueError(f"unknown fragment kind {self.kind!r}; expected one of {tuple(_DUMMIES)}")
        mol = _parse(self.smiles)
        if mol is None:
            raise ValueError(f"fragment SMILES does not parse: {self.smiles!r}")
        dummies = sum(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms())
        if dummies != _DUMMIES[self.kind]:
            raise ValueError(f"a {self.kind} needs {_DUMMIES[self.kind]} '*', {self.smiles!r} has {dummies}")


def to_fragment(text: str, opsin: NameResolver | None) -> Fragment | None:
    """The fragment a table cell describes, or None when no rule reads it."""
    fragment = _local(text)
    if fragment is None and opsin is not None and (name := _name_candidate(text)):
        fragment = _from_wildcards(opsin.to_smiles([name], radicals=True)[0])
    return fragment


def drawn_fragment(smiles: str | None) -> Fragment | None:
    """The substituent a drawn cell gives: its voted SMILES with exactly one attachment point (`*` with any atom
    map or isotope), as a canonical `*` fragment with its stereo; None for any other SMILES."""
    mol = _parse(smiles) if smiles else None
    if mol is None:
        return None
    dummies = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummies) != 1:
        return None
    dummies[0].SetAtomMapNum(0)
    dummies[0].SetIsotope(0)
    return _substituent(Chem.MolToSmiles(mol))


def to_fragments(texts: list[str], opsin: NameResolver | None) -> list[Fragment | None]:
    """`to_fragment` for many cells, with every name sent to OPSIN in one batch first."""
    if opsin is not None:
        names = [name for text in texts if _local(text) is None and (name := _name_candidate(text))]
        if names:
            opsin.to_smiles(list(dict.fromkeys(names)), radicals=True)
    return [to_fragment(text, opsin) for text in texts]


def _local(text: str) -> Fragment | None:
    cell = _normalize(text)
    if not cell.strip("-"):
        return None
    if len(cell) > 2 and cell[0] == "-" and cell[-1] == "-":
        return _linker(cell[1:-1])
    cell = cell.removeprefix("-").removesuffix("-")
    for rule in (_abbreviation, _ring_atom, _chain, _aryl, _positional, _smiles):
        if (fragment := rule(cell)) is not None:
            return fragment
    return None


def _abbreviation(cell: str) -> Fragment | None:
    smiles = ABBREVIATIONS.get(cell)
    return _substituent(smiles) if smiles else None


def _ring_atom(cell: str) -> Fragment | None:
    if _RING_ELEMENT.fullmatch(cell):
        return Fragment(cell, "atom")
    if match := _RING_HYDRIDE.fullmatch(cell):
        element, count = match["el"], int(match["n"] or 1)
        if count > _MAX_RING_H[element]:
            return None
        return Fragment(f"[{element}H{count if count > 1 else ''}]", "atom")
    if match := _RING_WITH_GROUP.fullmatch(cell):
        if match["bare"] is not None and match["bare"] not in _BARE_RING_GROUPS:
            return None
        group = _group(match["dash"] or match["paren"] or match["bare"])
        if group is None or group == ABBREVIATIONS["H"]:
            return None
        return Fragment(match["el"] + group[1:], "atom")
    return None


def _chain(cell: str) -> Fragment | None:
    keys = _chain_keys(cell)
    if not keys:
        return None
    valences = [_valence(key) for key in keys]
    body = "".join(_piece(key) for key in keys)
    if len(keys) == 1 and valences[0] == 2 and keys[0] not in _TWO_ATOM_UNITS:
        return Fragment(body, "atom")
    if any(valence != 2 for valence in valences[:-1]):
        return None
    if valences[-1] == 1:
        return _substituent("*" + body)
    return Fragment(f"*{body}*", "linker")


def _linker(inner: str) -> Fragment | None:
    keys = _chain_keys(inner)
    if not keys or any(_valence(key) != 2 for key in keys):
        return None
    return Fragment("*" + "".join(_piece(key) for key in keys) + "*", "linker")


def _chain_keys(cell: str) -> list[str] | None:
    keys: list[str] = []
    start = 0
    while start < len(cell):
        if repeated := _REPEATED_UNIT.match(cell, start):
            keys += [repeated[1]] * int(repeated[2])
            start = repeated.end()
            continue
        key = next((key for key in _CHAIN_KEYS if cell.startswith(key, start)), None)
        if key is None:
            return None
        keys.append(key)
        start += len(key)
    return keys


def _valence(key: str) -> int:
    return _UNITS[key][1] if key in _UNITS else 1


def _piece(key: str) -> str:
    return _UNITS[key][0] if key in _UNITS else ABBREVIATIONS[key][1:]


def _group(text: str) -> str | None:
    """Substituent SMILES (with its `*`) for an abbreviation or condensed formula."""
    if text in ABBREVIATIONS:
        return ABBREVIATIONS[text]
    fragment = _chain(text)
    return fragment.smiles if fragment is not None and fragment.kind == "substituent" else None


def _aryl(cell: str) -> Fragment | None:
    match = _ARYL.fullmatch(cell)
    if match is None:
        return None
    parts = match["prefix"].split("-")
    if len(parts) % 2:
        return None
    branches: dict[int, str] = {}
    for locants, group_text in zip(parts[::2], parts[1::2], strict=True):
        placed = _place(locants, group_text)
        if placed is None or branches.keys() & placed.keys():
            return None
        branches.update(placed)
    ring = ["c1"] + [f"c({branches[p]})" if p in branches else "c" for p in range(2, 6)]
    ring.append(f"c1({branches[6]})" if 6 in branches else "c1")
    return _substituent(_ARYL_BASES[match["base"]] + "".join(ring))


def _place(locants: str, group_text: str) -> dict[int, str] | None:
    """Ring position -> branch SMILES for one "<locants>-<group>" pair, or None if they disagree."""
    if not _LOCANTS.fullmatch(locants):
        return None
    positions = [_LETTER_POSITIONS.get(item) or int(item) for item in locants.split(",")]
    counted = _counted_group(group_text)
    if counted is None or len(set(positions)) != len(positions):
        return None
    smiles, count = counted
    if len(positions) != (count or 1):
        return None
    return dict.fromkeys(positions, smiles[1:])


def _counted_group(text: str) -> tuple[str, int | None] | None:
    if (smiles := _group(text)) is not None:
        return smiles, None
    for pattern in (_PREFIXED_COUNT, _PAREN_COUNT, _SUFFIX_COUNT):
        if (match := pattern.fullmatch(text)) is None:
            continue
        first, second = match.groups()
        group, count = (second, _COUNT_PREFIXES[first]) if pattern is _PREFIXED_COUNT else (first, int(second))
        if (smiles := _group(group)) is not None:
            return smiles, count
    return None


def _positional(cell: str) -> Fragment | None:
    match = _POSITIONAL.fullmatch(cell)
    if match is None or (smiles := _group(match["group"])) is None:
        return None
    location = match["loc"]
    return _substituent(smiles, _POSITION_NAMES.get(location, location))


def _smiles(cell: str) -> Fragment | None:
    if "*" not in cell:
        return None
    mol = _parse(cell)
    if mol is None:
        return None
    dummies = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    for atom in dummies:
        atom.SetAtomMapNum(0)
    if len(dummies) == 1:
        return _substituent(Chem.MolToSmiles(mol))
    if len(dummies) == 2:
        return Fragment(Chem.MolToSmiles(mol, canonical=False), "linker")
    return None


def _from_wildcards(smiles: str | None) -> Fragment | None:
    if smiles is None:
        return None
    count = smiles.count("*")
    if count == 1:
        return Fragment(smiles, "substituent")
    if count == 2:
        return Fragment(smiles, "linker")
    return None


def _substituent(smiles: str, position: str | None = None) -> Fragment | None:
    canonical = canonical_smiles(smiles)
    return Fragment(canonical, "substituent", position) if canonical else None


def _name_candidate(text: str) -> str | None:
    """The text to send to OPSIN: a word of three or more lowercase letters, not a linker cell."""
    name = " ".join(unicodedata.normalize("NFKC", text).translate(_DASHES).translate(_PRIMES).split())
    if not _NAME_WORD.search(name) or (name.startswith("-") and name.endswith("-")):
        return None
    return name


def _normalize(text: str) -> str:
    cell = unicodedata.normalize("NFKC", text).translate(_DASHES).translate(_PRIMES)
    cell = _MARKUP.sub("", _LATEX_COMMAND.sub("", cell))
    return _unwrap(cell)


def _unwrap(cell: str) -> str:
    """Drop one pair of parentheses around the whole text: "(CF3)" -> "CF3"."""
    if not (cell.startswith("(") and cell.endswith(")")):
        return cell
    depth = 0
    for index, char in enumerate(cell):
        depth += {"(": 1, ")": -1}.get(char, 0)
        if depth == 0:
            return cell[1:-1] if index == len(cell) - 1 else cell
    return cell


def _parse(smiles: str) -> Chem.Mol | None:
    with rdBase.BlockLogs():
        return Chem.MolFromSmiles(smiles)
