"""Named compounds: (label, systematic name) pairs from experimental-section text.

Patterns read: ``N-(4-fluorophenyl)-...-carboxamide (12a)``, ``12a: name`` and
``Compound 12a. name``. A name that starts a sentence counts anywhere; a name in the middle of a
sentence ("to afford <name> (7)") counts only inside an experimental section, because elsewhere it
usually names a compound class. A pair is kept only when the name looks systematic; OPSIN itself is
the final check.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from molscout.pipeline.labels import LABEL_PATTERN, is_code, plain_text

if TYPE_CHECKING:
    from molscout.pipeline.schema import LayoutBlock

TEXT_TYPES = {"text", "title"}
MAX_HEADING = 60
_SECTION_START_RE = re.compile(
    r"(?i)\b(?:experimental|synthes[ie]s|chemistry|general (?:procedures?|methods)"
    r"|materials and methods|preparation of)\b"
)
_SECTION_END_RE = re.compile(
    r"(?i)\b(?:references|acknowledge?ments?|conflicts? of interest|author information"
    r"|abbreviations|associated content|funding)\b"
)
MIN_NAME, MAX_NAME, MAX_WORDS = 4, 300, 5
_PAREN_LABEL_RE = re.compile(rf"\((?P<label>{LABEL_PATTERN})\)")
_LEADING_RE = re.compile(
    r"^(?:(?i:compound|compd\.?|cmpd\.?)\s+)?"
    rf"(?P<label>{LABEL_PATTERN})\s*[:.]\s+(?P<name>.+?)(?:\.\s|\.$|;|$)"
)
_TAIL_WORDS = frozenset(
    """
    acid hydrochloride dihydrochloride hydrobromide salt ester trifluoroacetate oxalate
    mesylate tosylate hydrate monohydrate formate maleate fumarate
    """.split()
)
_HEAD_WORDS = frozenset(
    """
    methyl ethyl propyl isopropyl butyl tert-butyl t-butyl isobutyl benzyl allyl phenyl
    cyclopropyl cyclohexyl dimethyl diethyl di-tert-butyl sodium potassium lithium
    """.split()
)
_PROSE = frozenset(
    """
    the a an was were is with and of to from in for by as at solution compound compounds
    mixture product yield using prepared title crude stirred procedure general residue solid
    oil afford afforded gave obtained mp white yellow
    """.split()
)
_GENERIC_RE = re.compile(
    r"(?i)\b(?:aryl|alkyl|heteroaryl|hetaryl|cycloalkyl|substituted|analogue)\b"
)
_MORPHEME_RE = re.compile(
    r"(?i)meth|eth|prop|but|pent|hex|phen|benz|pyr|pip|morph|indol|quinol|azol|azin|imid|thi|fur"
    r"|oxa|amin|amid|carb|chlor|fluor|brom|iod|hydr|cyclo|nitr|sulf|oxy|yl|ol\b|one\b|ane\b|ene\b"
    r"|ine\b|ate\b|ide\b|oic\b"
)
_LOCANT_RE = re.compile(r"\d+[A-Za-z]?[,\-]|\b[NOS]\d?-|\(\d|\[\d|\([RSEZ]\)")


def _balanced(text: str) -> bool:
    depth = 0
    for ch in text:
        depth += {"(": 1, "[": 1, ")": -1, "]": -1}.get(ch, 0)
        if depth < 0:
            return False
    return depth == 0


def looks_systematic(name: str) -> bool:
    """True when `name` reads like an IUPAC or CAS name rather than prose, a code or a class."""
    words = name.split()
    if not MIN_NAME <= len(name) <= MAX_NAME or not words or len(words) > MAX_WORDS:
        return False
    if any(w.lower().strip(",.;:") in _PROSE for w in words) or is_code(name):
        return False
    if _GENERIC_RE.search(name) or not _balanced(name):
        return False
    hits = len(_MORPHEME_RE.findall(name))
    return hits >= 2 or (hits >= 1 and bool(_LOCANT_RE.search(name)))


def _name_before(text: str, end: int) -> str:
    """The name that ends just before position `end`: last token plus salt and ester words."""
    tokens = text[:end].split()
    picked: list[str] = []
    while tokens:
        picked.insert(0, tokens.pop())
        if picked[0].lower() not in _TAIL_WORDS:
            break
    while tokens and tokens[-1].lower() in _HEAD_WORDS:
        picked.insert(0, tokens.pop())
    name = " ".join(picked).strip("\"'“”")
    while name[:1] in "([" and not _balanced(name):
        name = name[1:]
    return name


def _starts_sentence(text: str, start: int) -> bool:
    before = text[:start].rstrip()
    return not before or before.endswith((".", ";", ":"))


def _pairs_in(text: str, experimental: bool) -> list[tuple[str, str]]:
    """(label, name) candidates in one block of text, in text order."""
    s = plain_text(text)
    pairs: list[tuple[str, str]] = []
    lead = _LEADING_RE.match(s)
    if lead:
        pairs.append((lead["label"], lead["name"].strip()))
    for m in _PAREN_LABEL_RE.finditer(s):
        name = _name_before(s, m.start())
        start = s.rfind(name, 0, m.start()) if name else -1
        if name and (experimental or _starts_sentence(s, start)):
            pairs.append((m["label"], name))
    return pairs


def _section_change(text: str) -> bool | None:
    """True when a short heading opens an experimental section, False when one closes it."""
    heading = plain_text(text)
    if len(heading) > MAX_HEADING:
        return None
    if _SECTION_END_RE.search(heading):
        return False
    return True if _SECTION_START_RE.search(heading) else None


def named_compounds(blocks: Sequence[LayoutBlock]) -> list[tuple[str, str]]:
    """(label, name) pairs whose name looks systematic; the first name seen for a label wins."""
    found: dict[str, str] = {}
    experimental = False
    for block in sorted(blocks, key=lambda b: (b.page, b.index)):
        if block.type not in TEXT_TYPES:
            continue
        change = _section_change(block.text or "")
        if change is not None:
            experimental = change
        for label, name in _pairs_in(block.text or "", experimental):
            if label not in found and looks_systematic(name):
                found[label] = name
    return list(found.items())
