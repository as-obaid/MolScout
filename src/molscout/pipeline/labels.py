"""Compound labels: clean text, recognise labels, expand ranges and link labels to drawings.

A journal compound label is a number of up to three digits with an optional lower-case letter and
primes (``12``, ``12a``, ``12b'``) or a supplementary label (``S1``). Named codes such as
``CCT400028`` count only when they stand alone in a text block or are printed in bold.

Labels printed inside figure images reach linking through OCR (``ocr.jsonl``, the ocr_labels stage):
an OCR line is a candidate when it reads as a label, was read with at least `OCR_MIN_SCORE`, is not
text MinerU already read (its text, caption, table and similar blocks), not in the page margin, not
part of an axis scale, not a member of a range printed in the same figure (``8–17`` under a core:
its members label the R groups drawn beside it) and not inside a drawing's box. Atom symbols, R-group
letters and ring numbers sit inside the drawing; a label the drawing's box took in at its lower edge
(within `EDGE_PT`) still counts. MinerU and OCR candidates then compete under the same
nearest-and-mutual rule.

Codes that a table prints for fragments drawn elsewhere (``M0``, ``Ar1``) are no compound labels;
`link_codes` pairs them with drawings under the same rule, for the legends fix.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from molscout.pipeline.schema import OcrLine, read_jsonl

if TYPE_CHECKING:
    from molscout.pipeline.schema import LayoutBlock, Region

LABEL_PATTERN = r"(?:S\d{1,3}[a-z]?|\d{1,3}[a-z]?)(?:'{1,2})?"
LABEL_RE = re.compile(LABEL_PATTERN.replace("'{1,2}", "['′]{1,2}"))
CODE_RE = re.compile(r"(?P<letters>[A-Za-z]{2,8})-?\d{2,7}[A-Za-z]?")

# Letter prefixes of codes that name cell lines, targets, assays or residues, not compounds.
_CODE_DENY = frozenset(
    """
    IC EC GI CC ED LD TD MIC pIC pEC pKi HCT HEK HT MDA NCI SW HepG SK BT Calu Caco DU LNCaP
    MCF PC PDB CYP HER EGFR BRD PDE CDK JAK BTK ALK HDAC PARP ERK MEK COX HIV HCV SARS MERS
    COVID GPR CXCR CCR KRAS FGFR VEGFR
    """.split()
)

_DASH = r"\s*(?:[-–—−‒]|\bto\b)\s*"
_RANGE_RE = re.compile(rf"^(?P<a>{LABEL_PATTERN}){_DASH}(?P<b>{LABEL_PATTERN}|[a-z]'{{0,2}})$")
_PARTS_RE = re.compile(r"^(?P<s>S?)(?P<n>\d{1,3})(?P<l>[a-z]?)(?P<q>'*)$")
_SPLIT_RE = re.compile(r"\s*(?:,|;|&|\band\b)\s*")
MAX_RANGE = 100

_ITEM = rf"{LABEL_PATTERN}(?:{_DASH}(?:{LABEL_PATTERN}|[a-z]'{{0,2}}))?"
_END = r"(?![\w'])(?!\.\d)"
_LIST = rf"{_ITEM}{_END}(?:(?:\s*(?:;|&|\band\b)\s*|\s*,\s*){_ITEM}{_END}|,\s?[a-z]'{{0,2}}{_END})*"
_KEYWORD = (
    r"\b(?i:compounds?|cmpds?\.?|cpds?\.?|compd\.?|analogu?es?|derivatives?|inhibitors?|ligands?)"
)
_MENTION_RE = re.compile(
    rf"{_KEYWORD}\s+(?P<kw>{_LIST})"
    rf"|\*\*(?P<bold>{_LIST}|{CODE_RE.pattern})\*\*"
    rf"|\((?P<paren>{_LIST})\)"
    rf"|\((?:R|S|rac)\)-\s*(?P<stereo>{LABEL_PATTERN})(?![\w'\-(\[])(?![.,]\d)"
)
LABEL_LIST_PATTERN = _LIST
_START_RE = re.compile(
    r"^(?:(?i:compound|compd\.?|cmpd\.?|cpd\.?)\s+)?"
    rf"\(?(?P<label>{LABEL_PATTERN})\)?(?P<rest>(?:\s*[,:;(].*)?)$"
)
MAX_LABEL_TAIL = 60

_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉′″", "01234567890123456789'\"")
_SUPER_LETTERS = re.compile("[ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ]")
_COMMAND_RE = re.compile(
    r"\\(?:mathrm|mathbf|mathit|mathsf|mathbb|textbf|textit|textrm|text|boldsymbol"
    r"|operatorname|bf|rm)"
    r"\s*\{([^{}]*)\}"
)
_SYMBOLS = {
    "%": "%",
    "upmu": "µ",
    "mu": "µ",
    "pm": "±",
    "times": "×",
    "leq": "≤",
    "le": "≤",
    "geq": "≥",
    "ge": "≥",
    "sim": "~",
    "approx": "≈",
    "cdot": "·",
    "prime": "'",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "Delta": "Δ",
    "circ": "°",
    "dagger": "†",
    "ddagger": "‡",
    "S": "§",
    "ast": "*",
    "lt": "<",
    "gt": ">",
    "left": "",
    "right": "",
}
_SYMBOL_RE = re.compile(r"\\(%|[A-Za-z]+)(?![A-Za-z])")
_SPACE_CMD_RE = re.compile(r"\\[,;:! ]")
_FOOTNOTE = r"(?:[a-z]|[*†‡§¶#‖]+)"
FOOTNOTE_RE = re.compile(rf"^{_FOOTNOTE}(?:\s*,\s*{_FOOTNOTE})*$")
_SUPER_RE = re.compile(r"\^\{([^{}]*)\}|\^(\S)")
_SUB_RE = re.compile(r"_\{([^{}]*)\}|_(\w)")
_JOIN_50_RE = re.compile(r"\b(p?IC|p?EC|GI|CC|ED|LD|TD)\s+50\b")
_JOIN_K_RE = re.compile(r"\b(p?K)\s+([idm])\b")
_EQ_TAG_RE = re.compile(r"<eq>(.*?)</eq>", re.DOTALL)
_MATH_RE = re.compile(r"\$([^$]*)\$")

LABEL_BLOCK_TYPES = {"text", "title", "caption"}
DEFAULT_PAGE = (612.0, 792.0)
BELOW_MAX = 0.06
BESIDE_MAX = 0.05
TIE_PT = 2.0
PAGE_MARGIN = 0.04

OCR_FILE = "ocr.jsonl"
OCR_MIN_SCORE = 0.8
EDGE_PT = 4.0  # a line inside a drawing's box that reaches this close to its lower edge is a label
ALIGN_PT = 3.0  # tick labels of one axis share a row, a centre column or a right edge within this
TICK_MIN = 3
READ_BLOCK_TYPES = frozenset({"text", "title", "caption", "footnote", "table", "equation"})
_OCR_QUOTES = str.maketrans({"’": "'", "‘": "'", "`": "'", "′": "'", "´": "'", "″": "''"})
_OCR_PLAIN_RE = re.compile(
    rf"(?:(?i:compound|compd\.?|cmpd\.?|cpd\.?)\s*)?\(?\s*(?P<label>{LABEL_PATTERN})\s*\)?[.,:;]?"
)
_OCR_NAMED_RE = re.compile(rf"(?:[\w\-–]+\s+){{0,3}}[\w\-–]+\s*(?P<open>\()\s*(?P<label>{LABEL_PATTERN})\s*\)")
_TICK_RE = re.compile(r"[0-9]{1,6}")
_RESIDUE_RE = re.compile(
    r"(?i:ala|arg|asn|asp|cys|gln|glu|gly|his|ile|leu|lys|met|phe|pro|ser|thr|trp|tyr|val)[-\s]?\d{1,4}[A-Za-z]?"
)


def clean_text(text: str) -> str:
    """Flatten inline LaTeX and Unicode sub/superscripts, drop footnote markers, squeeze spaces.

    Math comes as ``$...$`` or MinerU ``<eq>...</eq>``, often with every token spaced out
    (``\\mathsf { I C } _ { 5 0 }``); spaces inside math are dropped as LaTeX does.
    """
    s = _EQ_TAG_RE.sub(lambda m: f"${m.group(1)}$", text)
    s = _MATH_RE.sub(lambda m: _squeeze_math(m.group(1)), s)
    s = s.translate(_SUPERSCRIPT)
    s = _SUPER_LETTERS.sub("", s)
    s = _SPACE_CMD_RE.sub(" ", s)
    s = re.sub(r"\\(?:upmu|mu)(?![A-Za-z])\s*", "µ", s)
    for _ in range(4):
        s, n = _COMMAND_RE.subn(r"\1", s)
        if not n:
            break
    s = _SYMBOL_RE.sub(lambda m: _SYMBOLS.get(m.group(1), m.group(0)), s)
    s = _SUPER_RE.sub(lambda m: _keep_superscript(m.group(1) or m.group(2)), s)
    s = _SUB_RE.sub(lambda m: m.group(1) or m.group(2), s)
    s = s.replace("$", "").replace("{", "").replace("}", "")
    s = _JOIN_50_RE.sub(r"\g<1>50", s)
    s = _JOIN_K_RE.sub(r"\1\2", s)
    return " ".join(s.split())


def _squeeze_math(content: str) -> str:
    """Math without spaces; ``{}`` keeps a command apart from a following letter or digit."""
    content = re.sub(r"(\\[A-Za-z]+)\s+(?=[A-Za-z0-9])", r"\1{}", content)
    return re.sub(r"\s+", "", content)


def _keep_superscript(content: str) -> str:
    """Superscript text to keep; footnote markers become empty."""
    return "" if FOOTNOTE_RE.match(content.strip()) else content


def is_code(text: str) -> bool:
    """True for a named compound code such as ``CCT400028``; cell lines and targets are refused."""
    m = CODE_RE.fullmatch(text)
    if not m:
        return False
    letters = m.group("letters")
    return letters not in _CODE_DENY and sum(c.isupper() for c in letters) >= 2


def expand_range(text: str) -> list[str]:
    """Expand ``12a–f``, ``12a-12f``, ``5–8`` and lists like ``12a,b and 14``; [] when unclear."""
    labels: list[str] = []
    for item in _SPLIT_RE.split(text.strip().replace("′", "'")):
        if re.fullmatch(r"[a-z]'*", item) and labels:
            item = _with_letter(labels[-1], item)
        expanded = _expand_item(item)
        if not expanded:
            return []
        labels.extend(expanded)
    return list(dict.fromkeys(labels))


def _with_letter(previous: str, suffix: str) -> str:
    """``12a`` + ``b`` -> ``12b``."""
    m = _PARTS_RE.match(previous)
    return f"{m['s']}{m['n']}{suffix}" if m else suffix


def _expand_item(item: str) -> list[str]:
    """One label or one range."""
    if LABEL_RE.fullmatch(item):
        return [item]
    m = _RANGE_RE.match(item)
    if not m:
        return []
    a = _PARTS_RE.match(m["a"])
    b = _PARTS_RE.match(m["b"]) or _PARTS_RE.match(f"{a['s']}{a['n']}{m['b']}")
    if not a or not b or a["s"] != b["s"] or a["q"] != b["q"]:
        return []
    if a["l"] and b["l"] and a["n"] == b["n"] and a["l"] <= b["l"]:
        letters = (chr(c) for c in range(ord(a["l"]), ord(b["l"]) + 1))
        return [f"{a['s']}{a['n']}{x}{a['q']}" for x in letters]
    lo, hi = int(a["n"]), int(b["n"])
    if not a["l"] and not b["l"] and lo < hi and hi - lo < MAX_RANGE:
        return [f"{a['s']}{n}{a['q']}" for n in range(lo, hi + 1)]
    return []


def plain_text(text: str) -> str:
    """`clean_text` without Markdown or HTML bold markers."""
    return re.sub(r"\*\*|</?b>|</?strong>", "", clean_text(text)).strip()


def label_of_text(text: str) -> str | None:
    """The label a short text block names (``12a``, ``(12a)``, ``12a, R = Me``), else None."""
    s = plain_text(text)
    core = s.strip("()[]").rstrip(".:;,").strip()
    if LABEL_RE.fullmatch(core) or is_code(core):
        return core.replace("′", "'")
    m = _START_RE.match(s)
    if m and len(m["rest"]) <= MAX_LABEL_TAIL:
        return m["label"]
    return None


def mentioned_labels(text: str) -> list[str]:
    """Labels a sentence names: after "compound(s)", in bold, in parentheses or after (R)-/(S)-."""
    found: list[str] = []
    for m in _MENTION_RE.finditer(clean_text(text)):
        span = m["kw"] or m["bold"] or m["paren"] or m["stereo"]
        found.extend([span] if is_code(span) else expand_range(span))
    return list(dict.fromkeys(found))


@dataclass(frozen=True)
class OcrLabel:
    """The label an OCR line names, and where it starts as a fraction of the line's width."""

    label: str
    start: float = 0.0


@dataclass(frozen=True)
class _Candidate:
    """A label that may name a drawing: its page, the box of the label text and the label."""

    page: int
    bbox: Sequence[float]
    label: str


def ocr_label(text: str) -> OcrLabel | None:
    """The label an OCR line reads as: ``12a``, ``(12a)``, ``5'``, ``Compound 7``, ``Roxadustat (1)``, or a
    compound code standing alone (``CCT400028``).

    Atom symbols (``N``, ``Cl``, ``NH``), R-group letters (``R1``, ``X``), ranges (``8–17``), values,
    numbers that start with 0 and amino-acid residues (``TYR-310``) are refused. For a name followed by a
    parenthesised label, `start` is where the parenthesis begins, so the label's own box can be told from
    the name's.
    """
    s = " ".join(text.translate(_OCR_QUOTES).split())
    match, start = _OCR_PLAIN_RE.fullmatch(s), 0.0
    if match is None:
        match = _OCR_NAMED_RE.fullmatch(s)
        if match is None:
            return OcrLabel(s) if is_code(s) and not _RESIDUE_RE.fullmatch(s) else None
        start = match.start("open") / len(s)
    label = match["label"]
    return None if label.lstrip("S").startswith("0") else OcrLabel(label, start)


def load_ocr_lines(paper_dir: Path | str) -> list[OcrLine]:
    """The paper's ocr.jsonl lines; [] when the ocr_labels stage did not run for it."""
    path = Path(paper_dir) / OCR_FILE
    return read_jsonl(path, OcrLine) if path.is_file() else []


def link_regions(
    regions: Sequence[Region],
    blocks: Sequence[LayoutBlock],
    page_sizes: Mapping[int, tuple[float, float]] | None = None,
    ocr_lines: Sequence[OcrLine] = (),
) -> dict[str, str]:
    """Map region_id to the label printed under or beside it, else to a lone caption label.

    A text label (a MinerU text block, or an OCR line from `_ocr_candidates`) links only when the drawing
    and the label are each other's nearest partner within `BELOW_MAX` of the page height (below) or
    `BESIDE_MAX` of the page width (beside). A label tied between a drawing and a larger box around it
    goes to the inner drawing. The caption fallback applies only when the figure holds one drawing and
    its caption names one label.
    """
    sizes = page_sizes or {}
    structures = [r for r in regions if r.is_structure]
    links = _nearest_links(structures, _label_candidates(blocks, ocr_lines, structures, sizes), sizes)
    for region_id, label in _caption_links(structures, blocks).items():
        links.setdefault(region_id, label)
    return links


def link_codes(
    regions: Sequence[Region],
    blocks: Sequence[LayoutBlock],
    code_of: Callable[[str], str | None],
    page_sizes: Mapping[int, tuple[float, float]] | None = None,
    ocr_lines: Sequence[OcrLine] = (),
) -> dict[str, str]:
    """Map region_id to the code printed under or beside it: a word such as "M0" that a table prints for a
    fragment drawn elsewhere (the legends fix). `code_of` gives the code a text names, or None.

    A text block or OCR line that is no compound label but names a code is a candidate, with an OCR label's
    filters. Codes and compound labels compete under `link_regions`' nearest-and-mutual rule, so a drawing
    whose own compound label is nearer keeps it; only drawings whose label names a code are returned.
    """
    sizes = page_sizes or {}
    structures = [r for r in regions if r.is_structure]

    def line_code(text: str) -> OcrLabel | None:
        code = None if ocr_label(text) else code_of(" ".join(text.translate(_OCR_QUOTES).split()))
        return OcrLabel(code) if code else None

    codes = [
        _Candidate(b.page, b.bbox, code)
        for b in blocks
        if b.type in LABEL_BLOCK_TYPES
        and not _in_margin(b, sizes)
        and not label_of_text(b.text or "")
        and (code := code_of(plain_text(b.text or "")))
    ]
    codes += _ocr_candidates(ocr_lines, blocks, structures, sizes, line_code)
    labelled = [*_label_candidates(blocks, ocr_lines, structures, sizes), *codes]
    return {region_id: code for region_id, label in _nearest_links(structures, labelled, sizes).items()
            if (code := code_of(label))}


def _label_candidates(
    blocks: Sequence[LayoutBlock],
    ocr_lines: Sequence[OcrLine],
    structures: Sequence[Region],
    sizes: Mapping[int, tuple[float, float]],
) -> list[_Candidate]:
    """The compound labels of MinerU's text blocks and of the OCR lines (`_ocr_candidates`)."""
    labelled = [
        _Candidate(b.page, b.bbox, label)
        for b in blocks
        if b.type in LABEL_BLOCK_TYPES
        and not _in_margin(b, sizes)
        and (label := label_of_text(b.text or ""))
    ]
    return labelled + _ocr_candidates(ocr_lines, blocks, structures, sizes)


def _ocr_candidates(
    lines: Sequence[OcrLine],
    blocks: Sequence[LayoutBlock],
    regions: Sequence[Region],
    sizes: Mapping[int, tuple[float, float]],
    read_label: Callable[[str], OcrLabel | None] = ocr_label,
) -> list[_Candidate]:
    """The OCR lines that may be compound labels (see the module docstring), boxed to the label text;
    `read_label` reads a line's label (`ocr_label`, or a code for `link_codes`)."""
    read = [b for b in blocks if b.type in READ_BLOCK_TYPES]
    ticks = _axis_ticks(lines)
    series = _series(lines)
    out = []
    for index, line in enumerate(lines):
        found = read_label(line.text) if line.score >= OCR_MIN_SCORE and index not in ticks else None
        if found is None or found.label in series.get((line.page, line.source_block), ()):
            continue
        x0, y0, x1, y1 = line.bbox
        candidate = _Candidate(line.page, [x0 + (x1 - x0) * found.start, y0, x1, y1], found.label)
        if not (_in_margin(candidate, sizes) or _on(candidate, read) or _in_a_drawing(candidate, regions)):
            out.append(candidate)
    return out


def _series(lines: Sequence[OcrLine]) -> dict[tuple[int, str], set[str]]:
    """Per figure area, the labels of the ranges printed in it (``8–17`` under a core drawn with R). A single
    label of such a range in the same figure names an R-group drawing of the series, not a compound."""
    found: dict[tuple[int, str], set[str]] = {}
    for line in lines:
        labels = expand_range(line.text.translate(_OCR_QUOTES)) if line.score >= OCR_MIN_SCORE else []
        if len(labels) > 1:
            found.setdefault((line.page, line.source_block), set()).update(labels)
    return found


def _on(candidate: _Candidate, blocks: Sequence[LayoutBlock]) -> bool:
    """True when the candidate is centred in one of the blocks on its page."""
    return any(b.page == candidate.page and _inside(candidate.bbox, b.bbox) for b in blocks)


def _in_a_drawing(candidate: _Candidate, regions: Sequence[Region]) -> bool:
    """True when the candidate is centred in a drawing's box and does not reach the box's lower edge."""
    return any(
        r.page == candidate.page and _inside(candidate.bbox, r.bbox) and candidate.bbox[3] < r.bbox[3] - EDGE_PT
        for r in regions
    )


def _axis_ticks(lines: Sequence[OcrLine]) -> set[int]:
    """Indices of number lines that form an axis scale: `TICK_MIN` or more numbers of one area in a row or
    column, evenly spaced in value, starting at 0 or stepping by 2 or more (a row of labels 8, 9, 10 is not)."""
    numbers = [i for i, line in enumerate(lines) if _TICK_RE.fullmatch(line.text.strip())]
    ticks: set[int] = set()
    for i in numbers:
        for key in (_centre_x, _centre_y, _right_edge):
            group = [
                k for k in numbers
                if lines[k].page == lines[i].page and lines[k].source_block == lines[i].source_block
                and abs(key(lines[k]) - key(lines[i])) <= ALIGN_PT
            ]
            if _is_scale(sorted(int(lines[k].text) for k in group)):
                ticks.update(group)
    return ticks


def _centre_x(line: OcrLine) -> float:
    return (line.bbox[0] + line.bbox[2]) / 2


def _centre_y(line: OcrLine) -> float:
    return (line.bbox[1] + line.bbox[3]) / 2


def _right_edge(line: OcrLine) -> float:
    return line.bbox[2]


def _is_scale(values: Sequence[int]) -> bool:
    if len(values) < TICK_MIN:
        return False
    steps = {b - a for a, b in zip(values, values[1:])}
    return len(steps) == 1 and (values[0] == 0 or min(steps) >= 2)


def _in_margin(block: LayoutBlock | _Candidate, sizes: Mapping[int, tuple[float, float]]) -> bool:
    """True for blocks centred in the top or bottom `PAGE_MARGIN` of the page (page numbers)."""
    height = sizes.get(block.page, DEFAULT_PAGE)[1]
    cy = (block.bbox[1] + block.bbox[3]) / 2
    return cy < PAGE_MARGIN * height or cy > (1 - PAGE_MARGIN) * height


def _nearest_links(
    regions: Sequence[Region],
    labelled: Sequence[_Candidate],
    sizes: Mapping[int, tuple[float, float]],
) -> dict[str, str]:
    """Mutual-nearest pairs of drawing and label, ties refused (but for a drawing inside a tied box)."""
    gaps: dict[tuple[int, int], float] = {}
    for i, region in enumerate(regions):
        width, height = sizes.get(region.page, DEFAULT_PAGE)
        for j, candidate in enumerate(labelled):
            if candidate.page == region.page:
                gap = _gap(region.bbox, candidate.bbox, width, height)
                if gap is not None:
                    gaps[(i, j)] = gap
    links: dict[str, str] = {}
    for i, region in enumerate(regions):
        j = _unique_best({jj: g for (ii, jj), g in gaps.items() if ii == i})
        if j is not None and _best_region({ii: g for (ii, jj), g in gaps.items() if jj == j}, regions) == i:
            links[region.region_id] = labelled[j].label
    return links


def _unique_best(options: Mapping[int, float]) -> int | None:
    """Key of the smallest value, or None when empty or tied within `TIE_PT`."""
    ranked = sorted(options.items(), key=lambda kv: kv[1])
    if not ranked or (len(ranked) > 1 and ranked[1][1] - ranked[0][1] < TIE_PT):
        return None
    return ranked[0][0]


def _best_region(options: Mapping[int, float], regions: Sequence[Region]) -> int | None:
    """The nearest drawing to a label; of drawings tied within `TIE_PT`, the one that lies inside all the
    others (a box segmentation drew around two drawings), else None."""
    best = _unique_best(options)
    if best is not None or not options:
        return best
    low = min(options.values())
    tied = [i for i, gap in options.items() if gap - low < TIE_PT]
    inner = [i for i in tied if all(k == i or _nested(regions[i].bbox, regions[k].bbox) for k in tied)]
    return inner[0] if len(inner) == 1 else None


def _nested(inner: Sequence[float], outer: Sequence[float]) -> bool:
    """True when `inner` is centred in `outer` and smaller."""
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return _inside(inner, outer) and area < (outer[2] - outer[0]) * (outer[3] - outer[1])


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _gap(
    region: Sequence[float], block: Sequence[float], width: float, height: float
) -> float | None:
    """Distance in points from a drawing to a label below, beside or inside it; None otherwise."""
    rx0, ry0, rx1, ry1 = region
    bx0, by0, bx1, by1 = block
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    if rx0 <= cx <= rx1 and ry0 <= cy <= ry1:
        return 0.0
    options = []
    aligned_x = rx0 <= cx <= rx1 or _overlap(rx0, rx1, bx0, bx1) >= 0.5 * max(bx1 - bx0, 1e-6)
    if aligned_x and cy >= (ry0 + ry1) / 2:
        gap = max(0.0, by0 - ry1)
        if gap <= BELOW_MAX * height:
            options.append(gap)
    aligned_y = ry0 <= cy <= ry1 or _overlap(ry0, ry1, by0, by1) >= 0.5 * max(by1 - by0, 1e-6)
    if aligned_y:
        gap = max(0.0, bx0 - rx1) if cx >= (rx0 + rx1) / 2 else max(0.0, rx0 - bx1)
        if gap <= BESIDE_MAX * width:
            options.append(gap)
    return min(options) if options else None


def _inside(inner: Sequence[float], outer: Sequence[float]) -> bool:
    """True when the centre of `inner` lies in `outer`."""
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _caption_links(regions: Sequence[Region], blocks: Iterable[LayoutBlock]) -> dict[str, str]:
    """Lone-drawing figures whose caption names exactly one label."""
    blocks = list(blocks)
    links: dict[str, str] = {}
    for figure in (b for b in blocks if b.type == "image"):
        inside = [r for r in regions if r.page == figure.page and _inside(r.bbox, figure.bbox)]
        captions = [
            b.text or ""
            for b in blocks
            if b.type == "caption" and b.page == figure.page and b.parent == figure.index
        ]
        labels = list(dict.fromkeys(x for c in captions for x in mentioned_labels(c)))
        if len(inside) == 1 and len(labels) == 1:
            links[inside[0].region_id] = labels[0]
    return links
