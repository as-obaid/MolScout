"""R-group tables: parse MinerU table HTML and find the tables that define compounds row by row.

An R-group table has a compound-label column and at least one variable column (R, R1, R', X, Ar,
Het, linker ...). Rows are keyed by column header in column order; variable columns are renamed to
their canonical names (``R<sub>1</sub>`` -> ``R1``). Rows whose label is a range (``12a–c``) are
split when every cell holds one value or exactly one value per label. A table printed as side-by-side
copies (``Cmpd R X IC50 | Cmpd R X IC50``) is read as one, the left copy's rows first.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import TYPE_CHECKING

from molscout.pipeline.labels import FOOTNOTE_RE, clean_text, expand_range, is_code

if TYPE_CHECKING:
    from molscout.pipeline.schema import LayoutBlock

IMAGE_CELL = "[image]"
MAX_SPAN = 100
MAX_HEADER_ROWS = 3
LABEL_SHARE = 0.5

_TRAILING_MARK_RE = re.compile(r"(?<=\S)[*†‡§¶‖]+$")
_VAR_RE = re.compile(  # R may carry one lowercase letter (Ra, Rb) in place of a number
    r"^(?P<name>R(?:\s*[a-z](?![a-z]))?|X|Y|Z|Ar|A|B|Het|[Ll]inker)\s*(?P<num>\d{0,2})\s*(?P<prime>'{0,3})$"
)
_LABEL_HEADER_RE = re.compile(
    r"^(?:compd|cmpd|cpd|compound|comp|product|no|id|code|example|ex|#)\.?"
    r"\s*(?:no\.?|number|id|#)?$",
    re.IGNORECASE,
)
_ENTRY_HEADER_RE = re.compile(r"^entry\.?$", re.IGNORECASE)
# The subscript 50 as OCR reads it: "50", "s0" or "5o" (MinerU prints IC50 as "ICs0").
ACTIVITY_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:p?IC|p?EC|GI|CC|ED|LD)(?:50|s0|5o)s?(?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9])(?:p?K[idb]|MIC|MBC)s?(?![A-Za-z0-9])"
    r"|(?i:inhibition)|%\s*(?i:inh)"
)
_IMAGE_TOKEN_RE = re.compile(r"\s*\[image(?::[^\]]*)?\]\s*")
_NUMERIC_RE = re.compile(
    r"^[<>≤≥~≈]?\s*[-+−]?\d+(?:[.,]\d+)?(?:\s*(?:±|\+/-)\s*\d+(?:\.\d+)?)?"
    r"\s*(?:%|nM|µM|μM|mM|pM|µg/mL|μg/mL)?$"
)
MISSING = {"nt", "n.t.", "nd", "n.d.", "na", "n.a.", "–", "—", "-", "−", "inactive", ""}
_VALUE_SPLIT_RE = re.compile(r"\s*;\s*|,\s+")


@dataclass(frozen=True)
class RGroupTable:
    """A table of compounds: label column, variable columns and activity columns."""

    table_id: str
    page: int
    bbox: list[float]
    label_col: int
    variables: dict[str, int]
    rows: list[dict[str, str]]
    activity_cols: list[str]

    @property
    def headers(self) -> list[str]:
        """Column headers in column order (empty when the table has no rows)."""
        return list(self.rows[0]) if self.rows else []

    def label_of(self, row: dict[str, str]) -> str:
        """The compound label of `row` as printed (a range when it could not be split)."""
        return list(row.values())[self.label_col]


@dataclass
class _Cell:
    rowspan: int = 1
    colspan: int = 1
    parts: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)

    def text(self) -> str:
        body = _TRAILING_MARK_RE.sub("", clean_text("".join(self.parts)))
        return " ".join(x for x in (body, *self.images) if x)


class _TableParser(HTMLParser):
    """Collects table rows as lists of cells; sub/superscripts are inlined, footnotes dropped."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[_Cell]] = []
        self._row: list[_Cell] | None = None
        self._cell: _Cell | None = None
        self._sup: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._close_row()
            self._row = []
        elif tag in ("td", "th"):
            self._close_cell()
            if self._row is None:
                self._row = []
            spans = dict(attrs)
            self._cell = _Cell(_span(spans.get("rowspan")), _span(spans.get("colspan")))
        elif tag == "sup" and self._cell is not None:
            self._sup = []
        elif tag == "br":
            self.handle_data(" ")
        elif tag == "eq":
            self.handle_data("$")
        elif tag == "img" and self._cell is not None:
            src = dict(attrs).get("src")
            self._cell.images.append(f"[image:{src}]" if src else IMAGE_CELL)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag == "sup" and self._sup is not None:
            content, self._sup = "".join(self._sup), None
            if not FOOTNOTE_RE.match(content.strip()):
                self.handle_data(content)
        elif tag == "eq":
            self.handle_data("$")
        elif tag in ("td", "th"):
            self._close_cell()
        elif tag in ("tr", "table"):
            self._close_row()

    def handle_data(self, data: str) -> None:
        if self._sup is not None:
            self._sup.append(data)
        elif self._cell is not None:
            self._cell.parts.append(data)

    def close(self) -> None:
        super().close()
        self._close_row()

    def _close_cell(self) -> None:
        if self._cell is not None and self._row is not None:
            self._row.append(self._cell)
        self._cell, self._sup = None, None

    def _close_row(self) -> None:
        self._close_cell()
        if self._row is not None:
            self.rows.append(self._row)
        self._row = None


def _span(value: str | None) -> int:
    """A rowspan/colspan attribute as an int in [1, MAX_SPAN]; bad values count as 1."""
    try:
        return min(max(int(value or 1), 1), MAX_SPAN)
    except ValueError:
        return 1


def parse_html_table(html: str) -> list[list[str]]:
    """Table HTML as a rectangular grid of cell text, rowspan and colspan copied into each slot."""
    parser = _TableParser()
    parser.feed(html or "")
    parser.close()
    grid: dict[tuple[int, int], str] = {}
    for r, row in enumerate(parser.rows):
        c = 0
        for cell in row:
            while (r, c) in grid:
                c += 1
            text = cell.text()
            for dr in range(cell.rowspan):
                for dc in range(cell.colspan):
                    grid.setdefault((r + dr, c + dc), text)
            c += cell.colspan
    n_rows = len(parser.rows)
    width = max((c for (r, c) in grid if r < n_rows), default=-1) + 1
    return [[grid.get((r, c), "") for c in range(width)] for r in range(n_rows)]


def cell_labels(cell: str) -> list[str]:
    """Labels a table cell holds: one label, a range or list, or a named code."""
    s = re.sub(r"\*\*|</?b>", "", cell).strip().strip("()").strip()
    if not s:
        return []
    first = s.split()[0].rstrip(",;:")
    for candidate in (s, first):
        labels = expand_range(candidate)
        if labels:
            return labels
    return [first] if is_code(first) else []


def _is_value(cell: str) -> bool:
    """True for cells that only data rows hold: labels, numbers or missing-value marks."""
    return (
        bool(cell_labels(cell)) or bool(_NUMERIC_RE.match(cell)) or cell.lower() in MISSING - {""}
    )


def _is_header_row(row: Sequence[str], above: Sequence[str]) -> bool:
    """A further header row: new text that holds no values and does not span the whole row."""
    new = [c for i, c in enumerate(row) if c and c != above[i]]
    spanning = len(set(row)) == 1 and len(row) > 1
    return bool(new) and not spanning and not any(_is_value(c) for c in new)


def _header_count(grid: Sequence[Sequence[str]]) -> int:
    count = 1
    while count < min(MAX_HEADER_ROWS, len(grid) - 1) and _is_header_row(
        grid[count], grid[count - 1]
    ):
        count += 1
    return count


def _headers(grid: Sequence[Sequence[str]], count: int) -> list[str]:
    """One header per column: the header rows joined, spanned parts kept once, images dropped."""
    headers = []
    for c in range(len(grid[0])):
        parts: list[str] = []
        for r in range(count):
            part = _IMAGE_TOKEN_RE.sub(" ", grid[r][c]).strip()
            if parts and part.startswith(parts[-1]):
                parts[-1] = part
            elif part and part not in parts:
                parts.append(part)
        headers.append(" ".join(parts))
    return headers


def variable_name(header: str) -> str | None:
    """Canonical variable name of a header (``R 1`` -> ``R1``, ``R a`` -> ``Ra``, ``[R]`` -> ``R``, ``Linker`` -> ``linker``) or None."""
    text = header.strip().rstrip("=").strip()
    if len(text) > 2 and text[0] + text[-1] in ("[]", "()"):  # MinerU's standard tier prints [R]
        text = text[1:-1].strip()
    m = _VAR_RE.match(text)
    if not m:
        return None
    name = "linker" if m["name"].lower() == "linker" else re.sub(r"\s+", "", m["name"])
    return f"{name}{m['num']}{m['prime']}"


def _label_column(
    headers: Sequence[str], data: Sequence[Sequence[str]], skip: set[int]
) -> int | None:
    """The compound-label column: a label header over label values, else a first column of labels."""
    for pattern in (_LABEL_HEADER_RE, _ENTRY_HEADER_RE):
        for c, header in enumerate(headers):
            if c not in skip and pattern.match(header) and _mostly_labels(data, c):
                return c
    return 0 if 0 not in skip and _mostly_labels(data, 0) else None


def _mostly_labels(data: Sequence[Sequence[str]], c: int) -> bool:
    """True when at least `LABEL_SHARE` of the filled cells in column `c` start with a label."""
    values = [row[c] for row in data if row[c]]
    labelled = sum(bool(cell_labels(v)) for v in values)
    return bool(values) and labelled >= LABEL_SHARE * len(values)


def _leading_label(cell: str) -> tuple[str, str]:
    """Split ``8 [image:...]`` or ``6 precursor`` into the label and the rest of the cell."""
    first, _, rest = cell.strip().partition(" ")
    return (first, rest.strip()) if cell_labels(first) else ("", cell)


def _split_shared_column(
    headers: list[str], data: Sequence[Sequence[str]], variables: dict[str, int], shared: int
) -> tuple[list[str], list[list[str]], dict[str, int]]:
    """Move the labels out of a variable column that also holds them into a new first column."""
    rows = []
    for values in data:
        label, rest = _leading_label(values[shared])
        rows.append([label, *values[:shared], rest, *values[shared + 1 :]])
    moved = {name: c + 1 for name, c in variables.items()}
    return _unique(["label", *headers]), rows, moved


def _unique(headers: Sequence[str]) -> list[str]:
    """Headers made unique and non-empty."""
    out: list[str] = []
    for c, header in enumerate(headers):
        name = header or f"column {c + 1}"
        n = 2
        base = name
        while name in out:
            name, n = f"{base} ({n})", n + 1
        out.append(name)
    return out


def _split_row(row: dict[str, str], label_key: str, labels: list[str]) -> list[dict[str, str]]:
    """One row per label when every cell holds one value or one value per label, else []."""
    n = len(labels)
    columns: dict[str, list[str]] = {}
    for key, value in row.items():
        if key == label_key:
            continue
        parts = _VALUE_SPLIT_RE.split(value) if value else [value]
        if len(parts) not in (1, n):
            return []
        columns[key] = parts
    if not any(len(parts) == n for parts in columns.values()):
        return []
    return [
        {key: (label if key == label_key else _pick(columns[key], i)) for key in row}
        for i, label in enumerate(labels)
    ]


def _pick(parts: list[str], i: int) -> str:
    return parts[i] if len(parts) > 1 else parts[0]


def _data_rows(
    data: Sequence[Sequence[str]], headers: list[str], label_col: int
) -> list[dict[str, str]]:
    """Rows as header -> value dicts; section and empty rows skipped, label ranges split."""
    rows: list[dict[str, str]] = []
    label_key = headers[label_col]
    for values in data:
        filled = [v for v in values if v]
        if not filled or (len(filled) > 1 and len(set(filled)) == 1):
            continue
        row = dict(zip(headers, values, strict=True))
        labels = cell_labels(row[label_key])
        if len(labels) == 1:
            row[label_key] = labels[0]
        split = _split_row(row, label_key, labels) if len(labels) > 1 else []
        rows.extend(split or [row])
    return rows


def _stacked_panels(grid: list[list[str]], count: int) -> list[list[str]] | None:
    """A table printed as side-by-side copies (the header repeated along the row, a copy's header cell sometimes
    left blank) as one copy: its header rows, then each copy's data rows in turn. None when the header does not
    repeat with a compound-label header in every copy."""
    raw = _headers(grid, count)
    width = len(raw)
    for size in range(2, width // 2 + 1):
        if width % size:
            continue
        first = raw[:size]
        labels = [i for i, header in enumerate(first) if _LABEL_HEADER_RE.match(header)]
        copies = range(1, width // size)
        if labels and all(
            raw[k * size + i] in (header, "") and (i not in labels or raw[k * size + i] == header)
            for k in copies for i, header in enumerate(first)
        ):
            head = [row[:size] for row in grid[:count]]
            return head + [row[k * size:(k + 1) * size] for k in range(width // size) for row in grid[count:]]
    return None


def interpret_grid(block: LayoutBlock, grid: list[list[str]]) -> RGroupTable | None:
    """Read a parsed grid as a labelled table (variables may be empty); None without labels. A table printed as
    side-by-side copies is read as one (rows of the left copy first)."""
    if len(grid) < 2 or not grid[0]:
        return None
    count = _header_count(grid)
    grid = _stacked_panels(grid, count) or grid
    raw = _headers(grid, count)
    data = grid[count:]
    canonical = [variable_name(h) for h in raw]
    variables = _variable_columns(canonical, data)
    headers = _unique(
        [
            name if name and variables.get(name) == c else h
            for c, (h, name) in enumerate(zip(raw, canonical, strict=True))
        ]
    )
    label_col = _label_column(headers, data, set(variables.values()))
    if label_col is None:
        shared = next((c for c in variables.values() if _mostly_labels(data, c)), None)
        if shared is None:
            return None
        headers, data, variables = _split_shared_column(headers, data, variables, shared)
        label_col = 0
    taken = {label_col, *variables.values()}
    activity = [h for c, h in enumerate(headers) if c not in taken and ACTIVITY_RE.search(h)]
    return RGroupTable(
        table_id=f"p{block.page}-t{block.index}",
        page=block.page,
        bbox=list(block.bbox),
        label_col=label_col,
        variables=variables,
        rows=_data_rows(data, headers, label_col),
        activity_cols=activity,
    )


def _variable_columns(canonical: Sequence[str | None], data: Sequence[Sequence[str]]) -> dict[str, int]:
    """Each variable's column: the first under its header that does not hold the labels (one header spanning the
    label column and the column beside it, as [R] over "8 | drawing", names the column beside it), else the first."""
    columns: dict[str, list[int]] = {}
    for c, name in enumerate(canonical):
        if name:
            columns.setdefault(name, []).append(c)
    return {name: next((c for c in cs if not _mostly_labels(data, c)), cs[0]) for name, cs in columns.items()}


def parse_table_block(block: LayoutBlock) -> RGroupTable | None:
    """One table block read on its own as a labelled table."""
    if block.type != "table" or not block.html:
        return None
    return interpret_grid(block, parse_html_table(block.html))


def _captions(blocks: Iterable[LayoutBlock]) -> dict[tuple[int, int], list[str]]:
    out: dict[tuple[int, int], list[str]] = {}
    for b in blocks:
        if b.type == "caption" and b.parent is not None:
            out.setdefault((b.page, b.parent), []).append(b.text or "")
    return out


def _continues(
    prev: tuple[list[list[str]], int],
    block: LayoutBlock,
    grid: list[list[str]],
    captions: list[str],
    first_on_page: bool,
) -> bool:
    """True when `block` carries on the table that ended on the previous page."""
    prev_grid, prev_page = prev
    if not first_on_page or block.page != prev_page + 1 or len(grid[0]) != len(prev_grid[0]):
        return False
    if any(not re.search(r"continued|cont['’]d", c, re.IGNORECASE) for c in captions):
        return False
    return grid[0] == prev_grid[0] or any(_is_value(c) for c in grid[0])


def _without_repeated_header(grid: list[list[str]], prev_grid: list[list[str]]) -> list[list[str]]:
    n = _header_count(prev_grid)
    if grid[:n] == prev_grid[:n]:
        return grid[n:]
    return grid[1:] if grid[0] == prev_grid[0] else grid


def merged_table_grids(blocks: Sequence[LayoutBlock]) -> list[tuple[LayoutBlock, list[list[str]]]]:
    """Table blocks with their grids; a table MinerU split across a page break is joined."""
    captions = _captions(blocks)
    tables = sorted(
        (b for b in blocks if b.type == "table" and b.html), key=lambda b: (b.page, b.index)
    )
    merged: list[tuple[LayoutBlock, list[list[str]], int]] = []
    last_page = None
    for block in tables:
        grid = parse_html_table(block.html or "")
        first_on_page, last_page = block.page != last_page, block.page
        if not grid or not grid[0]:
            continue
        if merged:
            head, prev_grid, prev_page = merged[-1]
            own = captions.get((block.page, block.index), [])
            if _continues((prev_grid, prev_page), block, grid, own, first_on_page):
                merged[-1] = (
                    head,
                    prev_grid + _without_repeated_header(grid, prev_grid),
                    block.page,
                )
                continue
        merged.append((block, grid, block.page))
    return [(block, grid) for block, grid, _ in merged]


def labelled_tables(blocks: Sequence[LayoutBlock]) -> list[RGroupTable]:
    """Every table with a compound-label column, R-group or not."""
    found = (interpret_grid(block, grid) for block, grid in merged_table_grids(blocks))
    return [t for t in found if t is not None]


def find_rgroup_tables(blocks: list[LayoutBlock]) -> list[RGroupTable]:
    """Tables with a compound-label column and at least one variable column."""
    return [t for t in labelled_tables(blocks) if t.variables]
