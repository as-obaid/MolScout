"""Drawn R groups in table cells: the drawings the cells stage cut, and the row each one belongs to.

The cells stage (workers/cells.py) writes cells.jsonl beside regions.jsonl, one record per drawing it cut from a
table: the MinerU image it came from (`src`; None for a drawing DECIMER found where MinerU kept no image), which
drawing of that image it is (`part` of `parts`), and its box on the page (None when the image was not found on
the page). A variable cell of a table row holds `[image:<src>]` for each image MinerU kept in it.

`column_drawings` gives each row of one variable column its drawing:

- by position, when the PDF's text places the rows' labels: every drawing in the column's x range goes to the row
  whose label is nearest vertically (within ROW_REACH of the gap to the next label that way). A row takes the one drawing
  it gets, or its own image's drawing when it gets several. The positions count only when every located
  image of a single drawing falls in its own row; then a drawing MinerU dropped, or one MinerU put in the
  image of the row above, reaches its row;
- else by the image in the row's own cell, when that image holds one drawing.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from molscout.pipeline.schema import read_jsonl

CELLS = "cells.jsonl"
KIND = "cell"  # Region.kind of a table-cell drawing
IMAGE_TOKEN = re.compile(r"\[image(?::([^\]]*))?\]")
ROW_REACH = 0.6  # a drawing belongs to the nearest label within this share of the gap to the next label
BAND_SLACK = 0.15  # the column's x range is widened by this share of its width
WORD_GAP = 0.6  # a gap between characters wider than this share of their height parts two words
_LABEL_TRIM = re.compile(r"^[(\[]+|[)\],;:.*†‡§]+$")


@dataclass(frozen=True)
class Cell:
    """One drawing the cells stage cut from a table (cells.jsonl)."""

    region_id: str
    table: str
    page: int
    row: int | None
    col: int | None
    src: str | None
    method: str
    part: int = 1
    parts: int = 1
    match: float | None = None
    bbox: list[float] | None = None


@dataclass(frozen=True)
class Word:
    """A word of the PDF's text and its box in points, origin top-left."""

    text: str
    box: tuple[float, float, float, float]

    @property
    def centre(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


@dataclass(frozen=True)
class ColumnDrawings:
    """The drawing each row of a variable column gets (row index -> Cell), why some rows get none, and how."""

    by_row: dict[int, Cell] = field(default_factory=dict)
    problems: dict[int, str] = field(default_factory=dict)
    by_position: bool = False


def read_cells(paper_dir: Path) -> list[Cell]:
    """The cells stage's records; [] when the stage did not run."""
    path = Path(paper_dir) / CELLS
    return read_jsonl(path, Cell) if path.is_file() else []


def image_sources(value: str) -> list[str]:
    """The MinerU images a table cell's text holds, in order (an image without a src counts as "")."""
    return [src or "" for src in IMAGE_TOKEN.findall(value or "")]


def text_beside_images(value: str) -> str:
    """A cell's text without its image tokens."""
    return " ".join(IMAGE_TOKEN.sub(" ", value or "").split())


def centre(box: Sequence[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


# --- where the rows are ----------------------------------------------------------------------------


def pdf_words(pdf: Path | str, page_number: int) -> list[Word]:
    """The words of one page of the PDF (1-based) with boxes in points from the page's top-left corner; [] for a
    rotated page or a page without text."""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(pdf))
    try:
        page = document[page_number - 1]
        try:
            if page.get_rotation() % 360:
                return []
            left, _, _, top = page.get_cropbox()
            textpage = page.get_textpage()
            try:
                chars = [(textpage.get_text_range(i, 1), textpage.get_charbox(i)) for i in range(textpage.count_chars())]
            finally:
                textpage.close()
        finally:
            page.close()
    finally:
        document.close()
    return _words(chars, left, top)


def _words(chars: Sequence[tuple[str, tuple[float, float, float, float]]], left: float, top: float) -> list[Word]:
    """Characters (text, (left, bottom, right, top) in PDF space) joined into words: a space, a gap wider than
    WORD_GAP of the character's height (ink boxes: a narrow "1" leaves a wide gap) or a jump in the baseline starts
    a new word."""
    words: list[Word] = []
    text, box = "", None
    for char, (x0, y0, x1, y1) in chars:
        blank = not char.strip() or (x1 <= x0 and y1 <= y0)
        height = y1 - y0
        jump = box is not None and abs((y0 + y1) / 2 - (box[1] + box[3]) / 2) > height / 2
        if box is not None and (blank or x0 - box[2] > WORD_GAP * height or jump):
            words.append(Word(text, (box[0] - left, top - box[3], box[2] - left, top - box[1])))
            text, box = "", None
        if blank:
            continue
        text += char
        box = (x0, y0, x1, y1) if box is None else (min(box[0], x0), min(box[1], y0), max(box[2], x1), max(box[3], y1))
    if box is not None:
        words.append(Word(text, (box[0] - left, top - box[3], box[2] - left, top - box[1])))
    return words


def _same_label(word: str, label: str) -> bool:
    return _LABEL_TRIM.sub("", word).casefold() == label.casefold()


def label_rows(words: Sequence[Word], area: Sequence[float], labels: Sequence[str | None]) -> list[float | None]:
    """The vertical centre of each row's label: the leftmost word inside area that reads as the label. None when a
    row has no label or its label is not found, and for every row when the labels found are not top to bottom."""
    inside = [w for w in words if area[0] - 2 <= w.centre[0] <= area[2] + 2 and area[1] - 2 <= w.centre[1] <= area[3] + 2]
    found: list[float | None] = []
    for label in labels:
        matches = [w for w in inside if label and _same_label(w.text, label)]
        found.append(min(matches, key=lambda w: (w.box[0], w.box[1])).centre[1] if matches else None)
    placed = [y for y in found if y is not None]
    if any(b <= a for a, b in zip(placed, placed[1:])):
        return [None] * len(labels)
    return found


def nearest_row(y: float, rows_y: Sequence[float | None]) -> int | None:
    """The row whose label is nearest y, when y is within ROW_REACH of the gap from that label to the next label on
    y's side (the label on the other side when there is none)."""
    placed = [(i, ry) for i, ry in enumerate(rows_y) if ry is not None]
    if len(placed) < 2:
        return None
    position = min(range(len(placed)), key=lambda k: abs(placed[k][1] - y))
    index, ry = placed[position]
    toward = position + 1 if y > ry else position - 1
    other = toward if 0 <= toward < len(placed) else 2 * position - toward
    return index if abs(y - ry) <= ROW_REACH * abs(placed[other][1] - ry) else None


# --- the drawing of each row ------------------------------------------------------------------------


def column_drawings(
    values: Sequence[str],
    table_id: str,
    area: Sequence[float],
    labels: Sequence[str | None],
    cells: Sequence[Cell],
    words: Callable[[], Sequence[Word]] | None = None,
) -> ColumnDrawings:
    """The drawing for each row of one variable column.

    `values` are the column's cells row by row, `labels` the rows' labels, `area` the table's box and `table_id`
    its id; `words()` gives the PDF words of the table's page (None: no PDF).
    """
    by_src: dict[str, list[Cell]] = {}
    for cell in cells:
        if cell.src is not None:
            by_src.setdefault(cell.src, []).append(cell)
    sources = [image_sources(value) for value in values]
    located = _assign_by_position(sources, table_id, area, labels, cells, by_src, words)
    result = ColumnDrawings(by_position=located is not None)
    for row, own in enumerate(sources):
        mine = [cell for src in own for cell in by_src.get(src, [])]
        got = located.get(row, []) if located is not None else []
        if len(got) == 1:
            result.by_row[row] = got[0]
        elif got and (ours := [cell for cell in got if cell in mine]) and len(ours) == 1:
            result.by_row[row] = ours[0]
        elif got:
            result.problems[row] = f"{len(got)} drawings fall in the row"
        elif len(own) > 1:
            result.problems[row] = f"the cell holds {len(own)} images"
        elif own and not mine:
            result.problems[row] = "no cell drawing was cut for its image (the cells stage did not run or failed)"
        elif own and len(mine) > 1:
            result.problems[row] = f"its image holds {len(mine)} drawings and the rows' positions are not known"
        elif own:
            result.by_row[row] = mine[0]
    return result


def _assign_by_position(
    sources: Sequence[list[str]],
    table_id: str,
    area: Sequence[float],
    labels: Sequence[str | None],
    cells: Sequence[Cell],
    by_src: Mapping[str, list[Cell]],
    words: Callable[[], Sequence[Word]] | None,
) -> dict[int, list[Cell]] | None:
    """Row -> the drawings whose position puts them in it, or None when positions cannot be trusted."""
    own_row = {src: row for row, found in enumerate(sources) for src in found}
    mine = [cell for src in own_row for cell in by_src.get(src, []) if cell.bbox is not None and cell.table == table_id]
    singles = [cell for cell in mine if cell.parts == 1]
    if words is None or not singles:
        return None
    rows_y = label_rows(words(), area, labels)
    x0 = min(cell.bbox[0] for cell in mine)
    x1 = max(cell.bbox[2] for cell in mine)
    slack = BAND_SLACK * (x1 - x0)
    candidates = mine + [
        cell for cell in cells
        if cell.table == table_id and cell.src is None and cell.bbox is not None
        and x0 - slack <= centre(cell.bbox)[0] <= x1 + slack
    ]
    placed: dict[int, list[Cell]] = {}
    for cell in candidates:
        row = nearest_row(centre(cell.bbox)[1], rows_y)
        if cell.parts == 1 and cell.src is not None and row != own_row[cell.src]:
            return None  # a located image of one drawing outside its own row: the positions are wrong
        if row is not None:
            placed.setdefault(row, []).append(cell)
    return placed
