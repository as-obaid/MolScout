"""Layout helpers of join_32b.py: BioMiner's request images and the table a 32B row belongs to.

Standard library only, Python 3.9+. Boxes and areas are (x, y, w, h) normalized to the page, as BioMiner
works; layout blocks are layout.jsonl records (PDF points).

Request planning follows BioMiner (jiaxianyan/BioMiner at 17c6161, runner/extractor.py and
commons/mineru_pdf.py; MIT licence, Copyright (c) 2025 jiaxianyan): per page and kind, boxes go in one
image per figure or table area when every box lies in one; a "full" image shows at most 4 boxes, cropped
without areas to their extent enlarged by 1/8; a "part" image shows all.

Table placement: a 32B row is placed in a table block when the first cell of one of the table's rows
reads as the row's identifier: the cell's whole text or its first word, compared after dropping case,
generic words ("compound 5" is "5"), brackets, trailing dots, spaces and dash variants. Only first cells
count, so a value such as an IC50 of "5" never places compound 5. Tables on the scaffold's page come
first, then those on the next page (a core drawn at the foot of the page before its table). A placed row
takes the table's id p{page}-t{index}, which is how the rules joiner (tables.py) names the table, and
its page and bbox. A row that cannot be placed (label column not first, label ranges, a table continued
from the page before) is left for merge to match by label alone.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import NamedTuple

SPLIT_FULL = 4  # structure_full_split_molecule_num
ENLARGE_FULL = 1 / 8  # structure_full_seg_enlarge
ENLARGE_AREA = 1 / 32  # get_mineru_complete_figure_table_bbox
GENERIC = re.compile(r"^(?:compounds?|compd|cpds?|ligands?|inhibitors?|analogues?|analogs?)\b\.?\s*")
DASHES = re.compile("[‐-―−]")


# ---------------------------------------------------------------- request planning (BioMiner's images)

@dataclass(frozen=True)
class Request:
    request_id: str
    kind: str
    page: int
    indices: tuple
    crop: tuple | None  # normalized x, y, w, h; None is the whole page


def enlarge(boxes, size):
    """get_segment_size: the boxes' extent, enlarged by size on each side and kept on the page."""
    x0 = max(min(b[0] for b in boxes) - size, 0)
    y0 = max(min(b[1] for b in boxes) - size, 0)
    x1 = min(max(b[0] + b[2] for b in boxes) + size, 1)
    y1 = min(max(b[1] + b[3] for b in boxes) + size, 1)
    return (x0, y0, x1 - x0, y1 - y0)


def inside(box, area):
    """check_bbox_in_area."""
    return (box[0] >= area[0] and box[0] + box[2] <= area[0] + area[2]
            and box[1] >= area[1] and box[1] + box[3] <= area[1] + area[3])


def page_areas(blocks, width_pt, height_pt):
    """A page's figure and table areas, tables first (get_mineru_complete_figure_table_bbox).

    An area is a table or image block with its children (captions, footnotes) when it has a caption.
    An uncaptioned image inside a table area is skipped; any other uncaptioned figure or table leaves
    the page without areas (None), as does a page with neither.
    """
    def norm(block):
        x0, y0, x1, y1 = (float(v) for v in block["bbox"])
        return (x0 / width_pt, y0 / height_pt, (x1 - x0) / width_pt, (y1 - y0) / height_pt)

    children = defaultdict(list)
    for block in blocks:
        if block.get("parent") is not None:
            children[block["parent"]].append(block)
    areas = {"table": [], "image": []}
    for kind in ("table", "image"):
        for block in sorted((b for b in blocks if b.get("type") == kind), key=lambda b: b["index"]):
            kids = children.get(block["index"], [])
            if any(kid.get("type") == "caption" for kid in kids):
                areas[kind].append(enlarge([norm(block), *map(norm, kids)], ENLARGE_AREA))
            elif not (kind == "image" and any(inside(norm(block), table) for table in areas["table"])):
                return None
    return areas["table"] + areas["image"] or None


def segments(kind, boxes, areas):
    """(indices, crop) of each image for one page and kind (segment_image_by_layout, segment_image_to_new_images)."""
    crop_groups = kind == "full" and areas is None  # structure_full_seg_image applies only without areas
    parts = [(None, boxes)]
    if areas is not None:
        assigned = [next((n for n, area in enumerate(areas) if inside(b.xywh, area)), None) for b in boxes]
        if None not in assigned:
            parts = [(area, [b for b, a in zip(boxes, assigned) if a == n]) for n, area in enumerate(areas)]
    images = []
    for area, members in parts:
        groups = [members[i:i + SPLIT_FULL] for i in range(0, len(members), SPLIT_FULL)] if kind == "full" else [members]
        for group in groups:
            if group:
                crop = enlarge([b.xywh for b in group], ENLARGE_FULL) if crop_groups else area
                images.append((tuple(b.index for b in group), crop))
    return images


def plan_requests(paper_id, boxes, areas):
    """BioMiner's structure requests for a paper: per page, the full images, then the part images."""
    requests = []
    for page in sorted({b.page for b in boxes}):
        for kind in ("full", "part"):
            members = [b for b in boxes if b.page == page and b.kind == kind]
            for n, (indices, crop) in enumerate(segments(kind, members, areas.get(page)) if members else []):
                requests.append(Request(f"p{page}-{kind}-{n}", kind, page, indices, crop))
    return requests


# ---------------------------------------------------------------- the table of a row

class Table(NamedTuple):
    table_id: str  # p{page}-t{index}
    page: int
    bbox: list  # PDF points
    labels: frozenset  # normalized first cells of its rows


def normalize(text) -> str:
    """'Compound 5a.', '(5a)' and '5–a' all read '5a'; merge.aliases does the same."""
    text = GENERIC.sub("", DASHES.sub("-", str(text or "").strip().casefold()))
    return re.sub(r"\s+", "", text.strip(" .,:;()[]{}"))


def aliases(identifier) -> set:
    """An identifier's normalized aliases (32b joins them with ';')."""
    return {alias for alias in (normalize(part) for part in str(identifier or "").split(";")) if alias}


class _FirstCells(HTMLParser):
    """The text of each table row's first cell."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cells: list = []
        self._count = 0
        self._text = None

    def handle_starttag(self, tag, attrs) -> None:
        if tag == "tr":
            self._count = 0
        elif tag in ("td", "th"):
            self._count += 1
            self._text = [] if self._count == 1 else None

    def handle_endtag(self, tag) -> None:
        if tag in ("td", "th") and self._text is not None:
            self.cells.append(" ".join("".join(self._text).split()))
            self._text = None

    def handle_data(self, data) -> None:
        if self._text is not None:
            self._text.append(data)


def first_cell_labels(html) -> frozenset:
    """The normalized first cells of a table's rows, whole and by first word."""
    parser = _FirstCells()
    parser.feed(html or "")
    parser.close()
    found = set()
    for cell in parser.cells:
        words = GENERIC.sub("", cell.casefold()).split()  # "Compound 3 (ZD1839)": "3"
        found |= aliases(cell) | (aliases(words[0]) if words else set())
    return frozenset(found)


def layout_tables(blocks) -> list:
    """The table blocks of a paper's layout.jsonl records that carry HTML."""
    return [Table(f"p{int(b['page'])}-t{int(b['index'])}", int(b["page"]), [float(v) for v in b["bbox"]],
                  first_cell_labels(b["html"]))
            for b in blocks if b.get("type") == "table" and b.get("html")]


def place(identifier, page: int, tables) -> Table | None:
    """The table whose row the identifier labels, on the scaffold's page first, then the next; else None."""
    wanted = aliases(identifier)
    for candidate in (page, page + 1):
        for table in tables:
            if table.page == candidate and wanted & table.labels:
                return table
    return None
