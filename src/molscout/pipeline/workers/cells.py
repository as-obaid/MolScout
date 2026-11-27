"""Cut the drawings of R-group table cells for the readers. Runs in the decimer_ai environment (Python 3.9).

MinerU keeps a drawn table cell as <img src="layout_images/..."> inside the cell's <td>: a 200 dpi JPEG without a
box. For each paper of the manifest this stage reads layout.jsonl (MinerU's tables), pages.jsonl (the 300 dpi
pages) and regions.jsonl (the segment stage's drawings), and:

1. finds each table image on the 300 dpi page: MinerU's image is matched against the table's area by normalized
   cross-correlation, at a few scales around the ratio of the two renders. A match of at least MATCH_MIN is cut
   from the page there ("matched"); otherwise MinerU's own image, scaled up by that ratio, is used ("upscaled");
2. segments the area of every table that holds an image with DECIMER Segmentation, as segment.py segments a page,
   and keeps each drawing that no located image covers ("segmented"): MinerU drops some drawn cells, and link
   gives such a drawing to the row it sits in;
3. cuts each of these into its drawings: ink split at white rows GAP_PX tall (MinerU and DECIMER sometimes take the
   drawings of two rows as one), leaving out the bits of neighbouring text a crop catches at its edges. Each
   drawing is saved alone on white.

Every drawing becomes a region in regions.jsonl with kind "cell". The classifier's score is kept, but a cell is
always is_structure true: small fragments score like non-structures, and the gate must not drop them.

    {"region_id": "p3-t5-r6-c0", "page": 3, "bbox": [328.1, 317.2, 374.6, 338.0], "crop": "cells/p3-t5-r6-c0.png",
     "structure_score": 3.1e-07, "is_structure": true, "kind": "cell"}

Region ids: p<page>-t<block index>-r<row>-c<column> for an image in grid row r and column c of the table (counted
as tables.parse_html_table counts them, with spans expanded; a second image in the same cell adds "-i2"), and
p<page>-t<block index>-d<k> for the k-th drawing segmentation found that no image covers; the second and later
drawings cut from one image or segment add "-p2", "-p3". cells.jsonl, this stage's output, describes every cell
region:

    {"region_id": "p3-t5-r6-c0", "table": "p3-t5", "page": 3, "row": 6, "col": 0,
     "src": "layout_images/p3-5-c5.jpg", "part": 1, "parts": 2, "method": "matched", "match": 0.97,
     "bbox": [x0, y0, x1, y1]}

bbox is in PDF points, origin top-left (an upscaled image has bbox null, since where it sits is not known, and its
region gets the table's box); row, col and src are null for a segmented drawing. The paper's counts go to
<paper_dir>/.counts/cells.json (see write_counts). A rerun replaces the paper's cell regions, crops, cells.jsonl and
counts, and removes the reads and votes made from the old regions. A table that fails is a line in errors.jsonl
(the other tables are kept and the paper counts as failed). Exit status 0, 4 when some papers failed, 1 when the models do not load or
files cannot be locked, 2 on a bad manifest.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _readers_common as common  # noqa: E402

STAGE = "cells"
CELLS = "cells.jsonl"
COUNTS = Path(".counts") / ("%s.json" % STAGE)  # the paper's counts, which the runner records
CROPS = "cells"
KIND = "cell"
MATCH_MIN = 0.6  # normalized cross-correlation a located image needs (real cells match at 0.9 and above)
MINERU_DPI = 200  # the scale MinerU renders at, when its table image cannot give the ratio
SCALE_STEPS = (0.97, 1.0, 1.03)
PEAKS = 4  # candidate places per image, so that two images of one drawing take different places
AREA_PAD_PT = 12.0  # margin around the table box when cutting its area from the page
OVERLAP = 0.5  # a drawing covered this much by a located image is that image
MAX_SHARE = 0.5  # a drawing larger than this share of the table area is the table, not a cell
INK = 160  # a gray level under this is ink
GAP_PX = 8  # white rows this tall (at 300 dpi) part two drawings stacked in one image
FRAGMENT = 0.4  # a piece at the crop's edge smaller than this share of the largest is a neighbour's text
CUT_SHARE = 0.1  # a connected piece of ink crossing the box's top or bottom edge under this share of the ink is text
PAD_PX = 12  # white around each cut drawing
MAX_SPAN = 100
CROP = re.compile(r"p\d+-t\d+-(?:r\d+-c\d+(?:-i\d+)?|d\d+)(?:-p\d+)?\.png")


# --- the images of a table's cells ------------------------------------------------------------------


class _Grid(HTMLParser):
    """The <img> of each cell with its grid row and column, rowspan and colspan expanded as
    tables.parse_html_table expands them."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[int, int, list[str]]]] = []
        self._row: list[tuple[int, int, list[str]]] | None = None
        self._cell: tuple[int, int, list[str]] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._close_row()
            self._row = []
        elif tag in ("td", "th"):
            self._close_cell()
            if self._row is None:
                self._row = []
            spans = dict(attrs)
            self._cell = (_span(spans.get("rowspan")), _span(spans.get("colspan")), [])
        elif tag == "img" and self._cell is not None:
            src = dict(attrs).get("src")
            if src:
                self._cell[2].append(src)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._close_cell()
        elif tag in ("tr", "table"):
            self._close_row()

    def close(self):
        super().close()
        self._close_row()

    def _close_cell(self):
        if self._cell is not None and self._row is not None:
            self._row.append(self._cell)
        self._cell = None

    def _close_row(self):
        self._close_cell()
        if self._row is not None:
            self.rows.append(self._row)
        self._row = None


def _span(value) -> int:
    try:
        return min(max(int(value or 1), 1), MAX_SPAN)
    except ValueError:
        return 1


def table_images(html: str) -> list[tuple[int, int, str]]:
    """(row, column, src) of every <img> in a table's cells, in document order; the place of a spanning cell is its
    top-left slot."""
    parser = _Grid()
    parser.feed(html or "")
    parser.close()
    taken: set[tuple[int, int]] = set()
    images = []
    for r, row in enumerate(parser.rows):
        c = 0
        for rowspan, colspan, sources in row:
            while (r, c) in taken:
                c += 1
            images += [(r, c, src) for src in sources]
            taken.update((r + dr, c + dc) for dr in range(rowspan) for dc in range(colspan))
            c += colspan
    return images


# --- locating an image on the page --------------------------------------------------------------------


def _gray(image):
    import numpy as np

    return np.asarray(image.convert("L"), dtype=np.float64)


def _resized(image, scale: float):
    from PIL import Image

    width, height = image.size
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(size, getattr(Image, "Resampling", Image).BICUBIC)


def _window_sums(values, height: int, width: int):
    import numpy as np

    total = np.pad(values.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return total[height:, width:] - total[:-height, width:] - total[height:, :-width] + total[:-height, :-width]


def correlation(area, template):
    """Normalized cross-correlation of a template at every place it fits in area (as OpenCV's TM_CCOEFF_NORMED),
    or None when the template is blank or larger than the area."""
    import numpy as np

    height, width = template.shape
    if height > area.shape[0] or width > area.shape[1]:
        return None
    centred = template - template.mean()
    norm = float(np.sqrt((centred * centred).sum()))
    if norm < 1e-6:
        return None
    shape = (area.shape[0] + height - 1, area.shape[1] + width - 1)
    product = np.fft.rfft2(area, shape) * np.fft.rfft2(centred[::-1, ::-1], shape)
    numerator = np.fft.irfft2(product, shape)[height - 1:area.shape[0], width - 1:area.shape[1]]
    sums = _window_sums(area, height, width)
    variance = _window_sums(area * area, height, width) - sums * sums / (height * width)
    denominator = norm * np.sqrt(np.maximum(variance, 0.0))
    return np.where(denominator > 1e-3 * norm, numerator / np.maximum(denominator, 1e-12), 0.0)


def peaks(scores, height: int, width: int, count: int = PEAKS) -> list[tuple[float, int, int]]:
    """The best (score, x, y) places, each at least half a template away from the better ones."""
    work = scores.copy()
    found = []
    for _ in range(count):
        index = int(work.argmax())
        y, x = divmod(index, work.shape[1])
        score = float(work[y, x])
        if score <= -1:
            break
        found.append((score, x, y))
        work[max(0, y - height // 2):y + height // 2 + 1, max(0, x - width // 2):x + width // 2 + 1] = -2
    return found


def candidates(area, image, scale: float) -> list[tuple[float, list[int]]]:
    """(score, [x0, y0, x1, y1] in area pixels) of the best places of image in area, over the scales around scale."""
    found = []
    for step in SCALE_STEPS:
        template = _gray(_resized(image, scale * step))
        scores = correlation(area, template)
        if scores is None:
            continue
        height, width = template.shape
        found += [(score, [x, y, x + width, y + height]) for score, x, y in peaks(scores, height, width)]
    return sorted(found, key=lambda item: -item[0])


def overlap(a, b) -> float:
    """The share of the smaller box that the two boxes share."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return width * height / smaller if smaller > 0 else 0.0


def place_images(area, images: list, scale: float) -> list:
    """For each image (or None when it could not be read), the (score, box) where it sits best in area, or None.
    Images claim places best match first, and a place another image took (overlap > OVERLAP) is not reused."""
    options = [candidates(area, image, scale) if image is not None else [] for image in images]
    order = sorted(range(len(images)), key=lambda i: -(options[i][0][0] if options[i] else -2.0))
    placed: list = [None] * len(images)
    for i in order:
        for score, box in options[i]:
            if score < MATCH_MIN:
                break
            if all(other is None or overlap(box, other[1]) <= OVERLAP for other in placed):
                placed[i] = (score, box)
                break
    return placed


# --- one paper ----------------------------------------------------------------------------------------


def _scale(paper_dir: Path, block: dict, table_width_px: float, page: dict) -> float:
    """Page pixels per pixel of MinerU's images: the table's width on the page over MinerU's table image width."""
    from PIL import Image

    if block.get("image"):
        try:
            with Image.open(paper_dir / block["image"]) as image:
                if image.size[0] > 0:
                    return table_width_px / image.size[0]
        except OSError:
            pass
    return float(page.get("dpi") or 300) / MINERU_DPI


def _open(path: Path):
    from PIL import Image

    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except OSError:
        return None


def _margin(box, size) -> list[int]:
    """box widened by 6 px plus 6% of its size on every side, kept inside size (width, height)."""
    x0, y0, x1, y1 = box
    dx, dy = 6 + 0.06 * (x1 - x0), 6 + 0.06 * (y1 - y0)
    return [max(0, int(x0 - dx)), max(0, int(y0 - dy)), min(size[0], int(x1 + dx)), min(size[1], int(y1 + dy))]


def _runs(marks, gap: int) -> list[tuple[int, int]]:
    """(start, end) of the runs of True in a 1-D array; runs fewer than `gap` False values apart are one run."""
    import numpy as np

    found = np.flatnonzero(marks)
    if found.size == 0:
        return []
    runs, start, previous = [], int(found[0]), int(found[0])
    for index in found[1:]:
        if index - previous > gap:
            runs.append((start, previous + 1))
            start = int(index)
        previous = int(index)
    runs.append((start, previous + 1))
    return runs


def _small_at_edge(start: int, end: int, low: int, high: int, largest: int) -> bool:
    """A piece from start to end, small next to the largest, that reaches the edge low..high of the box it was cut
    by."""
    return (start <= low + 1 or end >= high - 1) and end - start < FRAGMENT * largest


def _component(ink, start) -> list[tuple[int, int]]:
    """The 8-connected ink pixels joined to `start` (y, x)."""
    height, width = ink.shape
    seen, todo = {start}, [start]
    while todo:
        y, x = todo.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                near = (y + dy, x + dx)
                if near not in seen and 0 <= near[0] < height and 0 <= near[1] < width and ink[near]:
                    seen.add(near)
                    todo.append(near)
    return list(seen)


def _erase_cut_text(ink, row: int) -> None:
    """Clear each small piece of ink that crosses `row`, the top or bottom edge of the box a crop was widened from:
    the letters of the row above or below that the box cut. A piece is small under CUT_SHARE of the ink."""
    if not 0 <= row < ink.shape[0]:
        return
    total = int(ink.sum())
    done: set = set()
    for x in [int(x) for x in ink[row].nonzero()[0]]:
        if (row, x) in done:
            continue
        piece = _component(ink, (row, x))
        done.update(piece)
        if len(piece) < CUT_SHARE * total:
            ys, xs = zip(*piece)
            ink[list(ys), list(xs)] = False


def drawings_in(gray, inner=None) -> list[list[int]]:
    """[x0, y0, x1, y1] of each drawing in a crop, top to bottom.

    Ink is cut into bands at rows of white GAP_PX or wider (table rows stack their drawings), each band trimmed to
    its ink. A band, or a piece at a band's side, that reaches the edge of `inner` (the box the crop was widened
    from; default the whole crop) and is small next to the largest is a neighbour's text that the box cut, and is
    dropped. A small band inside is a label of the drawing nearest it (a detached "HO" or "CF3") and joins it.
    """
    import numpy as np

    ink = gray < INK
    height, width = ink.shape
    left, top, right, bottom = inner if inner is not None else (0, 0, width, height)
    for row in (top, bottom - 1):
        _erase_cut_text(ink, row)
    bands = []
    for y0, y1 in _runs(ink.any(axis=1), GAP_PX):
        pieces = _runs(ink[y0:y1].any(axis=0), GAP_PX)
        widest = max(end - start for start, end in pieces)
        kept = [(start, end) for start, end in pieces if not _small_at_edge(start, end, left, right, widest)]
        if not kept:
            continue
        x0, x1 = kept[0][0], kept[-1][1]
        rows = np.flatnonzero(ink[y0:y1, x0:x1].any(axis=1))
        bands.append([x0, y0 + int(rows[0]), x1, y0 + int(rows[-1]) + 1])
    if not bands:
        return []
    tallest = max(y1 - y0 for _, y0, _, y1 in bands)
    small = [band[3] - band[1] < FRAGMENT * tallest for band in bands]
    drawings = [band for band, little in zip(bands, small) if not little]
    for band, little in zip(bands, small):
        if little and not _small_at_edge(band[1], band[3], top, bottom, tallest):
            near = min(drawings, key=lambda d: max(0, d[1] - band[3], band[1] - d[3]))
            near[:] = [min(near[0], band[0]), min(near[1], band[1]), max(near[2], band[2]), max(near[3], band[3])]
    return drawings


def _clean(image, box):
    """The pixels of box alone on white, with PAD_PX of white around them."""
    from PIL import Image

    piece = image.crop(box)
    canvas = Image.new("RGB", (piece.size[0] + 2 * PAD_PX, piece.size[1] + 2 * PAD_PX), "white")
    canvas.paste(piece, (PAD_PX, PAD_PX))
    return canvas


def pieces(image, box) -> list[tuple[object, list[int]]]:
    """(clean crop, box in image pixels) of each drawing in box of image; box itself when no drawing stands out."""
    outer = _margin(box, image.size)
    inner = [box[0] - outer[0], box[1] - outer[1], box[2] - outer[0], box[3] - outer[1]]
    found = drawings_in(_gray(image.crop(outer)), inner)
    boxes = [[x0 + outer[0], y0 + outer[1], x1 + outer[0], y1 + outer[1]] for x0, y0, x1, y1 in found]
    return [(_clean(image, part), part) for part in (boxes or [list(box)])]


def _shift(box, origin) -> list[int]:
    return [box[0] + origin[0], box[1] + origin[1], box[2] + origin[0], box[3] + origin[1]]


def table_cells(paper: common.Paper, block: dict, page: dict, page_image, models) -> tuple[list, list]:
    """The cell regions and cells.jsonl records of one table block (structure scores still unset)."""
    images = table_images(block.get("html") or "")
    if not images:
        return [], []
    size = page_image.size
    sx, sy = size[0] / page["width_pt"], size[1] / page["height_pt"]
    x0, y0, x1, y1 = block["bbox"]
    area_box = [max(0, int((x0 - AREA_PAD_PT) * sx)), max(0, int((y0 - AREA_PAD_PT) * sy)),
                min(size[0], int((x1 + AREA_PAD_PT) * sx) + 1), min(size[1], int((y1 + AREA_PAD_PT) * sy) + 1)]
    area_image = page_image.crop(area_box)
    scale = _scale(paper.paper_dir, block, (x1 - x0) * sx, page)
    sources = [_open(paper.paper_dir / src) for _, _, src in images]
    placed = place_images(_gray(area_image), sources, scale)
    table = "p%d-t%d" % (block["page"], block["index"])
    out: list = []
    located, seen = [], {}
    for (row, col, src), image, where in zip(images, sources, placed):
        if image is None:
            raise OSError("cannot read the table image %s" % src)
        seen[(row, col)] = seen.get((row, col), 0) + 1
        base = "%s-r%d-c%d" % (table, row, col) + ("-i%d" % seen[(row, col)] if seen[(row, col)] > 1 else "")
        cell = {"table": table, "page": block["page"], "row": row, "col": col, "src": src}
        if where is not None:
            score, box = where
            located.append(box)
            parts = pieces(page_image, _shift(box, area_box))
            out += _parts(base, cell, parts, "matched", round(score, 4), page, size)
        else:
            enlarged = _resized(image, scale)
            parts = pieces(enlarged, [0, 0, enlarged.size[0], enlarged.size[1]])
            out += _parts(base, cell, [(crop, None) for crop, _ in parts], "upscaled", None, page, size)
    for k, box in enumerate(_drawings(area_image, located, models), 1):
        cell = {"table": table, "page": block["page"], "row": None, "col": None, "src": None}
        parts = pieces(page_image, _shift(box, area_box))
        out += _parts("%s-d%d" % (table, k), cell, parts, "segmented", None, page, size)
    for region_id, crop, _ in out:
        crop.save(paper.paper_dir / CROPS / ("%s.png" % region_id))
    regions = [_region(region_id, block, record["bbox"] or list(block["bbox"])) for region_id, _, record in out]
    return regions, [record for _, _, record in out]


def _parts(base: str, cell: dict, parts: list, method: str, match, page: dict, size) -> list:
    """(region_id, crop, cells.jsonl record) for each drawing of one image or segment."""
    out = []
    for k, (crop, box) in enumerate(parts, 1):
        region_id = base if k == 1 else "%s-p%d" % (base, k)
        bbox = common.px_to_points(box, page, size) if box is not None else None
        record = {"region_id": region_id, **cell, "part": k, "parts": len(parts), "method": method, "match": match,
                  "bbox": bbox}
        out.append((region_id, crop, record))
    return out


def _drawings(area_image, located: list, models) -> list:
    """[x0, y0, x1, y1] in area pixels of every drawing DECIMER finds in the table area that no located image
    covers and that is not most of the table."""
    handle, name = tempfile.mkstemp(suffix=".png", prefix="molscout_cells_")
    os.close(handle)
    try:
        area_image.save(name)
        _, segments = models.segment(Path(name))
    finally:
        os.unlink(name)
    width, height = area_image.size
    kept = []
    for _, (y0, x0, y1, x1) in segments:
        box = [int(x0), int(y0), int(x1), int(y1)]
        if (x1 - x0) * (y1 - y0) > MAX_SHARE * width * height:
            continue
        if any(overlap(box, other) > OVERLAP for other in located):
            continue
        kept.append(box)
    return kept


def _region(region_id: str, block: dict, bbox: list) -> dict:
    return {"region_id": region_id, "page": block["page"], "bbox": bbox, "crop": "%s/%s.png" % (CROPS, region_id),
            "structure_score": None, "is_structure": True, "kind": KIND}


def cells_paper(paper: common.Paper, models) -> tuple[list, int]:
    """Cut every table cell drawing of a paper; write its cell regions, cells.jsonl and its counts. Returns the
    cells.jsonl records and the number of tables that failed (each a line in errors.jsonl)."""
    from PIL import Image

    paper_dir = paper.paper_dir
    pages = {page["page"]: page for page in common.read_jsonl(paper_dir / "pages.jsonl")}
    blocks = [b for b in common.read_jsonl(paper_dir / "layout.jsonl") if b.get("type") == "table" and b.get("html")]
    drawings = [r for r in common.read_jsonl(paper_dir / common.REGIONS) if r.get("kind") != KIND]
    common.remove_outputs(paper_dir, (CELLS, COUNTS, *common.AFTER_REGIONS))
    crops = paper_dir / CROPS
    crops.mkdir(parents=True, exist_ok=True)
    for stale in crops.iterdir():
        if CROP.fullmatch(stale.name):
            stale.unlink()
    regions, cells, failed, lost = [], [], 0, 0
    opened: dict = {}
    for block in blocks:
        try:
            page = pages[block["page"]]
            if block["page"] not in opened:
                with Image.open(paper_dir / page["image"]) as image:
                    opened = {block["page"]: image.convert("RGB")}
            more_regions, more_cells = table_cells(paper, block, page, opened[block["page"]], models)
        except Exception as exc:
            failed += 1
            lost += len(table_images(block.get("html") or ""))
            common.append_error(paper_dir, STAGE, paper.paper_id, common.describe(exc),
                                table="p%s-t%s" % (block.get("page"), block.get("index")))
            continue
        regions += more_regions
        cells += more_cells
    for region in regions:
        region["structure_score"] = common.finite_or_none(models.score(paper_dir / region["crop"]))
    common.write_jsonl(paper_dir / common.REGIONS, drawings + regions)
    common.write_jsonl(paper_dir / CELLS, cells)
    write_counts(paper_dir, cells, lost)
    return cells, failed


def write_counts(paper_dir: Path, cells: list, lost: int) -> dict:
    """The paper's counts for the runner (.counts/cells.json): `in` images MinerU kept in table cells = `out`
    images cut (`matched` on the page or `upscaled`) + `dropped_failed_table`; `segmented` drawings found where
    MinerU kept no image, `split` images holding several drawings, and `drawings` the cell regions written."""
    images = {}
    for cell in cells:
        if cell["src"] is not None:
            images[(cell["table"], cell["row"], cell["col"], cell["src"])] = (cell["method"], cell["parts"])
    counts = {
        "in": len(images) + lost,
        "out": len(images),
        "dropped_failed_table": lost,
        "matched": sum(method == "matched" for method, _ in images.values()),
        "upscaled": sum(method == "upscaled" for method, _ in images.values()),
        "split": sum(parts > 1 for _, parts in images.values()),
        "segmented": sum(cell["method"] == "segmented" and cell["part"] == 1 for cell in cells),
        "drawings": len(cells),
    }
    (paper_dir / COUNTS).parent.mkdir(parents=True, exist_ok=True)
    (paper_dir / COUNTS).write_text(json.dumps(counts) + "\n", encoding="utf-8")
    return counts


def run(papers: list, models, logger) -> int:
    """Cut the cells of every paper; return how many failed, as a whole or on some table. LockError stops the run."""
    failed = 0
    for paper in papers:
        start = time.perf_counter()
        try:
            common.clear_stage_errors(paper.paper_dir, STAGE)
            cells, failed_tables = cells_paper(paper, models)
        except common.LockError:
            raise
        except Exception as exc:
            failed += 1
            common.append_error(paper.paper_dir, STAGE, paper.paper_id, common.describe(exc))
            logger.error("%s: %s", paper.paper_id, common.describe(exc))
            continue
        failed += failed_tables > 0
        methods = {}
        for cell in cells:
            methods[cell["method"]] = methods.get(cell["method"], 0) + 1
        logger.info("%s: %d cell drawings %s, %d tables failed, %.1f s", paper.paper_id, len(cells),
                    dict(sorted(methods.items())), failed_tables, time.perf_counter() - start)
    logger.info("%d papers, %d failed", len(papers), failed)
    return failed


def load_models(device: str, logger):
    import segment

    return segment.load_models(device, logger)


def main(argv=None) -> int:
    parser = common.worker_parser(__doc__.split("\n\n")[0])
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args(argv)
    logger = common.open_log(args.log.resolve(), STAGE)
    previous = os.getcwd()
    # The classifier writes "caffeine_mod.png" into the working directory: work from a scratch folder.
    scratch = tempfile.TemporaryDirectory(prefix="molscout_cells_")
    try:
        try:
            papers = common.read_manifest(args.manifest)
        except common.ManifestError as exc:
            logger.error("bad manifest: %s", exc)
            return 2
        os.chdir(scratch.name)
        try:
            models = load_models(args.device, logger)
        except Exception as exc:
            logger.error("cannot load DECIMER Segmentation and the classifier: %s", common.describe(exc))
            return 1
        try:
            failed = run(papers, models, logger)
        except common.LockError as exc:
            logger.error("%s", exc)
            return 1
        return common.PARTIAL if failed else 0
    finally:
        os.chdir(previous)
        scratch.cleanup()
        common.close_log(logger)


if __name__ == "__main__":
    sys.exit(main())
