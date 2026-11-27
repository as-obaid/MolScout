"""workers/cells.py (decimer_ai env): table-cell images found on the 300 dpi page, dropped drawings from DECIMER.

The page and MinerU's images are drawn with RDKit; DECIMER Segmentation and the classifier are fakes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

from molscout.pipeline.tables import parse_html_table

WORKERS = Path(__file__).resolve().parents[2] / "src" / "molscout" / "pipeline" / "workers"


def _load(name):
    sys.path.insert(0, str(WORKERS))
    try:
        if name.startswith("_"):
            return importlib.import_module(name)
        spec = importlib.util.spec_from_file_location(f"i2_{name}", WORKERS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(WORKERS))


common = _load("_readers_common")
cells = _load("cells")

PAGE_PT = (200.0, 300.0)  # a page 200 x 300 pt, rendered at 300 dpi: 834 x 1250 px
PX = 300 / 72


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def drawing(smiles, size):
    return Draw.MolToImage(Chem.MolFromSmiles(smiles), size=size).convert("RGB")


class FakeModels:
    """DECIMER Segmentation and the classifier: `boxes` are (y0, x0, y1, x1) in page pixels; segment() gets the table
    area and answers in area pixels, shifted by the area's top-left corner `origin`."""

    threshold = 0.000089

    def __init__(self, boxes=(), score=0.9):
        self.boxes, self.score_value, self.origin = list(boxes), score, (0, 0)
        self.segmented = []

    def segment(self, image):
        with Image.open(image) as area:
            size = area.size
        self.segmented.append(size)
        ox, oy = self.origin
        return size, [(np.zeros((4, 4, 3), dtype=np.uint8), (y0 - oy, x0 - ox, y1 - oy, x1 - ox))
                      for y0, x0, y1, x1 in self.boxes]

    def save(self, segment, path):
        Image.fromarray(segment).save(path)

    def score(self, path):
        return self.score_value


# Three drawings in the R column of a table at [20, 40, 180, 280] pt; MinerU kept the first two as images.
TABLE_PT = [20.0, 40.0, 180.0, 280.0]
DRAWN = {  # top-left corner in page pixels, SMILES, size in page pixels
    "row1": ((150, 300), "*Cc1ccccc1", (180, 120)),
    "row2": ((150, 560), "*Cc1ccc(Cl)cc1", (180, 120)),
    "row3": ((150, 820), "*CC1CCCCC1", (180, 120)),
}
HTML = ("<table><tr><td>No.</td><td>R</td><td>IC50</td></tr>"
        "<tr><td>1</td><td><img src='layout_images/a.jpg'/></td><td>3</td></tr>"
        "<tr><td>2</td><td><img src='layout_images/b.jpg'/></td><td>4</td></tr>"
        "<tr><td>3</td><td></td><td>5</td></tr></table>")


def make_paper(tmp_path, html=HTML, mineru=("row1", "row2"), regions=None):
    paper_dir = tmp_path / "p1"
    (paper_dir / "pages").mkdir(parents=True)
    (paper_dir / "layout_images").mkdir()
    page = Image.new("RGB", (834, 1250), "white")
    for (x, y), smiles, size in DRAWN.values():
        page.paste(drawing(smiles, size), (x, y))
    page.save(paper_dir / "pages" / "p001.png")
    for name, key in zip(("a", "b"), mineru):
        (x, y), _, (w, h) = DRAWN[key]
        image = page.crop((x, y, x + w, y + h))  # MinerU renders at 200 dpi: two thirds of the 300 dpi page
        image.resize((round(w * 2 / 3), round(h * 2 / 3)), Image.BICUBIC).save(paper_dir / "layout_images" / f"{name}.jpg")
    table_px = [round(v * PX) for v in TABLE_PT]
    page.crop(table_px).resize((round((table_px[2] - table_px[0]) * 2 / 3), round((table_px[3] - table_px[1]) * 2 / 3))
                               ).save(paper_dir / "layout_images" / "t.jpg")
    common.write_jsonl(paper_dir / "pages.jsonl", [{"page": 1, "width_pt": PAGE_PT[0], "height_pt": PAGE_PT[1],
                                                    "image": "pages/p001.png", "dpi": 300, "vector": True}])
    common.write_jsonl(paper_dir / "layout.jsonl", [
        {"page": 1, "index": 0, "type": "text", "bbox": [20, 10, 180, 30], "text": "Table 1", "html": None},
        {"page": 1, "index": 3, "type": "table", "bbox": TABLE_PT, "text": "", "html": html,
         "image": "layout_images/t.jpg"},
    ])
    drawings = regions if regions is not None else [
        {"region_id": "p1-r1", "page": 1, "bbox": [0, 0, 10, 10], "crop": "regions/p1-r1.png",
         "structure_score": 0.0, "is_structure": True},
    ]
    common.write_jsonl(paper_dir / "regions.jsonl", drawings)
    return common.Paper("p1", tmp_path / "p1.pdf", paper_dir)


def area_origin():
    """The table area's top-left corner in page pixels (the table box widened by AREA_PAD_PT)."""
    return (int((TABLE_PT[0] - cells.AREA_PAD_PT) * PX), int((TABLE_PT[1] - cells.AREA_PAD_PT) * PX))


def inside(box, outer, slack=1.0):
    """box lies in outer and covers more than half its width."""
    return (box[0] >= outer[0] - slack and box[1] >= outer[1] - slack and box[2] <= outer[2] + slack
            and box[3] <= outer[3] + slack and box[2] - box[0] > 0.5 * (outer[2] - outer[0]))


def boxes_of(*keys):
    out = []
    for key in keys:
        (x, y), _, (w, h) = DRAWN[key]
        out.append((y + 10, x + 10, y + h - 10, x + w - 10))
    return out


# --- the grid ----------------------------------------------------------------------------------


@pytest.mark.parametrize("html", [
    HTML,
    "<table><tr><td rowspan=2>R</td><td colspan=2><img src='h.jpg'/> IC50</td></tr><tr><td>A</td><td>B</td></tr>"
    "<tr><td><img src='x.jpg'/> 8</td><td>1</td><td>2</td></tr><tr><td>9 <img src='y.jpg'/></td><td>3</td>"
    "<td rowspan='2'><img src='z.jpg'/></td></tr><tr><td>10</td><td>4</td></tr></table>",
])
def test_table_images_have_the_grid_places_tables_gives_them(html):
    grid = parse_html_table(html)
    expected = [(r, c, token.removeprefix("[image:").removesuffix("]"))
                for r, row in enumerate(grid) for c, text in enumerate(row)
                for token in text.split() if token.startswith("[image:")]
    found = cells.table_images(html)
    # parse_html_table copies a spanning cell's image into every slot; the worker keeps its top-left one
    assert set(found) <= set(expected)
    assert {src for _, _, src in found} == {src for _, _, src in expected}
    for r, c, src in found:
        assert not any(src == s and (rr, cc) < (r, c) for rr, cc, s in expected)


def test_two_images_in_one_cell_keep_their_order():
    html = "<table><tr><td><img src='a.jpg'/><img src='b.jpg'/></td></tr></table>"
    assert cells.table_images(html) == [(0, 0, "a.jpg"), (0, 0, "b.jpg")]


# --- matching ----------------------------------------------------------------------------------


def test_correlation_is_one_where_the_template_sits_and_matches_a_direct_computation():
    rng = np.random.default_rng(6630)
    area = rng.random((40, 50))
    template = area[12:22, 30:45].copy()
    scores = cells.correlation(area, template)
    assert scores.shape == (31, 36)
    assert np.unravel_index(scores.argmax(), scores.shape) == (12, 30)
    assert scores[12, 30] == pytest.approx(1.0)
    window = area[5:15, 7:22]
    a, t = window - window.mean(), template - template.mean()
    assert scores[5, 7] == pytest.approx((a * t).sum() / np.sqrt((a * a).sum() * (t * t).sum()))


def test_a_blank_or_oversized_template_has_no_correlation():
    assert cells.correlation(np.ones((10, 10)), np.ones((3, 3))) is None
    assert cells.correlation(np.ones((10, 10)), np.eye(12)) is None


def test_two_images_of_one_drawing_take_two_places():
    page = Image.new("RGB", (400, 300), "white")
    benzyl = drawing("*Cc1ccccc1", (120, 90))
    page.paste(benzyl, (20, 20))
    page.paste(benzyl, (220, 170))
    same = benzyl.resize((80, 60), Image.BICUBIC)
    placed = cells.place_images(cells._gray(page), [same, same, None], 1.5)
    assert placed[2] is None
    boxes = sorted(box for _, box in placed[:2])
    assert all(score > 0.9 for score, _ in placed[:2])
    assert abs(boxes[0][0] - 20) <= 3 and abs(boxes[0][1] - 20) <= 3
    assert abs(boxes[1][0] - 220) <= 3 and abs(boxes[1][1] - 170) <= 3


# --- cutting drawings ---------------------------------------------------------------------------


def stacked(top="*Cc1ccccc1", bottom="*Cc1ccccn1", gap=30, text_on_top=True):
    image = Image.new("RGB", (220, 300), "white")
    image.paste(drawing(top, (200, 120)), (10, 20))
    image.paste(drawing(bottom, (200, 120)), (10, 20 + 120 + gap))
    if text_on_top:
        image.paste(Image.new("RGB", (60, 6), "black"), (20, 0))  # the bottom of a word of the row above
    return image


def test_drawings_in_parts_stacked_drawings_and_drops_cut_text_at_the_edge():
    found = cells.drawings_in(cells._gray(stacked()))
    assert len(found) == 2
    top, bottom = found
    assert top[3] < bottom[1] and top[1] >= 15 and bottom[3] <= 300


def test_drawings_in_keeps_one_drawing_whole():
    image = Image.new("RGB", (220, 160), "white")
    image.paste(drawing("OC[C@H](*)c1ccccc1", (200, 140)), (10, 10))
    assert len(cells.drawings_in(cells._gray(image))) == 1


def test_a_small_label_apart_from_its_drawing_joins_it():
    image = Image.new("RGB", (220, 200), "white")
    image.paste(Image.new("RGB", (30, 12), "black"), (60, 20))  # a label 18 px above the drawing
    image.paste(drawing("*Cc1ccccc1", (200, 120)), (10, 50))
    (found,) = cells.drawings_in(cells._gray(image))
    assert found[1] == 20 and found[3] > 100


def test_text_the_box_cut_is_dropped_even_close_to_the_drawing():
    image = Image.new("RGB", (220, 200), "white")
    benzyl = drawing("*Cc1ccccc1", (200, 120))
    image.paste(benzyl, (10, 60))
    top = 60 + int(np.flatnonzero((np.asarray(benzyl.convert("L")) < cells.INK).any(axis=1))[0])
    for x in range(20, 120, 12):  # letters of the row above, 3 px over the drawing, cut by the box's top edge
        image.paste(Image.new("RGB", (8, 20), "black"), (x, top - 23))
    (found,) = cells.drawings_in(cells._gray(image), inner=[0, top - 12, 220, 200])
    assert found[1] == top
    (kept,) = cells.drawings_in(cells._gray(image))  # without the box the letters stay with the drawing
    assert kept[1] == top - 23


def test_pieces_cut_each_drawing_alone_on_white():
    parts = cells.pieces(stacked(), [0, 0, 220, 300])
    assert len(parts) == 2
    for crop, (x0, y0, x1, y1) in parts:
        assert crop.size == (x1 - x0 + 2 * cells.PAD_PX, y1 - y0 + 2 * cells.PAD_PX)
        assert np.asarray(crop.convert("L"))[: cells.PAD_PX].min() == 255


def test_a_blank_box_is_its_own_piece():
    ((crop, box),) = cells.pieces(Image.new("RGB", (100, 80), "white"), [10, 10, 50, 40])
    assert box == [10, 10, 50, 40] and crop.size == (40 + 2 * cells.PAD_PX, 30 + 2 * cells.PAD_PX)


def test_an_image_holding_two_rows_drawings_gives_two_cell_regions(tmp_path):
    paper = make_paper(tmp_path)
    with Image.open(paper.paper_dir / "pages" / "p001.png") as page:
        page = page.convert("RGB")
    (x, y), _, (w, h) = DRAWN["row2"]
    both = page.crop((x, y, x + w, y + 2 * h + 140))  # row2's image also holds row3's drawing below it
    both.resize((round(w * 2 / 3), round((2 * h + 140) * 2 / 3)), Image.BICUBIC).save(
        paper.paper_dir / "layout_images" / "b.jpg")

    records, _ = cells.cells_paper(paper, FakeModels())

    assert [(c["region_id"], c["part"], c["parts"], c["src"]) for c in records] == [
        ("p1-t3-r1-c1", 1, 1, "layout_images/a.jpg"),
        ("p1-t3-r2-c1", 1, 2, "layout_images/b.jpg"),
        ("p1-t3-r2-c1-p2", 2, 2, "layout_images/b.jpg"),
    ]
    (x3, y3), _, (w3, h3) = DRAWN["row3"]
    assert inside(records[2]["bbox"], [x3 / PX, y3 / PX, (x3 + w3) / PX, (y3 + h3) / PX])


# --- a paper -----------------------------------------------------------------------------------


def test_cells_paper_cuts_located_images_from_the_page_and_adds_cell_regions(tmp_path):
    paper = make_paper(tmp_path)
    for name in ("reads.jsonl", "voted.jsonl", ".reads.lock"):
        (paper.paper_dir / name).write_text("{}\n")
    models = FakeModels()

    records, failed = cells.cells_paper(paper, models)

    assert failed == 0 and records == read_jsonl(paper.paper_dir / "cells.jsonl")
    assert [(c["region_id"], c["row"], c["col"], c["src"], c["method"], c["parts"]) for c in records] == [
        ("p1-t3-r1-c1", 1, 1, "layout_images/a.jpg", "matched", 1),
        ("p1-t3-r2-c1", 2, 1, "layout_images/b.jpg", "matched", 1),
    ]
    assert all(c["match"] > 0.9 and c["table"] == "p1-t3" and c["page"] == 1 for c in records)
    (x, y), _, (w, h) = DRAWN["row2"]
    assert inside(records[1]["bbox"], [x / PX, y / PX, (x + w) / PX, (y + h) / PX])
    regions = read_jsonl(paper.paper_dir / "regions.jsonl")
    assert [r["region_id"] for r in regions] == ["p1-r1", "p1-t3-r1-c1", "p1-t3-r2-c1"]
    cell = regions[2]
    # the classifier calls a fragment no structure (0.9), but the gate never drops a cell
    assert (cell["kind"], cell["is_structure"], cell["structure_score"], cell["crop"]) == (
        "cell", True, 0.9, "cells/p1-t3-r2-c1.png")
    with Image.open(paper.paper_dir / cell["crop"]) as crop:  # the page's 300 dpi pixels, alone on white
        x0, y0, x1, y1 = (v * PX for v in records[1]["bbox"])
        assert crop.size == pytest.approx((x1 - x0 + 2 * cells.PAD_PX, y1 - y0 + 2 * cells.PAD_PX), abs=2)
    left = {p.name for p in paper.paper_dir.iterdir()}
    assert not left & {"reads.jsonl", "voted.jsonl", ".reads.lock"}


def test_a_drawing_mineru_dropped_becomes_a_segmented_cell_and_covered_ones_do_not(tmp_path):
    paper = make_paper(tmp_path)
    models = FakeModels(boxes_of("row1", "row2", "row3"))
    models.origin = area_origin()

    records, _ = cells.cells_paper(paper, models)

    counts = json.loads((paper.paper_dir / ".counts" / "cells.json").read_text())
    assert counts == {"in": 2, "out": 2, "dropped_failed_table": 0, "matched": 2, "upscaled": 0, "split": 0,
                      "segmented": 1, "drawings": 3}

    assert [(c["region_id"], c["method"], c["src"], c["row"]) for c in records][-1] == (
        "p1-t3-d1", "segmented", None, None)
    assert len(records) == 3
    (x, y), _, (w, h) = DRAWN["row3"]
    assert inside(records[2]["bbox"], [x / PX, y / PX, (x + w) / PX, (y + h) / PX])
    assert (paper.paper_dir / "cells" / "p1-t3-d1.png").is_file()
    assert models.segmented  # the table area, not the page
    assert models.segmented[0][0] < 834


def test_an_image_not_found_on_the_page_is_scaled_up_instead(tmp_path):
    paper = make_paper(tmp_path)
    Image.new("RGB", (60, 40), "white").save(paper.paper_dir / "layout_images" / "b.jpg")  # blank: matches nowhere
    records, failed = cells.cells_paper(paper, FakeModels())
    assert failed == 0
    assert [(c["method"], c["bbox"] is None, c["match"]) for c in records][1] == ("upscaled", True, None)
    with Image.open(paper.paper_dir / "cells" / "p1-t3-r2-c1.png") as crop:
        # MinerU's image times the ratio of the renders (1.5), with white around it
        assert crop.size == (90 + 2 * cells.PAD_PX, 60 + 2 * cells.PAD_PX)
    region = read_jsonl(paper.paper_dir / "regions.jsonl")[2]
    assert region["bbox"] == TABLE_PT and region["is_structure"] is True


def test_a_rerun_replaces_the_cell_regions_and_crops(tmp_path):
    paper = make_paper(tmp_path)
    cells.cells_paper(paper, FakeModels())
    (paper.paper_dir / "cells" / "p1-t3-r9-c9.png").write_bytes(b"old")
    (paper.paper_dir / "cells" / "notes.txt").write_text("keep")

    cells.cells_paper(paper, FakeModels())

    regions = read_jsonl(paper.paper_dir / "regions.jsonl")
    assert [r["region_id"] for r in regions] == ["p1-r1", "p1-t3-r1-c1", "p1-t3-r2-c1"]
    assert sorted(p.name for p in (paper.paper_dir / "cells").iterdir()) == [
        "notes.txt", "p1-t3-r1-c1.png", "p1-t3-r2-c1.png"]


def test_tables_without_images_give_no_cells_and_are_not_segmented(tmp_path):
    paper = make_paper(tmp_path, html="<table><tr><td>1</td><td>Me</td></tr></table>")
    models = FakeModels(boxes_of("row1"))
    assert cells.cells_paper(paper, models) == ([], 0)
    assert models.segmented == []
    assert read_jsonl(paper.paper_dir / "cells.jsonl") == []
    assert [r["region_id"] for r in read_jsonl(paper.paper_dir / "regions.jsonl")] == ["p1-r1"]


def test_a_missing_table_image_fails_that_table_only(tmp_path):
    paper = make_paper(tmp_path)
    (paper.paper_dir / "layout_images" / "a.jpg").unlink()
    logger = common.open_log(tmp_path / "cells.log", "cells")
    try:
        assert cells.run([paper], FakeModels(), logger) == 1
    finally:
        common.close_log(logger)
    (error,) = read_jsonl(paper.paper_dir / "errors.jsonl")
    assert error["stage"] == "cells" and error["table"] == "p1-t3" and "a.jpg" in error["error"]
    assert read_jsonl(paper.paper_dir / "cells.jsonl") == []
    counts = json.loads((paper.paper_dir / ".counts" / "cells.json").read_text())
    assert (counts["in"], counts["out"], counts["dropped_failed_table"]) == (2, 0, 2)


def test_main_exits_2_on_a_bad_manifest(tmp_path):
    manifest = tmp_path / "work.jsonl"
    manifest.write_text("not json\n")
    assert cells.main(["--manifest", str(manifest), "--log", str(tmp_path / "cells.log"), "--device", "cpu"]) == 2
