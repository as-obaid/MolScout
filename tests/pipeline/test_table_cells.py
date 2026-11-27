"""Drawn R groups in table cells: rows placed from the PDF's text, each row's drawing, and link building the rows.

The paper is synthetic: an R-group table on a benzamide core whose R column holds MinerU images, an image that
holds two rows' drawings, a drawing MinerU dropped, a lone "N" MinerU kept from one, and a whole molecule.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import pytest

from molscout.pipeline import link, table_cells
from molscout.pipeline.table_cells import Cell, Word
from molscout.scoring.smiles import canonical_smiles

CORE = "O=C(N[*:1])c1ccccc1"
WHOLE = "Cc1cccnc1CN1CCC2(CC1)C(=O)N(c1ccc(-c3ccccc3)cc1)C(=O)N2c1ncccn1"
TABLE = [300.0, 100.0, 550.0, 500.0]
ROW_Y = [130 + 40 * i for i in range(8)]  # the labels 1..8, one row each
HTML = ("<table><tr><td>Compd</td><td>R</td><td>IC<sub>50</sub> (nM)</td></tr>"
        "<tr><td>1</td><td><img src='layout_images/a.jpg'/></td><td>3</td></tr>"
        "<tr><td>2</td><td><img src='layout_images/b.jpg'/></td><td>4</td></tr>"
        "<tr><td>3</td><td></td><td>5</td></tr>"
        "<tr><td>4</td><td>N</td><td>6</td></tr>"
        "<tr><td>5</td><td></td><td>7</td></tr>"
        "<tr><td>6</td><td><img src='layout_images/c.jpg'/></td><td>8</td></tr>"
        "<tr><td>7</td><td><img src='layout_images/e.jpg'/></td><td>9</td></tr>"
        "<tr><td>8</td><td>Me</td><td>10</td></tr></table>")


def box(row, height=24, x0=360.0, x1=420.0):
    return [x0, ROW_Y[row] - height / 2, x1, ROW_Y[row] + height / 2]


CELLS = [  # (region_id, src, part, parts, row of its box, vote SMILES, agree)
    ("p1-t2-r1-c1", "layout_images/a.jpg", 1, 1, 0, "[*:1]Cc1ccccc1", 3),
    ("p1-t2-r2-c1", "layout_images/b.jpg", 1, 2, 1, "[*:1]Cc1ccccn1", 2),
    ("p1-t2-r2-c1-p2", "layout_images/b.jpg", 2, 2, 2, "[*:1]C1CCCCC1", 3),
    ("p1-t2-d1", None, 1, 1, 3, "[*:1]Cc1ccncc1", 3),
    ("p1-t2-r6-c1", "layout_images/c.jpg", 1, 1, 5, WHOLE, 3),
    ("p1-t2-r7-c1", "layout_images/e.jpg", 1, 1, 6, "CC(CO)c1ccccc1", 2),
]


def words():
    found = [Word("Compd", (305.0, 105.0, 330.0, 112.0)), Word("R", (380.0, 105.0, 384.0, 112.0))]
    found += [Word(str(i + 1), (305.0, y - 3.0, 310.0, y + 3.0)) for i, y in enumerate(ROW_Y)]
    found += [Word(str(i + 3), (470.0, y - 3.0, 476.0, y + 3.0)) for i, y in enumerate(ROW_Y)]  # activity values
    return found


def write(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def voted(region_id, smiles, agree=3, slots=None, reason=None):
    return {"region_id": region_id, "smiles": smiles, "inchikey": None, "agree": agree, "stereo_kept": True,
            "confidence": agree / 3, "readers": ["molglyph", "molscribe", "molnextr"][:agree],
            "slots": slots or ({"1": "*"} if smiles and "*" in smiles else {}), "reason": reason}


@pytest.fixture
def paper(tmp_path, monkeypatch):
    folder = tmp_path / "papers" / "cells"
    folder.mkdir(parents=True)
    write(folder / "layout.jsonl", [
        {"page": 1, "index": 1, "type": "caption", "bbox": [300, 30, 550, 38], "text": "Table 1. IC50 values.",
         "html": None, "parent": 2, "image": None},
        {"page": 1, "index": 2, "type": "table", "bbox": TABLE, "text": "", "html": HTML, "parent": None,
         "image": None},
    ])
    regions = [{"region_id": "p1-r1", "page": 1, "bbox": [320, 40, 500, 95], "crop": "regions/p1-r1.png",
                "structure_score": 0.0, "is_structure": True},
               {"region_id": "p1-r2", "page": 1, "bbox": box(0, 20, 362, 418), "crop": "regions/p1-r2.png",
                "structure_score": 0.0, "is_structure": True}]
    regions += [{"region_id": rid, "page": 1, "bbox": box(row), "crop": f"cells/{rid}.png", "structure_score": None,
                 "is_structure": True, "kind": "cell"} for rid, _, _, _, row, _, _ in CELLS]
    write(folder / "regions.jsonl", regions)
    write(folder / "voted.jsonl", [voted("p1-r1", CORE, slots={"1": "R"}), voted("p1-r2", "Cc1ccccc1"),
                                   *(voted(rid, smiles, agree) for rid, _, _, _, _, smiles, agree in CELLS)])
    write(folder / "cells.jsonl", [
        {"region_id": rid, "table": "p1-t2", "page": 1, "row": None, "col": None, "src": src, "part": part,
         "parts": parts, "method": "matched" if src else "segmented", "match": 0.95 if src else None,
         "bbox": box(row)}
        for rid, src, part, parts, row, _, _ in CELLS
    ])
    monkeypatch.setattr(table_cells, "pdf_words", lambda pdf, page: words())
    return folder


def run(paper, pdf=True, fixes=None):
    return link.link_and_build(paper, None, fixes, pdf=paper / "paper.pdf" if pdf else None)


def rows(molecules):
    return {m.label: m for m in molecules if m.source == "table"}


def built(r):
    return canonical_smiles(CORE.replace("[*:1]", r))


def rewrite(path, edit):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    write(path, edit(records))


# --- positions -----------------------------------------------------------------------------------


def test_label_rows_takes_the_leftmost_word_that_reads_as_each_label():
    labels = [str(i + 1) for i in range(8)]
    assert table_cells.label_rows(words(), TABLE, labels) == pytest.approx(ROW_Y)
    assert table_cells.label_rows(words(), TABLE, ["1", None, "99"]) == [ROW_Y[0], None, None]


def test_label_rows_out_of_order_are_not_trusted():
    shuffled = [Word("1", (305.0, 200.0, 310.0, 206.0)), Word("2", (305.0, 100.0, 310.0, 106.0))]
    assert table_cells.label_rows(shuffled, TABLE, ["1", "2"]) == [None, None]


def test_labels_with_marks_and_words_outside_the_table():
    found = [Word("3a*", (305.0, 120.0, 312.0, 126.0)), Word("(3b)", (305.0, 150.0, 315.0, 156.0)),
             Word("3c", (5.0, 180.0, 10.0, 186.0))]
    assert table_cells.label_rows(found, TABLE, ["3a", "3b", "3c"]) == [123.0, 153.0, None]


def test_nearest_row_reaches_part_of_the_gap_to_the_next_label():
    rows_y = [100.0, 140.0, None, 220.0]
    assert table_cells.nearest_row(110.0, rows_y) == 0
    assert table_cells.nearest_row(170.0, rows_y) == 1  # 30 below 140: within 0.6 of the 80 gap down to 220
    assert table_cells.nearest_row(128.0, rows_y) == 1  # 12 above 140: within 0.6 of the 40 gap up to 100
    assert table_cells.nearest_row(75.0, rows_y) is None  # above the first label by more than 0.6 of its gap
    assert table_cells.nearest_row(245.0, rows_y) == 3
    assert table_cells.nearest_row(100.0, [100.0]) is None


def test_words_join_characters_and_split_at_spaces_gaps_and_lines():
    chars = [("1", (10, 100, 11.5, 105)), ("1", (13.5, 100, 15, 105)), (" ", (15, 100, 15, 100)),
             ("N", (20, 100, 24, 105)), ("O", (40, 100, 44, 105)), ("x", (44, 80, 47, 85))]
    found = table_cells._words(chars, left=0, top=200)
    assert [(w.text, w.box) for w in found] == [
        ("11", (10, 95, 15, 100)), ("N", (20, 95, 24, 100)), ("O", (40, 95, 44, 100)), ("x", (44, 115, 47, 120))]


def test_pdf_words_reads_a_page_with_its_boxes_from_the_top_left(tmp_path):
    pdfium = pytest.importorskip("pypdfium2")
    raw = pdfium.raw
    document = pdfium.PdfDocument.new()
    page = document.new_page(200, 300)
    text = raw.FPDFPageObj_NewTextObj(document.raw, b"Helvetica", 10.0)
    buffer = ctypes.create_string_buffer(("12a").encode("utf-16-le") + b"\x00\x00")
    raw.FPDFText_SetText(text, ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ushort)))
    raw.FPDFPageObj_Transform(text, 1, 0, 0, 1, 50, 250)  # baseline 50 pt from the left, 250 pt up
    raw.FPDFPage_InsertObject(page.raw, text)
    raw.FPDFPage_GenerateContent(page.raw)
    path = tmp_path / "words.pdf"
    document.save(str(path))
    document.close()

    (word,) = table_cells.pdf_words(path, 1)

    assert word.text == "12a"
    x0, y0, x1, y1 = word.box
    assert 49 <= x0 < 52 and 60 < x1 < 70 and 40 < y0 < y1 <= 51  # 300 - 250 = 50 pt from the top


# --- each row's drawing --------------------------------------------------------------------------


def cells_of(**changes):
    return [Cell(rid, "p1-t2", 1, None, None, src, "matched" if src else "segmented", part, parts, 0.9,
                 changes.get(rid, box(row)))
            for rid, src, part, parts, row, _, _ in CELLS]


VALUES = ["[image:layout_images/a.jpg]", "[image:layout_images/b.jpg]", "", "N", "",
          "[image:layout_images/c.jpg]", "[image:layout_images/e.jpg]", "Me"]
LABELS = [str(i + 1) for i in range(8)]


def test_by_position_each_drawing_reaches_the_row_it_sits_in():
    result = table_cells.column_drawings(VALUES, "p1-t2", TABLE, LABELS, cells_of(), words)
    assert result.by_position
    assert {row: cell.region_id for row, cell in result.by_row.items()} == {
        0: "p1-t2-r1-c1", 1: "p1-t2-r2-c1", 2: "p1-t2-r2-c1-p2", 3: "p1-t2-d1", 5: "p1-t2-r6-c1", 6: "p1-t2-r7-c1"}


def test_without_the_pdf_a_row_reads_its_own_image_when_it_holds_one_drawing():
    result = table_cells.column_drawings(VALUES, "p1-t2", TABLE, LABELS, cells_of(), None)
    assert not result.by_position
    assert {row: cell.region_id for row, cell in result.by_row.items()} == {
        0: "p1-t2-r1-c1", 5: "p1-t2-r6-c1", 6: "p1-t2-r7-c1"}
    assert "holds 2 drawings" in result.problems[1]


def test_positions_that_put_a_located_image_in_another_row_are_not_trusted():
    result = table_cells.column_drawings(VALUES, "p1-t2", TABLE, LABELS, cells_of(**{"p1-t2-r1-c1": box(1)}), words)
    assert not result.by_position and 2 not in result.by_row


def test_two_strange_drawings_in_one_row_are_a_problem_and_its_own_image_wins_over_one():
    extra = Cell("p1-t2-d9", "p1-t2", 1, None, None, None, "segmented", bbox=box(2, 10))
    result = table_cells.column_drawings(VALUES, "p1-t2", TABLE, LABELS, [*cells_of(), extra], words)
    assert result.problems[2] == "2 drawings fall in the row"
    stray = Cell("p1-t2-d9", "p1-t2", 1, None, None, None, "segmented", bbox=box(0, 10))
    result = table_cells.column_drawings(VALUES, "p1-t2", TABLE, LABELS, [*cells_of(), stray], words)
    assert result.by_row[0].region_id == "p1-t2-r1-c1"


def test_drawings_outside_the_column_or_of_another_table_are_left_alone():
    elsewhere = Cell("p1-t2-d9", "p1-t2", 1, None, None, None, "segmented", bbox=box(4, x0=480, x1=520))
    other = Cell("p1-t7-d1", "p1-t7", 1, None, None, None, "segmented", bbox=box(4))
    result = table_cells.column_drawings(VALUES, "p1-t2", TABLE, LABELS, [*cells_of(), elsewhere, other], words)
    assert 4 not in result.by_row and 4 not in result.problems


def test_an_image_without_a_cut_drawing_and_a_cell_of_two_images_are_problems():
    values = ["[image:layout_images/zz.jpg]", "[image:layout_images/a.jpg] [image:layout_images/c.jpg]"]
    result = table_cells.column_drawings(values, "p1-t2", TABLE, ["1", "2"], cells_of(), None)
    assert "did not run" in result.problems[0] and result.problems[1] == "the cell holds 2 images"


def test_image_sources_and_the_text_beside_them():
    value = "8 [image:layout_images/p3-5-c2.jpg] (S) [image]"
    assert table_cells.image_sources(value) == ["layout_images/p3-5-c2.jpg", ""]
    assert table_cells.text_beside_images(value) == "8 (S)"
    assert table_cells.read_cells(Path("/nonexistent")) == []


# --- link ----------------------------------------------------------------------------------------


def test_link_builds_every_drawn_row(paper):
    found = rows(run(paper))
    assert (found["1"].smiles, found["1"].status) == (built("Cc1ccccc1"), "accepted")
    assert (found["2"].smiles, found["2"].status) == (built("Cc1ccccn1"), "accepted")
    assert (found["3"].smiles, found["3"].status) == (built("C1CCCCC1"), "accepted")
    assert "p1-t2-r2-c1-p2 placed in the row by its position" in found["3"].reasons[0]
    assert (found["4"].smiles, found["4"].status) == (built("Cc1ccncc1"), "accepted")  # the drawing, not "N"
    assert (found["5"].status, found["5"].reasons) == ("abstained", ["R cell is empty"])
    assert (found["6"].smiles, found["6"].status) == (canonical_smiles(WHOLE), "accepted")
    assert found["6"].reasons == ["the R cell holds a whole molecule"]
    assert found["7"].status == "abstained" and "has no attachment point" in found["7"].reasons[0]
    assert (found["8"].smiles, found["8"].status) == (built("C"), "accepted")
    assert found["1"].confidence == pytest.approx(1.0) and found["2"].confidence == pytest.approx(2 / 3)


def test_cell_regions_give_no_drawn_record_and_a_page_drawing_of_a_cell_is_abstained(paper):
    drawn = {m.bbox[1]: m for m in run(paper) if m.source == "drawn"}
    assert len(drawn) == 2  # the core and the page drawing; no record for the six cell drawings
    page_drawing = drawn[box(0, 20)[1]]
    assert page_drawing.status == "abstained"
    assert page_drawing.reasons == ["drawn in table p1-t2 row 0 (1), read there as p1-t2-r1-c1"]
    assert any("R-group core for table p1-t2" in r for r in drawn[40].reasons)


def test_without_the_pdf_rows_read_only_their_own_images(paper):
    found = rows(run(paper, pdf=False))
    assert found["1"].smiles == built("Cc1ccccc1")
    assert found["2"].status == "abstained" and "holds 2 drawings" in found["2"].reasons[0]
    assert found["3"].reasons == ["R cell is empty"]
    assert found["4"].smiles == built("N") and found["4"].status == "review"  # "N" read as text, flagged
    assert any("letter of a drawing MinerU dropped" in r for r in found["4"].reasons)


def test_a_drawing_read_by_one_reader_goes_to_review(paper):
    rewrite(paper / "voted.jsonl", lambda rs: [{**r, "agree": 1} if r["region_id"] == "p1-t2-r1-c1" else r for r in rs])
    first = rows(run(paper))["1"]
    assert first.status == "review" and "read by 1 agreeing reader(s)" in first.reasons[0]
    assert rows(run(paper, fixes={"vote": False}))["1"].status == "accepted"


def test_a_drawing_without_a_vote_explains_why(paper):
    rewrite(paper / "voted.jsonl", lambda rs: [
        {**r, "smiles": None, "reason": "tie between 2 fragments"} if r["region_id"] == "p1-t2-r1-c1" else r
        for r in rs])
    first = rows(run(paper))["1"]
    assert first.status == "abstained"
    assert first.reasons == ["R is drawn in the table, but its drawing p1-t2-r1-c1 gave no vote "
                             "(tie between 2 fragments)"]


def test_without_the_cells_stage_drawn_rows_say_so(paper):
    (paper / "cells.jsonl").unlink()
    found = rows(run(paper))
    assert found["1"].status == "abstained"
    assert "no drawing of it was found (no cell drawing was cut" in found["1"].reasons[0]
    assert found["8"].smiles == built("C")


def test_cell_drawings_are_never_cores_or_labelled_drawings(paper):
    rewrite(paper / "voted.jsonl", lambda rs: [r for r in rs if r["region_id"] != "p1-r1"])
    found = rows(run(paper))
    assert all(m.status == "abstained" for m in found.values())
    assert found["1"].reasons == ["no R-group core drawing found for table p1-t2"]
