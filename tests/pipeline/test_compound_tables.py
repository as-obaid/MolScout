"""Whole-compound tables: a label column beside a column of drawn whole compounds, one molecule per row.

The paper is synthetic: a Ki table whose Structure column holds MinerU images, with a label cell that spans two
grid rows, a drawing read by one reader, an untested compound, a fragment, a drawing without a vote, a row without
a drawing and a label range.
"""

from __future__ import annotations

import json

import pytest

from molscout.pipeline import compound_tables, link, table_cells
from molscout.pipeline.finalize import finalize
from molscout.pipeline.schema import LayoutBlock
from molscout.pipeline.table_cells import Word

TABLE = [300.0, 100.0, 550.0, 500.0]
ROW_Y = [130 + 40 * i for i in range(7)]  # the compounds 1..6 and 7–8, one row each
HTML = ("<table><tr><td>Compd</td><td>Structure</td><td>K<sub>i</sub> (nM)</td></tr>"
        "<tr><td rowspan='2'>1</td><td><img src='layout_images/w1.jpg'/></td><td>0.5</td></tr>"
        "<tr><td></td><td></td></tr>"
        "<tr><td>2</td><td><img src='layout_images/w2.jpg'/></td><td>1.2</td></tr>"
        "<tr><td>3</td><td><img src='layout_images/w3.jpg'/></td><td>NT</td></tr>"
        "<tr><td>4</td><td><img src='layout_images/w4.jpg'/></td><td>3.1</td></tr>"
        "<tr><td>5</td><td><img src='layout_images/w5.jpg'/></td><td>2.0</td></tr>"
        "<tr><td>6</td><td></td><td>7.0</td></tr>"
        "<tr><td>7–8</td><td><img src='layout_images/w7.jpg'/></td><td>4.0</td></tr></table>")
CHLORO, FLUORO, BROMO = "O=C(Nc1ccccc1)c1ccc(Cl)cc1", "O=C(Nc1ccccc1)c1ccc(F)cc1", "O=C(Nc1ccccc1)c1ccc(Br)cc1"
CELLS = [  # (region_id, src, compound row of its box, vote SMILES, agree, vote reason)
    ("p1-t2-r1-c1", "layout_images/w1.jpg", 0, CHLORO, 3, None),
    ("p1-t2-r3-c1", "layout_images/w2.jpg", 1, FLUORO, 1, None),
    ("p1-t2-r4-c1", "layout_images/w3.jpg", 2, BROMO, 3, None),
    ("p1-t2-r5-c1", "layout_images/w4.jpg", 3, "[*:1]c1ccc(Cl)cc1", 3, None),
    ("p1-t2-r6-c1", "layout_images/w5.jpg", 4, None, 0, "tie between 2 structures"),
    ("p1-t2-r8-c1", "layout_images/w7.jpg", 6, "O=C(Nc1ccccc1)c1ccc(I)cc1", 3, None),
]


def box(row, height=24, x0=360.0, x1=440.0):
    return [x0, ROW_Y[row] - height / 2, x1, ROW_Y[row] + height / 2]


def write(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def voted(region_id, smiles, agree=3, reason=None):
    return {"region_id": region_id, "smiles": smiles, "inchikey": None, "agree": agree, "stereo_kept": True,
            "confidence": agree / 3, "readers": ["molglyph", "molscribe", "molnextr"][:agree],
            "slots": {"1": "*"} if smiles and "*" in smiles else {}, "reason": reason}


def table_block(html=HTML, index=2):
    return {"page": 1, "index": index, "type": "table", "bbox": TABLE, "text": "", "html": html, "parent": None,
            "image": None}


@pytest.fixture
def paper(tmp_path):
    folder = tmp_path / "papers" / "whole"
    folder.mkdir(parents=True)
    write(folder / "layout.jsonl", [
        {"page": 1, "index": 1, "type": "caption", "bbox": [300, 80, 550, 95], "text": "Table 1. Ki values.",
         "html": None, "parent": 2, "image": None},
        table_block(),
    ])
    regions = [{"region_id": "p1-r1", "page": 1, "bbox": box(0, 20, 362, 438), "crop": "regions/p1-r1.png",
                "structure_score": 0.9, "is_structure": True}]
    regions += [{"region_id": rid, "page": 1, "bbox": box(row), "crop": f"cells/{rid}.png", "structure_score": None,
                 "is_structure": True, "kind": "cell"} for rid, _, row, _, _, _ in CELLS]
    write(folder / "regions.jsonl", regions)
    write(folder / "voted.jsonl", [voted("p1-r1", CHLORO),
                                   *(voted(rid, smiles, agree, why) for rid, _, _, smiles, agree, why in CELLS)])
    write(folder / "cells.jsonl", [
        {"region_id": rid, "table": "p1-t2", "page": 1, "row": None, "col": 1, "src": src, "part": 1, "parts": 1,
         "method": "matched", "match": 0.95, "bbox": box(row)}
        for rid, src, row, _, _, _ in CELLS
    ])
    return folder


def blocks_of(*html):
    return [LayoutBlock(**table_block(text, index)) for index, text in enumerate(html, start=2)]


def compounds(molecules):
    return {m.label: m for m in molecules if m.source == "table"}


def rewrite(path, edit):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    write(path, edit(records))


# --- finding the tables ----------------------------------------------------------------------------


def test_a_labelled_table_with_a_column_of_drawings_and_no_variables_is_a_compound_table():
    (found,) = compound_tables.find_compound_tables(blocks_of(HTML))
    assert (found.table.table_id, found.column) == ("p1-t2", "Structure")
    assert [(row.index, row.label, row.drawing) for row in found.rows] == [
        (0, "1", "[image:layout_images/w1.jpg]"), (2, "2", "[image:layout_images/w2.jpg]"),
        (3, "3", "[image:layout_images/w3.jpg]"), (4, "4", "[image:layout_images/w4.jpg]"),
        (5, "5", "[image:layout_images/w5.jpg]"), (6, "6", ""), (7, "7–8", "[image:layout_images/w7.jpg]")]


def test_r_group_tables_and_tables_with_few_or_no_drawings_are_not_compound_tables():
    rgroup = HTML.replace("<td>Structure</td>", "<td>R</td>")
    few = ("<table><tr><td>Compd</td><td>Structure</td><td>IC50</td></tr>"
           "<tr><td>1</td><td><img src='a.jpg'/></td><td>5</td></tr><tr><td>2</td><td></td><td>6</td></tr>"
           "<tr><td>3</td><td></td><td>7</td></tr><tr><td>4</td><td></td><td>8</td></tr></table>")
    plain = "<table><tr><td>Compd</td><td>IC50</td></tr><tr><td>1</td><td>2</td></tr></table>"
    assert compound_tables.find_compound_tables(blocks_of(rgroup, few, plain)) == []


def test_a_repeated_label_is_one_compound_only_when_its_rows_do_not_disagree():
    html = ("<table><tr><td>Compd</td><td>Structure</td><td>IC50</td></tr>"
            "<tr><td>1</td><td></td><td></td></tr><tr><td>1</td><td><img src='a.jpg'/></td><td>4</td></tr>"
            "<tr><td>2</td><td><img src='b.jpg'/></td><td>5</td></tr><tr><td>2</td><td></td><td>6</td></tr>"
            "<tr><td></td><td><img src='c.jpg'/></td><td>7</td></tr><tr><td></td><td></td><td></td></tr></table>")
    (found,) = compound_tables.find_compound_tables(blocks_of(html))
    assert [(row.index, row.label, row.drawing) for row in found.rows] == [
        (0, "1", "[image:a.jpg]"), (2, "2", "[image:b.jpg]"), (3, "2", ""), (4, "", "[image:c.jpg]")]


# --- link ----------------------------------------------------------------------------------------


def test_each_compound_gives_one_record_from_its_drawing(paper):
    found = compounds(link.link_and_build(paper, None))
    assert (found["1"].smiles, found["1"].status, found["1"].row) == (CHLORO, "accepted", 0)
    assert found["1"].reasons == ["drawn whole in column Structure (p1-t2-r1-c1)"]
    assert (found["1"].table_id, found["1"].bbox, found["1"].confidence) == ("p1-t2", TABLE, pytest.approx(1.0))
    assert (found["2"].smiles, found["2"].status) == (FLUORO, "review")
    assert found["2"].reasons[1] == "drawing p1-t2-r3-c1 read by 1 agreeing reader(s)"
    assert (found["3"].smiles, found["3"].status) == (BROMO, "accepted")
    assert found["4"].status == "abstained" and found["4"].smiles is None
    assert found["4"].reasons == ["drawing p1-t2-r5-c1 is a fragment with an attachment point "
                                  "([*:1]c1ccc(Cl)cc1), not a whole compound"]
    assert found["5"].reasons == ["drawing p1-t2-r6-c1 gave no vote (tie between 2 structures)"]
    assert found["6"].status == "abstained" and found["6"].reasons == ["no drawing in column Structure"]
    assert found["7"].status == found["8"].status == "abstained"
    assert found["8"].reasons == ["label range 7–8 not split: one drawing for 2 labels"]


def test_a_column_whose_drawings_vote_mostly_as_fragments_holds_r_groups(paper):
    # An R header MinerU misread leaves an R-group table without variables: its drawings carry attachment points.
    fragments = {"p1-t2-r3-c1": "[*:1]c1ccc(F)cc1", "p1-t2-r4-c1": "[*:1]c1ccc(Br)cc1"}
    rewrite(paper / "voted.jsonl", lambda rs: [{**r, "smiles": fragments[r["region_id"]], "slots": {"1": "*"}}
                                               if r["region_id"] in fragments else r for r in rs])
    found = compounds(link.link_and_build(paper, None))
    assert {m.status for m in found.values()} == {"abstained"}
    why = ["column Structure holds R-group fragments (3 of 5 voted drawings), not whole compounds"]
    assert found["1"].reasons == found["6"].reasons == found["8"].reasons == why


def test_cell_drawings_give_no_drawn_record_and_a_page_drawing_of_a_row_is_abstained(paper):
    drawn = [m for m in link.link_and_build(paper, None) if m.source == "drawn"]
    assert len(drawn) == 1  # the page drawing; the six cell drawings give none
    assert (drawn[0].status, drawn[0].reasons) == ("abstained", ["drawn in table p1-t2 row 0 (1), read there as "
                                                                 "p1-t2-r1-c1"])


def test_a_page_drawing_over_a_row_that_was_not_built_is_kept(paper):
    rewrite(paper / "regions.jsonl", lambda rs: [{**r, "bbox": box(3, 20, 362, 438)} if r["region_id"] == "p1-r1"
                                                 else r for r in rs])
    (drawn,) = [m for m in link.link_and_build(paper, None) if m.source == "drawn"]
    assert (drawn.smiles, drawn.status) == (CHLORO, "accepted")


def test_the_vote_fix_off_accepts_a_drawing_one_reader_read(paper):
    assert compounds(link.link_and_build(paper, None, {"vote": False}))["2"].status == "accepted"
    assert not compounds(link.link_and_build(paper, None, {"tables": False}))


def test_without_the_cells_stage_rows_abstain_and_say_so(paper):
    (paper / "cells.jsonl").unlink()
    found = compounds(link.link_and_build(paper, None))
    assert {m.status for m in found.values()} == {"abstained"}
    assert found["1"].reasons == ["no drawing in column Structure (no cell drawing was cut for its image "
                                  "(the cells stage did not run or failed))"]


def test_the_pdf_text_places_a_drawing_minerU_dropped_in_its_row(paper, monkeypatch):
    dropped = ("p1-t2-d1", None, 5, CHLORO.replace("Cl", "C#N"), 3, None)
    rewrite(paper / "cells.jsonl", lambda rs: [*rs, {**rs[0], "region_id": dropped[0], "src": None,
                                                     "method": "segmented", "match": None, "bbox": box(5)}])
    rewrite(paper / "regions.jsonl", lambda rs: [*rs, {**rs[-1], "region_id": dropped[0], "bbox": box(5)}])
    rewrite(paper / "voted.jsonl", lambda rs: [*rs, voted(dropped[0], dropped[3])])
    labels = ["1", "2", "3", "4", "5", "6", "7–8"]
    words = [Word(label, (305.0, y - 3.0, 315.0, y + 3.0)) for label, y in zip(labels, ROW_Y, strict=True)]
    monkeypatch.setattr(table_cells, "pdf_words", lambda pdf, page: words)
    sixth = compounds(link.link_and_build(paper, None, pdf=paper / "paper.pdf"))["6"]
    assert (sixth.smiles, sixth.status) == (dropped[3], "accepted")
    assert sixth.reasons[1] == "drawing p1-t2-d1 placed in the row by its position"


def test_link_counts_add_up_and_count_the_whole_compound_rows(paper):
    molecules, counts = link.link_paper(paper, None)
    dropped = sum(value for key, value in counts.items() if key.startswith("dropped_"))
    assert counts["in"] == len(molecules) == counts["out"] + dropped
    assert (counts["whole_compound_rows"], counts["whole_compound_molecules"]) == (8, 3)
    assert (counts["dropped_drawn"], counts["dropped_table"]) == (1, 5)


def test_finalize_names_the_compounds_the_table_reports_activity_for(paper):
    molecules, _ = finalize(paper, link.link_and_build(paper, None))
    found = compounds(molecules)
    assert (found["1"].status, found["1"].reasons[-1]) == ("accepted", "label 1 is in an activity table or sentence")
    assert found["3"].status == "accepted" and not any("activity table" in r for r in found["3"].reasons)
