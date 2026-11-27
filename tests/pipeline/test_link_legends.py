"""Legend codes (the legends fix): an R cell holding a code such as "M0" is filled from the drawing labelled M0.

The synthetic paper's table (p2-t2, core [*:1]c1ccc(C(=O)N[*:2])cc1) holds R1 = M0 in row 5a, and a drawing on the
table's page carries the label M0 printed below it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from molscout.pipeline import link
from molscout.scoring.smiles import canonical_smiles

_A2 = Path(__file__).parent / "fixtures" / "a2" / "paper_synthetic"
TABLE = (
    "<table><tr><th>Compd</th><th>R<sup>1</sup></th><th>R<sup>2</sup></th><th>IC<sub>50</sub> (nM)</th></tr>"
    "<tr><td>5a</td><td>M0</td><td>Me</td><td>14</td></tr>"
    "<tr><td>5b</td><td>F</td><td>Me</td><td>25</td></tr></table>"
)
LEGEND_BOX = [100, 450, 200, 520]
MORPHOLINYL_ROW = canonical_smiles("CNC(=O)c1ccc(N2CCOCC2)cc1")


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _vote(region_id: str, smiles: str, agree: int = 3, slots: dict[str, str] | None = None) -> dict:
    return {"region_id": region_id, "smiles": smiles, "inchikey": None, "agree": agree, "stereo_kept": True,
            "confidence": agree / 3, "readers": ["molglyph", "molscribe", "molnextr"][:agree], "slots": slots or {},
            "reason": None}


def _add_drawing(paper: Path, region_id: str, page: int, bbox: list[float], label: str, smiles: str) -> None:
    """A page drawing, its vote and its label printed just below it."""
    region = {"region_id": region_id, "page": page, "bbox": bbox, "crop": f"crops/{region_id}.png",
              "structure_score": 0.9, "is_structure": True}
    _write(paper / "regions.jsonl", [*_read(paper / "regions.jsonl"), region])
    _write(paper / "voted.jsonl", [*_read(paper / "voted.jsonl"), _vote(region_id, smiles)])
    layout = _read(paper / "layout.jsonl")
    index = 1 + max(block["index"] for block in layout if block["page"] == page)
    middle = (bbox[0] + bbox[2]) / 2
    below = [middle - 15, bbox[3] + 5, middle + 15, bbox[3] + 17]
    block = {"page": page, "index": index, "type": "text", "bbox": below, "text": label, "html": None, "parent": None,
             "image": None}
    _write(paper / "layout.jsonl", [*layout, block])


def _drop_drawing(paper: Path, region_id: str) -> None:
    for name in ("regions.jsonl", "voted.jsonl"):
        _write(paper / name, [row for row in _read(paper / name) if row["region_id"] != region_id])


def _edit_vote(paper: Path, region_id: str, **changes) -> None:
    rows = _read(paper / "voted.jsonl")
    _write(paper / "voted.jsonl", [{**row, **changes} if row["region_id"] == region_id else row for row in rows])


def _replace_table(paper: Path, html: str) -> None:
    rows = _read(paper / "layout.jsonl")
    _write(paper / "layout.jsonl", [{**row, "html": html} if row["type"] == "table" else row for row in rows])


@pytest.fixture
def paper(tmp_path) -> Path:
    """The synthetic paper with R1 = M0 in row 5a and a morpholine drawn as CN1CCOCC1 labelled M0 on page 2."""
    target = tmp_path / "papers" / "synthetic"
    shutil.copytree(_A2, target)
    _replace_table(target, TABLE)
    _add_drawing(target, "p2-r2", 2, LEGEND_BOX, "M0", "CN1CCOCC1")
    return target


def _run(paper: Path, fixes=None):
    return link.link_paper(paper, None, fixes)


def _rows(molecules) -> dict:
    return {m.label: m for m in molecules if m.source == "table"}


def _drawn(molecules, bbox):
    (found,) = [m for m in molecules if m.source == "drawn" and m.bbox == bbox]
    return found


def test_switch_off_keeps_the_code_unreadable_and_the_drawing_a_compound(paper):
    molecules, counts = _run(paper, {"legends": False})
    row = _rows(molecules)["5a"]
    assert (row.status, row.smiles, row.reasons) == ("abstained", None, ["cannot read R1 = 'M0'"])
    drawing = _drawn(molecules, LEGEND_BOX)
    assert (drawing.status, drawing.smiles, drawing.label, drawing.reasons) == ("accepted", "CN1CCOCC1", None, [])
    assert counts["legend_drawings"] == 0


def test_a_plain_molecule_legend_gives_its_methyl_as_the_attachment(paper):
    rows = _rows(_run(paper)[0])
    assert (rows["5a"].status, rows["5a"].smiles) == ("accepted", MORPHOLINYL_ROW)
    assert rows["5a"].reasons == ["R1 = M0 from drawing p2-r2"]
    assert rows["5a"].confidence == pytest.approx(0.9)
    assert rows["5b"].smiles == canonical_smiles("CNC(=O)c1ccc(F)cc1")


def test_a_legend_drawn_with_its_attachment_point_fills_the_slot(paper):
    _edit_vote(paper, "p2-r2", smiles="[*:1]N1CCOCC1", slots={"1": "*"})
    molecules, _ = _run(paper)
    assert _rows(molecules)["5a"].smiles == MORPHOLINYL_ROW
    assert _drawn(molecules, LEGEND_BOX).reasons == ["legend fragment M0 for table p2-t2"]


def test_a_legend_with_two_methyls_stays_unreadable(paper):
    _edit_vote(paper, "p2-r2", smiles="CN1CCC(C)CC1")
    molecules, counts = _run(paper)
    assert _rows(molecules)["5a"].reasons == ["cannot read R1 = 'M0'"]
    assert _drawn(molecules, LEGEND_BOX).status == "accepted"
    assert counts["legend_drawings"] == 0


def test_the_legend_drawing_is_abstained_not_exported(paper):
    molecules, counts = _run(paper)
    drawing = _drawn(molecules, LEGEND_BOX)
    assert (drawing.status, drawing.smiles) == ("abstained", None)
    assert drawing.reasons == ["legend fragment M0 for table p2-t2"]
    assert counts["legend_drawings"] == 1


def test_a_code_without_a_drawing_stays_unreadable(paper):
    _replace_table(paper, TABLE.replace("M0", "M1"))
    molecules, counts = _run(paper)
    assert _rows(molecules)["5a"].reasons == ["cannot read R1 = 'M1'"]
    assert _drawn(molecules, LEGEND_BOX).status == "accepted"
    assert counts["legend_drawings"] == 0


def test_a_linker_slot_does_not_use_legends(paper):
    _edit_vote(paper, "p2-r1", smiles="[*:1]c1ccc(C(=O)N[*:2]C)cc1")  # R2 sits between two atoms
    _replace_table(paper, TABLE.replace("<td>M0</td><td>Me</td>", "<td>H</td><td>M0</td>"))
    molecules, _ = _run(paper)
    assert "cannot read R2 = 'M0'" in _rows(molecules)["5a"].reasons
    assert _drawn(molecules, LEGEND_BOX).status == "accepted"


def test_the_tables_page_comes_first_then_the_page_before_then_the_page_after(paper):
    _add_drawing(paper, "p1-r5", 1, [350, 650, 450, 720], "M0", "CN1CCCCC1")
    _add_drawing(paper, "p3-r1", 3, [100, 300, 200, 370], "M0", "CN1CCCC1")
    assert _rows(_run(paper)[0])["5a"].smiles == MORPHOLINYL_ROW
    _drop_drawing(paper, "p2-r2")
    assert _rows(_run(paper)[0])["5a"].smiles == canonical_smiles("CNC(=O)c1ccc(N2CCCCC2)cc1")
    _drop_drawing(paper, "p1-r5")
    assert _rows(_run(paper)[0])["5a"].smiles == canonical_smiles("CNC(=O)c1ccc(N2CCCC2)cc1")


def test_a_legend_read_by_one_reader_sends_its_row_to_review(paper):
    _edit_vote(paper, "p2-r2", agree=1, confidence=0.4)
    row = _rows(_run(paper)[0])["5a"]
    assert (row.status, row.smiles) == ("review", MORPHOLINYL_ROW)
    assert row.reasons == ["R1 drawing p2-r2 read by 1 agreeing reader(s)", "R1 = M0 from drawing p2-r2"]
    assert row.confidence == pytest.approx(0.4)
    assert _rows(_run(paper, {"vote": False})[0])["5a"].status == "accepted"


def test_a_compound_number_in_an_r_cell_is_not_a_legend_code(paper):
    _replace_table(paper, TABLE.replace("M0", "1"))  # drawing p1-r1 is compound 1, with one methyl
    molecules, _ = _run(paper)
    assert _rows(molecules)["5a"].reasons == ["cannot read R1 = '1'"]
    assert _drawn(molecules, [100, 120, 250, 225]).status == "accepted"
