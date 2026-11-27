"""Compound labels read by OCR in the figures (ocr.jsonl): which lines are labels and which drawing each names."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from molscout.pipeline import link
from molscout.pipeline.labels import link_regions, load_ocr_lines, ocr_label
from molscout.pipeline.schema import OcrLine

from .fixtures.a2.fakes import Block, Region, table, text

_SYNTHETIC = Path(__file__).parent / "fixtures" / "a2" / "paper_synthetic"


def line(bbox, words, score=0.99, page=1, source="image:p1-1") -> OcrLine:
    return OcrLine(page=page, bbox=list(bbox), text=words, score=score, source_block=source)


def drawing(rid, bbox, page=1) -> Region:
    return Region(region_id=rid, page=page, bbox=list(bbox))


@pytest.mark.parametrize(
    ("words", "label"),
    [
        ("12a", "12a"), ("(12a)", "12a"), ("5'", "5'"), ("5’", "5'"), ("6b", "6b"), ("(18)", "18"), ("7.", "7"),
        ("Compound 7", "7"), ("Roxadustat (1)", "1"), ("FG-2216 (4)", "4"), ("Ci FG-2216 (4)", "4"), ("S3", "S3"),
        ("CCT400028", "CCT400028"), ("AZD-9291", "AZD-9291"),
    ],
)
def test_ocr_label_reads_labels_and_named_labels(words, label):
    assert ocr_label(words).label == label


@pytest.mark.parametrize(
    "words",
    ["N", "O", "Cl", "NH", "OH", "HN", "R1", "R", "X", "R=", "8- 17", "2-oxoglutarate", "0", "05", "IC50", "2OG",
     "CO2", "0.256", "1234", "TYR-310", "ARG-38", "Lys101", "HEK293", "PHD1-3", "N11", "Daprodustat", ">25", ""],
)
def test_ocr_label_refuses_atoms_r_groups_ranges_and_values(words):
    assert ocr_label(words) is None


def test_ocr_label_gives_where_a_named_label_starts():
    found = ocr_label("Roxadustat (1)")
    assert found.start == pytest.approx(len("Roxadustat ") / len("Roxadustat (1)"))
    assert ocr_label("(1)").start == 0.0


# --- linking ---------------------------------------------------------------------------------------


def test_ocr_label_under_a_drawing_is_linked():
    regions = [drawing("p1-r1", [100, 100, 200, 200])]
    assert link_regions(regions, [], ocr_lines=[line([140, 206, 160, 216], "6")]) == {"p1-r1": "6"}


def test_atom_labels_and_ring_numbers_inside_the_drawing_are_not_labels():
    regions = [drawing("p1-r1", [100, 100, 200, 200])]
    lines = [line([120, 140, 128, 150], "4"), line([150, 150, 160, 160], "N"), line([170, 120, 180, 130], "OH")]
    assert link_regions(regions, [], ocr_lines=lines) == {}


def test_a_number_inside_one_drawing_is_not_given_to_its_neighbour():
    regions = [drawing("p1-r1", [100, 100, 200, 200]), drawing("p1-r2", [220, 100, 320, 200])]
    assert link_regions(regions, [], ocr_lines=[line([190, 150, 196, 158], "5")]) == {}


def test_a_label_the_drawing_box_took_in_at_its_lower_edge_is_linked():
    regions = [drawing("p1-r1", [100, 100, 200, 200])]
    assert link_regions(regions, [], ocr_lines=[line([140, 191, 146, 199], "7")]) == {"p1-r1": "7"}


def test_a_named_label_counts_from_its_parenthesised_part():
    regions = [drawing("p1-r1", [100, 100, 200, 180]), drawing("p1-r2", [80, 160, 104, 230])]
    lines = [line([99, 183, 190, 192], "Molidustat (5)")]  # the name starts beside p1-r2, the label is under p1-r1
    assert link_regions(regions, [], ocr_lines=lines) == {"p1-r1": "5"}


def test_a_tie_between_a_drawing_and_a_box_around_it_goes_to_the_inner_drawing():
    regions = [drawing("p1-r5", [100, 50, 200, 200]), drawing("p1-r7", [100, 120, 200, 200])]
    assert link_regions(regions, [], ocr_lines=[line([140, 202, 152, 212], "(8)")]) == {"p1-r7": "8"}


def test_text_mineru_already_read_is_not_read_twice():
    regions = [drawing("p1-r1", [100, 100, 200, 200])]
    blocks = [text(1, 0, [135, 206, 165, 216], "12a")]
    lines = [line([136, 206, 164, 216], "12a")]  # the same text: a duplicate would tie and link nothing
    assert link_regions(regions, blocks, ocr_lines=lines) == {"p1-r1": "12a"}


def test_text_inside_a_table_is_left_to_the_table():
    regions = [drawing("p1-r1", [330, 320, 380, 360])]
    blocks = [table(1, 0, "<table></table>", bbox=[300, 300, 550, 600])]
    assert link_regions(regions, blocks, ocr_lines=[line([305, 330, 315, 340], "11")]) == {}


def test_a_line_read_with_low_confidence_is_not_a_label():
    regions = [drawing("p1-r1", [100, 100, 200, 200])]
    assert link_regions(regions, [], ocr_lines=[line([140, 206, 160, 216], "6", score=0.4)]) == {}


def test_a_number_in_the_page_margin_is_not_a_label():
    regions = [drawing("p1-r1", [100, 700, 200, 750])]
    assert link_regions(regions, [], ocr_lines=[line([140, 762, 160, 772], "27")]) == {}


def test_axis_ticks_are_not_labels():
    regions = [drawing("p1-r1", [100, 100, 200, 140])]
    ticks = [line([215 - 6 * len(str(v)), 100 + 30 * k, 215, 108 + 30 * k], str(v)) for k, v in enumerate((0, 20, 40, 60))]
    assert link_regions(regions, [], ocr_lines=ticks[1:2]) == {"p1-r1": "20"}  # alone, "20" would be the label
    assert link_regions(regions, [], ocr_lines=ticks) == {}


def test_a_row_of_compound_labels_is_not_taken_for_ticks():
    regions = [drawing(f"p1-r{k}", [60 + 100 * k, 100, 140 + 100 * k, 180]) for k in range(1, 4)]
    labels = [line([95 + 100 * k, 186, 105 + 100 * k, 196], str(7 + k)) for k in range(1, 4)]
    assert link_regions(regions, [], ocr_lines=labels) == {"p1-r1": "8", "p1-r2": "9", "p1-r3": "10"}


def test_labels_of_a_series_printed_in_the_same_figure_name_r_groups_not_compounds():
    regions = [drawing("p2-r5", [420, 600, 490, 640], page=2), drawing("p2-r10", [190, 680, 226, 700], page=2),
               drawing("p2-r3", [250, 600, 320, 640], page=2)]
    lines = [
        line([449, 641, 470, 649], "8- 17", page=2, source="image:p2-7"),
        line([215, 690, 226, 700], "15", page=2, source="image:p2-7"),  # an R-group drawn in the scheme's R = box
        line([280, 643, 288, 651], "6", page=2, source="image:p2-7"),
    ]
    assert link_regions(regions, [], ocr_lines=lines) == {"p2-r3": "6"}
    assert link_regions(regions, [], ocr_lines=[*lines[1:], line([449, 641, 470, 649], "8- 17", page=2,
                                                                 source="region:p2-r5")]) == {"p2-r10": "15", "p2-r3": "6"}


def test_a_mineru_label_and_an_ocr_label_compete_by_distance():
    regions = [drawing("p1-r1", [100, 100, 200, 200])]
    blocks = [text(1, 0, [135, 230, 165, 240], "3")]
    assert link_regions(regions, blocks, ocr_lines=[line([140, 204, 160, 214], "6")]) == {"p1-r1": "6"}


def test_without_ocr_lines_linking_is_unchanged():
    regions = [drawing("p1-r1", [100, 100, 200, 200])]
    blocks = [Block(page=1, index=0, type="text", bbox=[135, 206, 165, 216], text="12a")]
    assert link_regions(regions, blocks) == link_regions(regions, blocks, ocr_lines=[]) == {"p1-r1": "12a"}


# --- ocr.jsonl in link and build ---------------------------------------------------------------------


def test_load_ocr_lines_reads_the_file_or_gives_none(tmp_path):
    assert load_ocr_lines(tmp_path) == []
    (tmp_path / "ocr.jsonl").write_text(json.dumps(
        {"page": 2, "bbox": [1, 2, 3, 4], "text": "7", "score": 0.9, "source_block": "region:p2-r1"}) + "\n")
    assert load_ocr_lines(tmp_path) == [OcrLine(2, [1, 2, 3, 4], "7", 0.9, "region:p2-r1")]


class NoOpsin:
    def to_smiles(self, names, radicals=False):
        return [None] * len(names)


def test_link_and_build_labels_a_drawing_from_ocr_jsonl(tmp_path):
    paper = tmp_path / "papers" / "synthetic"
    shutil.copytree(_SYNTHETIC, paper)
    before = {tuple(m.bbox[:2]): m.label for m in link.link_and_build(paper, NoOpsin()) if m.source == "drawn"}
    assert before[(100, 500)] is None
    lines = [line([165, 606, 180, 616], "5"), line([150, 540, 160, 550], "N")]
    (paper / "ocr.jsonl").write_text("".join(json.dumps(r.__dict__) + "\n" for r in lines))
    after = {tuple(m.bbox[:2]): m.label for m in link.link_and_build(paper, NoOpsin()) if m.source == "drawn"}
    assert after[(100, 500)] == "5" and after[(100, 120)] == before[(100, 120)] == "1"
    unlabelled = [m.label for m in link.link_and_build(paper, NoOpsin(), {"labels": False}) if m.source == "drawn"]
    assert set(unlabelled) == {None}
