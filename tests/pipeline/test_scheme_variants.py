"""scheme_variants: drawn cores filled from the R definitions printed beside them in a figure."""

from __future__ import annotations

import pytest

from molscout.pipeline.link import PaperInputs, drawn_molecules, link_paper
from molscout.pipeline.scheme_variants import parse_definition, scheme_molecules
from molscout.pipeline.schema import OcrLine, Region, VotedRead, write_jsonl
from molscout.scoring.smiles import canonical_smiles

PAPER = "26_test"
CORE = "[*:1]Oc1ccccc1"  # a phenyl ether; slot 1 is R


@pytest.mark.parametrize(
    ("text", "label", "values"),
    [
        ("20, R = Boc", "20", {"R": "Boc"}),
        ("R= Ph, 16b", "16b", {"R": "Ph"}),
        ("25 (R = H)", "25", {"R": "H"}),
        ("2 R = isobutyl", "2", {"R": "isobutyl"}),
        ("3 R = (S)-2-methylbutyl", "3", {"R": "(S)-2-methylbutyl"}),
        ("R1 = Me, R2 = H, 7a", "7a", {"R1": "Me", "R2": "H"}),
        ("9a: R1 = OMe; R2 = F", "9a", {"R1": "OMe", "R2": "F"}),
    ],
)
def test_a_definition_names_one_label_and_its_values(text, label, values):
    definition = parse_definition(text)
    assert (definition.label, definition.values) == (label, values)


@pytest.mark.parametrize(
    "text",
    ["R = Boc or Cbz", "12-17 R1 = CH3", "R = isobutyl, (S)-2-methylbutyl", "Scheme 1", "16a", "16a, 16b, R = Me",
     "R = Me", "Reagents and conditions: (a) R = Me", "20, R = "],
)
def test_text_that_is_not_one_definition_gives_none(text):
    assert parse_definition(text) is None


def region(region_id, page=1, bbox=(100.0, 100.0, 200.0, 160.0)):
    return Region(region_id, page, list(bbox), f"regions/{region_id}.png", 0.9, True)


def core(region_id, smiles=CORE, slots=None, agree=3):
    return VotedRead(region_id, smiles, None, agree, True, agree / 3, ["molglyph"], slots or {"1": "R"})


def line(text, page=1, bbox=(110.0, 165.0, 190.0, 175.0)):
    return OcrLine(page, list(bbox), text, 0.99, "image:p1-2")


def inputs(*votes, regions=None):
    regions = regions or [region(v.region_id) for v in votes]
    return PaperInputs(PAPER, [], {r.region_id: r for r in regions}, list(votes))


def run(paper, lines, used=None):
    return scheme_molecules(paper, lines, None, used or {})


def test_each_definition_fills_the_nearest_core_whose_slots_it_names():
    paper = inputs(core("p1-r1"), core("p1-r2", "[*:1]c1ccccc1[*:2]", {"1": "R1", "2": "R2"}),
                   regions=[region("p1-r1"), region("p1-r2", bbox=(300.0, 100.0, 400.0, 160.0))])
    molecules, cores = run(paper, [line("20, R = Me"), line("21, R = Et", bbox=(110.0, 178.0, 190.0, 188.0))])
    assert [(m.label, m.smiles, m.status) for m in molecules] == [
        ("20", canonical_smiles("COc1ccccc1"), "accepted"), ("21", canonical_smiles("CCOc1ccccc1"), "accepted")]
    assert molecules[0].source == "drawn" and molecules[0].reasons == ["core p1-r1 filled from '20, R = Me'"]
    assert cores == {"p1-r1": ["20", "21"]}


def test_a_definition_on_another_page_or_without_a_matching_core_gives_nothing():
    paper = inputs(core("p1-r1"))
    assert run(paper, [line("20, R = Me", page=2), line("21, R1 = Me")]) == ([], {})


def test_a_core_a_table_uses_is_not_filled():
    assert run(inputs(core("p1-r1")), [line("20, R = Me")], used={"p1-r1": ["p1-t3"]}) == ([], {})


def test_a_core_read_by_one_reader_gives_review():
    (molecule,), _ = run(inputs(core("p1-r1", agree=1)), [line("20, R = Me")])
    assert molecule.status == "review"
    assert molecule.reasons[-1] == "core read by 1 agreeing reader(s), fewer than 2"


def test_an_unreadable_value_abstains_with_the_reason():
    (molecule,), cores = run(inputs(core("p1-r1")), [line("20, R = zz9")])
    assert (molecule.status, molecule.smiles) == ("abstained", None)
    assert molecule.reasons == ["core p1-r1 filled from '20, R = zz9'", "cannot read R = 'zz9'"]
    assert cores == {"p1-r1": ["20"]}


def test_the_filled_core_names_its_compounds():
    paper = inputs(core("p1-r1"))
    (record,) = drawn_molecules(paper, {}, {}, scheme_cores={"p1-r1": ["20", "21"]})
    assert (record.status, record.reasons) == ("abstained", ["R-group core for scheme compounds 20, 21"])


def test_link_paper_fills_scheme_cores_from_the_figure_text(tmp_path):
    folder = tmp_path / PAPER
    write_jsonl(folder / "layout.jsonl", [])
    write_jsonl(folder / "regions.jsonl", [region("p1-r1")])
    write_jsonl(folder / "voted.jsonl", [core("p1-r1")])
    write_jsonl(folder / "ocr.jsonl", [line("20, R = Me"), line("21, R = Et", bbox=(110.0, 178.0, 190.0, 188.0))])
    molecules, counts = link_paper(folder, None)
    assert sorted((m.label, m.status) for m in molecules if m.label) == [("20", "accepted"), ("21", "accepted")]
    assert counts["scheme_variants"] == 2 and counts["in"] == 3
    off, off_counts = link_paper(folder, None, fixes={"tables": False})
    assert [m.reasons for m in off] == [["open R-group slots and no table uses this core"]]
    assert off_counts["scheme_variants"] == 0


def test_a_drawing_voted_whole_fills_scheme_compounds_from_its_alternative_core():
    alternative = {"smiles": CORE, "slots": {"1": "R"}, "agree": 3, "readers": ["molglyph", "molscribe", "molnextr"],
                   "confidence": 1.0, "kind": "attachment"}
    plain = VotedRead("p1-r1", "COc1ccccc1", None, 2, True, 2 / 3, ["molscribe", "molnextr"],
                      alternatives=[alternative])
    (molecule,), cores = run(inputs(plain), [line("20, R = Et")])
    assert (molecule.label, molecule.smiles, molecule.status) == ("20", canonical_smiles("CCOc1ccccc1"), "accepted")
    assert molecule.confidence == 1.0 and cores == {"p1-r1": ["20"]}
    (record,) = drawn_molecules(inputs(plain), {}, {}, scheme_cores=cores)
    assert (record.status, record.smiles) == ("abstained", None)
    assert record.reasons == ["R-group core for scheme compounds 20"]
    assert scheme_molecules(inputs(plain), [line("20, R = Et")], None, {}, {"attachments": False}) == ([], {})


def test_a_core_drawn_from_table_cells_is_not_a_scheme_core():
    cell = Region("p1-t3-r1-c1", 1, [100.0, 100.0, 200.0, 160.0], "cells/x.png", 0.9, True, kind="cell")
    assert run(inputs(core("p1-t3-r1-c1"), regions=[cell]), [line("20, R = Me")]) == ([], {})
