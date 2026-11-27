import json
from types import SimpleNamespace

import pytest

from molscout.pipeline.schema import (
    FIXES,
    LayoutBlock,
    Molecule,
    PageRecord,
    Read,
    Region,
    VotedRead,
    fix_enabled,
    read_jsonl,
    to_points,
    write_jsonl,
)

RECORDS = {
    PageRecord: [
        PageRecord(page=1, width_pt=612.0, height_pt=792.0, image="pages/p1.png", dpi=200, vector=True),
        PageRecord(page=2, width_pt=595.276, height_pt=841.89, image="pages/p2.png", dpi=300, vector=False),
    ],
    LayoutBlock: [
        LayoutBlock(page=1, index=0, type="title", bbox=[72.0, 60.5, 540.0, 90.25], text="Discovery of 12a"),
        LayoutBlock(
            page=1,
            index=4,
            type="table",
            bbox=[72.0, 300.0, 540.0, 500.0],
            text="",
            html="<table><tr><td>R1</td></tr></table>",
            image="images/t1.jpg",
        ),
        LayoutBlock(page=1, index=5, type="caption", bbox=[72, 505, 540, 520], text="Table 1. IC50 (µM) of 12a–f", parent=4),
    ],
    Region: [
        Region("p3-r2", page=3, bbox=[10.5, 20.0, 110.0, 90.0], crop="crops/p3-r2.png", structure_score=0.97, is_structure=True),
    ],
    Read: [
        Read(region_id="p3-r2", reader="molglyph", smiles="[*:1]c1ccccc1", slots={"1": "R1"}, error=None),
        Read(region_id="p3-r2", reader="molscribe", smiles=None, slots={}, error="timeout"),
    ],
    VotedRead: [
        VotedRead(
            region_id="p3-r2",
            smiles="CCO",
            inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            agree=2,
            stereo_kept=True,
            confidence=2 / 3,
            readers=["molglyph", "molnextr"],
        ),
        VotedRead(
            region_id="p3-r3",
            smiles="[*:1]c1ccc([*:2])cc1",
            inchikey=None,
            agree=3,
            stereo_kept=True,
            confidence=1.0,
            readers=["molglyph", "molnextr", "molscribe"],
            slots={"1": "R1", "2": "R2"},
        ),
        VotedRead("p3-r4", None, None, 0, False, 0.0, [], reason="tie between 2 structures"),
        VotedRead(
            "p3-r5", None, None, 0, False, 0.0, [], reason="tie between 3 structures",
            alternatives=[{"smiles": "Cn1nccc1[*:1]", "slots": {"1": "R"}, "agree": 3,
                           "readers": ["molglyph", "molnextr", "molscribe"], "confidence": 1.0, "kind": "attachment"}],
        ),
        VotedRead(
            "p3-r6", "[*:1]c1ccc([*:2])cc1", None, 1, True, 0.5, ["molglyph"], slots={"1": "R1", "2": "R2"},
            skeleton_agree=2,
            alternatives=[{"smiles": "[*:1]c1ccc([*:2])cc1", "slots": {"1": "R2", "2": "R1"}, "agree": 1,
                           "readers": ["molnextr"], "confidence": 0.5, "kind": "slot_labels"}],
        ),
    ],
    Molecule: [
        Molecule("26_6st3", "12a", "CCO", page=3, bbox=[1.0, 2.0, 3.0, 4.0], source="drawn", status="accepted", confidence=1.0),
        Molecule(
            "26_6st3", "12b", "CCN", page=4, bbox=None, source="table", table_id="p4-t1", row=2, reasons=["no activity"], confidence=0.5
        ),
        Molecule("26_6st3", None, None, page=None, bbox=None, source="name", status="abstained", reasons=["OPSIN failed"]),
    ],
}


@pytest.mark.parametrize("cls", list(RECORDS))
def test_jsonl_round_trip_is_exact(tmp_path, cls):
    path = tmp_path / f"{cls.__name__}.jsonl"
    write_jsonl(path, RECORDS[cls])
    assert read_jsonl(path, cls) == RECORDS[cls]


def test_write_jsonl_writes_one_object_per_line(tmp_path):
    path = tmp_path / "nested" / "reads.jsonl"
    write_jsonl(path, RECORDS[Read])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {
        "region_id": "p3-r2",
        "reader": "molglyph",
        "smiles": "[*:1]c1ccccc1",
        "slots": {"1": "R1"},
        "error": None,
    }


def test_write_jsonl_keeps_non_ascii_text(tmp_path):
    path = tmp_path / "layout.jsonl"
    write_jsonl(path, RECORDS[LayoutBlock][2:])
    assert "12a–f" in path.read_text(encoding="utf-8")


def test_write_jsonl_accepts_plain_dicts(tmp_path):
    path = tmp_path / "pages.jsonl"
    write_jsonl(path, [{"page": 1, "width_pt": 1.0, "height_pt": 2.0, "image": "a.png", "dpi": 72, "vector": False}])
    assert read_jsonl(path, PageRecord) == [PageRecord(1, 1.0, 2.0, "a.png", 72, False)]


def test_write_jsonl_empty_list_gives_empty_file(tmp_path):
    path = tmp_path / "regions.jsonl"
    write_jsonl(path, [])
    assert path.read_text() == ""
    assert read_jsonl(path, Region) == []


def test_write_jsonl_leaves_no_temporary_file(tmp_path):
    write_jsonl(tmp_path / "regions.jsonl", RECORDS[Region])
    assert [p.name for p in tmp_path.iterdir()] == ["regions.jsonl"]


def test_write_jsonl_rejects_nan(tmp_path):
    region = Region("p1-r1", 1, [0, 0, 1, 1], "c.png", float("nan"), False)
    with pytest.raises(ValueError):
        write_jsonl(tmp_path / "regions.jsonl", [region])
    assert not (tmp_path / "regions.jsonl").exists()


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "reads.jsonl"
    line = json.dumps({"region_id": "p1-r1", "reader": "molscribe", "smiles": "C", "slots": {}, "error": None})
    path.write_text(f"{line}\n\n{line}\n", encoding="utf-8")
    assert len(read_jsonl(path, Read)) == 2


def test_read_jsonl_reports_bad_json_with_line_number(tmp_path):
    path = tmp_path / "reads.jsonl"
    path.write_text('{"region_id": "p1-r1", "reader": "x", "smiles": null}\n{not json}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"reads.jsonl:2"):
        read_jsonl(path, Read)


def test_read_jsonl_reports_unknown_or_missing_fields(tmp_path):
    path = tmp_path / "reads.jsonl"
    path.write_text('{"region_id": "p1-r1", "reader": "x", "smiles": "C", "colour": "red"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"reads.jsonl:1.*Read"):
        read_jsonl(path, Read)


def test_read_defaults_for_slots_and_error():
    read = Read(region_id="p1-r1", reader="molnextr", smiles="CCO")
    assert read.slots == {}
    assert read.error is None


def test_molecule_rejects_unknown_status_and_source():
    with pytest.raises(ValueError, match="status"):
        Molecule("p", "1", "C", 1, None, "drawn", status="maybe")
    with pytest.raises(ValueError, match="source"):
        Molecule("p", "1", "C", 1, None, "ocr")


def test_records_are_immutable():
    page = RECORDS[PageRecord][0]
    with pytest.raises(AttributeError):
        page.dpi = 72


def test_to_points_scales_normalized_box():
    assert to_points([0.1, 0.25, 0.5, 1.0], 600.0, 800.0) == pytest.approx([60.0, 200.0, 300.0, 800.0])


def test_to_points_rejects_wrong_length():
    with pytest.raises(ValueError, match="4 numbers"):
        to_points([0.1, 0.2, 0.3], 600.0, 800.0)


def test_voted_read_without_slots_or_reason_reads_with_defaults(tmp_path):
    path = tmp_path / "voted.jsonl"
    line = {"region_id": "p1-r1", "smiles": "CCO", "inchikey": "", "agree": 3, "stereo_kept": True,
            "confidence": 1.0, "readers": ["molglyph"]}
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    (voted,) = read_jsonl(path, VotedRead)
    assert (voted.slots, voted.reason) == ({}, None)


def test_voted_read_written_before_alternatives_reads_with_defaults(tmp_path):
    path = tmp_path / "voted.jsonl"
    line = {"region_id": "p1-r1", "smiles": "[*:1]CC", "inchikey": None, "agree": 2, "stereo_kept": True,
            "confidence": 1.0, "readers": ["molglyph", "molscribe"], "slots": {"1": "R"}, "reason": None}
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    (voted,) = read_jsonl(path, VotedRead)
    assert (voted.alternatives, voted.skeleton_agree, voted.slots) == ([], 0, {"1": "R"})


def test_fixes_lists_the_ladder_rungs():
    assert FIXES == (
        "labels", "relevance", "tables", "core_check", "vote", "stereo", "junk", "names", "formulas",
        "attachments", "slot_labels", "legends", "partial_stereo",
    )


@pytest.mark.parametrize(
    ("fixes", "on"),
    [
        (None, True),
        ({"junk": False}, False),
        ({"junk": True}, True),
        ({"stereo": False}, True),
        ({"junk"}, True),
        ({"stereo"}, False),
        (["junk"], True),
        (SimpleNamespace(junk=False), False),
        (SimpleNamespace(stereo=False), True),
    ],
)
def test_fix_enabled_reads_mappings_sets_and_objects(fixes, on):
    assert fix_enabled(fixes, "junk") is on


def test_fix_enabled_rejects_unknown_names():
    with pytest.raises(ValueError, match="junkk"):
        fix_enabled(None, "junkk")


@pytest.mark.parametrize("fixes", [{"junkk": False}, {"junkk"}, ["labels", "relabel"]])
def test_fix_enabled_rejects_unknown_flags_in_the_settings(fixes):
    with pytest.raises(ValueError, match="unknown fix"):
        fix_enabled(fixes, "junk")
