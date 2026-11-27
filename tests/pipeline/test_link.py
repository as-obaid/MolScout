"""Link and build on a synthetic paper: drawn, table and named molecules, with the real chemistry.

OPSIN is replaced by a lookup table; R-group cells use the local rules of `rgroups`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from molscout.pipeline import link
from molscout.scoring.smiles import canonical_smiles

_A2 = Path(__file__).parent / "fixtures" / "a2"
NAMES = {
    "N-Phenylbenzamide": "O=C(Nc1ccccc1)c1ccccc1",
    "4-Methoxy-N-phenylbenzamide": "COc1ccc(C(=O)Nc2ccccc2)cc1",
}
# The fixture's table on the core [*:1]c1ccc(C(=O)N[*:2])cc1 (R1 on the ring, R2 on the amide N).
BUILT = {
    "3a": canonical_smiles("CNC(=O)c1ccccc1"),
    "3b": canonical_smiles("CCNC(=O)c1ccc(F)cc1"),
    "3d": canonical_smiles("CNC(=O)c1ccc(Cl)cc1"),
}


class FakeOpsin:
    """Looks names up in a table; records each batch."""

    def __init__(self, names: dict[str, str] | None = None) -> None:
        self.names = NAMES if names is None else names
        self.batches: list[list[str]] = []

    def to_smiles(self, names: list[str], radicals: bool = False) -> list[str | None]:
        self.batches.append(list(names))
        return [self.names.get(n) for n in names]


@pytest.fixture
def paper(tmp_path) -> Path:
    """A copy of the synthetic paper that a test may edit."""
    target = tmp_path / "papers" / "synthetic"
    shutil.copytree(_A2 / "paper_synthetic", target)
    return target


def _rewrite(path: Path, edit) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    path.write_text("".join(json.dumps(r) + "\n" for r in edit(rows)))


def _edit_core(paper: Path, **changes) -> None:
    _rewrite(paper / "voted.jsonl", lambda rows: [{**r, **changes} if r["region_id"] == "p2-r1" else r for r in rows])


def _by(molecules, source):
    return [m for m in molecules if m.source == source]


def _rows(molecules):
    return {m.label: m for m in _by(molecules, "table")}


def _run(paper, fixes=None, opsin="default"):
    return link.link_and_build(paper, FakeOpsin() if opsin == "default" else opsin, fixes)


# --- drawings -------------------------------------------------------------------------------


def test_drawn_molecules_get_labels_and_are_accepted_when_built(paper):
    drawn = {tuple(m.bbox): m for m in _by(_run(paper), "drawn")}
    first = drawn[(100, 120, 250, 225)]
    assert (first.label, first.status, first.paper_id, first.page) == ("1", "accepted", "synthetic", 1)
    assert first.smiles == "COc1ccc(C(=O)Nc2ccccc2)cc1"
    assert first.confidence == pytest.approx(0.95)
    assert drawn[(350, 120, 500, 225)].label == "2"
    unlabelled = drawn[(100, 500, 250, 600)]
    assert (unlabelled.label, unlabelled.status) == (None, "accepted")


def test_regions_the_gate_rejected_give_no_record(paper):
    assert all(m.bbox != [300, 500, 450, 600] for m in _run(paper))


def test_core_drawing_is_abstained_with_its_table(paper):
    (core,) = [m for m in _by(_run(paper), "drawn") if m.page == 2]
    assert (core.status, core.smiles) == ("abstained", None)
    assert any("p2-t2" in r for r in core.reasons)


def test_drawing_without_a_vote_is_recorded_with_the_votes_reason(paper):
    _rewrite(
        paper / "voted.jsonl",
        lambda rows: [{**r, "smiles": None, "reason": "tie between 3 structures"} if r["region_id"] == "p1-r3" else r
                      for r in rows],
    )
    (unagreed,) = [m for m in _by(_run(paper), "drawn") if m.bbox == [100, 500, 250, 600]]
    assert (unagreed.status, unagreed.smiles) == ("abstained", None)
    assert unagreed.reasons == ["no vote: tie between 3 structures"]


# --- table rows -----------------------------------------------------------------------------


def test_every_table_row_gives_a_record_built_from_the_core(paper):
    rows = _by(_run(paper), "table")
    assert [(m.label, m.row, m.table_id) for m in rows] == [
        ("3a", 0, "p2-t2"), ("3b", 1, "p2-t2"), ("3c", 2, "p2-t2"), ("3d", 3, "p2-t2"),
    ]
    assert [m.status for m in rows] == ["accepted", "accepted", "abstained", "accepted"]
    assert {m.label: m.smiles for m in rows if m.smiles} == BUILT
    assert rows[0].page == 2 and rows[0].bbox == [72, 215, 540, 400]
    assert rows[0].confidence == pytest.approx(0.9)
    assert rows[2].smiles is None
    assert rows[2].reasons == ["cannot read R2 = 'Q'"]


def test_slots_numbered_the_other_way_build_the_same_rows(paper):
    _edit_core(paper, smiles="[*:2]c1ccc(C(=O)N[*:1])cc1", slots={"1": "R2", "2": "R1"})
    assert {k: m.smiles for k, m in _rows(_run(paper)).items() if m.smiles} == BUILT


def test_slot_labels_decide_which_cell_goes_where(paper):
    _edit_core(paper, slots={"1": "R2", "2": "R1"})  # R1 drawn on the amide N, R2 on the ring
    rows = _rows(_run(paper))
    assert rows["3b"].smiles == canonical_smiles("CCc1ccc(C(=O)NF)cc1")


def test_label_spellings_match_the_column_headers(paper):
    _edit_core(paper, slots={"1": "R¹", "2": "R<sup>2</sup>"})
    assert _rows(_run(paper))["3b"].smiles == BUILT["3b"]


def test_a_repeated_label_gets_a_copy_of_its_fragment_at_each_place(paper):
    _edit_core(paper, smiles="[*:1]c1cc([*:2])cc(C(=O)N[*:3])c1", slots={"1": "R1", "2": "R1", "3": "R2"})
    rows = _rows(_run(paper))
    assert rows["3b"].smiles == canonical_smiles("CCNC(=O)c1cc(F)cc(F)c1")
    assert rows["3b"].status == "accepted"


def test_core_labelled_with_a_variable_the_table_lacks_abstains_its_rows(paper):
    _edit_core(paper, slots={"1": "R1", "2": "R3"})
    rows = _rows(_run(paper))
    assert rows["3a"].status == "abstained"
    assert "no slot in the core for R2" in rows["3a"].reasons
    assert "core slot(s) R3 have no table column" in rows["3a"].reasons
    assert all(m.smiles is None for m in rows.values())


def test_lone_unlabelled_slot_fills_a_one_variable_table(paper):
    _edit_core(paper, smiles="[*:1]c1ccc(C(=O)NC)cc1", slots={"1": "*"})
    _replace_table(paper, "<table><tr><th>Compd</th><th>R</th><th>IC50 (nM)</th></tr>"
                          "<tr><td>8a</td><td>F</td><td>3</td></tr></table>")
    assert _rows(_run(paper))["8a"].smiles == canonical_smiles("CNC(=O)c1ccc(F)cc1")


def test_a_core_without_a_slot_map_abstains_its_rows(paper):
    _edit_core(paper, slots=None)
    assert {m.status for m in _rows(_run(paper)).values()} == {"abstained"}


def test_unlabelled_slots_do_not_fill_a_two_variable_table(paper):
    _edit_core(paper, slots={"1": "*", "2": "*"})
    rows = _rows(_run(paper))
    assert {m.status for m in rows.values()} == {"abstained"}
    assert "core slot(s) * have no table column" in rows["3a"].reasons


def test_core_that_passes_the_check_beats_a_nearer_one_that_fails(paper):
    def edit(rows):
        near = {**rows[-1], "region_id": "p2-r2", "smiles": "[*:1]c1ccc(C(=O)O[*:2])cc1", "slots": {"1": "R1", "2": "X"}}
        return [*rows, near]

    _rewrite(paper / "voted.jsonl", edit)
    _rewrite(paper / "regions.jsonl", lambda rows: [*rows, {**rows[-1], "region_id": "p2-r2", "bbox": [210, 190, 390, 212]}])
    rows = _rows(_run(paper))
    assert rows["3b"].smiles == BUILT["3b"] and rows["3b"].status == "accepted"
    unchecked = _rows(_run(paper, fixes={"core_check": False}))
    assert unchecked["3b"].status == "abstained"
    assert "core slot(s) X have no table column" in unchecked["3b"].reasons


def test_failed_core_check_is_recorded_on_the_rows(paper):
    _edit_core(paper, slots={"1": "R1", "2": "X"})
    rows = _rows(_run(paper))
    assert rows["3a"].status == "abstained"
    assert "core check failed: no slot for R2; slots no column fills: X" in rows["3a"].reasons
    assert "core slot(s) X have no table column" in rows["3a"].reasons
    unchecked = _rows(_run(paper, fixes={"core_check": False}))["3a"]
    assert not any("core check" in r for r in unchecked.reasons)


def test_core_without_reader_agreement_goes_to_review(paper):
    _edit_core(paper, agree=1)
    rows = _rows(_run(paper))
    assert rows["3a"].status == "review"
    assert any("agree" in r for r in rows["3a"].reasons)
    assert _rows(_run(paper, fixes={"vote": False}))["3a"].status == "accepted"


CORE = "[*:1]c1ccc(C(=O)N[*:2])cc1"
R12 = {"1": "R1", "2": "R2"}


def _alternative(kind, slots=None, agree=3, readers=("molglyph", "molscribe", "molnextr")):
    return {"smiles": CORE, "slots": slots or R12, "agree": agree, "readers": list(readers)[:agree],
            "confidence": agree / 3, "kind": kind}


def _drawn_core(molecules):
    (record,) = [m for m in _by(molecules, "drawn") if m.page == 2]
    return record


@pytest.mark.parametrize("main", [
    {"smiles": "CNC(=O)c1ccccc1", "slots": {}, "agree": 2, "confidence": 2 / 3},  # the slots read as H and methyl
    {"smiles": None, "slots": {}, "agree": 0, "confidence": 0.0, "reason": "tie between 3 structures"},
])
def test_a_table_builds_from_the_core_a_drawing_votes_as_an_alternative(paper, main):
    _edit_core(paper, **main, alternatives=[_alternative("attachment")])
    molecules = _run(paper)
    rows = _rows(molecules)
    assert {k: m.smiles for k, m in rows.items() if m.smiles} == BUILT
    assert rows["3a"].status == "accepted" and rows["3a"].confidence == pytest.approx(1.0)
    core = _drawn_core(molecules)
    assert (core.status, core.smiles, core.reasons) == ("abstained", None, ["R-group core for table p2-t2"])
    off = _run(paper, fixes={"attachments": False})
    assert {m.status for m in _rows(off).values()} == {"abstained"}
    assert _drawn_core(off).smiles == main["smiles"]  # the plain reading is the drawing's molecule again


def test_a_drawn_core_beats_a_nearer_drawings_attachment_alternative(paper):
    # an R-group drawing beside the table, read as a plain molecule, whose alternative core also fits the columns
    def edit(rows):
        alternative = {**_alternative("attachment"), "smiles": "[*:1]c1ccc(S(=O)(=O)N[*:2])cc1"}
        near = {**rows[-1], "region_id": "p2-r2", "smiles": "CNS(=O)(=O)c1ccccc1", "slots": {}, "inchikey": None,
                "alternatives": [alternative]}
        return [*rows, near]

    _rewrite(paper / "voted.jsonl", edit)
    _rewrite(paper / "regions.jsonl", lambda rows: [*rows, {**rows[-1], "region_id": "p2-r2", "bbox": [210, 190, 390, 212]}])
    rows = _rows(_run(paper))
    assert {k: m.smiles for k, m in rows.items() if m.smiles} == BUILT  # the drawn core, not the nearer alternative


def test_the_table_columns_choose_between_labellings_the_readers_split_on(paper):
    _edit_core(paper, slots={"1": "R1", "2": "X"}, agree=1, confidence=1 / 3, skeleton_agree=2,
               readers=["molglyph"], alternatives=[_alternative("slot_labels", agree=1, readers=["molnextr"])])
    rows = _rows(_run(paper))
    assert {k: m.smiles for k, m in rows.items() if m.smiles} == BUILT
    assert rows["3a"].status == "accepted"  # one labelling fits the table, so both readers of the skeleton count
    off = _rows(_run(paper, fixes={"slot_labels": False}))
    assert {m.status for m in off.values()} == {"abstained"}
    assert "core slot(s) X have no table column" in off["3a"].reasons


def test_when_several_labellings_fit_the_table_the_core_keeps_its_own_agreement(paper):
    _edit_core(paper, agree=1, confidence=1 / 3, skeleton_agree=2, readers=["molglyph"],
               alternatives=[_alternative("slot_labels", slots={"1": "R2", "2": "R1"}, agree=1, readers=["molnextr"])])
    rows = _rows(_run(paper))
    assert rows["3a"].smiles == BUILT["3a"]  # the drawing's own labelling comes first
    assert rows["3a"].status == "review"
    assert "core read by 1 agreeing reader(s), fewer than 2" in rows["3a"].reasons


def test_missing_core_abstains_every_row(paper):
    _rewrite(paper / "voted.jsonl", lambda rows: [r for r in rows if r["region_id"] != "p2-r1"])
    rows = _by(_run(paper), "table")
    assert len(rows) == 4
    assert {m.status for m in rows} == {"abstained"}
    assert all(any("core" in r for r in m.reasons) for m in rows)


def test_link_does_not_read_reads_jsonl(paper):
    (paper / "reads.jsonl").write_text("not json\n")
    assert {k: m.smiles for k, m in _rows(_run(paper)).items() if m.smiles} == BUILT


def _replace_table(paper: Path, html: str) -> None:
    _rewrite(paper / "layout.jsonl", lambda rows: [{**r, "html": html} if r["type"] == "table" else r for r in rows])


def test_unsplit_label_range_gives_one_abstained_record_per_label(paper):
    _replace_table(
        paper,
        "<table><tr><th>Compd</th><th>R1</th><th>R2</th><th>IC50 (nM)</th></tr>"
        "<tr><td>5a–c</td><td>H, Me</td><td>Cl</td><td>3</td></tr>"
        "<tr><td>6</td><td>Et</td><td>Me</td><td>4</td></tr></table>",
    )
    rows = _by(_run(paper), "table")
    assert [(m.label, m.row, m.status) for m in rows] == [
        ("5a", 0, "abstained"), ("5b", 0, "abstained"), ("5c", 0, "abstained"), ("6", 1, "accepted"),
    ]
    assert all(any("range" in r for r in m.reasons) for m in rows[:3])
    assert rows[3].smiles == canonical_smiles("CCc1ccc(C(=O)NC)cc1")


def test_drawn_cell_and_empty_cell_are_explained(paper):
    _replace_table(
        paper,
        "<table><tr><th>Compd</th><th>R1</th><th>R2</th><th>IC50 (nM)</th></tr>"
        "<tr><td>7a</td><td><img src='images/cell.jpg'/></td><td>Me</td><td>3</td></tr>"
        "<tr><td>7b</td><td>H</td><td>–</td><td>4</td></tr></table>",
    )
    first, second = _by(_run(paper), "table")
    assert first.status == second.status == "abstained"
    assert any("drawn" in r for r in first.reasons)
    assert any("empty" in r for r in second.reasons)


def test_named_substituents_go_to_opsin_in_one_batch(paper):
    _replace_table(
        paper,
        "<table><tr><th>Compd</th><th>R1</th><th>R2</th><th>IC50 (nM)</th></tr>"
        "<tr><td>9a</td><td>morpholin-4-yl</td><td>Me</td><td>3</td></tr>"
        "<tr><td>9b</td><td>piperidin-1-yl</td><td>Me</td><td>4</td></tr></table>",
    )
    opsin = FakeOpsin({**NAMES, "morpholin-4-yl": "*N1CCOCC1", "piperidin-1-yl": "*N1CCCCC1"})
    rows = _rows(_run(paper, opsin=opsin))
    assert opsin.batches[0] == ["morpholin-4-yl", "piperidin-1-yl"]
    assert rows["9a"].smiles == canonical_smiles("CNC(=O)c1ccc(N2CCOCC2)cc1")


# --- names ----------------------------------------------------------------------------------


def test_names_become_molecules_and_repeats_are_kept_for_dedupe(paper):
    opsin = FakeOpsin()
    names = {m.label: m for m in _by(_run(paper, opsin=opsin), "name")}
    assert opsin.batches == [["N-Phenylbenzamide", "4-Methoxy-N-phenylbenzamide"]]
    assert (names["4"].smiles, names["4"].status, names["4"].page) == ("O=C(Nc1ccccc1)c1ccccc1", "accepted", 3)
    assert (names["1"].status, names["1"].reasons) == ("accepted", [])


def test_name_that_disagrees_with_the_drawing_goes_to_review(paper):
    opsin = FakeOpsin({**NAMES, "4-Methoxy-N-phenylbenzamide": "CCOc1ccc(C(=O)Nc2ccccc2)cc1"})
    names = {m.label: m for m in _by(_run(paper, opsin=opsin), "name")}
    assert names["1"].status == "review"
    assert any("differs" in r for r in names["1"].reasons)


def test_unparsed_name_is_abstained(paper):
    names = {m.label: m for m in _by(_run(paper, opsin=FakeOpsin({})), "name")}
    assert (names["4"].status, names["4"].smiles) == ("abstained", None)
    assert any("OPSIN" in r for r in names["4"].reasons)


def test_names_without_opsin_are_abstained(paper):
    names = _by(_run(paper, opsin=None), "name")
    assert {m.status for m in names} == {"abstained"}
    assert len(names) == 2


# --- fixes ----------------------------------------------------------------------------------


def test_labels_off_leaves_drawings_unlabelled(paper):
    drawn = [m for m in _by(_run(paper, fixes={"labels": False}), "drawn") if m.smiles]
    assert {m.label for m in drawn} == {None}


def test_drawings_only_rung_keeps_drawings_alone(paper):
    molecules = _run(paper, fixes={"labels": False, "relevance": False, "tables": False, "names": False})
    assert {m.source for m in molecules} == {"drawn"}
    statuses = {m.smiles: m.status for m in molecules}
    assert statuses["Cc1ccc(C(=O)Nc2ccccc2)cc1"] == "accepted"
    assert statuses[None] == "abstained"  # the core, used by no table


@pytest.mark.parametrize(
    "fixes",
    [
        {"labels", "relevance", "core_check", "vote", "stereo", "junk", "names", "formulas"},
        SimpleNamespace(tables=False),
        {"tables": False},
    ],
)
def test_fix_flags_accept_sets_objects_and_mappings(paper, fixes):
    assert _by(_run(paper, fixes=fixes), "table") == []


def test_names_off_gives_no_named_molecules(paper):
    assert _by(_run(paper, fixes={"names": False}), "name") == []


@pytest.mark.parametrize("name", ["layout.jsonl", "regions.jsonl", "voted.jsonl"])
def test_missing_stage_output_fails_loudly(paper, name):
    (paper / name).unlink()
    with pytest.raises(FileNotFoundError, match=name):
        _run(paper)


def test_empty_paper_gives_no_molecules(tmp_path):
    for name in ("layout", "regions", "voted"):
        (tmp_path / f"{name}.jsonl").write_text("")
    assert link.link_and_build(tmp_path, FakeOpsin(), None) == []
