"""The vote for table-cell drawings: one attachment point from readers that draw it differently.

The reads are what the three readers gave on BioVista 26_6st3's Table 1 cells (a public paper) and on RDKit
drawings of R groups with wavy bonds.
"""

from __future__ import annotations

import json

import pytest

from molscout.pipeline import cell_vote, rgroups, vote
from molscout.pipeline.schema import Read
from molscout.scoring.smiles import canonical_smiles


def reads(*items, region="p3-t5-r4-c0"):
    return [Read(region, reader, smiles, slots or {}) for reader, smiles, slots in items]


def cast(*items):
    return cell_vote.vote_cell(reads(*items), fallback=vote.vote_with_reason)


# --- candidates ----------------------------------------------------------------------------------


def test_a_drawn_attachment_is_the_one_candidate():
    drawn, found = cell_vote.candidates(Read("x", "molglyph", "c1ccc(Oc2ccc(C[*:1])cc2)cc1", {"1": "*"}))
    assert drawn and list(found.values()) == [canonical_smiles("[*:1]Cc1ccc(Oc2ccccc2)cc1")]


def test_an_attachment_read_as_a_carbon_gives_a_candidate_per_methyl():
    drawn, found = cell_vote.candidates(Read("x", "molscribe", "CCc1ccc(C)cc1"), fixes={"attachments": False})
    assert not drawn
    assert set(found.values()) == {canonical_smiles("[*:1]Cc1ccc(C)cc1"), canonical_smiles("[*:1]c1ccc(CC)cc1")}


def test_a_radical_left_at_the_attachment_is_cleared_or_bonded_to_the_attachment():
    _, found = cell_vote.candidates(Read("x", "molnextr", "C[CH]c1ccc(-c2ccccc2)cc1"))
    assert canonical_smiles("[*:1]Cc1ccc(-c2ccccc2)cc1") in found.values()  # the methyl: [CH] gets its H back
    assert canonical_smiles("[*:1]C(C)c1ccc(-c2ccccc2)cc1") in found.values()  # the radical carbon
    _, found = cell_vote.candidates(Read("x", "molnextr", "[CH2]c1ccccc1"), fixes={"attachments": False})
    assert set(found.values()) == {canonical_smiles("[*:1]Cc1ccccc1"), canonical_smiles("[*:1]c1ccccc1")}


@pytest.mark.parametrize("smiles", ["CCc1ccnn1C", "CC(C)c1ccnn1C", "CCCc1ccnn1C"])
def test_an_attachment_read_as_an_ethyl_or_propyl_gives_the_chain_as_one_candidate(smiles):
    _, found = cell_vote.candidates(Read("x", "molnextr", smiles))
    assert canonical_smiles("Cn1nccc1[*:1]") in found.values()
    _, old = cell_vote.candidates(Read("x", "molnextr", smiles), fixes={"attachments": False})
    assert canonical_smiles("Cn1nccc1[*:1]") not in old.values()


@pytest.mark.parametrize("smiles", ["C=Cc1ccccc1", "OCc1ccccc1", "CC(C)(C)c1ccccc1", "C1CC1c1ccccc1", "CCCCc1ccccc1"])
def test_only_a_saturated_chain_of_two_or_three_carbons_stands_for_the_attachment(smiles):
    _, found = cell_vote.candidates(Read("x", "molscribe", smiles))
    assert canonical_smiles("[*:1]c1ccccc1") not in found.values()


def test_an_attachment_left_out_gives_a_candidate_per_hydrogen():
    _, found = cell_vote.candidates(Read("x", "molscribe", "c1ccc(Oc2ccccc2)cc1"))
    assert set(found.values()) == {canonical_smiles(smiles) for smiles in (
        "[*:1]c1ccc(Oc2ccccc2)cc1", "[*:1]c1cccc(Oc2ccccc2)c1", "[*:1]c1ccccc1Oc1ccccc1")}
    assert cell_vote.candidates(Read("x", "molscribe", "c1ccc(Oc2ccccc2)cc1"), fixes={"attachments": False}) == (
        False, {})
    _, found = cell_vote.candidates(Read("x", "molscribe", "c1cc[nH]c1"))
    assert canonical_smiles("[*:1]n1cccc1") in found.values()


def test_a_stray_word_is_dropped_and_two_drawings_give_nothing():
    _, found = cell_vote.candidates(Read("x", "molscribe", "CCCCC.CCc1ccc(Oc2ccccc2)cc1"))
    assert canonical_smiles("[*:1]Cc1ccc(Oc2ccccc2)cc1") in found.values()
    assert cell_vote.candidates(Read("x", "molglyph", "c1ccc(C[*:1])cc1.c1ccc(C[*:2])nc1", {"1": "*", "2": "*"})) == (
        False, {})
    assert cell_vote.candidates(Read("x", "molglyph", "[*:1]CC[*:2]", {"1": "*", "2": "*"})) == (False, {})


@pytest.mark.parametrize("read", [Read("x", "a", None, error="NoMolecule: empty"), Read("x", "a", "C1CC"),
                                  Read("x", "a", "")])
def test_failed_reads_give_no_candidate(read):
    assert cell_vote.candidates(read) == (False, {})


# --- vote_cell -----------------------------------------------------------------------------------


def test_readers_that_read_the_bond_as_a_carbon_agree_with_the_drawn_attachment():
    voted, reason = cast(("molglyph", "c1ccc(Oc2ccc(C[*:1])cc2)cc1", {"1": "*"}),
                         ("molnextr", "C[CH]c1ccc(Oc2ccccc2)cc1", None),
                         ("molscribe", "C[CH]c1ccc(Oc2ccccc2)cc1", None))
    assert reason is None
    assert voted.smiles == canonical_smiles("[*:1]Cc1ccc(Oc2ccccc2)cc1")
    assert (voted.agree, voted.confidence, voted.slots, voted.inchikey) == (3, 1.0, {"1": "*"}, None)
    assert voted.readers == ["molglyph", "molnextr", "molscribe"]


def test_a_reader_that_reads_another_structure_does_not_count():
    voted, _ = cast(("molglyph", "C1CCC(C[*:1])CC1", {"1": "*"}), ("molnextr", "CCC1CCCCC1", None),
                    ("molscribe", "CC(C)C1CCCCC1", None))
    assert voted.smiles == canonical_smiles("[*:1]CC1CCCCC1") and voted.agree == 2
    assert voted.confidence == pytest.approx(2 / 3)


def test_stereo_is_kept_only_when_the_readers_agree_on_it():
    voted, _ = cast(("molglyph", "OC[C@H](c1ccccc1)[*:1]", {"1": "*"}), ("molscribe", "C[C@@H](CO)c1ccccc1", None))
    assert voted.agree == 2 and voted.stereo_kept is False and "@" not in voted.smiles
    same, _ = cast(("molglyph", "OC[C@H](c1ccccc1)[*:1]", {"1": "*"}), ("molscribe", "C[C@H](CO)c1ccccc1", None))
    assert same.stereo_kept is True and "@" in same.smiles
    unstereo = cell_vote.vote_cell(reads(("molglyph", "OC[C@H](c1ccccc1)[*:1]", {"1": "*"})), stereo=False,
                                   fallback=vote.vote_with_reason)[0]
    assert "@" not in unstereo.smiles


def test_a_reader_without_stereo_leaves_the_stereo_of_the_others():
    items = (("molglyph", "OC[C@H](c1ccccc1)[*:1]", {"1": "*"}), ("molscribe", "CC(CO)c1ccccc1", None))
    voted, _ = cast(*items)
    assert (voted.agree, voted.stereo_kept, voted.smiles) == (2, True, canonical_smiles("OC[C@H](c1ccccc1)[*:1]"))
    off, _ = cell_vote.vote_cell(reads(*items), fallback=vote.vote_with_reason, fixes={"partial_stereo": False})
    assert off.stereo_kept is False and "@" not in off.smiles
    # The stereo may come from a reader that did not draw the attachment.
    other, _ = cast(("molglyph", "OCC(c1ccccc1)[*:1]", {"1": "*"}), ("molscribe", "C[C@H](CO)c1ccccc1", None))
    assert (other.stereo_kept, other.smiles) == (True, canonical_smiles("OC[C@H](c1ccccc1)[*:1]"))


def test_readers_that_read_the_bond_as_an_ethyl_or_isopropyl_agree_with_the_drawn_attachment():
    items = (("molglyph", "Cn1nccc1[*:1]", {"1": "*"}), ("molnextr", "CCc1ccnn1C", None),
             ("molscribe", "CC(C)c1ccnn1C", None))
    voted, _ = cast(*items)
    assert (voted.smiles, voted.agree, voted.readers) == (
        canonical_smiles("Cn1nccc1[*:1]"), 3, ["molglyph", "molnextr", "molscribe"])
    off, _ = cell_vote.vote_cell(reads(*items), fallback=vote.vote_with_reason, fixes={"attachments": False})
    assert (off.smiles, off.agree) == (canonical_smiles("Cn1nccc1[*:1]"), 1)


def test_a_reader_that_leaves_the_attachment_out_agrees():
    items = (("molglyph", "c1ccc(Oc2ccc([*:1])cc2)cc1", {"1": "*"}), ("molscribe", "c1ccc(Oc2ccccc2)cc1", None))
    assert cast(*items)[0].agree == 2
    off, _ = cell_vote.vote_cell(reads(*items), fallback=vote.vote_with_reason, fixes={"attachments": False})
    assert off.agree == 1


def test_one_drawing_reader_alone_is_a_vote_of_one():
    voted, _ = cast(("molglyph", "c1ccc(C[*:1])cc1", {"1": "*"}), ("molnextr", "[Ti-]Cc1ccccc1", None),
                    ("molscribe", "[V]Cc1ccccc1", None))
    assert voted.smiles == canonical_smiles("[*:1]Cc1ccccc1") and voted.agree == 1


def test_two_drawn_fragments_with_equal_support_tie():
    voted, reason = cast(("molglyph", "c1ccc(C[*:1])cc1", {"1": "*"}), ("molscribe", "c1ccc([*:1])cc1", {"1": "X"}))
    assert voted is None and reason == "tie between 2 fragments"


def test_without_a_drawn_attachment_the_readers_may_agree_on_a_whole_molecule():
    whole = "Cc1cccnc1CN1CCC2(CC1)C(=O)N(c1ccc(-c3ccccc3)cc1)C(=O)N2c1ncccn1"
    voted, _ = cast(("molglyph", whole, None), ("molnextr", whole, None), ("molscribe", whole, None))
    assert voted.smiles == canonical_smiles(whole) and voted.agree == 3 and voted.inchikey
    voted, reason = cast(("molglyph", "CCc1ccccc1", None), ("molscribe", "Cc1ccccc1", None))
    assert voted is None and reason == cell_vote.NO_ATTACHMENT


def test_attachments_counts_dummies():
    assert [cell_vote.attachments(s) for s in ("[*:1]CC", "C", None, "C1CC", "*C*")] == [1, 0, 0, 0, 2]


def test_drawn_fragment_is_a_substituent_with_its_stereo():
    fragment = rgroups.drawn_fragment("OC[C@H](c1ccccc1)[*:1]")
    assert fragment.kind == "substituent" and fragment.smiles == canonical_smiles("OC[C@H](c1ccccc1)*")
    assert "@" in fragment.smiles
    assert rgroups.drawn_fragment("[2*]Cc1ccccc1").smiles == canonical_smiles("*Cc1ccccc1")
    assert [rgroups.drawn_fragment(s) for s in ("CC", "*C*", None, "C1C")] == [None] * 4


# --- vote_paper ----------------------------------------------------------------------------------


def write(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def region(region_id, kind="drawing", structure=True):
    return {"region_id": region_id, "page": 3, "bbox": [0, 0, 10, 10], "crop": f"x/{region_id}.png",
            "structure_score": 0.9, "is_structure": structure, "kind": kind}


def read(region_id, reader, smiles, slots=None):
    return {"region_id": region_id, "reader": reader, "smiles": smiles, "slots": slots or {}, "error": None}


def test_vote_paper_votes_cells_as_fragments_and_counts_them(tmp_path):
    write(tmp_path / "regions.jsonl", [region("p3-r1"), region("p3-t5-r2-c0", "cell"), region("p3-t5-d1", "cell"),
                                       region("p3-t5-r3-c0", "cell")])
    write(tmp_path / "reads.jsonl", [
        read("p3-r1", "molglyph", "O=C(N[*:1])c1ccccc1", {"1": "R"}),
        read("p3-r1", "molscribe", "O=C(N[*:1])c1ccccc1", {"1": "R"}),
        read("p3-t5-r2-c0", "molglyph", "c1ccc(C[*:1])cc1", {"1": "*"}),
        read("p3-t5-r2-c0", "molscribe", "CCc1ccccc1"),
        read("p3-t5-d1", "molglyph", "CCO"),
        read("p3-t5-d1", "molscribe", "CCN"),
        read("p3-t5-r3-c0", "molglyph", "CC(C)(C)c1ccccc1"),
        read("p3-t5-r3-c0", "molscribe", "CC(C)(C)c1ccccc1"),
    ])

    counts = vote.vote_paper(tmp_path)

    voted = {r["region_id"]: r for r in map(json.loads, (tmp_path / "voted.jsonl").read_text().splitlines())}
    assert voted["p3-t5-r2-c0"]["smiles"] == canonical_smiles("[*:1]Cc1ccccc1")
    assert voted["p3-t5-r2-c0"]["agree"] == 2 and voted["p3-t5-r2-c0"]["slots"] == {"1": "*"}
    assert voted["p3-t5-d1"]["smiles"] is None and voted["p3-t5-d1"]["reason"] == cell_vote.NO_ATTACHMENT
    assert voted["p3-t5-r3-c0"]["smiles"] == "CC(C)(C)c1ccccc1"  # a whole molecule; link decides on its size
    assert counts == {"in": 4, "out": 3, "cores": 1, "dropped_cell_no_attachment": 1, "cells": 3,
                      "cell_fragments": 1, "cell_molecules": 1}


def test_a_cell_the_classifier_scores_badly_is_still_voted(tmp_path):
    write(tmp_path / "regions.jsonl", [{**region("p3-t5-r2-c0", "cell"), "structure_score": 0.99}])
    write(tmp_path / "reads.jsonl", [read("p3-t5-r2-c0", "molglyph", "c1ccc(C[*:1])cc1", {"1": "*"})])
    assert vote.vote_paper(tmp_path)["out"] == 1


def test_with_the_vote_fix_off_a_cell_takes_the_primary_readers_read(tmp_path):
    write(tmp_path / "regions.jsonl", [region("p3-t5-r2-c0", "cell")])
    write(tmp_path / "reads.jsonl", [read("p3-t5-r2-c0", "molglyph", "c1ccc(C[*:1])cc1", {"1": "*"}),
                                     read("p3-t5-r2-c0", "molscribe", "c1ccc([*:1])cc1", {"1": "X"})])
    vote.vote_paper(tmp_path, fixes={"vote": False}, primary_reader="molscribe")
    (record,) = [json.loads(line) for line in (tmp_path / "voted.jsonl").read_text().splitlines()]
    assert record["smiles"] == canonical_smiles("[*:1]c1ccccc1") and record["agree"] == 1


def test_a_cell_without_reads_has_no_vote(tmp_path):
    write(tmp_path / "regions.jsonl", [region("p3-t5-r2-c0", "cell")])
    write(tmp_path / "reads.jsonl", [])
    assert vote.vote_paper(tmp_path) == {"in": 1, "out": 0, "cores": 0, "dropped_no_reads": 1, "cells": 1,
                                         "cell_fragments": 0, "cell_molecules": 0}


def test_regions_written_before_cells_existed_read_as_drawings(tmp_path):
    old = {k: v for k, v in region("p1-r1").items() if k != "kind"}
    write(tmp_path / "regions.jsonl", [old])
    write(tmp_path / "reads.jsonl", [read("p1-r1", "molglyph", "c1ccc(C[*:1])cc1", {"1": "R"})])
    assert "cells" not in vote.vote_paper(tmp_path)
