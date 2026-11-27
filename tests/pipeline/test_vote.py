import json
from dataclasses import replace

import pytest
from rdkit import Chem

from molscout.pipeline.schema import Read, Region, VotedRead, read_jsonl, write_jsonl
from molscout.pipeline.vote import vote, vote_paper, vote_record, vote_with_reason
from molscout.scoring.smiles import canonical_smiles

READERS = ("molglyph", "molscribe", "molnextr")

# One R-group core drawn with R1 on the ring and R2 on the amide N, numbered two ways.
CORE = "[*:1]c1ccc(C(=O)N[*:2])cc1"
SWAPPED = "[*:2]c1ccc(C(=O)N[*:1])cc1"
R12 = {"1": "R1", "2": "R2"}
R21 = {"1": "R2", "2": "R1"}


def reads(*smiles, errors=None):
    errors = errors or {}
    return [
        Read(region_id="p2-r1", reader=reader, smiles=text, error=errors.get(reader))
        for reader, text in zip(READERS, smiles, strict=False)
    ]


def core_reads(*pairs, readers=READERS):
    return [Read("p2-r1", reader, smiles, slots) for reader, (smiles, slots) in zip(readers, pairs, strict=False)]


def key(smiles):
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def labelled(smiles, slots):
    """The core with each slot's label in place of its number, to compare two numberings."""
    mol = Chem.MolFromSmiles(smiles)
    names = sorted(set(slots.values()))
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(names.index(slots[str(atom.GetAtomMapNum())]) + 1)
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


# --- plain molecules ------------------------------------------------------------------------


def test_three_of_three_agree_across_spellings():
    got = vote(reads("OCC", "CCO", "C(O)C"))
    assert got == VotedRead("p2-r1", "CCO", key("CCO"), 3, True, 1.0, list(READERS))


def test_two_of_three_agree():
    got = vote(reads("c1ccccc1O", "Oc1ccccc1", "Nc1ccccc1"))
    assert got == VotedRead("p2-r1", "Oc1ccccc1", key("Oc1ccccc1"), 2, True, pytest.approx(2 / 3), ["molglyph", "molscribe"])


def test_kekule_and_aromatic_spellings_agree():
    got = vote(reads("C1=CC=CC=C1C(=O)O", "OC(=O)c1ccccc1", "CC"))
    assert got.agree == 2
    assert got.smiles == canonical_smiles("OC(=O)c1ccccc1")


def test_three_different_reads_tie():
    voted, reason = vote_with_reason(reads("CCO", "CCN", "CCC"))
    assert voted is None
    assert reason == "tie between 3 structures"
    assert vote(reads("CCO", "CCN", "CCC")) is None


def test_two_usable_reads_that_differ_tie():
    voted, reason = vote_with_reason(reads("CCO", "CCN", None))
    assert voted is None and reason == "tie between 2 structures"


def test_reads_with_errors_or_without_smiles_are_ignored():
    got = vote(reads("CCOC(=O)C", None, "CCOC(C)=O", errors={"molglyph": "timeout"}))
    assert got.readers == ["molnextr"]
    assert got.agree == 1
    assert got.confidence == pytest.approx(1 / 3)


def test_invalid_smiles_is_ignored():
    got = vote(reads("C1CC", "CCN", "NCC"))
    assert got.agree == 2 and got.smiles == "CCN"


def test_no_usable_read():
    voted, reason = vote_with_reason(reads(None, "", "not smiles", errors={"molglyph": "crash"}))
    assert voted is None and reason == "no usable read"


def test_no_reads():
    assert vote_with_reason([]) == (None, "no reads")


def test_single_reader():
    got = vote([Read("p1-r1", "molglyph", "CCO")])
    assert (got.agree, got.confidence, got.stereo_kept, got.slots) == (1, 1.0, True, {})


def test_stereo_kept_when_all_agreeing_readers_agree_on_it():
    got = vote(reads("C[C@H](N)C(=O)O", "N[C@@H](C)C(=O)O", "CCCC"))
    assert got.stereo_kept is True
    assert got.smiles == canonical_smiles("C[C@H](N)C(=O)O")
    assert got.inchikey == key("C[C@H](N)C(=O)O")


def test_stereo_only_disagreement_strips_stereo():
    got = vote(reads("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O", "C[C@H](N)C(=O)O"))
    assert got.agree == 3
    assert got.stereo_kept is False
    assert got.smiles == "CC(N)C(=O)O"
    assert got.inchikey == key("CC(N)C(=O)O")


def test_a_reader_without_stereo_also_strips_it_without_the_partial_stereo_fix():
    got = vote(reads("C[C@H](N)C(=O)O", "CC(N)C(=O)O", "C[C@H](N)C(=O)O"), fixes={"partial_stereo": False})
    assert (got.agree, got.stereo_kept, got.smiles) == (3, False, "CC(N)C(=O)O")


def test_reads_without_stereo_leave_the_stereo_the_others_agree_on():
    given = reads("CC(N)C(=O)O", "C[C@H](N)C(=O)O", "N[C@@H](C)C(=O)O")
    got = vote(given)
    assert (got.agree, got.stereo_kept, got.smiles) == (3, True, canonical_smiles("C[C@H](N)C(=O)O"))
    assert got.inchikey == key("C[C@H](N)C(=O)O")  # from a stereo read, though molglyph's sorts first
    off = vote(given, fixes={"partial_stereo": False})
    assert (off.stereo_kept, off.smiles, off.inchikey) == (False, "CC(N)C(=O)O", key("CC(N)C(=O)O"))


def test_stereo_reads_that_disagree_strip_it_beside_a_read_without_stereo():
    got = vote(reads("CC(N)C(=O)O", "C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"))
    assert (got.agree, got.stereo_kept, got.smiles) == (3, False, "CC(N)C(=O)O")


def test_double_bond_stereo_disagreement_strips_it():
    got = vote(reads("C/C=C/C(=O)O", "C/C=C\\C(=O)O", "CCCC"))
    assert (got.agree, got.stereo_kept, got.smiles) == (2, False, "CC=CC(=O)O")


def test_connectivity_majority_wins_over_stereo_split():
    got = vote(reads("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O", "CCO"))
    assert got.agree == 2 and got.stereo_kept is False


def test_stereo_off_strips_even_agreed_stereo():
    got = vote(reads("C[C@H](N)C(=O)O", "N[C@@H](C)C(=O)O", "C[C@H](N)C(=O)O"), stereo=False)
    assert (got.agree, got.stereo_kept, got.smiles) == (3, False, "CC(N)C(=O)O")
    assert got.inchikey == key("CC(N)C(=O)O")


def test_reads_from_two_regions_are_refused():
    mixed = [Read("p1-r1", "molglyph", "C"), Read("p1-r2", "molscribe", "C")]
    with pytest.raises(ValueError, match="regions"):
        vote(mixed)


def test_a_reader_counted_twice_is_refused():
    twice = [Read("p1-r1", "molglyph", "C"), Read("p1-r1", "molglyph", "C")]
    with pytest.raises(ValueError, match="molglyph"):
        vote(twice)


# --- R-group cores (Markush) ----------------------------------------------------------------


def test_three_identical_cores_give_the_core_and_its_slots():
    got = vote(core_reads((CORE, R12), (CORE, R12), (CORE, R12)))
    assert (got.smiles, got.slots) == (canonical_smiles(CORE), R12)
    assert (got.agree, got.confidence, got.stereo_kept, got.inchikey) == (3, 1.0, True, None)
    assert got.readers == list(READERS)


def test_cores_numbered_differently_agree():
    got = vote(core_reads((CORE, R12), (SWAPPED, R21), (CORE, R12)))
    assert got.agree == 3
    assert (got.smiles, got.slots) == (canonical_smiles(CORE), R12)


@pytest.mark.parametrize("order", [READERS, tuple(reversed(READERS))])
def test_smiles_and_slots_come_from_one_read_picked_by_reader_name(order):
    # molglyph and molscribe each write a different numbering; on a 1-1 count the reader whose
    # name sorts first wins, whatever the order of the reads.
    pairs = {"molglyph": (SWAPPED, R21), "molscribe": (CORE, R12)}
    got = vote(core_reads(*(pairs[r] for r in order if r in pairs), readers=[r for r in order if r in pairs]))
    assert (got.smiles, got.slots) == (canonical_smiles(SWAPPED), R21)
    assert labelled(got.smiles, got.slots) == labelled(CORE, R12)


def test_labels_on_swapped_positions_disagree_without_the_slot_labels_fix():
    voted, reason = vote_with_reason(core_reads((CORE, R12), (CORE, R21)), fixes={"slot_labels": False})
    assert voted is None and reason == "slot labels disagree"


def test_the_label_majority_within_one_structure_wins():
    got = vote(core_reads((CORE, R12), (CORE, R21), (SWAPPED, R21)))
    assert got.agree == 2
    assert got.readers == ["molglyph", "molnextr"]
    assert labelled(got.smiles, got.slots) == labelled(CORE, R12)
    assert got.confidence == pytest.approx(2 / 3)


def test_label_spellings_are_compared_normalized():
    got = vote(core_reads((CORE, {"1": "R¹", "2": "R²"}), (CORE, R12)))
    assert got.agree == 2


def test_a_core_and_a_full_molecule_are_different_structures():
    voted, reason = vote_with_reason(core_reads((CORE, R12), ("Cc1ccc(C(=O)NC)cc1", {})))
    assert voted is None and reason == "tie between 2 structures"


def test_a_core_wins_over_a_lone_full_reading():
    got = vote(core_reads((CORE, R12), ("Cc1ccc(C(=O)NC)cc1", {}), (SWAPPED, R21)))
    assert got.agree == 2 and got.slots in (R12, R21)


def test_repeated_labels_vote():
    smiles, slots = "[*:1]c1cc([*:2])cc(C(=O)N[*:3])c1", {"1": "R1", "2": "R1", "3": "R2"}
    got = vote(core_reads((smiles, slots), (smiles, slots), ("[*:2]c1cc([*:1])cc(C(=O)N[*:3])c1", slots)))
    assert (got.agree, got.slots) == (3, slots)


def test_a_core_without_a_slot_map_votes_with_unlabelled_slots():
    got = vote([Read("p2-r1", "molglyph", CORE, None), Read("p2-r1", "molscribe", CORE, {})])
    assert (got.agree, got.slots) == (2, {})


def test_unlabelled_attachment_points_vote():
    got = vote(core_reads(("[*:1]CC(=O)Nc1ccccc1", {"1": "*"}), ("[*:1]CC(=O)Nc1ccccc1", {"1": "*"})))
    assert (got.agree, got.slots) == (2, {"1": "*"})


def test_core_stereo_disagreement_keeps_the_slots():
    got = vote(core_reads(("[*:1][C@H](F)c1ccccc1", {"1": "R1"}), ("[*:1][C@@H](F)c1ccccc1", {"1": "R1"})))
    assert got.stereo_kept is False
    assert (got.smiles, got.slots) == (canonical_smiles("[*:1]C(F)c1ccccc1"), {"1": "R1"})


def test_core_stereo_is_kept_when_clear_and_stripped_when_off():
    pair = ("[*:1][C@H](F)c1ccccc1", {"1": "R1"})
    assert vote(core_reads(pair, pair)).smiles == canonical_smiles(pair[0])
    off = vote(core_reads(pair, pair), stereo=False)
    assert (off.smiles, off.slots, off.stereo_kept) == (canonical_smiles("[*:1]C(F)c1ccccc1"), {"1": "R1"}, False)


def test_a_core_keeps_the_stereo_of_its_stereo_reads():
    pairs = (("[*:1]C(F)c1ccccc1", {"1": "R1"}), ("[*:2][C@H](F)c1ccccc1", {"2": "R1"}))
    got = vote(core_reads(*pairs))
    assert (got.agree, got.stereo_kept, got.smiles, got.slots) == (2, True, canonical_smiles(pairs[1][0]), {"2": "R1"})
    off = vote(core_reads(*pairs), fixes={"partial_stereo": False})
    assert (off.stereo_kept, off.smiles, off.slots) == (False, canonical_smiles("[*:1]C(F)c1ccccc1"), {"1": "R1"})


# --- slot labels the readers split on (the slot_labels fix) -------------------------------------

# One core from a public BioVista paper: MolGlyph and MolNexTR draw the same skeleton and label one slot differently.
GLYPH_CORE = "O=C(c1nc(C2C[*:5]([*:2])c3cccc([*:3])c3O2)sc1[*:4])[*:1]"
GLYPH_SLOTS = {"1": "R1", "2": "R2", "3": "R3", "4": "R4", "5": "X"}
NEXTR_CORE = "O=C(c1nc(C2C[*:5]([*:1])c3cccc([*:2])c3O2)sc1[*:3])[*:4]"
NEXTR_SLOTS = {"1": "R2", "2": "R3", "3": "R4", "4": "Rf", "5": "X"}


def test_tied_slot_labels_give_the_first_readers_labelling_and_keep_the_other():
    given = [Read("p2-r1", "molnextr", NEXTR_CORE, NEXTR_SLOTS), Read("p2-r1", "molglyph", GLYPH_CORE, GLYPH_SLOTS)]
    got, reason = vote_with_reason(given)
    assert reason is None
    assert (got.smiles, got.slots, got.readers) == (canonical_smiles(GLYPH_CORE), GLYPH_SLOTS, ["molglyph"])
    assert (got.agree, got.confidence, got.skeleton_agree, got.inchikey) == (1, 0.5, 2, None)
    assert got.alternatives == [{"smiles": canonical_smiles(NEXTR_CORE), "slots": NEXTR_SLOTS, "agree": 1,
                                 "readers": ["molnextr"], "confidence": 0.5, "kind": "slot_labels"}]
    assert vote_with_reason(given, fixes={"slot_labels": False}) == (None, "slot labels disagree")


def test_tied_slot_labels_go_to_the_reader_of_highest_priority():
    got = vote(core_reads((CORE, R21), (CORE, R12), readers=("molnextr", "molscribe")))
    assert (got.readers, got.slots, got.agree, got.skeleton_agree) == (["molscribe"], R12, 1, 2)
    assert [(a["readers"], a["slots"]) for a in got.alternatives] == [(["molnextr"], R21)]


def test_a_label_majority_records_no_alternative():
    got = vote(core_reads((CORE, R12), (CORE, R21), (SWAPPED, R21)))
    assert (got.agree, got.skeleton_agree, got.alternatives) == (2, 0, [])


# --- alternative cores: the attachment read as H or a short chain (the attachments fix) ----------

# Single drawings from public BioVista papers: MolGlyph draws the wavy attachment bond as a dummy, MolScribe and
# MolNexTR read it as a short carbon chain.
PYRROLE = "COC(=O)c1ccc([*:1])n1C"
PYRAZOLE = "Cn1nccc1[*:1]"
R = {"1": "R"}


def mixed(*items, region_id="p2-r1"):
    return [Read(region_id, reader, smiles, slots or {}) for reader, smiles, slots in items]


def test_a_plain_win_with_masked_support_records_the_drawn_core_and_keeps_the_molecule():
    given = mixed(("molglyph", PYRROLE, R), ("molscribe", "COC(=O)c1ccc(C(C)C)n1C", None),
                  ("molnextr", "COC(=O)c1ccc(C(C)C)n1C", None))
    got = vote(given)
    assert (got.smiles, got.agree, got.reason) == (canonical_smiles("COC(=O)c1ccc(C(C)C)n1C"), 2, None)
    assert got.alternatives == [{"smiles": canonical_smiles(PYRROLE), "slots": R, "agree": 3,
                                 "readers": ["molglyph", "molscribe", "molnextr"], "confidence": 1.0,
                                 "kind": "attachment"}]
    assert replace(got, alternatives=[]) == vote(given, fixes={"attachments": False})


def test_an_ethyl_reading_of_the_attachment_supports_the_core():
    got = vote(mixed(("molglyph", "CCCn1nccc1[*:1]", R), ("molscribe", "CCCn1nccc1CC", None),
                     ("molnextr", "CCCn1nccc1CC", None)))
    assert [(a["smiles"], a["agree"], a["kind"]) for a in got.alternatives] == [
        (canonical_smiles("CCCn1nccc1[*:1]"), 3, "attachment")]


@pytest.mark.parametrize(("core", "first", "second"), [
    (PYRAZOLE, "CCc1ccnn1C", "CC(C)c1ccnn1C"),  # ethyl and isopropyl
    ("Cc1ncc([*:1])n1C", "Cc1cnc(C)n1C", "Cc1ncc(C(C)C)n1C"),  # methyl and isopropyl
])
def test_a_tie_with_masked_support_keeps_no_result_and_records_the_core(core, first, second):
    given = mixed(("molglyph", core, R), ("molnextr", first, None), ("molscribe", second, None))
    assert vote_with_reason(given) == (None, "tie between 3 structures")
    record = vote_record("p2-r1", given)
    assert (record.smiles, record.agree, record.confidence, record.reason) == (None, 0, 0.0, "tie between 3 structures")
    assert record.alternatives == [{"smiles": canonical_smiles(core), "slots": R, "agree": 3,
                                    "readers": ["molglyph", "molnextr", "molscribe"], "confidence": 1.0,
                                    "kind": "attachment"}]
    assert vote_record("p2-r1", given, fixes={"attachments": False}).alternatives == []


@pytest.mark.parametrize(("core", "plain", "supported"), [
    ("[*:1]c1ccc(C(=O)N[*:2])cc1", "CC(C)c1ccc(C(=O)NCCC)cc1", True),  # two slots: H to isopropyl
    ("[*:1]c1cc([*:2])cc(C(=O)N[*:3])c1", "Cc1cc(C)cc(C(=O)NC)c1", True),  # three slots: H and methyl
    ("[*:1]c1cc([*:2])cc(C(=O)N[*:3])c1", "CCc1cc(C)cc(C(=O)NC)c1", False),
    ("[*:1]c1c([*:2])c([*:3])c([*:4])cc1C(=O)O", "OC(=O)c1ccccc1", True),
    ("[*:1]c1c([*:2])c([*:3])c([*:4])c([*:5])c1C(=O)O", "OC(=O)c1ccccc1", False),  # five slots: no fills
])
def test_masked_support_fills_the_slots_within_the_cap(core, plain, supported):
    slots = {str(n): f"R{n}" for n in range(1, 6)}
    got = vote(mixed(("molglyph", core, slots), ("molscribe", plain, None), ("molnextr", plain, None)))
    assert got.smiles == canonical_smiles(plain)
    assert bool(got.alternatives) is supported


def test_no_alternative_without_masked_support_or_when_a_core_wins():
    unsupported = mixed(("molglyph", PYRAZOLE, R), ("molscribe", "CCO", None), ("molnextr", "OCC", None))
    core_wins = mixed(("molglyph", PYRAZOLE, R), ("molscribe", PYRAZOLE, R), ("molnextr", "CCc1ccnn1C", None))
    assert vote(unsupported).alternatives == vote(core_wins).alternatives == []


# --- vote_paper -----------------------------------------------------------------------------


def region(region_id, is_structure=True):
    page = int(region_id[1 : region_id.index("-")])
    return Region(region_id, page, [10.0, 10.0, 100.0, 100.0], f"regions/{region_id}.png", 0.01, is_structure)


def paper(tmp_path, regions, reads_):
    write_jsonl(tmp_path / "regions.jsonl", regions)
    write_jsonl(tmp_path / "reads.jsonl", reads_)
    return tmp_path


def all_read(region_id, *smiles):
    return [Read(region_id, reader, text) for reader, text in zip(READERS, smiles, strict=True)]


@pytest.fixture
def one_paper(tmp_path):
    regions = [region("p1-r1"), region("p1-r2"), region("p1-r3", is_structure=False), region("p2-r1"), region("p2-r2")]
    reads_ = [
        *all_read("p1-r1", "CCO", "OCC", "CCN"),
        *all_read("p1-r2", "CCO", "CCN", "CCC"),
        *[Read("p2-r1", r, s, sl) for r, (s, sl) in zip(READERS, [(CORE, R12), (SWAPPED, R21), (CORE, R12)], strict=True)],
        *all_read("p2-r2", "C[C@H](N)C(=O)O", "C[C@H](N)C(=O)O", "C[C@H](N)C(=O)O"),
    ]
    return paper(tmp_path, regions, reads_)


def test_vote_paper_writes_one_record_per_region(one_paper):
    counts = vote_paper(one_paper)
    voted = read_jsonl(one_paper / "voted.jsonl", VotedRead)
    assert [v.region_id for v in voted] == ["p1-r1", "p1-r2", "p1-r3", "p2-r1", "p2-r2"]
    by_id = {v.region_id: v for v in voted}
    assert (by_id["p1-r1"].smiles, by_id["p1-r1"].agree, by_id["p1-r1"].reason) == ("CCO", 2, None)
    assert (by_id["p1-r2"].smiles, by_id["p1-r2"].reason) == (None, "tie between 3 structures")
    assert (by_id["p1-r3"].smiles, by_id["p1-r3"].reason) == (None, "not a structure")
    assert (by_id["p2-r1"].smiles, by_id["p2-r1"].slots) == (canonical_smiles(CORE), R12)
    assert by_id["p2-r2"].stereo_kept is True
    assert counts == {"in": 5, "out": 3, "dropped_tie": 1, "dropped_not_structure": 1, "cores": 1}


def test_vote_paper_counts_balance(one_paper):
    counts = vote_paper(one_paper)
    dropped = sum(v for k, v in counts.items() if k.startswith("dropped_"))
    assert counts["in"] == counts["out"] + dropped


def test_vote_paper_without_voting_uses_one_reader(one_paper):
    vote_paper(one_paper, fixes={"vote": False})
    by_id = {v.region_id: v for v in read_jsonl(one_paper / "voted.jsonl", VotedRead)}
    assert (by_id["p1-r1"].readers, by_id["p1-r1"].agree, by_id["p1-r1"].confidence) == (["molglyph"], 1, 1.0)
    assert by_id["p1-r2"].smiles == "CCO"  # molglyph's own reading, no tie without a vote
    assert (by_id["p2-r1"].smiles, by_id["p2-r1"].slots) == (canonical_smiles(CORE), R12)


def test_vote_paper_without_voting_takes_the_primary_reader(one_paper):
    vote_paper(one_paper, fixes={"vote": False}, primary_reader="molnextr")
    by_id = {v.region_id: v for v in read_jsonl(one_paper / "voted.jsonl", VotedRead)}
    assert (by_id["p1-r1"].readers, by_id["p1-r1"].smiles) == (["molnextr"], "CCN")


def test_vote_paper_without_voting_falls_back_when_the_primary_reader_is_absent(one_paper):
    vote_paper(one_paper, fixes={"vote": False}, primary_reader="decimer")
    by_id = {v.region_id: v for v in read_jsonl(one_paper / "voted.jsonl", VotedRead)}
    assert by_id["p1-r1"].readers == ["molglyph"]


def test_vote_paper_uses_only_the_configured_readers(one_paper):
    vote_paper(one_paper, readers=("molscribe", "molnextr"))
    by_id = {v.region_id: v for v in read_jsonl(one_paper / "voted.jsonl", VotedRead)}
    assert by_id["p1-r1"].reason == "tie between 2 structures"
    assert (by_id["p2-r2"].readers, by_id["p2-r2"].confidence) == (["molscribe", "molnextr"], 1.0)


def test_vote_paper_without_the_stereo_fix_strips_stereo(one_paper):
    vote_paper(one_paper, fixes={"stereo": False})
    by_id = {v.region_id: v for v in read_jsonl(one_paper / "voted.jsonl", VotedRead)}
    assert (by_id["p2-r2"].smiles, by_id["p2-r2"].stereo_kept) == ("CC(N)C(=O)O", False)


def test_vote_paper_records_a_structure_region_nobody_read(tmp_path):
    paper(tmp_path, [region("p1-r1")], [])
    counts = vote_paper(tmp_path)
    (voted,) = read_jsonl(tmp_path / "voted.jsonl", VotedRead)
    assert (voted.smiles, voted.reason, voted.agree, voted.readers) == (None, "no reads", 0, [])
    assert counts == {"in": 1, "out": 0, "dropped_no_reads": 1, "cores": 0}


def test_vote_paper_records_an_unusable_region(tmp_path):
    paper(tmp_path, [region("p1-r1")], [Read("p1-r1", "molglyph", None, error="NoMolecule: empty")])
    counts = vote_paper(tmp_path)
    assert read_jsonl(tmp_path / "voted.jsonl", VotedRead)[0].reason == "no usable read"
    assert counts["dropped_no_usable_read"] == 1


def test_vote_paper_counts_disagreeing_slot_labels(tmp_path):
    paper(tmp_path, [region("p1-r1")], [Read("p1-r1", "molglyph", CORE, R12), Read("p1-r1", "molscribe", CORE, R21)])
    assert vote_paper(tmp_path, fixes={"slot_labels": False})["dropped_slot_labels_disagree"] == 1


def test_vote_paper_writes_the_alternatives_and_reads_them_back(tmp_path):
    paper(tmp_path, [region("p1-r1"), region("p1-r2")], [
        *mixed(("molglyph", PYRAZOLE, R), ("molnextr", "CCc1ccnn1C", None), ("molscribe", "CC(C)c1ccnn1C", None),
               region_id="p1-r1"),
        Read("p1-r2", "molglyph", CORE, R12), Read("p1-r2", "molscribe", CORE, R21),
    ])
    counts = vote_paper(tmp_path)
    tie, labels = read_jsonl(tmp_path / "voted.jsonl", VotedRead)
    assert (tie.smiles, tie.reason, [a["kind"] for a in tie.alternatives]) == (None, "tie between 3 structures",
                                                                               ["attachment"])
    assert (labels.smiles, labels.slots, labels.skeleton_agree) == (canonical_smiles(CORE), R12, 2)
    assert [(a["kind"], a["slots"]) for a in labels.alternatives] == [("slot_labels", R21)]
    assert counts == {"in": 2, "out": 1, "cores": 1, "dropped_tie": 1}
    vote_paper(tmp_path, fixes={"attachments": False, "slot_labels": False})
    off = read_jsonl(tmp_path / "voted.jsonl", VotedRead)
    assert [(v.alternatives, v.skeleton_agree, v.reason) for v in off] == [
        ([], 0, "tie between 3 structures"), ([], 0, "slot labels disagree")]


def test_vote_paper_without_structures_needs_no_reads(tmp_path):
    write_jsonl(tmp_path / "regions.jsonl", [region("p1-r1", is_structure=False)])
    counts = vote_paper(tmp_path)
    assert counts == {"in": 1, "out": 0, "dropped_not_structure": 1, "cores": 0}
    assert len(read_jsonl(tmp_path / "voted.jsonl", VotedRead)) == 1


def test_vote_paper_needs_reads_when_a_region_is_a_structure(tmp_path):
    write_jsonl(tmp_path / "regions.jsonl", [region("p1-r1")])
    with pytest.raises(FileNotFoundError, match="reads.jsonl"):
        vote_paper(tmp_path)


def test_vote_paper_needs_regions(tmp_path):
    with pytest.raises(FileNotFoundError, match="regions.jsonl"):
        vote_paper(tmp_path)


def test_vote_paper_refuses_reads_of_unknown_regions(tmp_path):
    paper(tmp_path, [region("p1-r1")], [Read("p1-r1", "molglyph", "CCO"), Read("p9-r9", "molglyph", "CCO")])
    with pytest.raises(ValueError, match="p9-r9"):
        vote_paper(tmp_path)


def test_vote_paper_reads_what_the_reader_workers_write(tmp_path):
    write_jsonl(tmp_path / "regions.jsonl", [region("p1-r1")])
    line = {"region_id": "p1-r1", "reader": "molscribe", "smiles": CORE, "slots": R12, "error": None}
    (tmp_path / "reads.jsonl").write_text(json.dumps(line) + "\n", encoding="utf-8")
    vote_paper(tmp_path)
    record = json.loads((tmp_path / "voted.jsonl").read_text(encoding="utf-8"))
    assert record["slots"] == R12 and record["reason"] is None
