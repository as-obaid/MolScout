"""merge(): the `both` joiner mode, rules first, the 32B model's molecules checking and filling them."""

import copy

import pytest

from molscout.pipeline.merge import MergePolicy, counts_for, junk_free, merge
from molscout.pipeline.schema import Molecule

ASPIRIN, ASPIRIN_OTHER_WRITING = "CC(=O)Oc1ccccc1C(=O)O", "OC(=O)c1ccccc1OC(C)=O"
NAPHTHALENE, CHLOROBENZENE, ANISOLE, TOLUENE, PHENOL = (
    "c1ccc2ccccc2c1", "Clc1ccccc1", "COc1ccccc1", "Cc1ccccc1", "Oc1ccccc1")
L_ALANINE, D_ALANINE, ALANINE = "C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O", "CC(N)C(=O)O"
ALWAYS, NEVER = (lambda m: True), (lambda m: False)


def mol(smiles, label=None, source="drawn", **kw):
    values = {"paper_id": "p", "label": label, "smiles": smiles, "page": 1, "bbox": [0.0, 0.0, 10.0, 10.0],
              "source": source, "table_id": None, "row": None, "status": "accepted", "reasons": [], "confidence": 0.5}
    values.update(kw)
    return Molecule(**values)


def m32(smiles, label=None, **kw):
    return mol(smiles, label, source="32b", **kw)


def row(smiles, label, table_id, number, source="table", **kw):
    return mol(smiles, label, source=source, table_id=table_id, row=number, **kw)


def summary(molecules):
    return [(m.label, m.source, m.status, m.table_id, m.row) for m in molecules]


def merged(rules, found, policy=None, **kw):
    """The merged molecules, after checking that the counts account for every input."""
    molecules, counts = merge(rules, found, policy or MergePolicy(fill_check=ALWAYS), **kw)
    dropped = sum(n for key, n in counts.items() if key.startswith("dropped_"))
    assert counts["in"] == len(rules) + len(found) and counts["out"] == len(molecules)
    assert counts["in"] == counts["out"] + dropped
    return molecules


# ---------------------------------------------------------------- policy and result

def test_filling_tables_needs_a_fill_check_and_the_default_is_the_junk_check():
    with pytest.raises(ValueError, match="fill_check"):
        MergePolicy(fill_check=None)
    assert MergePolicy(fill_tables=False, fill_check=None).fill_check is None
    assert MergePolicy().fill_check is junk_free
    assert junk_free(mol(ANISOLE)) and not junk_free(mol("CC")) and not junk_free(mol(None))


def test_the_default_policy_checks_fills_for_junk():
    rules = [row(None, "3b", "T1", 0, status="abstained"), row(None, "3c", "T1", 1, status="abstained")]
    molecules, _ = merge(rules, [m32(ANISOLE, "3b"), m32("CC", "3c")])
    assert [(m.label, m.status) for m in molecules] == [("3b", "accepted"), ("3c", "review")]
    assert molecules[1].reasons[-1] == "fill check failed"


def test_counts_that_do_not_add_up_are_refused():
    with pytest.raises(AssertionError, match="lost track"):
        counts_for(2, [], {"dropped_32b_abstained": 1})
    with pytest.raises(AssertionError, match="lost track"):
        counts_for(1, [], {"dropped_for_no_reason": 1})
    assert counts_for(2, [], {"dropped_32b_abstained": 2, "dropped_32b_duplicate": 0}) == {
        "in": 2, "out": 0, "dropped_32b_abstained": 2}


def test_counts_name_the_reason_for_every_dropped_record():
    rules = [mol(ASPIRIN, "1"), row(None, "3b", "T1", 1, status="abstained"), mol(PHENOL, "5")]
    found = [m32(ASPIRIN, "1"), m32(ANISOLE, "3b", table_id="T1", row=1), m32(None, "9", status="abstained"),
             m32("C1CC", "8"), m32(NAPHTHALENE, "5"), m32(TOLUENE, "6"), m32(TOLUENE, "6b")]
    molecules, counts = merge(rules, found, MergePolicy(fill_check=ALWAYS))
    assert counts == {"in": 10, "out": 4, "dropped_32b_agrees": 1, "dropped_rules_row_filled": 1,
                      "dropped_32b_abstained": 1, "dropped_32b_unreadable": 1, "dropped_32b_conflict": 1,
                      "dropped_32b_duplicate": 1}
    assert summary(molecules) == [("1", "drawn", "accepted", None, None), ("3b", "32b", "accepted", "T1", 1),
                                         ("5", "drawn", "accepted", None, None), ("6", "32b", "review", None, None)]


# ---------------------------------------------------------------- agreement

def test_agreement_by_inchikey_raises_confidence_and_keeps_one_copy():
    (out,) = merged([mol(ASPIRIN, "1", confidence=0.5)], [m32(ASPIRIN_OTHER_WRITING, "1")],
                    MergePolicy(agree_boost=0.2, fill_check=ALWAYS))
    assert out.source == "drawn" and out.smiles == ASPIRIN
    assert out.confidence == pytest.approx(0.7)
    assert "32b agrees" in out.reasons


def test_agreement_caps_confidence_at_one():
    (out,) = merged([mol(ASPIRIN, "1", confidence=0.95)], [m32(ASPIRIN, "1")])
    assert out.confidence == 1.0


@pytest.mark.parametrize("rules_smiles, smiles_32b", [(L_ALANINE, D_ALANINE), (ALANINE, L_ALANINE)])
def test_a_stereo_only_difference_is_agreement_with_a_reason_not_a_conflict(rules_smiles, smiles_32b):
    (out,) = merged([mol(rules_smiles, "4", confidence=0.5)], [m32(smiles_32b, "4")])
    assert out.smiles == rules_smiles and out.confidence == pytest.approx(0.7)
    assert out.reasons == ["32b agrees", "32b differs in stereo only"]


def test_the_32b_molecule_with_the_same_stereo_speaks_for_the_structure():
    (out,) = merged([mol(L_ALANINE, None)], [m32(D_ALANINE, "4d"), m32(L_ALANINE, "4l")])
    assert out.label == "4l" and out.reasons == ["32b agrees", "label from 32b"]


def test_agreement_gives_an_unlabelled_drawing_32bs_label():
    rules = [mol(ASPIRIN, None, status="review", reasons=["no label"])]
    (out,) = merged(rules, [m32(ASPIRIN, "1")])
    assert (out.label, out.status) == ("1", "review")
    assert out.reasons == ["no label", "32b agrees", "label from 32b"]
    (promoted,) = merged(rules, [m32(ASPIRIN, "1")], MergePolicy(promote_agreed=True, fill_check=ALWAYS))
    assert promoted.status == "accepted"


def test_a_review_32b_molecule_does_not_promote():
    rules = [mol(ASPIRIN, None, status="review")]
    (out,) = merged(rules, [m32(ASPIRIN, None, status="review")], MergePolicy(promote_agreed=True, fill_check=ALWAYS))
    assert out.status == "review" and out.label is None


# ---------------------------------------------------------------- 32b-only molecules

def test_32b_only_molecule_joins_as_review_or_is_dropped():
    rules, extra = [mol(ASPIRIN, "1")], [m32(NAPHTHALENE, "2", reasons=["32b labelled drawing p1-r2"])]
    out = merged(rules, extra)
    assert summary(out) == [("1", "drawn", "accepted", None, None), ("2", "32b", "review", None, None)]
    assert out[1].reasons == ["32b labelled drawing p1-r2", "32b only"]
    molecules, counts = merge(rules, extra, MergePolicy(add_unmatched=False, fill_check=ALWAYS))
    assert len(molecules) == 1 and counts == {"in": 2, "out": 1, "dropped_32b_unmatched": 1}


def test_32b_duplicates_are_added_once():
    assert [m.label for m in merged([], [m32(NAPHTHALENE, "2"), m32(NAPHTHALENE, "2b")])] == ["2"]


def test_32b_stereoisomers_are_not_duplicates():
    assert [m.label for m in merged([], [m32(L_ALANINE, "4l"), m32(D_ALANINE, "4d")])] == ["4l", "4d"]


def test_label_conflict_keeps_the_rules_molecule_and_notes_it():
    (out,) = merged([mol(ASPIRIN, "5")], [m32(NAPHTHALENE, "5")])
    assert out.smiles == ASPIRIN and "32b reads a different structure for this label" in out.reasons


def test_abstained_32b_rows_are_never_added():
    out = merged([mol(ASPIRIN, "1")], [m32(None, "9", status="abstained", table_id="32b-p1-part-0", row=0)])
    assert summary(out) == [("1", "drawn", "accepted", None, None)]


def test_merge_does_not_filter_for_relevance():
    rules = [mol(NAPHTHALENE, "7"), mol(TOLUENE, None, status="review")]
    out = merged(rules, [m32(ASPIRIN, "1")], MergePolicy(add_unmatched=False, fill_check=ALWAYS))
    assert [(m.label, m.status, m.reasons) for m in out] == [("7", "accepted", []), (None, "review", [])]
    assert not hasattr(MergePolicy(), "relevance")


# ---------------------------------------------------------------- tables

def test_rules_row_that_did_not_build_is_filled_by_the_32b_molecule_with_its_label():
    rules = [row(CHLOROBENZENE, "3a", "p1-t5", 0), row(None, "3b", "p1-t5", 1, status="abstained", page=2,
                                                         bbox=[1.0, 2.0, 3.0, 4.0], reasons=["R1 unknown"])]
    found = [m32(CHLOROBENZENE, "3a"), m32(ANISOLE, "compound 3b", table_id="32b-p1-part-0", row=4)]
    out = merged(rules, found)
    assert summary(out) == [("3a", "table", "accepted", "p1-t5", 0), ("compound 3b", "32b", "accepted", "p1-t5", 1)]
    assert (out[1].smiles, out[1].page, out[1].bbox) == (ANISOLE, 2, [1.0, 2.0, 3.0, 4.0])
    assert out[1].reasons[-1] == "fills a rules row that did not build"


@pytest.mark.parametrize("check, status_32b, status", [
    (ALWAYS, "accepted", "accepted"), (NEVER, "accepted", "review"), (ALWAYS, "review", "review"),
])
def test_a_fill_is_accepted_only_when_the_check_passes_and_32b_accepted_it(check, status_32b, status):
    rules = [row(None, "3b", "T1", 0, status="abstained")]
    (out,) = merged(rules, [m32(ANISOLE, "3b", status=status_32b)], MergePolicy(fill_check=check))
    assert out.status == status
    assert ("fill check failed" in out.reasons) == (check is NEVER)


def test_a_row_fill_prefers_the_32b_row_of_the_same_table():
    rules = [row(None, "3b", "p2-t1", 0, status="abstained")]
    found = [m32(ANISOLE, "3b", table_id="p5-t0", row=0), m32(TOLUENE, "3b", table_id="p2-t1", row=0)]
    assert merged(rules, found, MergePolicy(add_unmatched=False, fill_check=ALWAYS))[0].smiles == TOLUENE


def table_32b(table_id, *rows_, page=1):
    return [m32(smiles, label, table_id=table_id, row=n, page=page) for n, (smiles, label) in enumerate(rows_)]


def test_short_table_is_filled_up_to_its_row_count_with_32b_rows_of_that_table():
    rules = [row(CHLOROBENZENE, "3a", "p1-t2", 0)]
    found = table_32b("p1-t2", (CHLOROBENZENE, "3a"), (ANISOLE, "3b"), (TOLUENE, "3c"), (PHENOL, "3d"))
    out = merged(rules, found, table_rows={"p1-t2": 3})
    assert summary(out) == [("3a", "table", "accepted", "p1-t2", 0), ("3b", "32b", "accepted", "p1-t2", 1),
                            ("3c", "32b", "accepted", "p1-t2", 2), ("3d", "32b", "review", "p1-t2", 3)]
    assert out[1].reasons[-1] == "fills a short rules table" and out[3].reasons[-1] == "32b only"
    checked = merged(rules, found, MergePolicy(fill_check=NEVER), table_rows={("p", "p1-t2"): 3})
    assert [m.status for m in checked] == ["accepted", "review", "review", "review"]


def test_a_core_on_the_page_before_its_table_still_fills_it():
    rules = [row(CHLOROBENZENE, "3a", "p4-t1", 0, page=4)]
    found = table_32b("p4-t1", (ANISOLE, "3b"), page=3)  # join placed the row although its scaffold is on page 3
    assert summary(merged(rules, found, table_rows={"p4-t1": 2}))[1] == ("3b", "32b", "accepted", "p4-t1", 0)


def test_rows_join_could_not_place_in_a_table_never_fill_a_short_one():
    rules = [row(CHLOROBENZENE, "3a", "p1-t2", 0)]
    found = table_32b("32b-p1-part-0", (ANISOLE, "3b"))
    out = merged(rules, found, table_rows={"p1-t2": 2})
    assert summary(out)[1] == ("3b", "32b", "review", "32b-p1-part-0", 0) and out[1].reasons[-1] == "32b only"


def test_table_size_defaults_to_the_rows_rules_saw():
    rules = [row(CHLOROBENZENE, "3a", "T1", 0), row(None, "3b", "T1", 1, status="abstained")]
    found = table_32b("T1", (TOLUENE, "3x"))  # a label rules never saw
    assert summary(merged(rules, found))[2] == ("3x", "32b", "accepted", "T1", 0)


def test_full_table_takes_no_fills():
    rules = [row(CHLOROBENZENE, "3a", "T1", 0), row(ANISOLE, "3b", "T1", 1)]
    out = merged(rules, table_32b("T1", (TOLUENE, "3c")), table_rows={"T1": 2})
    assert summary(out)[2] == ("3c", "32b", "review", "T1", 0) and out[2].reasons[-1] == "32b only"


def test_a_second_32b_claim_on_a_filled_label_is_a_conflict():
    rules = [row(None, "3b", "T1", 0, status="abstained")]
    molecules, counts = merge(rules, [m32(ANISOLE, "3b"), m32(TOLUENE, "3b")], MergePolicy(fill_check=ALWAYS))
    assert [m.smiles for m in molecules] == [ANISOLE] and counts["dropped_32b_conflict"] == 1


def test_fill_tables_off_leaves_tables_alone():
    rules = [row(CHLOROBENZENE, "3a", "T1", 0), row(None, "3b", "T1", 1, status="abstained")]
    out = merged(rules, table_32b("T1", (ANISOLE, "3b")), MergePolicy(fill_tables=False, add_unmatched=False))
    assert MergePolicy(fill_tables=False, fill_check=None).fill_tables is False
    assert summary(out) == [("3a", "table", "accepted", "T1", 0), ("3b", "table", "abstained", "T1", 1)]


@pytest.mark.parametrize("rules_label, label_32b", [
    ("5a", "compound 5a"), ("1", "Gefitinib;ZD1839;1"), ("12b", "12b."), ("Cpd 4", "(4)"), ("7-a", "7–a"),
])
def test_labels_match_through_aliases_and_generic_words(rules_label, label_32b):
    (out,) = merged([mol(NAPHTHALENE, rules_label)], [m32(ASPIRIN, label_32b)])
    assert out.status == "accepted" and "32b reads a different structure for this label" in out.reasons


# ---------------------------------------------------------------- hygiene

def test_inputs_are_not_modified():
    rules = [mol(ASPIRIN, None, status="review"), row(None, "3b", "T1", 1, status="abstained")]
    found = [m32(ASPIRIN, "1"), *table_32b("T1", (ANISOLE, "3b"), (TOLUENE, "3c"))]
    before = copy.deepcopy((rules, found))
    merged(rules, found, MergePolicy(promote_agreed=True, fill_check=ALWAYS), table_rows={"T1": 3})
    assert (rules, found) == before


def test_papers_are_merged_separately():
    out = merged([mol(ASPIRIN, "1", paper_id="a")], [m32(ASPIRIN, "1", paper_id="b")])
    assert [(m.paper_id, m.source, m.status) for m in out] == [("a", "drawn", "accepted"), ("b", "32b", "review")]
    assert "32b agrees" not in out[0].reasons


def test_rules_molecule_without_a_valid_smiles_passes_through():
    (out, extra) = merged([mol("not a smiles", "1")], [m32(ASPIRIN, "2")])
    assert out.smiles == "not a smiles" and out.reasons == [] and extra.label == "2"


def test_tuple_reasons_stay_tuples():
    (out,) = merged([mol(ASPIRIN, "1", reasons=("drawn",))], [m32(ASPIRIN, "1")])
    assert out.reasons == ("drawn", "32b agrees")
