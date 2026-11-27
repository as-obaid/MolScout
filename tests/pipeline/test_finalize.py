"""finalize: junk, printed formulas, relevance, dedupe and counts, once per paper."""

from __future__ import annotations

import pytest

from molscout.pipeline.finalize import activity_names, finalize
from molscout.pipeline.schema import LayoutBlock, Molecule, write_jsonl
from molscout.scoring.smiles import canonical_smiles

PAPER = "26_test"
ANISANILIDE = "COc1ccc(C(=O)Nc2ccccc2)cc1"  # compound 1, C14H13NO2
BENZANILIDE = "O=C(Nc1ccccc1)c1ccccc1"  # compound 4, C13H11NO
TOLUANILIDE = "Cc1ccc(C(=O)Nc2ccccc2)cc1"  # compound 2


def block(page, index, text, kind="text"):
    return LayoutBlock(page=page, index=index, type=kind, bbox=[72.0, 90.0, 540.0, 130.0], text=text)


ACTIVITY = block(1, 5, "Compound 1 inhibited the kinase with an IC50 of 12 nM, while 2 was inactive.")
FORMULAS = [
    block(3, 0, "Experimental section", "title"),
    block(3, 1, "N-Phenylbenzamide (4). HRMS (ESI) calcd for C13H12NO [M+H]+ 198.0913, found 198.0911."),
    block(3, 2, "4-Methoxy-N-phenylbenzamide (1). HRMS (ESI) calcd for C14H14NO2 [M+H]+ 228.1025, found 228.1022."),
]


def paper_dir(tmp_path, blocks=(ACTIVITY, *FORMULAS)):
    folder = tmp_path / PAPER
    write_jsonl(folder / "layout.jsonl", list(blocks))
    return folder


def mol(label, smiles, source="drawn", status="accepted", **fields):
    defaults = {"page": 1, "bbox": [1.0, 2.0, 3.0, 4.0], "confidence": 0.8}
    return Molecule(PAPER, label, smiles, source=source, status=status, **{**defaults, **fields})


def run(tmp_path, molecules, fixes=None, blocks=(ACTIVITY, *FORMULAS)):
    return finalize(paper_dir(tmp_path, blocks), molecules, fixes=fixes)


def by_label(molecules):
    return {m.label: m for m in molecules}


def balanced(counts):
    dropped = sum(v for k, v in counts.items() if k == "dropped" or k.startswith("dropped_"))
    return counts["in"] == counts["out"] + dropped


# --- relevance ------------------------------------------------------------------------------


DIANISANILIDE_CL = "COc1ccc(C(=O)Nc2ccc(OC)c(Cl)c2)cc1"  # 20 heavy atoms
DIANISANILIDE = "COc1ccc(C(=O)Nc2ccc(OC)cc2)cc1"  # 19 heavy atoms


def test_labelled_molecules_stay_accepted_and_small_unlabelled_ones_go_to_review(tmp_path):
    out, counts = run(tmp_path, [mol("1", ANISANILIDE), mol("2", TOLUANILIDE), mol(None, "CC(=O)Nc1ccc(O)cc1")])
    got = by_label(out)
    assert (got["1"].status, got["1"].reasons[-1]) == ("accepted", "label 1 is in an activity table or sentence")
    assert (got["2"].status, got["2"].reasons) == ("accepted", [])
    assert (got[None].status, got[None].reasons[-1]) == ("review", "no compound label")
    assert counts["not_relevant"] == 1


@pytest.mark.parametrize(("smiles", "status"), [(DIANISANILIDE_CL, "accepted"), (DIANISANILIDE, "review")])
def test_an_unlabelled_structure_stays_accepted_from_twenty_heavy_atoms(tmp_path, smiles, status):
    out, counts = run(tmp_path, [mol(None, smiles)])
    assert out[0].status == status
    if status == "accepted":
        assert out[0].reasons == ["no compound label, but drug-sized (20 heavy atoms)"]
        assert counts["unlabelled_kept"] == 1


@pytest.mark.parametrize("source", ["drawn", "table", "name", "32b"])
def test_a_small_unlabelled_molecule_goes_to_review_from_every_source(tmp_path, source):
    out, _ = run(tmp_path, [mol(None, TOLUANILIDE, source=source)])
    assert out[0].status == "review"


@pytest.mark.parametrize(("source", "status"), [("drawn", "accepted"), ("table", "accepted"), ("32b", "accepted"),
                                                ("name", "review")])
def test_only_a_named_molecule_needs_its_label_in_an_activity_table_or_sentence(tmp_path, source, status):
    out, _ = run(tmp_path, [mol("2", TOLUANILIDE, source=source)])
    assert out[0].status == status
    if status == "review":
        assert out[0].reasons[-1] == "label 2 is not in an activity table or sentence"


def test_a_drug_sized_name_without_a_label_goes_to_review(tmp_path):
    out, _ = run(tmp_path, [mol(None, DIANISANILIDE_CL, source="name")])
    assert (out[0].status, out[0].reasons[-1]) == ("review", "no compound label")


def test_a_named_molecule_needs_no_tested_label_when_the_paper_names_none(tmp_path):
    out, _ = run(tmp_path, [mol("2", TOLUANILIDE, source="name")], blocks=[block(1, 0, "The control had an IC50 of 5 nM.")])
    assert (out[0].status, out[0].reasons) == ("accepted", [])


def test_32b_label_aliases_count(tmp_path):
    out, _ = run(tmp_path, [mol("2;1", ANISANILIDE, source="32b")])
    assert (out[0].status, out[0].reasons[-1]) == ("accepted", "label 2;1 is in an activity table or sentence")


@pytest.mark.parametrize("label", ["compound 1", "Compd. 1", "(1)", " 1 "])
def test_label_spellings_match_the_tested_label(tmp_path, label):
    out, _ = run(tmp_path, [mol(label, ANISANILIDE, source="32b")])
    assert out[0].status == "accepted"
    assert "printed formula C14H13NO2 matches" in out[0].reasons
    assert f"label {label} is in an activity table or sentence" in out[0].reasons


def test_a_32b_label_spelling_merges_with_the_rules_label(tmp_path):
    out, counts = run(tmp_path, [mol("1", ANISANILIDE), mol("compound 1", ANISANILIDE, source="32b")])
    assert len(out) == 1 and counts["dropped_duplicate"] == 1


@pytest.mark.parametrize("label", ["compound", "()", ";"])
def test_a_label_of_only_generic_words_is_no_label(tmp_path, label):
    out, _ = run(tmp_path, [mol(label, ANISANILIDE, source="32b")])
    assert out[0].reasons[-1] == "no compound label"


def test_review_and_abstained_molecules_are_not_promoted(tmp_path):
    out, _ = run(tmp_path, [mol("1", ANISANILIDE, status="review"), mol("1", None, source="table", status="abstained")])
    assert [m.status for m in out] == ["review", "abstained"]


def test_paper_without_activity_needs_labels_only(tmp_path):
    out, _ = run(tmp_path, [mol("2", TOLUANILIDE), mol(None, ANISANILIDE)], blocks=[block(1, 0, "We made amides.")])
    got = by_label(out)
    assert (got["2"].status, got["2"].reasons) == ("accepted", [])
    assert got[None].status == "review"


def test_relevance_off_accepts_unlabelled_and_names_no_tested_label(tmp_path):
    out, counts = run(tmp_path, [mol("1", ANISANILIDE), mol(None, TOLUANILIDE)], fixes={"relevance": False})
    assert {m.status for m in out} == {"accepted"}
    assert not any("activity table" in reason for m in out for reason in m.reasons)
    assert "not_relevant" not in counts


def test_activity_table_rows_are_named_as_tested(tmp_path):
    table = LayoutBlock(2, 2, "table", [72.0, 215.0, 540.0, 400.0], "",
                        html="<table><tr><th>Compd</th><th>R1</th><th>IC50 (nM)</th></tr>"
                             "<tr><td>3a</td><td>H</td><td>14</td></tr><tr><td>3b</td><td>F</td><td>NT</td></tr></table>")
    out, _ = run(tmp_path, [mol("3a", BENZANILIDE, source="table"), mol("3b", TOLUANILIDE, source="table")],
                 blocks=[table])
    assert [m.status for m in out] == ["accepted", "accepted"]
    assert [m.reasons for m in out] == [["label 3a is in an activity table or sentence"], []]


def test_activity_names_are_the_labels_the_paper_reports_activity_for():
    assert activity_names([ACTIVITY]) == {"1"}


# --- junk -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smiles", "reason"),
    [("CCO", "junk: too few heavy atoms"), ("[*:1]c1ccccc1", "junk: leftover dummy"), ("C1CC", "junk: invalid"),
     ("c1ccccc1.c1ccncc1", "junk: multiple fragments")],
)
def test_junk_is_abstained_without_smiles(tmp_path, smiles, reason):
    out, counts = run(tmp_path, [mol("1", smiles)])
    assert (out[0].status, out[0].smiles, out[0].reasons[-1]) == ("abstained", None, reason)
    assert counts["junk"] == 1


def test_salts_are_stripped_to_the_parent(tmp_path):
    out, counts = run(tmp_path, [mol("1", ANISANILIDE + ".Cl")])
    assert (out[0].smiles, out[0].status) == (canonical_smiles(ANISANILIDE), "accepted")
    assert "salt or solvent removed" in out[0].reasons
    assert counts["salt_stripped"] == 1


def test_junk_off_keeps_junk(tmp_path):
    molecules = [mol("1", "CCO"), mol("2", "[*:1]c1ccccc1", status="review")]
    out, counts = run(tmp_path, molecules, fixes={"junk": False, "formulas": False})
    assert [(m.smiles, m.status) for m in out] == [("CCO", "accepted"), ("[*:1]c1ccccc1", "review")]
    assert "junk" not in counts


# --- printed formulas -----------------------------------------------------------------------


def test_matching_formula_supports_the_molecule(tmp_path):
    out, counts = run(tmp_path, [mol("1", ANISANILIDE)])
    assert out[0].status == "accepted"
    assert "printed formula C14H13NO2 matches" in out[0].reasons
    assert out[0].confidence == pytest.approx(0.9)
    assert counts["formula_match"] == 1


def test_mismatching_formula_sends_the_molecule_to_review(tmp_path):
    out, counts = run(tmp_path, [mol("1", "COc1ccc(C(=O)Nc2ccc(Cl)cc2)cc1")])
    assert out[0].status == "review"
    assert "printed formula C14H13NO2 does not match" in out[0].reasons
    assert counts["formula_mismatch"] == 1


def test_formulas_off_ignores_printed_formulas(tmp_path):
    out, counts = run(tmp_path, [mol("1", "COc1ccc(C(=O)Nc2ccc(Cl)cc2)cc1")], fixes={"formulas": False})
    assert out[0].status == "accepted" and "formula_mismatch" not in counts


def test_formula_support_does_not_lift_an_untested_named_label(tmp_path):
    out, _ = run(tmp_path, [mol("4", BENZANILIDE, source="name")])
    assert out[0].status == "review"
    assert "printed formula C13H11NO matches" in out[0].reasons


# --- dedupe ---------------------------------------------------------------------------------


def test_same_label_and_structure_from_two_sources_merge(tmp_path):
    drawn = mol("1", ANISANILIDE, confidence=0.7)
    named = mol("1", ANISANILIDE, source="name", confidence=0.9, page=3)
    out, counts = run(tmp_path, [drawn, named])
    (merged,) = out
    assert (merged.source, merged.page, merged.confidence) == ("drawn", 1, pytest.approx(1.0))
    assert "duplicate merged: name (accepted)" in merged.reasons
    assert counts["dropped_duplicate"] == 1 and balanced(counts)


def test_dedupe_keeps_the_highest_status(tmp_path):
    review = mol("1", ANISANILIDE, source="table", status="review", table_id="p2-t2", row=0)
    accepted = mol("1", ANISANILIDE, source="name")
    (merged,) = run(tmp_path, [review, accepted])[0]
    assert (merged.source, merged.status) == ("name", "accepted")
    assert "duplicate merged: table p2-t2 row 0 (review)" in merged.reasons


def test_stereo_variants_of_one_label_merge(tmp_path):
    out, _ = run(tmp_path, [mol("1", "C[C@H](N)c1ccc(OC)cc1"), mol("1", "CC(N)c1ccc(OC)cc1", source="name")],
                 fixes={"formulas": False})
    assert len(out) == 1


def test_different_labels_or_structures_stay_apart(tmp_path):
    out, counts = run(tmp_path, [mol("1", ANISANILIDE), mol("2", ANISANILIDE), mol("1", TOLUANILIDE, source="name")])
    assert len(out) == 3 and counts["dropped_duplicate"] == 0


def test_unlabelled_duplicates_merge(tmp_path):
    out, counts = run(tmp_path, [mol(None, ANISANILIDE), mol(None, ANISANILIDE, page=2)])
    assert len(out) == 1 and counts["dropped_duplicate"] == 1


def test_abstained_records_without_smiles_never_merge(tmp_path):
    rows = [mol("5a", None, source="table", status="abstained", row=i) for i in range(2)]
    out, counts = run(tmp_path, rows)
    assert len(out) == 2 and counts["dropped_duplicate"] == 0


# --- records and counts ---------------------------------------------------------------------


def test_abstained_smiles_becomes_none(tmp_path):
    out, _ = run(tmp_path, [mol("1", "", source="table", status="abstained"),
                            mol("x", "[*:1]c1ccccc1", status="abstained")], fixes={"junk": False})
    assert [m.smiles for m in out] == [None, None]


def test_a_built_molecule_without_smiles_is_abstained(tmp_path):
    out, counts = run(tmp_path, [mol("1", None)], fixes={"junk": False})
    assert (out[0].status, out[0].reasons[-1]) == ("abstained", "no SMILES")
    assert counts["no_smiles"] == 1


def test_counts_add_up_and_report_statuses(tmp_path):
    molecules = [mol("1", ANISANILIDE), mol("1", ANISANILIDE, source="name"), mol("2", TOLUANILIDE),
                 mol("5", "CCO"), mol(None, None, status="abstained")]
    out, counts = run(tmp_path, molecules)
    assert balanced(counts)
    assert (counts["in"], counts["out"], counts["dropped_duplicate"]) == (5, 4, 1)
    assert (counts["accepted"], counts["review"], counts["abstained"]) == (2, 0, 2)
    assert len(out) == counts["out"]


def test_inputs_are_not_changed(tmp_path):
    molecules = [mol("2", TOLUANILIDE), mol("5", "CCO")]
    before = list(molecules)
    run(tmp_path, molecules)
    assert molecules == before and molecules[0].status == "accepted"


def test_empty_paper(tmp_path):
    assert run(tmp_path, []) == ([], {"in": 0, "out": 0, "dropped_duplicate": 0, "accepted": 0, "review": 0,
                                      "abstained": 0})


def test_molecules_of_another_paper_are_refused(tmp_path):
    other = Molecule("other", "1", ANISANILIDE, 1, None, "drawn", status="accepted")
    with pytest.raises(ValueError, match="other"):
        run(tmp_path, [other])


def test_missing_layout_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="layout.jsonl"):
        finalize(tmp_path / PAPER, [])
