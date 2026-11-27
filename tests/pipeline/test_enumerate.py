import pytest

from molscout.pipeline.enumerate import CoreCheck, align_slots, build, check_core, label_key
from molscout.pipeline.rgroups import Fragment, to_fragment
from molscout.scoring.smiles import canonical_smiles


def frag(text):
    fragment = to_fragment(text, None)
    assert fragment is not None, text
    return fragment


# --- check_core -----------------------------------------------------------------------------


def test_core_matching_the_table():
    assert check_core("[*:1]c1ccc([*:2])cc1", ["R1", "R2"]) == CoreCheck(True, {1, 2}, set(), set())


def test_core_missing_a_table_variable():
    assert check_core("[*:1]c1ccccc1", ["R1", "R2"]) == CoreCheck(False, {1}, {2}, set())


def test_core_with_a_slot_the_table_lacks():
    assert check_core("[*:1]c1ccc([*:2])cc1[*:3]", ["R1", "R2"]) == CoreCheck(False, {1, 2, 3}, set(), {3})


def test_unlabelled_dummy_is_an_extra_slot_zero():
    assert check_core("*c1ccc([*:1])cc1", ["R1"]) == CoreCheck(False, {0, 1}, set(), {0})


def test_unparseable_core_misses_every_variable():
    assert check_core("c1ccc(", ["R1", "R2"]) == CoreCheck(False, set(), {1, 2}, set())


def test_core_without_variables():
    assert check_core("c1ccccc1", []) == CoreCheck(True, set(), set(), set())


# --- check_core with the reader's slot labels ------------------------------------------------


def test_check_core_with_slots_counts_a_repeated_label_once():
    core = "[*:1]c1cc([*:2])cc(C(=O)N[*:3])c1"
    got = check_core(core, ["R1", "R2"], {"1": "R1", "2": "R1", "3": "R2"})
    assert got == CoreCheck(True, {1, 2}, set(), set())


def test_check_core_with_slots_matches_labels_not_numbers():
    assert check_core("[*:1]c1ccc([*:3])cc1", ["R1", "R2"], {"1": "R1", "3": "R2"}).ok
    got = check_core("[*:1]c1ccc([*:2])cc1", ["R1", "R2"], {"1": "R1", "2": "R3"})
    assert got == CoreCheck(False, {1, 3}, {2}, {3})


def test_check_core_with_slots_counts_an_unknown_repeated_label_once():
    got = check_core("[*:1]c1cc([*:2])cc([*:3])c1", ["R1"], {"1": "R1", "2": "X", "3": "X"})
    assert got == CoreCheck(False, {1, 2}, set(), {2})


def test_label_key_ignores_markup_case_and_primes():
    assert label_key("R<sup>1</sup>") == label_key("R¹") == label_key(" r1 ") == "r1"
    assert label_key("R′") == label_key("R'")


# --- align_slots ----------------------------------------------------------------------------


def test_align_slots_numbers_dummies_by_table_column_order():
    core = align_slots("[*:7]C(=O)c1ccc([*:3])cc1", {"7": "R2", "3": "R1"}, ["R1", "R2"])
    assert canonical_smiles(core) == canonical_smiles("[*:2]C(=O)c1ccc([*:1])cc1")


@pytest.mark.parametrize(("drawn", "header"), [("R¹", "R1"), ("R 1", "R<sup>1</sup>"), ("R′", "R'"), ("r1", "R1")])
def test_align_slots_matches_label_spellings(drawn, header):
    core = align_slots("[*:5]c1ccccc1", {"5": drawn}, [header])
    assert check_core(core, [header]).ok


def test_align_slots_gives_unknown_labels_extra_numbers():
    core = align_slots("[*:1]c1ccc([*:2])cc1", {"1": "R1", "2": "Ar"}, ["R1"])
    assert check_core(core, ["R1"]) == CoreCheck(False, {1, 2}, set(), {2})


def test_align_slots_rejects_unparseable_core():
    assert align_slots("c1ccc(", {}, ["R1"]) is None


def test_align_slots_gives_a_repeated_label_one_number():
    core = align_slots("[*:1]c1cc([*:2])cc(C(=O)N[*:3])c1", {"1": "R1", "2": "R1", "3": "R2"}, ["R1", "R2"])
    assert canonical_smiles(core) == canonical_smiles("[*:1]c1cc([*:1])cc(C(=O)N[*:2])c1")


def test_align_slots_gives_one_unknown_label_one_number():
    core = align_slots("[*:1]c1cc([*:2])cc([*:3])c1", {"1": "R1", "2": "X", "3": "X"}, ["R1"])
    assert canonical_smiles(core) == canonical_smiles("[*:1]c1cc([*:2])cc([*:2])c1")


def test_align_slots_keeps_unlabelled_attachments_apart():
    core = align_slots("[*:1]c1cc([*:2])cc([*:3])c1", {"1": "R1", "2": "*", "3": "*"}, ["R1"])
    assert check_core(core, ["R1"]).extra == {2, 3}


def test_build_from_an_aligned_core_with_a_repeated_label():
    core = align_slots("[*:1]c1cc([*:2])cc(C(=O)N[*:3])c1", {"1": "R1", "2": "R1", "3": "R2"}, ["R1", "R2"])
    got = build(core, {1: frag("F"), 2: frag("Me")})
    assert got == canonical_smiles("Fc1cc(F)cc(C(=O)NC)c1")


# --- build: substituents ----------------------------------------------------------------------


def test_build_joins_substituents():
    got = build("[*:1]c1ccc([*:2])cc1", {1: frag("OMe"), 2: frag("Cl")})
    assert got == canonical_smiles("COc1ccc(Cl)cc1")


def test_build_with_hydrogen():
    assert build("[*:1]c1ccc([*:2])cc1", {1: frag("H"), 2: frag("Cl")}) == canonical_smiles("Clc1ccccc1")


def test_build_repeats_a_slot_drawn_twice():
    got = build("[*:1]c1cc([*:1])cc([*:2])c1", {1: frag("F"), 2: frag("Me")})
    assert got == canonical_smiles("Fc1cc(F)cc(C)c1")


def test_build_with_aryl_and_name_fragments():
    got = build("O=C(N[*:1])c1ccc([*:2])cc1", {1: frag("4-F-Ph"), 2: Fragment("*N1CCOCC1", "substituent")})
    assert got == canonical_smiles("O=C(Nc1ccc(F)cc1)c1ccc(N2CCOCC2)cc1")


def test_build_keeps_stereo_next_to_a_slot():
    assert build("C[C@H](N)[*:1]", {1: frag("OH")}) == canonical_smiles("C[C@H](N)O")


# --- build: linkers ---------------------------------------------------------------------------


def test_build_inserts_linker_left_end_to_first_neighbour():
    got = build("c1ccccc1[*:1]c1ccncc1", {1: frag("-NHCO-")})
    assert got == canonical_smiles("c1ccccc1NC(=O)c1ccncc1")


@pytest.mark.parametrize(("text", "expected"), [("-O-", "c1ccccc1Oc1ccccc1"), ("-CH2CH2-", "c1ccccc1CCc1ccccc1")])
def test_build_symmetric_linkers(text, expected):
    assert build("c1ccccc1[*:1]c1ccccc1", {1: frag(text)}) == canonical_smiles(expected)


@pytest.mark.parametrize(
    "core",
    [
        "c1ccccc1[*:1][C{s}H](F)Cl",
        # Four explicit neighbours: moving the slot bond from first to last is an odd
        # permutation, so the configuration tag must be inverted to keep the stereo.
        "c1ccccc1[*:1][C{s}](F)(Cl)Br",
    ],
)
@pytest.mark.parametrize("stereo", ["@", "@@"])
def test_build_linker_keeps_stereo_on_the_far_neighbour(core, stereo):
    got = build(core.format(s=stereo), {1: frag("-O-")})
    assert got == canonical_smiles(core.format(s=stereo).replace("[*:1]", "O"))


@pytest.mark.parametrize("stereo", ["@", "@@"])
def test_build_linker_keeps_stereo_on_the_near_neighbour(stereo):
    got = build(f"F[C{stereo}H](Cl)[*:1]c1ccccc1", {1: frag("-NHCO-")})
    assert got == canonical_smiles(f"F[C{stereo}H](Cl)NC(=O)c1ccccc1")


def test_build_rejects_linker_on_a_terminal_slot():
    assert build("c1ccccc1[*:1]", {1: frag("-O-")}) is None


def test_build_rejects_substituent_on_a_bridging_slot():
    assert build("c1ccccc1[*:1]c1ccccc1", {1: frag("OMe")}) is None


# --- build: ring atoms and one-atom bridges --------------------------------------------------


@pytest.mark.parametrize(
    ("core", "text", "expected"),
    [
        ("OC(=O)c1cc[*:1]cc1", "N", "OC(=O)c1ccncc1"),
        ("OC(=O)c1cc[*:1]cc1", "CH", "OC(=O)c1ccccc1"),
        ("OC(=O)C1=CC=[*:1]C=C1", "N", "OC(=O)c1ccncc1"),
        ("OC(=O)c1cc[*:1]cc1", "CF", "OC(=O)c1ccc(F)cc1"),
        ("OC(=O)c1cc[*:1]cc1", "C-OMe", "OC(=O)c1ccc(OC)cc1"),
        ("c1ccc2[*:1]ccc2c1", "NH", "c1ccc2[nH]ccc2c1"),
        ("c1ccc2[*:1]ccc2c1", "O", "c1ccc2occc2c1"),
        ("c1ccc2[*:1]ccc2c1", "S", "c1ccc2sccc2c1"),
        ("c1ccccc1[*:1]c1ccccc1", "O", "c1ccccc1Oc1ccccc1"),
        ("c1ccccc1[*:1]c1ccccc1", "CH2", "c1ccccc1Cc1ccccc1"),
        ("c1ccccc1[*:1]c1ccccc1", "SO2", "O=S(=O)(c1ccccc1)c1ccccc1"),
        ("c1ccccc1[*:1]c1ccccc1", "C=O", "O=C(c1ccccc1)c1ccccc1"),
        ("C1CC[*:1]CC1", "NMe", "CN1CCCCC1"),
    ],
)
def test_build_replaces_variable_atoms(core, text, expected):
    assert build(core, {1: frag(text)}) == canonical_smiles(expected)


@pytest.mark.parametrize(("core", "text"), [("c1cc[*:1]cc1", "CH2"), ("c1ccc2[*:1]ccc2c1", "N")])
def test_build_rejects_atoms_that_break_valence_or_aromaticity(core, text):
    assert build(core, {1: frag(text)}) is None


def test_build_mixes_kinds():
    core = "[*:1]c1cc[*:2]c([*:3])c1"
    got = build(core, {1: frag("Cl"), 2: frag("N"), 3: frag("NH2")})
    assert got == canonical_smiles("Clc1ccnc(N)c1")


# --- build: refusals ------------------------------------------------------------------------


def test_build_rejects_a_slot_without_a_fragment():
    assert build("[*:1]c1ccc([*:2])cc1", {1: frag("F")}) is None


def test_build_rejects_a_fragment_without_a_slot():
    assert build("[*:1]c1ccccc1", {1: frag("F"), 2: frag("Cl")}) is None


def test_build_rejects_unlabelled_dummies():
    assert build("*c1ccc([*:1])cc1", {1: frag("F")}) is None


def test_build_rejects_unparseable_core():
    assert build("c1ccc(", {}) is None


def test_build_core_without_slots():
    assert build("OCC", {}) == "CCO"


def test_build_output_has_no_dummies_or_atom_maps():
    got = build("[*:1]c1ccc([*:2])cc1", {1: frag("Et"), 2: frag("CN")})
    assert "*" not in got and ":" not in got
