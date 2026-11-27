import pytest
from rdkit import rdBase

from molscout.scoring import smiles as smiles_module
from molscout.scoring.smiles import (
    RDKIT_VERSION,
    canonical_smiles,
    check_rdkit_version,
    is_exact_match,
    stereo_stripped_smiles,
)


def test_pinned_rdkit_is_installed():
    assert rdBase.rdkitVersion == RDKIT_VERSION
    check_rdkit_version()


def test_version_check_rejects_other_rdkit(monkeypatch):
    monkeypatch.setattr(rdBase, "rdkitVersion", "2025.09.1")
    with pytest.raises(RuntimeError, match="needs RDKit 2026.03.2"):
        check_rdkit_version()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("OCC", "CCO"),
        ("C1=CC=CC=C1", "c1ccccc1"),
        ("  CCO\n", "CCO"),
        ("c1ccccc1*", "*c1ccccc1"),
        ("[*]C", "*C"),
        ("[1*]c1ccccc1", "[1*]c1ccccc1"),
        ("Oc1cc([*:1])cc([*:2])c1", "Oc1cc([*:1])cc([*:2])c1"),
        ("C[C@H](N)C(=O)O", "C[C@H](N)C(=O)O"),
    ],
)
def test_canonical_smiles(raw, expected):
    assert canonical_smiles(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "C1CC", "not_a_smiles", "C(C)(C)(C)(C)C"])
def test_unparsable_input_is_none(raw):
    assert canonical_smiles(raw) is None
    assert stereo_stripped_smiles(raw) is None


def test_parse_errors_are_not_logged(capfd):
    canonical_smiles("C1CC")
    assert "SMILES Parse Error" not in capfd.readouterr().err


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("C[C@@H](N)C(=O)O", "CC(N)C(=O)O"),
        ("C/C=C\\C", "CC=CC"),
        ("[13CH3]C[C@H](N)O", "[13CH3]CC(N)O"),
        ("[1*]c1ccccc1", "[1*]c1ccccc1"),
    ],
)
def test_stereo_stripped_keeps_everything_but_stereo(raw, expected):
    assert stereo_stripped_smiles(raw) == expected


def test_exact_match_stereo_aware_and_stripped():
    assert is_exact_match("OCC", "CCO")
    assert not is_exact_match("C[C@@H](N)C(=O)O", "C[C@H](N)C(=O)O")
    assert is_exact_match("C[C@@H](N)C(=O)O", "C[C@H](N)C(=O)O", stereo=False)


def test_star_and_labelled_star_do_not_match():
    assert not is_exact_match("*c1ccccc1", "[1*]c1ccccc1")


def test_invalid_never_matches_even_itself():
    assert not is_exact_match("C1CC", "C1CC")
    assert not is_exact_match("", "")


@pytest.mark.parametrize("raw", ["CCO CCN", "CCO\tCCN", "CCO junk"])
def test_text_after_whitespace_makes_output_invalid(raw):
    assert canonical_smiles(raw) is None


def test_cxsmiles_extension_still_parses():
    assert canonical_smiles("*C |$R1;$|") == "*C"


# Public references whose RDKit canonical SMILES changes when it is canonicalized again.
# MolRecBench-Wild 10.1002_anie.202411707_3_figure_0_mol_3:
WILD_STAR_RINGS = "OC1=[CH]*=[CH]*=[C]1C1=NN=C([C]2=C(O)[CH]=*[CH]=*2)N=N1"
# The CARBON graph in test_data_carbon's ring test:
CARBON_STAR_RING = "OC1=[CH]*=[CH]*=[CH]1"
# USPTO US07317016-20080108-C00012:
USPTO_EXPLICIT_H = "[H]/N=C(\\OC)c1nn(Cc2ccccc2F)c2ncccc12"
# MolRecBench-Wild 10.1021_acscatal.5c02249_7_figure_0_mol_4:
WILD_EXPLICIT_H = "[H]/C=C/C1=CC=CC=C1"
NOT_IDEMPOTENT = [WILD_STAR_RINGS, CARBON_STAR_RING, USPTO_EXPLICIT_H, WILD_EXPLICIT_H]


@pytest.mark.parametrize("raw", NOT_IDEMPOTENT)
@pytest.mark.parametrize("key", [canonical_smiles, stereo_stripped_smiles], ids=["stereo", "stripped"])
def test_canonical_form_is_a_fixed_point(key, raw):
    once = key(raw)
    assert once is not None and key(once) == once


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [
        # Aromatic rings with * atoms: RDKit writes them aromatic, then reads them back Kekulé.
        ("Oc1c*c*c1-c1nnc(-c2*c*cc2O)nn1", WILD_STAR_RINGS),
        ("Oc1c*c*c1", CARBON_STAR_RING),
        ("[H]/N=C(\\OC)c1nn(Cc2ccccc2F)c2ncccc12", USPTO_EXPLICIT_H),
        ("[H]/C=C/c1ccccc1", WILD_EXPLICIT_H),
    ],
)
@pytest.mark.parametrize("stereo", [True, False], ids=["stereo", "stripped"])
def test_rdkit_canonical_prediction_matches_raw_reference(prediction, reference, stereo):
    # Each prediction is RDKit's own canonical SMILES of the reference.
    assert is_exact_match(prediction, reference, stereo=stereo)


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [("COC(=N)c1nn(Cc2ccccc2F)c2ncccc12", USPTO_EXPLICIT_H), ("C=Cc1ccccc1", WILD_EXPLICIT_H)],
)
def test_prediction_without_stereo_matches_in_stereo_stripped_score(prediction, reference):
    # Stripping stereo leaves an explicit [H] that the next pass drops.
    assert is_exact_match(prediction, reference, stereo=False)


def test_canonicalization_stops_after_max_passes(monkeypatch):
    # A writer that never settles: each pass adds one atom. The last string is used.
    monkeypatch.setattr(smiles_module.Chem, "MolToSmiles", lambda mol: "C" * (mol.GetNumAtoms() + 1))
    assert canonical_smiles("C") == "C" * (smiles_module.MAX_PASSES + 1)


def test_written_string_that_cannot_be_read_back_is_kept(monkeypatch):
    # Validity is decided by the first parse alone: a string RDKit writes but cannot read back
    # ends the passes and is used, so the input still counts as valid output.
    monkeypatch.setattr(smiles_module.Chem, "MolToSmiles", lambda mol: "C1CC")
    assert canonical_smiles("C") == "C1CC"
    assert stereo_stripped_smiles("C") == "C1CC"
