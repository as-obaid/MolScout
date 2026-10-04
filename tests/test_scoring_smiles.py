import pytest
from rdkit import rdBase

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
