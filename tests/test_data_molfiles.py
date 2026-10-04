import pytest
from rdkit import Chem

from molscout.data.molfiles import Reference, load_references, read_molfile, read_sdfile


def molblock(smiles):
    return Chem.MolToMolBlock(Chem.MolFromSmiles(smiles))


# Pentavalent carbon: parses unsanitized, fails sanitization, as in some CLEF and USPTO files.
HYPERVALENT = Chem.MolToMolBlock(Chem.MolFromSmiles("C(C)(C)(C)(C)C", sanitize=False), kekulize=False)


def test_reads_molfile_to_smiles(tmp_path):
    path = tmp_path / "US07314883-20080101-C00573.MOL"
    path.write_text(molblock("OCC"))
    assert read_molfile(path) == Reference("US07314883-20080101-C00573", "CCO")


def test_molfile_keeps_stereo_and_star_atoms(tmp_path):
    (tmp_path / "a.mol").write_text(molblock("C[C@H](N)C(=O)O"))
    (tmp_path / "b.mol").write_text(molblock("*c1ccccc1"))
    assert read_molfile(tmp_path / "a.mol").smiles == "C[C@H](N)C(=O)O"
    assert read_molfile(tmp_path / "b.mol").smiles == "*c1ccccc1"


def test_unreadable_molfile_gives_rdkit_reason(tmp_path):
    path = tmp_path / "bad.mol"
    path.write_text(HYPERVALENT)
    reference = read_molfile(path)
    assert reference.smiles is None
    assert "valence" in reference.error.lower()


def test_garbage_molfile_is_unreadable(tmp_path):
    path = tmp_path / "junk.mol"
    path.write_text("not a molfile\n")
    reference = read_molfile(path)
    assert reference.smiles is None
    assert reference.error


def test_molfile_without_atoms_is_unreadable(tmp_path):
    path = tmp_path / "empty.mol"
    path.write_text(Chem.MolToMolBlock(Chem.Mol()))
    assert read_molfile(path) == Reference("empty", None, "file has no atoms")


def test_reads_sdfile(tmp_path):
    path = tmp_path / "2008063265_97_chem.sdf"
    path.write_text(molblock("c1ccccc1") + "$$$$\n")
    assert read_sdfile(path) == Reference("2008063265_97_chem", "c1ccccc1")


def test_sdfile_must_hold_one_record(tmp_path):
    path = tmp_path / "two.sdf"
    path.write_text((molblock("C") + "$$$$\n") * 2)
    assert read_sdfile(path) == Reference("two", None, "expected 1 record, found 2")


def test_bad_sdfile_record_gives_rdkit_reason(tmp_path):
    path = tmp_path / "bad.sdf"
    path.write_text(HYPERVALENT + "$$$$\n")
    reference = read_sdfile(path)
    assert reference.smiles is None
    assert "valence" in reference.error.lower()


def test_load_references_reads_a_directory(tmp_path):
    (tmp_path / "c1.MOL").write_text(molblock("CCO"))
    (tmp_path / "c2.sdf").write_text(molblock("CCN") + "$$$$\n")
    (tmp_path / "c3.mol").write_text(HYPERVALENT)
    (tmp_path / ".DS_Store").write_text("x")
    (tmp_path / "notes.txt").write_text("x")
    references = load_references(tmp_path)
    assert sorted(references) == ["c1", "c2", "c3"]
    assert references["c1"].smiles == "CCO"
    assert references["c2"].smiles == "CCN"
    assert references["c3"].smiles is None


def test_load_references_rejects_two_files_for_one_crop(tmp_path):
    (tmp_path / "c1.mol").write_text(molblock("C"))
    (tmp_path / "c1.sdf").write_text(molblock("C") + "$$$$\n")
    with pytest.raises(ValueError, match="two reference files for crop 'c1'"):
        load_references(tmp_path)


def test_load_references_needs_files(tmp_path):
    with pytest.raises(ValueError, match=r"no \.mol or \.sdf files"):
        load_references(tmp_path)


def test_load_references_needs_the_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_references(tmp_path / "missing")
