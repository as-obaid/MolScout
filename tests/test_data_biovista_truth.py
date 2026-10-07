import shutil
from pathlib import Path

import pytest

from molscout.data import biovista_truth as bt

FIXTURE = Path(__file__).parent / "fixtures" / "biovista"
MANIFEST = FIXTURE / "manifest.csv"


@pytest.fixture
def root(tmp_path):
    shutil.copytree(FIXTURE, tmp_path / "biovista")
    return tmp_path / "biovista"


@pytest.fixture(scope="module")
def truth():
    return bt.load_biovista_truth(FIXTURE, MANIFEST)


def test_scored_papers_are_ok_with_structures_in_index_order():
    papers = bt.scored_papers(MANIFEST)
    assert [p.paper_id for p in papers] == ["1_aaaa", "2_bbbb", "3_cccc"]
    assert papers[0] == bt.ScoredPaper("1_aaaa", "aaaa", "publishedVersion", papers[0].sha256, 4)


def test_scored_papers_reject_bad_manifests(tmp_path):
    bad = tmp_path / "m.csv"
    bad.write_text("paper_id,pdb_id\n1_aaaa,aaaa\n")
    with pytest.raises(ValueError, match="status"):
        bt.scored_papers(bad)
    text = MANIFEST.read_text().replace(",test,4,", ",test,four,", 1)
    bad.write_text(text)
    with pytest.raises(ValueError, match="1_aaaa"):
        bt.scored_papers(bad)


def test_references_split_into_drawn_and_enumerated(truth):
    assert dict(truth.references) == {"1_aaaa": ("CCO", "c1ccccc1", "CCCCl"), "2_bbbb": ("CCN",), "3_cccc": ("CCCl",)}
    assert dict(truth.drawn) == {"1_aaaa": ("CCO", "c1ccccc1"), "2_bbbb": ("CCN",)}
    assert dict(truth.enumerated) == {"1_aaaa": ("CCCCl",), "2_bbbb": (), "3_cccc": ("CCCl",)}
    assert truth.labels == 7


def test_unreadable_rows_name_the_csv_line_and_reason(truth):
    assert dict(truth.unreadable) == {
        "1_aaaa:4": "NA: the label has no SMILES",
        "2_bbbb:3": "RDKit cannot parse it; text after whitespace is part of the SMILES",
    }


def test_groups(truth):
    assert truth.groups() == {
        "all": frozenset({"1_aaaa", "2_bbbb", "3_cccc"}),
        "without_submitted": frozenset({"1_aaaa", "3_cccc"}),
    }
    assert truth.drawn_groups() == {
        "all": frozenset({"1_aaaa", "2_bbbb"}),
        "without_submitted": frozenset({"1_aaaa"}),
    }


def test_the_tab_in_the_label_is_kept():
    text = (FIXTURE / bt.LABELS / "2_bbbb_structure.csv").read_text()
    assert "CC(C)O\t67" in text


def test_changed_row_count_names_the_paper(root):
    path = root / bt.LABELS / "3_cccc_structure.csv"
    path.write_text(path.read_text() + "CCBr,8,NA,NA\n")
    with pytest.raises(ValueError, match="3_cccc"):
        bt.load_biovista_truth(root, MANIFEST)


def test_missing_label_file_and_columns(root):
    (root / bt.LABELS / "3_cccc_structure.csv").write_text("smiles,ligand,groups\nCCCl,7a,R=Cl\n")
    with pytest.raises(ValueError, match="backbone"):
        bt.load_biovista_truth(root, MANIFEST)
    (root / bt.LABELS / "3_cccc_structure.csv").unlink()
    with pytest.raises(ValueError, match="3_cccc"):
        bt.load_biovista_truth(root, MANIFEST)


def test_paper_without_readable_label_is_an_error(root):
    (root / bt.LABELS / "3_cccc_structure.csv").write_text("smiles,ligand,backbone,groups\nNA,7a,CC*,R=Cl\n")
    with pytest.raises(ValueError, match="3_cccc"):
        bt.load_biovista_truth(root, MANIFEST)


def test_checksum_changes_with_a_label_file(root):
    before = bt.reference_set_sha256(root, MANIFEST)
    assert before == bt.reference_set_sha256(FIXTURE, MANIFEST)
    path = root / bt.LABELS / "3_cccc_structure.csv"
    path.write_text(path.read_text().replace("CCCl", "CCCCl"))
    assert bt.reference_set_sha256(root, MANIFEST) != before


def test_blank_backbone_is_enumerated_not_drawn(tmp_path):
    labels = tmp_path / bt.LABELS
    labels.mkdir(parents=True)
    (labels / "1_aaaa_structure.csv").write_text("smiles,backbone\nCCO,NA\nCCN,\nCCC, \nCCCl,CC*\n")
    manifest = tmp_path / "manifest.csv"
    lines = MANIFEST.read_text().splitlines()
    manifest.write_text("\n".join([lines[0], lines[1].replace(",4,", ",4,", 1)]) + "\n")
    truth = bt.load_biovista_truth(tmp_path, manifest)
    assert dict(truth.drawn) == {"1_aaaa": ("CCO",)}
    assert dict(truth.enumerated) == {"1_aaaa": ("CCN", "CCC", "CCCl")}
