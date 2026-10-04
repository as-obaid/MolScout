import hashlib
from pathlib import Path

import pytest

from molscout.data.internal import (
    GROUND_TRUTH_PATH,
    SPLIT_PATH,
    GroundTruthMolecule,
    check_split_counts,
    load_internal_ground_truth,
    load_internal_split,
    references_by_paper,
)

REPO = Path(__file__).resolve().parents[1]
needs_internal = pytest.mark.skipif(
    not (REPO / GROUND_TRUTH_PATH).exists(), reason="private internal set not in data/internal/"
)

GROUND_TRUTH = (
    "paperID,name of molecule,canonical_SMILES\r\n"
    "1,ethanol,CCO\r\n"
    "1,phenyl R-group,*c1ccccc1\r\n"
    '2,"acid, acetic",CC(=O)O\r\n'
    "2,scaffold,Oc1cc([*:1])cc([*:2])c1\r\n"
)
SPLIT = "paperID,split,molecules\n1,dev,2\n2,test,2\n"


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_bytes(text.encode("utf-8"))
    return path


def test_reads_ground_truth_with_crlf_quotes_and_star_atoms(tmp_path):
    molecules = load_internal_ground_truth(write(tmp_path, "gt.csv", GROUND_TRUTH))
    assert len(molecules) == 4
    assert molecules[0] == GroundTruthMolecule("1", "ethanol", "CCO")
    assert molecules[1].smiles == "*c1ccccc1"
    assert molecules[2].name == "acid, acetic"


def test_references_by_paper(tmp_path):
    molecules = load_internal_ground_truth(write(tmp_path, "gt.csv", GROUND_TRUTH))
    assert references_by_paper(molecules) == {
        "1": ("CCO", "*c1ccccc1"),
        "2": ("CC(=O)O", "Oc1cc([*:1])cc([*:2])c1"),
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("paperID,name,canonical_SMILES\n1,a,CCO\n", "expected columns"),
        ("paperID,name of molecule,canonical_SMILES\n1,a,C1CC\n", "line 2: RDKit cannot parse 'C1CC'"),
        ("paperID,name of molecule,canonical_SMILES\n1,,CCO\n", "line 2: empty field"),
        ("paperID,name of molecule,canonical_SMILES\n1,a\n", "expected 3 fields"),
    ],
)
def test_rejects_bad_ground_truth(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        load_internal_ground_truth(write(tmp_path, "gt.csv", text))


def test_reads_split(tmp_path):
    split = load_internal_split(write(tmp_path, "split.csv", SPLIT))
    assert split.papers("dev") == {"1"}
    assert split.papers("test") == {"2"}
    assert split.papers("all") == {"1", "2"}
    assert list(split.groups()) == ["dev", "test", "all"]
    assert dict(split.molecules) == {"1": 2, "2": 2}


def test_unknown_split_name(tmp_path):
    split = load_internal_split(write(tmp_path, "split.csv", SPLIT))
    with pytest.raises(ValueError, match="train"):
        split.papers("train")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("paperID,split,molecules\n1,train,2\n", "split must be dev or test"),
        ("paperID,split,molecules\n1,dev,2\n1,test,3\n", "listed twice"),
        ("paperID,split,molecules\n1,dev,zero\n", "molecules"),
        ("paperID,split\n1,dev\n", "expected columns"),
        ("paperID,split,molecules\n", "no papers"),
    ],
)
def test_rejects_bad_split(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        load_internal_split(write(tmp_path, "split.csv", text))


def test_check_split_counts(tmp_path):
    molecules = load_internal_ground_truth(write(tmp_path, "gt.csv", GROUND_TRUTH))
    check_split_counts(molecules, load_internal_split(write(tmp_path, "ok.csv", SPLIT)))
    wrong = load_internal_split(write(tmp_path, "wrong.csv", "paperID,split,molecules\n1,dev,3\n2,test,2\n"))
    with pytest.raises(ValueError, match="do not match"):
        check_split_counts(molecules, wrong)


def test_frozen_internal_split():
    """Frozen before MolScout development: dev 1, 16, 19 (126 molecules); test 2, 4, 6 (96)."""
    split = load_internal_split(REPO / SPLIT_PATH)
    assert split.papers("dev") == {"1", "16", "19"}
    assert split.papers("test") == {"2", "4", "6"}
    assert sum(split.molecules[p] for p in split.papers("dev")) == 126
    assert sum(split.molecules[p] for p in split.papers("test")) == 96


def test_split_manifest_bytes_are_unchanged():
    digest = hashlib.sha256((REPO / SPLIT_PATH).read_bytes()).hexdigest()
    assert digest == "64c313cdef2cb9ebcaeaf213185245fb7c71b0bd1276ea8ddc7f0cdb26204d54"


@needs_internal
def test_real_ground_truth_matches_split():
    molecules = load_internal_ground_truth(REPO / GROUND_TRUTH_PATH)
    assert len(molecules) == 222
    check_split_counts(molecules, load_internal_split(REPO / SPLIT_PATH))


@needs_internal
def test_real_ground_truth_keeps_star_molecules():
    molecules = load_internal_ground_truth(REPO / GROUND_TRUTH_PATH)
    assert sum("*" in m.smiles for m in molecules) == 5


@needs_internal
def test_real_ground_truth_has_220_unique_structures():
    """Papers 4 and 19 each list one structure under two names."""
    references = references_by_paper(load_internal_ground_truth(REPO / GROUND_TRUTH_PATH))
    assert sum(len(set(smiles)) for smiles in references.values()) == 220
