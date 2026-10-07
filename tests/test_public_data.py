"""Fetched public datasets match their manifests and the unreadable counts in docs/dev.md."""

from pathlib import Path

import pytest

from molscout.data import biovista_truth, molrecbench
from molscout.data.fetch import load_manifest
from molscout.data.molfiles import load_references

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"


@pytest.mark.parametrize(
    ("dataset", "folder", "unreadable"),
    [("uspto", "USPTO", 15), ("uob", "UOB", 0), ("jpo", "JPO", 1), ("clef", "CLEF", 15)],
)
def test_fetched_references_match_manifest(dataset, folder, unreadable):
    references_dir = RAW / dataset / f"{folder}_mol_ref"
    if not references_dir.is_dir():
        pytest.skip("run scripts/fetch_data.py first")
    references = load_references(references_dir)
    assert len(references) == load_manifest(REPO / "data" / "manifests" / f"{dataset}.json").items
    assert {path.stem for path in (RAW / dataset / folder).iterdir()} == set(references)
    assert sum(reference.smiles is None for reference in references.values()) == unreadable


@pytest.fixture(scope="module")
def wild_references():
    root = RAW / "molrecbench_wild"
    try:
        molrecbench.shard_paths(root)
    except FileNotFoundError:
        pytest.skip("run scripts/fetch_data.py molrecbench_wild first")
    return root, molrecbench.load_references(root)


def test_molrecbench_wild_reference_counts(wild_references):
    _, references = wild_references
    assert len(references) == 5024
    assert sum(reference.smiles is not None for reference in references.values()) == 2371
    assert all(reference.error for reference in references.values() if reference.smiles is None)


def test_molrecbench_wild_images_match_reference_ids(wild_references):
    root, references = wild_references
    images = root / molrecbench.IMAGES
    if not images.is_dir():
        pytest.skip("run scripts/fetch_data.py molrecbench_wild first")
    assert {path.stem for path in images.glob("*.png")} == set(references)


def test_biovista_truth_counts():
    root = RAW / "biovista"
    if not (root / biovista_truth.LABELS).is_dir():
        pytest.skip("run scripts/fetch_data.py biovista first")
    truth = biovista_truth.load_biovista_truth(root, REPO / biovista_truth.BIOVISTA_PAPERS_PATH)
    assert len(truth.references) == 163
    assert len(truth.groups()["without_submitted"]) == 122
    assert truth.labels == 3086
    assert len(truth.unreadable) == 72
    assert len(truth.drawn) == 159
    assert sum(len(smiles) for smiles in truth.drawn.values()) == 1584
    assert [p for p in truth.references if p not in truth.drawn] == ["77_6r49", "129_6q4q", "156_6pka", "230_6o5t"]
