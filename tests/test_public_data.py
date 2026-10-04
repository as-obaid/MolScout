"""Fetched public datasets match their manifests and the unreadable counts in docs/dev.md."""

from pathlib import Path

import pytest

from molscout.data import molrecbench
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
