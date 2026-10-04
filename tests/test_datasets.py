import pytest

from molscout.datasets import DATASETS, Kind, dataset_kind


def test_crop_datasets():
    crops = {name for name, kind in DATASETS.items() if kind is Kind.CROP}
    assert crops == {"uspto", "uob", "jpo", "clef", "molrecbench_wild"}


def test_paper_datasets():
    papers = {name for name, kind in DATASETS.items() if kind is Kind.PAPER}
    assert papers == {"biovista", "internal"}


def test_unknown_dataset_names_the_options():
    with pytest.raises(ValueError, match="unknown dataset 'USPTO'.*uspto"):
        dataset_kind("USPTO")


def test_registry_is_read_only():
    with pytest.raises(TypeError):
        DATASETS["new"] = Kind.CROP  # type: ignore[index]
