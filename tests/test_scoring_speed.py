import pytest

from molscout.predictions import Prediction
from molscout.scoring import seconds_per_item


def crop(item, seconds):
    return Prediction("uspto", item, "C", None, None, None, "t", seconds)


def paper(item, seconds):
    return Prediction("internal", item, "C", 1, None, None, "t", seconds)


def test_crops_mean_and_median():
    assert seconds_per_item([crop("a", 0.5), crop("b", 1.5), crop("c", 4.0)]) == {
        "mean": 2.0,
        "median": 1.5,
        "items": 3,
        "total": 6.0,
    }


def test_a_paper_counts_once_with_its_time():
    # Every row of a paper carries the paper's time, so the paper is counted once.
    rows = [paper("1", 3.0), paper("1", 3.0), paper("2", 5.0)]
    assert seconds_per_item(rows) == {"mean": 4.0, "median": 4.0, "items": 2, "total": 8.0}


def test_no_rows_gives_no_speed():
    assert seconds_per_item([]) == {"mean": None, "median": None, "items": 0, "total": 0.0}


def test_item_seconds_count_papers_without_rows():
    assert seconds_per_item([paper("1", 3.0)], {"1": 3.0, "2": 5.0}) == {
        "mean": 4.0,
        "median": 4.0,
        "items": 2,
        "total": 8.0,
    }


def test_item_seconds_must_cover_every_predicted_item():
    with pytest.raises(ValueError, match="3"):
        seconds_per_item([paper("1", 3.0), paper("3", 1.0)], {"1": 3.0})
