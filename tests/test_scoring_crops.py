import pytest

from molscout.predictions import Prediction
from molscout.scoring.crops import score_crops


def crop(item_id, smiles):
    return Prediction("uspto", item_id, smiles, None, None, None, "T 1", 0.1)


# c5's reference file is unreadable; c6 has no prediction row; c7's prediction is invalid.
REFERENCES = {
    "c1": "CCO",
    "c2": "C[C@H](N)C(=O)O",
    "c3": "c1ccccc1",
    "c4": "*c1ccccc1",
    "c5": None,
    "c6": "CC(=O)O",
    "c7": "CCN",
}
PREDICTIONS = [
    crop("c1", "OCC"),               # right
    crop("c2", "C[C@@H](N)C(=O)O"),  # wrong stereo: right only when stripped
    crop("c3", "C1=CC=CC=C1"),       # right
    crop("c4", "*c1ccccc1"),         # right
    crop("c5", "CCC"),               # excluded crop: ignored
    crop("c7", "C1CC"),              # invalid: wrong
]


def test_scores_hand_counted_crops():
    scores = score_crops(PREDICTIONS, REFERENCES)
    assert scores["kind"] == "crop"
    assert scores["items_scored"] == 6
    assert scores["items_excluded"] == ["c5"]
    assert scores["items_without_prediction"] == 1
    assert (scores["accuracy"]["successes"], scores["accuracy"]["trials"]) == (3, 6)
    assert scores["accuracy_stereo_stripped"]["successes"] == 4
    assert scores["valid_output_rate"]["successes"] == 4
    assert scores["accuracy"]["ci95"] == pytest.approx([0.187616306483, 0.812383693517], abs=1e-9)


def test_empty_output_is_wrong_and_not_valid():
    scores = score_crops([crop("c1", "")], {"c1": "CCO"})
    assert scores["accuracy"]["value"] == 0.0
    assert scores["valid_output_rate"]["value"] == 0.0


def test_star_atoms_are_scored_not_rejected():
    assert score_crops([crop("c1", "c1ccccc1*")], {"c1": "*c1ccccc1"})["accuracy"]["value"] == 1.0


def test_unknown_crop_raises():
    with pytest.raises(ValueError, match="not in the references.*zz"):
        score_crops([crop("zz", "CCO")], {"c1": "CCO"})


def test_duplicate_crop_raises():
    with pytest.raises(ValueError, match="more than one prediction for crop 'c1'"):
        score_crops([crop("c1", "CCO"), crop("c1", "CCO")], {"c1": "CCO"})


def test_no_scoreable_crops_gives_zero_with_uninformative_interval():
    scores = score_crops([], {"c1": None})
    assert scores["items_scored"] == 0
    assert scores["accuracy"]["ci95"] == [0.0, 1.0]
