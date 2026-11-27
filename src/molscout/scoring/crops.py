"""Crop scoring: one reference molecule per crop image."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from molscout.predictions import Prediction
from molscout.scoring.smiles import canonical_smiles, stereo_stripped_smiles
from molscout.scoring.stats import proportion


def score_crops(predictions: Sequence[Prediction], references: Mapping[str, str | None]) -> dict[str, object]:
    """Score one prediction per crop against its reference SMILES.

    `references` maps crop ID to reference SMILES, or to None when RDKit cannot
    read the reference; those crops are excluded and listed. A crop without a
    prediction row, or with an empty or unparsable one, is wrong and not valid.
    """
    by_item = _one_per_item(predictions)
    unknown = sorted(set(by_item) - set(references))
    if unknown:
        raise ValueError(
            f"{len(unknown)} prediction(s) for crops not in the references, e.g. {', '.join(unknown[:5])}"
        )
    truth = {item: None if ref is None else canonical_smiles(ref) for item, ref in references.items()}
    scored = sorted(item for item, ref in truth.items() if ref is not None)
    excluded = sorted(item for item, ref in truth.items() if ref is None)
    correct = correct_stripped = valid = missing = 0
    for item in scored:
        prediction = by_item.get(item)
        if prediction is None:
            missing += 1
            continue
        predicted = canonical_smiles(prediction.smiles)
        if predicted is None:
            continue
        valid += 1
        correct += int(predicted == truth[item])
        correct_stripped += int(stereo_stripped_smiles(predicted) == stereo_stripped_smiles(truth[item]))
    return {
        "kind": "crop",
        "items_scored": len(scored),
        "items_excluded": excluded,
        "items_without_prediction": missing,
        "accuracy": proportion(correct, len(scored)),
        "accuracy_stereo_stripped": proportion(correct_stripped, len(scored)),
        "valid_output_rate": proportion(valid, len(scored)),
    }


def _one_per_item(predictions: Sequence[Prediction]) -> dict[str, Prediction]:
    by_item: dict[str, Prediction] = {}
    for prediction in predictions:
        if prediction.item_id in by_item:
            raise ValueError(f"more than one prediction for crop {prediction.item_id!r}")
        by_item[prediction.item_id] = prediction
    return by_item
