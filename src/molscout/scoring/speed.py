"""Wall-clock seconds per item, from the `seconds` column of predictions.csv."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from molscout.predictions import Prediction


def seconds_per_item(predictions: Sequence[Prediction]) -> dict[str, object]:
    """Mean and median seconds over items with at least one row.

    Every row carries its item's time, so an item's time is the largest value among
    its rows: a crop has one row, a paper one row per molecule.
    """
    per_item: dict[str, float] = {}
    for prediction in predictions:
        per_item[prediction.item_id] = max(prediction.seconds, per_item.get(prediction.item_id, 0.0))
    values = list(per_item.values())
    if not values:
        return {"mean": None, "median": None, "items": 0, "total": 0.0}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "items": len(values),
        "total": math.fsum(values),
    }
