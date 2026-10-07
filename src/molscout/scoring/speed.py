"""Wall-clock seconds per item, from the `seconds` column of predictions.csv."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence

from molscout.predictions import Prediction


def seconds_per_item(
    predictions: Sequence[Prediction], item_seconds: Mapping[str, float] | None = None
) -> dict[str, object]:
    """Mean and median seconds over items with at least one row.

    Every row carries its item's time, so an item's time is the largest value among
    its rows: a crop has one row, a paper one row per molecule.

    With `item_seconds` (item to seconds, including items that have no rows) the items are its
    keys and the times are its values; every predicted item must be one of them.
    """
    per_item: dict[str, float] = {}
    for prediction in predictions:
        per_item[prediction.item_id] = max(prediction.seconds, per_item.get(prediction.item_id, 0.0))
    if item_seconds is not None:
        unknown = sorted(set(per_item) - set(item_seconds))
        if unknown:
            raise ValueError(f"predictions for item(s) with no time in the timing file: {', '.join(unknown[:5])}")
        per_item = dict(item_seconds)
    values = list(per_item.values())
    if not values:
        return {"mean": None, "median": None, "items": 0, "total": 0.0}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "items": len(values),
        "total": math.fsum(values),
    }
