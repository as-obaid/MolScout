"""A run's outputs: molecules.csv (every molecule with its status) and the eight-column predictions.csv.

molecules.csv keeps every molecule, accepted, review and abstained, with its label, source and status, so a reader
can filter it. predictions.csv holds only the statuses the run exports (`accepted`, or `accepted+review`) in the
format every benchmarked tool writes (molscout.predictions), one row per molecule, each row carrying its paper's
seconds.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from molscout.pipeline.config import PipelineConfig
from molscout.pipeline.stages import read_records
from molscout.predictions import Prediction, read_predictions, write_predictions

MOLECULE_COLUMNS = ("paper_id", "label", "smiles", "page", "bbox", "source", "status", "confidence")


def molecule_row(record: Mapping[str, Any]) -> list[str]:
    """One molecules.jsonl record as a molecules.csv row; the bbox is `x0,y0,x1,y1` in PDF points."""
    bbox = _bbox(record.get("bbox"))
    return [
        str(record.get("paper_id") or ""),
        str(record.get("label") or ""),
        str(record.get("smiles") or ""),
        "" if record.get("page") is None else str(record["page"]),
        "" if bbox is None else ",".join(repr(value) for value in bbox),
        str(record.get("source") or ""),
        str(record.get("status") or ""),
        "" if _confidence(record.get("confidence")) is None else repr(float(record["confidence"])),
    ]


def write_molecules_csv(path: Path, sources: Iterable[Path]) -> int:
    """molecules.csv from molecules.jsonl files, in order; returns the number of rows."""
    rows = [molecule_row(record) for source in sources for record in read_records(source)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MOLECULE_COLUMNS)
        writer.writerows(rows)
    return len(rows)


def predictions(
    molecules: Mapping[str, Path],
    config: PipelineConfig,
    *,
    dataset: str,
    tool: str,
    seconds: Mapping[str, float],
) -> list[Prediction]:
    """The exported molecules of each paper (paper ID -> its molecules.jsonl) as predictions.csv rows.

    A molecule's paper is the paper it was found in, whatever its record says. A page below 1 and a bbox that is
    not four finite numbers in order are left empty rather than written in a form the scorer would reject.
    """
    rows = []
    for paper_id, path in molecules.items():
        for record in read_records(path):
            if record.get("status") not in config.export_statuses:
                continue
            page = record.get("page")
            rows.append(
                Prediction(
                    dataset=dataset,
                    item_id=paper_id,
                    smiles=str(record.get("smiles") or ""),
                    page=page if isinstance(page, int) and not isinstance(page, bool) and page >= 1 else None,
                    bbox=_bbox(record.get("bbox")),
                    confidence=_confidence(record.get("confidence")),
                    tool=tool,
                    seconds=float(seconds.get(paper_id, 0.0)),
                )
            )
    return rows


def write_predictions_csv(path: Path, rows: Sequence[Prediction]) -> None:
    """Write predictions.csv and read it back through the format checks."""
    write_predictions(path, rows)
    read_predictions(path)


def _bbox(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    try:
        box = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in box) or box[0] > box[2] or box[1] > box[3]:
        return None
    return box  # type: ignore[return-value]


def _confidence(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)
