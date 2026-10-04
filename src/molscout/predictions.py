"""Read, validate and write predictions.csv, the format every tool and MolScout emit.

One row per predicted molecule; see docs/dev.md, "Predictions Format".
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from molscout.datasets import Kind, dataset_kind

COLUMNS = ("dataset", "item_id", "smiles", "page", "bbox", "confidence", "tool", "seconds")
MAX_ERRORS_SHOWN = 20

BBox = tuple[float, float, float, float]
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Prediction:
    """One predicted molecule. `smiles` is the raw tool output and may be empty or invalid."""

    dataset: str
    item_id: str
    smiles: str
    page: int | None
    bbox: BBox | None
    confidence: float | None
    tool: str
    seconds: float


class PredictionsFormatError(ValueError):
    """predictions.csv breaks the format; `errors` lists every problem found."""

    def __init__(self, path: Path, errors: Sequence[str]) -> None:
        self.path = path
        self.errors = tuple(errors)
        shown = "\n".join(f"  - {error}" for error in self.errors[:MAX_ERRORS_SHOWN])
        hidden = len(self.errors) - MAX_ERRORS_SHOWN
        tail = f"\n  ... and {hidden} more" if hidden > 0 else ""
        super().__init__(f"{path}: {len(self.errors)} format error(s)\n{shown}{tail}")


def read_predictions(path: str | Path) -> tuple[Prediction, ...]:
    """Read and validate predictions.csv; raise PredictionsFormatError listing every problem."""
    path = Path(path)
    errors: list[str] = []
    predictions: list[Prediction] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header_errors = _check_header(reader.fieldnames)
        if header_errors:
            raise PredictionsFormatError(path, header_errors)
        for row in reader:
            prediction, row_errors = _parse_row(row)
            errors.extend(f"line {reader.line_num}: {error}" for error in row_errors)
            if prediction is not None:
                predictions.append(prediction)
    errors.extend(_check_file(predictions))
    if errors:
        raise PredictionsFormatError(path, errors)
    return tuple(predictions)


def write_predictions(path: str | Path, predictions: Iterable[Prediction]) -> None:
    """Write predictions.csv with the columns in format order."""
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for p in predictions:
            bbox = "" if p.bbox is None else ",".join(str(v) for v in p.bbox)
            writer.writerow(
                [p.dataset, p.item_id, p.smiles, _text(p.page), bbox, _text(p.confidence), p.tool, str(p.seconds)]
            )


def _text(value: object | None) -> str:
    return "" if value is None else str(value)


def _check_header(fieldnames: Sequence[str] | None) -> list[str]:
    if fieldnames is None:
        return ["file is empty; expected a header row"]
    errors = []
    missing = [c for c in COLUMNS if c not in fieldnames]
    unknown = [n for n in fieldnames if n not in COLUMNS]
    repeated = sorted({n for n in fieldnames if list(fieldnames).count(n) > 1})
    if missing:
        errors.append(f"header is missing column(s): {', '.join(missing)}")
    if unknown:
        errors.append(f"header has unknown column(s): {', '.join(unknown)}")
    if repeated:
        errors.append(f"header repeats column(s): {', '.join(repeated)}")
    return errors


def _parse_row(row: dict[str | None, str | list[str] | None]) -> tuple[Prediction | None, list[str]]:
    if None in row:
        return None, [f"{len(row[None])} extra field(s) beyond the {len(COLUMNS)} columns"]  # type: ignore[arg-type]
    if any(value is None for value in row.values()):
        return None, [f"expected {len(COLUMNS)} fields"]
    fields: dict[str, str] = row  # type: ignore[assignment]
    errors: list[str] = []
    dataset = fields["dataset"].strip()
    kind: Kind | None = None
    try:
        kind = dataset_kind(dataset)
    except ValueError as exc:
        errors.append(str(exc))
    item_id = fields["item_id"].strip()
    if not item_id:
        errors.append("item_id is empty")
    tool = fields["tool"].strip()
    if not tool:
        errors.append("tool is empty")
    page = _optional(fields["page"], _parse_page, "page", errors)
    bbox = _optional(fields["bbox"], _parse_bbox, "bbox", errors)
    confidence = _optional(fields["confidence"], _parse_finite, "confidence", errors)
    seconds = _parse_seconds(fields["seconds"], errors)
    if kind is Kind.CROP and page is not None:
        errors.append("page must be empty for a crop dataset")
    if kind is Kind.CROP and bbox is not None:
        errors.append("bbox must be empty for a crop dataset")
    if errors or seconds is None:
        return None, errors
    prediction = Prediction(dataset, item_id, fields["smiles"], page, bbox, confidence, tool, seconds)
    return prediction, []


def _optional(text: str, parse: Callable[[str], T], name: str, errors: list[str]) -> T | None:
    if not text.strip():
        return None
    try:
        return parse(text)
    except ValueError as exc:
        errors.append(f"{name}: {exc}")
        return None


def _parse_finite(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text.strip()!r} is not a finite number")
    return value


def _parse_page(text: str) -> int:
    page = int(text.strip())
    if page < 1:
        raise ValueError(f"got {page}; pages are 1-based")
    return page


def _parse_bbox(text: str) -> BBox:
    parts = text.split(",")
    if len(parts) != 4:
        raise ValueError(f"expected x0,y0,x1,y1, got {text!r}")
    x0, y0, x1, y1 = (_parse_finite(part) for part in parts)
    if x0 > x1 or y0 > y1:
        raise ValueError(f"expected x0 <= x1 and y0 <= y1, got {text!r}")
    return (x0, y0, x1, y1)


def _parse_seconds(text: str, errors: list[str]) -> float | None:
    if not text.strip():
        errors.append("seconds is empty")
        return None
    try:
        seconds = _parse_finite(text)
    except ValueError as exc:
        errors.append(f"seconds: {exc}")
        return None
    if seconds < 0:
        errors.append(f"seconds must be >= 0, got {seconds}")
        return None
    return seconds


def _check_file(predictions: Sequence[Prediction]) -> list[str]:
    errors = []
    datasets = sorted({p.dataset for p in predictions})
    if len(datasets) > 1:
        errors.append(f"a run covers one dataset; found {', '.join(datasets)}")
    tools = sorted({p.tool for p in predictions})
    if len(tools) > 1:
        errors.append(f"a run covers one tool; found {', '.join(tools)}")
    crop_ids = Counter(p.item_id for p in predictions if dataset_kind(p.dataset) is Kind.CROP)
    repeated = sorted(item for item, count in crop_ids.items() if count > 1)
    if repeated:
        errors.append(f"crop dataset has more than one row for item(s): {', '.join(repeated[:10])}")
    return errors
