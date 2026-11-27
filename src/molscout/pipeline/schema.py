"""Records passed between pipeline stages, and their JSONL files.

Each stage writes one JSON object per line into the paper's work directory. Boxes are
`[x0, y0, x1, y1]` in PDF points with the origin at the top left; pages are 1-based. Lists
and dicts are stored as JSON arrays and objects, so a record written and read back compares
equal to the original when its boxes are lists (not tuples).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

SOURCES = ("drawn", "table", "name", "32b")
STATUSES = ("accepted", "review", "abstained")
# The fixes each ladder rung switches on; every one is on unless switched off.
FIXES = (
    "labels", "relevance", "tables", "core_check", "vote", "stereo", "junk", "names", "formulas",
    "attachments", "slot_labels", "legends", "partial_stereo",
)

T = TypeVar("T")


def fix_enabled(fixes: Any, name: str) -> bool:
    """Whether fix `name` is on.

    `fixes` may be None (all on), a mapping of flags (a missing flag is on), a set, list or
    tuple of the enabled names, or an object with boolean attributes (a missing attribute is
    on). A name outside FIXES, asked for or given in a mapping or collection, raises
    ValueError, so a misspelt flag never leaves a rung silently on.
    """
    if name not in FIXES:
        raise ValueError(f"unknown fix {name!r}; expected one of {FIXES}")
    if fixes is None:
        return True
    if isinstance(fixes, (Mapping, set, frozenset, list, tuple)):
        unknown = sorted(set(fixes) - set(FIXES))
        if unknown:
            raise ValueError(f"unknown fix flag(s) {unknown}; expected names from {FIXES}")
        return bool(fixes.get(name, True)) if isinstance(fixes, Mapping) else name in fixes
    return bool(getattr(fixes, name, True))


@dataclass(frozen=True)
class PageRecord:
    """One rendered page (pages.jsonl)."""

    page: int
    width_pt: float
    height_pt: float
    image: str
    dpi: int
    vector: bool


@dataclass(frozen=True)
class LayoutBlock:
    """One block from the layout parser (layout.jsonl).

    `type` is text, title, table, image, caption, footnote or equation; `html` holds a
    table's HTML; `parent` is the index of the figure or table a caption belongs to.
    """

    page: int
    index: int
    type: str
    bbox: list[float]
    text: str
    html: str | None = None
    parent: int | None = None
    image: str | None = None


@dataclass(frozen=True)
class Region:
    """One candidate structure drawing (regions.jsonl); `region_id` looks like `p3-r2`.

    `kind` is "drawing" for a drawing the segment stage found on a page, "cell" for one the cells stage cut from
    an R-group table cell (`p3-t5-r4-c0`; see workers/cells.py).
    """

    region_id: str
    page: int
    bbox: list[float]
    crop: str
    structure_score: float | None
    is_structure: bool
    kind: str = "drawing"


@dataclass(frozen=True)
class Read:
    """One reader's SMILES for a region (reads.jsonl).

    R slots are atom-mapped dummies `[*:n]`; `slots` maps `"n"` to the drawn label, e.g. `"R1"`.
    """

    region_id: str
    reader: str
    smiles: str | None
    slots: dict[str, str] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class VotedRead:
    """The readers' agreed SMILES for a region (voted.jsonl); `agree` counts agreeing readers.

    For an R-group core, `smiles` keeps its `[*:n]` slots and `slots` maps `"n"` to the drawn
    label, both taken from one reader's read; `slots` is empty without dummies and `inchikey`
    is None for a core. Without a result, `smiles` is None and `reason` says why.

    `alternatives` are other cores the readers may mean (vote.py), each a dict with `smiles`,
    `slots`, `agree`, `readers`, `confidence` and `kind`: "attachment" for a drawn core that
    plain reads support once its slots are read as H or a short chain, "slot_labels" for another
    labelling of the winning core. `skeleton_agree` counts the readers of the core's skeleton
    when they split on its slot labels, else 0. Files written before these fields read with the
    defaults.
    """

    region_id: str
    smiles: str | None
    inchikey: str | None
    agree: int
    stereo_kept: bool
    confidence: float
    readers: list[str]
    slots: dict[str, str] = field(default_factory=dict)
    reason: str | None = None
    skeleton_agree: int = 0
    alternatives: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Molecule:
    """One output molecule (molecules.jsonl).

    `source` is drawn, table, name or 32b; `status` is accepted, review or abstained.
    """

    paper_id: str
    label: str | None
    smiles: str | None
    page: int | None
    bbox: list[float] | None
    source: str
    table_id: str | None = None
    row: int | None = None
    status: str = "review"
    reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source!r}; expected one of {SOURCES}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}; expected one of {STATUSES}")


@dataclass(frozen=True)
class OcrLine:
    """One text line OCR read in a figure area (ocr.jsonl).

    `score` is the recogniser's confidence (0-1); `source_block` names the area that read it,
    `image:p<page>-<layout index>` or `region:<region_id>`.
    """

    page: int
    bbox: list[float]
    text: str
    score: float
    source_block: str


def write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    """Write dataclass records (or dicts) one JSON object per line, replacing the file at once.

    The text goes to a temporary file that is renamed into place, so a stage that stops
    midway never leaves a partial file. NaN and infinity raise ValueError.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_as_dict(record), ensure_ascii=False, allow_nan=False) + "\n" for record in records]
    partial = target.with_name(target.name + ".tmp")
    try:
        partial.write_text("".join(lines), encoding="utf-8")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def read_jsonl(path: str | Path, cls: type[T]) -> list[T]:
    """Read a JSONL file into `cls` records; blank lines are skipped.

    A line that is not JSON, or whose keys do not fit `cls`, raises ValueError naming the
    file and line.
    """
    source = Path(path)
    records = []
    with source.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            where = f"{source}:{number}"
            try:
                data = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"{where}: not JSON ({err.msg})") from err
            if not isinstance(data, dict):
                raise ValueError(f"{where}: expected a JSON object")
            try:
                records.append(cls(**data))
            except TypeError as err:
                raise ValueError(f"{where}: does not fit {cls.__name__} ({err})") from err
    return records


def need(path: str | Path) -> Path:
    """`path` when it is a file, else FileNotFoundError saying an earlier stage did not run."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} is missing in {path.parent}: an earlier stage did not run")
    return path


def to_points(bbox01: list[float], w_pt: float, h_pt: float) -> list[float]:
    """Scale a 0–1 box `[x0, y0, x1, y1]` to PDF points on a page of `w_pt` by `h_pt`."""
    if len(bbox01) != 4:
        raise ValueError(f"a box needs 4 numbers, got {len(bbox01)}")
    x0, y0, x1, y1 = bbox01
    return [x0 * w_pt, y0 * h_pt, x1 * w_pt, y1 * h_pt]


def _as_dict(record: Any) -> dict[str, Any]:
    if is_dataclass(record) and not isinstance(record, type):
        return asdict(record)
    if isinstance(record, Mapping):
        return dict(record)
    raise TypeError(f"cannot write {type(record).__name__} as a JSONL record")
