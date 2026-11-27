"""The BioVista development papers: 20 of the 145 benchmark papers, frozen before any MolScout BioVista run.

MolScout is tuned on these and on Internal papers 1, 16 and 19 only; the other 125 BioVista papers are held out.
The 20 are drawn with a fixed seed from two strata of the benchmark papers, in manifest order: table-heavy papers,
where most labels are enumerated (backbone other than `NA`, so they come from an R-group table), and drawn-heavy
papers, the rest. `scripts/make_biovista_dev.py` regenerates data/manifests/biovista_dev.csv from the labels and
checks that the committed file matches.
"""

from __future__ import annotations

import csv
import random
from dataclasses import astuple, dataclass, fields
from pathlib import Path

from molscout.data.biovista_truth import BioVistaTruth

BIOVISTA_DEV_PATH = Path("data/manifests/biovista_dev.csv")
DEV_SEED = 6630
PER_KIND = 10
KINDS = ("table", "drawn")


@dataclass(frozen=True, slots=True)
class DevPaper:
    """One development paper: its kind and its label counts (drawn: backbone `NA`; enumerated: the rest)."""

    paper_id: str
    pdb_id: str
    kind: str
    labels: int
    drawn: int
    enumerated: int


COLUMNS = tuple(field.name for field in fields(DevPaper))


def paper_kind(*, drawn: int, enumerated: int) -> str:
    """`table` when most of a paper's labels are enumerated, else `drawn`."""
    return "table" if enumerated > drawn else "drawn"


def select_dev_papers(
    truth: BioVistaTruth, *, seed: int = DEV_SEED, per_kind: int = PER_KIND
) -> tuple[DevPaper, ...]:
    """`per_kind` table-heavy and `per_kind` drawn-heavy benchmark papers, sampled with `seed`, in manifest order.

    One random.Random(seed) samples the table-heavy papers first, then the drawn-heavy ones, each from its
    papers in manifest order; ValueError when a kind has fewer than `per_kind` papers.
    """
    pools: dict[str, list[DevPaper]] = {kind: [] for kind in KINDS}
    for paper in truth.papers:
        labels = len(truth.references[paper.paper_id])
        enumerated = len(truth.enumerated[paper.paper_id])
        drawn = labels - enumerated
        kind = paper_kind(drawn=drawn, enumerated=enumerated)
        pools[kind].append(DevPaper(paper.paper_id, paper.pdb_id, kind, labels, drawn, enumerated))
    rng = random.Random(seed)
    chosen: set[str] = set()
    for kind in KINDS:
        pool = pools[kind]
        if len(pool) < per_kind:
            raise ValueError(f"only {len(pool)} {kind}-heavy papers; {per_kind} are needed")
        chosen.update(paper.paper_id for paper in rng.sample(pool, per_kind))
    order = {paper.paper_id: i for i, paper in enumerate(truth.papers)}
    picked = [paper for pool in pools.values() for paper in pool if paper.paper_id in chosen]
    return tuple(sorted(picked, key=lambda paper: order[paper.paper_id]))


def write_dev_manifest(path: str | Path, papers: tuple[DevPaper, ...]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(astuple(paper) for paper in papers)


def read_dev_papers(
    path: str | Path = BIOVISTA_DEV_PATH, *, per_kind: int | None = PER_KIND
) -> tuple[DevPaper, ...]:
    """The development papers; ValueError naming the file and line for anything malformed.

    `per_kind` is how many papers of each kind the file must hold; None skips that check (the scorer reads
    a dev file beside any papers manifest, test fixtures included).
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"{path}: columns must be {','.join(COLUMNS)}, got {reader.fieldnames}")
        papers = [_dev_paper(row, f"{path}: line {reader.line_num}") for row in reader]
    ids = [paper.paper_id for paper in papers]
    repeated = sorted({paper for paper in ids if ids.count(paper) > 1})
    if repeated:
        raise ValueError(f"{path}: {', '.join(repeated)} listed more than once")
    counts = {kind: sum(paper.kind == kind for paper in papers) for kind in KINDS}
    if per_kind is not None and any(count != per_kind for count in counts.values()):
        raise ValueError(
            f"{path}: expected {per_kind} table-heavy and {per_kind} drawn-heavy papers, got {counts['table']} "
            f"and {counts['drawn']}"
        )
    return tuple(papers)


def dev_paper_ids(path: str | Path = BIOVISTA_DEV_PATH) -> frozenset[str]:
    return frozenset(paper.paper_id for paper in read_dev_papers(path, per_kind=None))


def dev_manifest_for(papers_manifest: str | Path) -> Path | None:
    """The dev manifest beside a BioVista papers manifest (`biovista_dev.csv`), or None when there is none."""
    path = Path(papers_manifest).with_name(BIOVISTA_DEV_PATH.name)
    return path if path.is_file() else None


def _dev_paper(row: dict[str, str], where: str) -> DevPaper:
    numbers = {}
    for name in ("labels", "drawn", "enumerated"):
        text = (row[name] or "").strip()
        if not text.isdecimal():
            raise ValueError(f"{where}: {name} must be an integer, got {text!r}")
        numbers[name] = int(text)
    if row["kind"] not in KINDS:
        raise ValueError(f"{where}: kind must be one of {', '.join(KINDS)}, got {row['kind']!r}")
    if numbers["drawn"] + numbers["enumerated"] != numbers["labels"]:
        raise ValueError(f"{where}: drawn plus enumerated must equal labels")
    if paper_kind(drawn=numbers["drawn"], enumerated=numbers["enumerated"]) != row["kind"]:
        raise ValueError(f"{where}: kind {row['kind']} does not match its label counts")
    return DevPaper(row["paper_id"], row["pdb_id"], row["kind"], **numbers)
