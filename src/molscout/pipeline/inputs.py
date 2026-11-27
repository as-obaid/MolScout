"""The papers a run reads: one PDF, a folder of PDFs, or a `paper_id,pdf` CSV (the harness's papers.csv)."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from molscout.hashing import sha256_file

PAPER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
CSV_COLUMNS = ["paper_id", "pdf"]


@dataclass(frozen=True, slots=True)
class Paper:
    """One paper: its ID (a folder name under <out>/papers/) and its PDF."""

    paper_id: str
    pdf: Path

    def sha256(self) -> str:
        return sha256_file(self.pdf)


def resolve_papers(source: str | Path, *, recursive: bool = False) -> tuple[Paper, ...]:
    """The papers named by `source`, in order; ValueError when it names none or a paper is unusable.

    A PDF is one paper named by its file stem; a folder gives every `*.pdf` in it, sorted by name (with
    `recursive`, in its subfolders too, sorted by path; two PDFs with one stem are an error); a CSV with the
    header `paper_id,pdf` lists papers, and a relative PDF path in it starts at the CSV's folder.
    """
    source = Path(source)
    if source.is_dir():
        found = source.rglob("*") if recursive else source.iterdir()
        pdfs = sorted(path for path in found if path.suffix.lower() == ".pdf" and path.is_file())
        if not pdfs:
            raise ValueError(f"{source}: no PDF files in this folder" + (" or its subfolders" if recursive else ""))
        papers = [Paper(path.stem, path.absolute()) for path in pdfs]
    elif source.suffix.lower() == ".csv":
        papers = _from_csv(source)
    elif source.suffix.lower() == ".pdf":
        if not source.is_file():
            raise ValueError(f"{source}: no such PDF")
        papers = [Paper(source.stem, source.absolute())]
    else:
        raise ValueError(f"{source}: expected a PDF, a folder of PDFs or a paper_id,pdf CSV")
    return _checked(papers, source)


def _from_csv(path: Path) -> list[Paper]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] != CSV_COLUMNS:
        raise ValueError(f"{path}: the header must be {','.join(CSV_COLUMNS)}, got {rows[0] if rows else 'nothing'!r}")
    papers = []
    for line, row in enumerate(rows[1:], start=2):
        if len(row) != 2 or not row[0].strip() or not row[1].strip():
            raise ValueError(f"{path}: line {line}: a row needs a paper ID and a PDF path, got {row!r}")
        pdf = Path(row[1])
        papers.append(Paper(row[0].strip(), (pdf if pdf.is_absolute() else path.parent / pdf).absolute()))
    if not papers:
        raise ValueError(f"{path}: lists no papers")
    return papers


def _checked(papers: list[Paper], source: Path) -> tuple[Paper, ...]:
    seen: set[str] = set()
    for paper in papers:
        if not PAPER_ID.fullmatch(paper.paper_id):
            raise ValueError(f"{source}: paper ID {paper.paper_id!r} must be letters, digits, '.', '_' or '-'")
        if paper.paper_id in seen:
            raise ValueError(f"{source}: paper ID {paper.paper_id!r} appears more than once")
        if not paper.pdf.is_file():
            raise ValueError(f"{source}: the PDF of {paper.paper_id!r} is not a file: {paper.pdf}")
        seen.add(paper.paper_id)
    return tuple(papers)
