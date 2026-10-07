"""Shared loop for the run.py of every complete system: PDFs in, predictions.csv out.

The twin of crop_runner for whole papers. A paper yields any number of molecules, each with an
optional page and bounding box, and the loop writes them as rows of the eight predictions.csv
columns. Beside predictions.csv it writes predictions.errors.json (the papers whose predict call
raised) and predictions.timing.json (seconds per paper, failed ones included). With --resume it
also keeps each finished paper in a checkpoint CSV plus <checkpoint stem>.papers.jsonl, so a
stopped run can carry on. Standard library only, Python 3.8+, so tool environments can import it
without molscout. It reuses crop_runner's helpers; tools put benchmarks/tools on sys.path.
"""

import argparse
import csv
import io
import json
import math
import operator
import os
import signal
import sys
from pathlib import Path
from time import perf_counter
from typing import NamedTuple, Optional, Tuple

from crop_runner import COLUMNS, _confidence, _describe, _exit_143, _read_rows, _replace

MAX_CONSECUTIVE_FAILURES = 5
PAPER_COLUMNS = ["paper_id", "pdf"]


class Paper(NamedTuple):
    paper_id: str
    pdf: Path


class Molecule(NamedTuple):
    """One structure found in a paper; bbox is (x0, y0, x1, y1) in PDF points, origin top-left."""

    smiles: Optional[str]
    page: Optional[int] = None  # 1-based
    bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: Optional[float] = None


def base_parser(description):
    """Argument parser with the arguments the harness always passes."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--papers", type=Path, required=True, help="CSV with columns paper_id,pdf")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tool", required=True, help='"<name> <version>"')
    parser.add_argument("--output", type=Path, required=True, help="predictions.csv to write")
    parser.add_argument(
        "--resume", type=Path, metavar="CHECKPOINT",
        help="CSV of finished rows: papers finished in it are skipped, and each new paper is appended to it",
    )
    return parser


def read_papers(path):
    """The papers listed in a paper_id,pdf CSV, in file order."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] != PAPER_COLUMNS:
        raise ValueError(f"{path}: the header must be {','.join(PAPER_COLUMNS)}, got {rows[0] if rows else 'nothing'!r}")
    papers, seen = [], set()
    for row in rows[1:]:
        if len(row) != 2 or not row[0].strip():
            raise ValueError(f"{path}: a row needs a paper ID and a PDF path, got {row!r}")
        paper_id, pdf = row[0], Path(row[1])
        if paper_id in seen:
            raise ValueError(f"{path}: paper ID {paper_id!r} appears more than once")
        if not pdf.is_file():
            raise ValueError(f"{path}: the PDF for {paper_id!r} is not a file: {pdf}")
        seen.add(paper_id)
        papers.append(Paper(paper_id, pdf))
    return papers


def errors_path(output):
    """The errors file beside an output file: predictions.csv -> predictions.errors.json."""
    output = Path(output)
    return output.with_name(output.stem + ".errors.json")


def timing_path(output):
    """The timing file beside an output file: predictions.csv -> predictions.timing.json."""
    output = Path(output)
    return output.with_name(output.stem + ".timing.json")


def papers_log_path(checkpoint):
    """The finished-papers log beside a checkpoint: checkpoint.csv -> checkpoint.papers.jsonl."""
    checkpoint = Path(checkpoint)
    return checkpoint.with_name(checkpoint.stem + ".papers.jsonl")


def run_papers(predict, args, *, warmup=True):
    """Run predict(paper) -> iterable of Molecule (or 4-tuples) on every paper; return how many raised.

    A paper that raises, or whose molecules fail the page and bbox checks, gets no rows and an
    entry in the errors file; its time is in the timing file either way. After
    MAX_CONSECUTIVE_FAILURES in a row the runtime is taken to be broken: RuntimeError, and none
    of the three files is written.

    With --resume, each finished paper appends its rows to the checkpoint CSV and then one line
    {"item_id", "seconds", "rows", "error"} to <checkpoint stem>.papers.jsonl; that line marks the
    paper done. Failures in a row are held back until a paper succeeds, so the papers behind a
    stop for failures run again. SIGTERM exits with status 143. The checkpoint is never deleted.
    """
    papers = read_papers(args.papers)
    checkpoint = getattr(args, "resume", None)
    if not checkpoint:
        return _run(predict, args, papers, {}, lambda finished: None, warmup)
    checkpoint = Path(checkpoint)
    previous = signal.signal(signal.SIGTERM, _exit_143)
    try:
        done = _resume(checkpoint, papers)
        log = papers_log_path(checkpoint)
        with checkpoint.open("a", newline="", encoding="utf-8") as rows, log.open("a", encoding="utf-8") as lines:
            writer = csv.writer(rows)

            def append(finished):
                writer.writerows(finished["rows"])
                rows.flush()
                lines.write(_log_line(finished) + "\n")  # the line is what marks the paper done
                lines.flush()

            return _run(predict, args, papers, done, append, warmup)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)


def _run(predict, args, papers, done, append, warmup):
    """run_papers after the papers already done (paper ID -> finished record) are known."""
    output = Path(args.output)
    part = Path(f"{output}.part")
    if done:
        print(f"resuming: {len(done)} of {len(papers)} papers already done", file=sys.stderr)
    first = next((paper for paper in papers if paper.paper_id not in done), None)
    if warmup and first is not None:
        try:
            list(predict(first))
        except Exception as exc:
            print(f"warmup on {first.paper_id} failed: {_describe(exc)}", file=sys.stderr)
    results = {}
    streak = []  # the latest failures in a row: appended only once a paper succeeds
    try:
        with part.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            for count, paper in enumerate(papers, 1):
                finished = done.get(paper.paper_id)
                if finished is None:
                    finished = _predict(predict, paper, args)
                    streak.append(finished)
                    if finished["error"] is None:
                        for held in streak:
                            append(held)
                        streak = []
                    else:
                        print(f"{paper.paper_id}: {finished['error']}", file=sys.stderr)
                        if len(streak) >= MAX_CONSECUTIVE_FAILURES:
                            raise RuntimeError(
                                "{} papers failed in a row, the last {}: {}; stopping, since the tool's runtime "
                                "looks broken".format(len(streak), paper.paper_id, finished["error"])
                            )
                results[paper.paper_id] = finished
                writer.writerows(finished["rows"])
                print(f"{count}/{len(papers)} papers", file=sys.stderr)
        for held in streak:
            append(held)
        failed = {pid: r["error"] for pid, r in results.items() if r["error"] is not None}
        timing = {"papers": len(papers), "seconds": {pid: r["seconds"] for pid, r in results.items()}}
        _replace(timing_path(output), json.dumps(timing, indent=2) + "\n")
        report = {"papers": len(papers), "failed": len(failed), "errors": failed}
        _replace(errors_path(output), json.dumps(report, indent=2) + "\n")
        os.replace(part, output)
    except BaseException:
        if part.exists():
            part.unlink()
        raise
    print(f"{len(papers)} papers, {len(failed)} failed", file=sys.stderr)
    return len(failed)


def _predict(predict, paper, args):
    """The finished record of one paper: its rows, seconds and error (None when it did not raise)."""
    start = perf_counter()
    try:
        molecules = [_check(Molecule(*molecule)) for molecule in predict(paper)]
        error = None
    except Exception as exc:
        molecules, error = [], _describe(exc)
    seconds = max(0.0, perf_counter() - start)
    rows = [
        [
            args.dataset, paper.paper_id, "" if m.smiles is None else str(m.smiles),
            "" if m.page is None else str(m.page),
            "" if m.bbox is None else ",".join(repr(float(v)) for v in m.bbox),
            _confidence(m.confidence), args.tool, repr(seconds),
        ]
        for m in molecules
    ]
    return {"item_id": paper.paper_id, "seconds": seconds, "rows": rows, "error": error}


def _check(molecule):
    """The molecule (its page as a plain int), or ValueError when its page or bbox is not one the scorer can read."""
    page, bbox = molecule.page, molecule.bbox
    if page is not None:
        # numpy integers and the like pass through operator.index; bool, floats and text do not
        if isinstance(page, bool) or not hasattr(page, "__index__") or operator.index(page) < 1:
            raise ValueError(f"page must be an integer >= 1, got {page!r}")
        molecule = molecule._replace(page=operator.index(page))
    if bbox is not None:
        if len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox):
            raise ValueError(f"bbox must be four finite numbers, got {bbox!r}")
        if float(bbox[0]) > float(bbox[2]) or float(bbox[1]) > float(bbox[3]):
            raise ValueError(f"bbox must have x0 <= x1 and y0 <= y1, got {bbox!r}")
    return molecule


def _log_line(finished):
    return json.dumps(
        {"item_id": finished["item_id"], "seconds": finished["seconds"],
         "rows": len(finished["rows"]), "error": finished["error"]}
    )


def _resume(checkpoint, papers):
    """The papers already done in a checkpoint (paper ID -> finished record).

    A paper is done when its log line is complete and its row count equals the rows the CSV holds
    for it. Rows of papers not done and a last line that a kill cut short are dropped, and both
    files are rewritten to match, header first. A row or line for a paper not in the list is a
    ValueError.
    """
    ids = {paper.paper_id for paper in papers}
    held = {}
    for row in _read_rows(checkpoint):
        if len(row) != len(COLUMNS):
            raise ValueError(f"{checkpoint}: a row has {len(row)} fields instead of {len(COLUMNS)}: {row!r}")
        if row[1] not in ids:
            raise ValueError(f"{checkpoint} has a row for paper {row[1]!r}, which is not in the papers list")
        held.setdefault(row[1], []).append(row)
    log = papers_log_path(checkpoint)
    lines = log.read_text(encoding="utf-8").split("\n")[:-1] if log.exists() else []
    entries = {}
    for line in lines:  # split leaves "" after the last newline, or a line cut short: [:-1] drops it
        entry = json.loads(line)
        if entry["item_id"] not in ids:
            raise ValueError(f"{log} has a line for paper {entry['item_id']!r}, which is not in the papers list")
        entries[entry["item_id"]] = entry
    done = {}
    for paper in papers:
        entry = entries.get(paper.paper_id)
        rows = held.get(paper.paper_id, [])
        if entry is not None and entry["rows"] == len(rows):
            done[paper.paper_id] = {
                "item_id": paper.paper_id, "seconds": entry["seconds"], "rows": rows, "error": entry["error"],
            }
    table = io.StringIO()
    csv.writer(table).writerows([COLUMNS, *(row for finished in done.values() for row in finished["rows"])])
    _replace(checkpoint, table.getvalue())
    _replace(log, "".join(_log_line(finished) + "\n" for finished in done.values()))
    return done
