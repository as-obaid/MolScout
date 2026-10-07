"""What the harness does for paper datasets (BioVista, Internal): find and pin the PDFs, then check what run.py wrote.

The paper loop (benchmarks/tools/paper_runner.py) writes predictions.csv, predictions.errors.json and
predictions.timing.json; the checks here are the paper twins of the crop checks in harness.py. The
timing file is the proof that every paper was run, so a paper with no molecules has no row.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

from molscout.bench import BenchError
from molscout.bench.config import RunConfig
from molscout.data.biovista_truth import BIOVISTA_PAPERS_PATH, scored_papers
from molscout.data.internal import load_internal_split
from molscout.hashing import sha256_file
from molscout.predictions import Prediction, PredictionsFormatError, read_predictions
from molscout.runs import read_paper_seconds

TIMING_FILE = "predictions.timing.json"  # what benchmarks/tools/paper_runner.py writes beside predictions.csv
ERRORS_KEYS = frozenset({"papers", "failed", "errors"})
EXAMPLES_SHOWN = 10


def paper_pdfs(config: RunConfig) -> dict[str, Path]:
    """Paper ID to its PDF, in paper order; a missing PDF, or a BioVista PDF that is not the frozen copy, is a BenchError."""
    run = config.run_name
    if config.pdfs is None or config.papers is None:
        raise BenchError(f"{run}: a paper dataset needs pdfs and papers in its config")
    if not config.papers.is_file():
        raise BenchError(f"{run}: papers file not found: {config.papers}")
    if not config.pdfs.is_dir():
        raise BenchError(f"{run}: pdfs folder not found: {config.pdfs}")
    pins: dict[str, str] = {}
    if config.dataset == "biovista":
        names = {}
        for paper in scored_papers(config.papers):
            names[paper.paper_id] = f"{paper.pdb_id}.pdf"
            pins[paper.paper_id] = paper.sha256
    else:
        names = {paper: f"{paper}.pdf" for paper in load_internal_split(config.papers).split_of}
    pdfs = {paper: config.pdfs / name for paper, name in names.items()}
    for paper, path in pdfs.items():
        if not path.is_file():
            raise BenchError(f"{run}: paper {paper} has no PDF at {path}")
        if paper in pins and (actual := sha256_file(path)) != pins[paper]:
            raise BenchError(
                f"{run}: {paper} PDF {path} has sha256 {actual}, but {BIOVISTA_PAPERS_PATH.as_posix()} pins "
                f"{pins[paper]}; restore the frozen copy (never re-fetch BioVista)"
            )
    return pdfs


def paper_inputs(config: RunConfig, pdfs: Mapping[str, Path]) -> dict[str, object]:
    """meta.json's `inputs` for a paper run: the PDF folder and the paper list, each with a sha256."""
    assert config.pdfs is not None and config.papers is not None
    lines = sorted(f"{sha256_file(path)}  {paper}\n" for paper, path in pdfs.items())
    return {
        "pdfs": str(config.pdfs),
        "pdfs_sha256": hashlib.sha256("".join(lines).encode("utf-8")).hexdigest(),
        "papers": str(config.papers),
        "papers_sha256": sha256_file(config.papers),
    }


def write_paper_list(path: Path, pdfs: Mapping[str, Path]) -> None:
    """The `--papers` CSV (paper_id,pdf) with absolute PDF paths."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["paper_id", "pdf"])
        writer.writerows((paper, str(pdf.absolute())) for paper, pdf in pdfs.items())


def read_timing(path: Path, run: str, paper_ids: Collection[str]) -> dict[str, float]:
    """Seconds per paper from run.py's timing file, which must exist and name exactly the run's papers."""
    if not path.is_file():
        raise BenchError(f"{run}: run.py exited with status 0 but wrote no {path.name}")
    try:
        seconds = read_paper_seconds(path)
    except (ValueError, UnicodeDecodeError) as exc:
        raise BenchError(f"{run}: run.py's {path.name} is invalid: {exc}") from None
    missing = sorted(set(paper_ids) - set(seconds))
    if missing:
        raise BenchError(
            f"{run}: {path.name} misses {len(missing)} of {len(paper_ids)} papers, e.g. {_examples(missing)}; "
            "a run that stopped early is not scored"
        )
    extra = sorted(set(seconds) - set(paper_ids))
    if extra:
        raise BenchError(f"{run}: {path.name} times {len(extra)} paper(s) not in the run, e.g. {_examples(extra)}")
    return {paper: seconds[paper] for paper in paper_ids}


def read_paper_errors(path: Path, run: str, paper_ids: Collection[str]) -> int | None:
    """The number of papers whose predict call raised, from paper_runner's errors file; None without one.

    The file is `{"papers": n, "failed": m, "errors": {paper ID: "<Error>: <message>"}}`; its paper
    IDs must be papers of this run.
    """
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchError(f"{run}: run.py's {path.name} is not valid JSON: {exc}") from None
    if not (
        isinstance(data, dict)
        and set(data) == ERRORS_KEYS
        and all(isinstance(data[key], int) and not isinstance(data[key], bool) for key in ("papers", "failed"))
        and isinstance(data["errors"], dict)
        and all(isinstance(message, str) for message in data["errors"].values())
    ):
        raise BenchError(
            f"{run}: {path.name} must be a JSON object with papers, failed and errors "
            '({"papers": n, "failed": m, "errors": {paper ID: message}})'
        )
    errors = data["errors"]
    if data["failed"] != len(errors):
        raise BenchError(f"{run}: in {path.name}, failed is {data['failed']}, but errors lists {len(errors)} paper(s)")
    if data["papers"] != len(paper_ids):
        raise BenchError(f"{run}: in {path.name}, papers is {data['papers']}, but the run has {len(paper_ids)} papers")
    unknown = sorted(set(errors) - set(paper_ids))
    if unknown:
        raise BenchError(f"{run}: {path.name} lists {len(unknown)} paper(s) not in the run, e.g. {_examples(unknown)}")
    return data["failed"]


def checked_paper_predictions(path: Path, config: RunConfig, paper_ids: Collection[str]) -> tuple[Prediction, ...]:
    """Read run.py's predictions.csv; it must be valid, for this dataset and tool, with rows only for the run's papers.

    Completeness is not checked here: a paper with no molecules has no row, and the timing file shows it was run.
    """
    run = config.run_name
    if not path.is_file():
        raise BenchError(f"{run}: run.py exited with status 0 but wrote no predictions.csv")
    try:
        predictions = read_predictions(path)
    except PredictionsFormatError as exc:
        raise BenchError(f"{run}: run.py wrote an invalid predictions.csv: {exc}") from None
    datasets = sorted({p.dataset for p in predictions} - {config.dataset})
    if datasets:
        raise BenchError(f"{run}: rows name dataset {datasets[0]!r}, but the config's dataset is {config.dataset!r}")
    tools = sorted({p.tool for p in predictions} - {config.tool_label})
    if tools:
        raise BenchError(f"{run}: rows name tool {tools[0]!r}, but the config's tool is {config.tool_label!r}")
    extra = sorted({p.item_id for p in predictions} - set(paper_ids))
    if extra:
        raise BenchError(
            f"{run}: predictions.csv has {len(extra)} paper(s) that are not in the run, e.g. {_examples(extra)}"
        )
    return predictions


def check_scored_papers(report: Mapping[str, object], run: str, paper_ids: Collection[str]) -> None:
    """The papers the report scored must be exactly the run's papers."""
    scored = set(report["scores"]["papers"])  # type: ignore[index]
    if scored != set(paper_ids):
        differ = sorted(scored ^ set(paper_ids))
        raise BenchError(
            f"{run}: the scored papers differ from the run's papers in {len(differ)} paper(s), e.g. {_examples(differ)}; "
            "check that the config's papers and references are the same dataset"
        )


def _examples(items: Sequence[str]) -> str:
    return ", ".join(items[:EXAMPLES_SHOWN])
