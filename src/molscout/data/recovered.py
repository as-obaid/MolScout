"""Pinned PDF sources for BioVista papers that the open-access APIs do not reach.

Some papers have a legal free copy that no API links to directly: an author manuscript in a
university repository or OSTI, a HAL deposit, or a preprint. Each copy found by hand is pinned
in data/manifests/biovista_recovered.csv with its direct PDF URL, its version, the sha256 and
size of the copy that was checked against the paper's title, and a short note. The biovista
step fetches these only for papers the open-access pass leaves without a PDF.
"""

from __future__ import annotations

import csv
import re
import urllib.parse
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path

from molscout.data import openaccess as oa

SOURCE_TYPES = ("pmc", "europepmc", "repository", "author_site", "publisher_free", "preprint", "other")
TITLE_CHECKS = ("match", "partial", "no_text")
PAPER_ID = re.compile(r"(\d+)_([0-9a-z]{4})")
HOST = re.compile(r"[a-z0-9-]+(\.[a-z0-9-]+)+")
SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class RecoveredSource:
    """One row of data/manifests/biovista_recovered.csv."""

    paper_id: str
    pdb_id: str
    doi: str
    url: str
    source_host: str
    source_type: str
    version: str
    sha256: str
    bytes: int
    pages: int
    title_check: str
    notes: str


COLUMNS = tuple(field.name for field in fields(RecoveredSource))


def read_recovered_sources(path: str | Path) -> dict[str, RecoveredSource]:
    """Rows of a pinned-source CSV keyed by PDB ID; {} if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return {}
    sources: dict[str, RecoveredSource] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"{path}: columns must be {', '.join(COLUMNS)}")
        for row in reader:
            where = f"{path}: line {reader.line_num}"
            source = _source(row, where)
            if source.pdb_id in sources:
                raise ValueError(f"{where}: {source.pdb_id} is listed twice")
            sources[source.pdb_id] = source
    return sources


def write_recovered_sources(path: str | Path, sources: Iterable[RecoveredSource]) -> None:
    """Write rows sorted by BioVista index, LF line ends, replacing the file atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(sources, key=lambda source: (_index(source.paper_id), source.pdb_id))
    partial = path.with_name(path.name + ".part")
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        for source in rows:
            writer.writerow(getattr(source, column) for column in COLUMNS)
    partial.replace(path)


def _source(row: dict[str | None, str | None], where: str) -> RecoveredSource:
    if None in row or None in row.values():
        raise ValueError(f"{where}: expected {len(COLUMNS)} fields")
    text: dict[str, str] = {str(key): str(value) for key, value in row.items()}
    paper = PAPER_ID.fullmatch(text["paper_id"])
    if not paper or paper[2] != text["pdb_id"]:
        raise ValueError(f"{where}: paper_id {text['paper_id']!r} must be <index>_<pdb_id>, lower case")
    if text["doi"] and oa.normalize_doi(text["doi"]) != text["doi"]:
        raise ValueError(f"{where}: doi {text['doi']!r} is not a lower-case DOI")
    url = urllib.parse.urlsplit(text["url"])
    plain = text["url"].isascii() and text["url"].isprintable() and " " not in text["url"]
    if url.scheme != "https" or not url.hostname or url.username is not None or not plain:
        raise ValueError(f"{where}: url must be an https URL without spaces or credentials")
    host = text["source_host"]
    on_host = url.hostname == host or url.hostname.endswith("." + host)
    _check(where, "source_host", host, bool(HOST.fullmatch(host)) and on_host)
    _check(where, "source_type", text["source_type"], text["source_type"] in SOURCE_TYPES)
    _check(where, "version", text["version"], text["version"] in oa.VERSIONS)
    _check(where, "sha256", text["sha256"], bool(SHA256.fullmatch(text["sha256"])))
    _check(where, "bytes", text["bytes"], _positive(text["bytes"]))
    _check(where, "pages", text["pages"], _positive(text["pages"]))
    _check(where, "title_check", text["title_check"], text["title_check"] in TITLE_CHECKS)
    sizes = {"bytes": int(text["bytes"]), "pages": int(text["pages"])}
    return RecoveredSource(**{**text, **sizes})  # type: ignore[arg-type]


def _check(where: str, column: str, value: str, ok: bool) -> None:
    if not ok:
        raise ValueError(f"{where}: {column} {value!r} is not valid")


def _positive(text: str) -> bool:
    return text.isascii() and text.isdigit() and int(text) > 0


def _index(paper_id: str) -> int:
    head = paper_id.partition("_")[0]
    return int(head) if head.isdigit() else -1
