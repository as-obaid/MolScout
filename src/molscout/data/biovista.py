"""BioVista papers: the 500-paper list, each PDB entry's primary citation, and open-access PDFs.

BioVista names papers `<index>_<pdb id>` and ships no PDFs. For each paper this module looks up
the PDB entry's primary citation (RCSB), its PubMed record (Europe PMC), its copy in the PMC
Article Datasets (when it has a PMCID), its OpenAlex work and, when UNPAYWALL_EMAIL is set,
Unpaywall. It then downloads the first direct open-access PDF URL those sources offer, published
versions first, and records every paper in a committable CSV. Landing pages are never scraped.

A paper the open-access pass leaves without a PDF falls back to its pinned copy in
data/manifests/biovista_recovered.csv, if it has one (see molscout.data.recovered). A valid PDF
already at pdfs/<pdb id>.pdf is used as is; otherwise the pinned URL is downloaded. A copy whose
sha256 differs from the pinned one is still kept when it is a valid PDF, because repositories
regenerate cover pages, but its row says so in `detail` and the run logs a warning.
"""

from __future__ import annotations

import csv
import hashlib
import os
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from molscout.data import openaccess as oa
from molscout.data.httpclient import NetworkError, PoliteClient, Response
from molscout.data.recovered import RecoveredSource, read_recovered_sources
from molscout.hashing import sha256_file

PAPER_LIST = Path("bioactivity_extraction/data/file_names.txt")
SPLIT_LISTS = {"test": Path("test_set_names.txt"), "valid": Path("valid_set_names.txt")}
LABELS = Path("bioactivity_extraction/labels")
PDF_DIR = Path("pdfs")
STATUSES = (
    "ok",
    "no_doi",
    "no_oa",
    "html_not_pdf",
    "blocked",
    "not_pdf",
    "truncated_pdf",
    "http_error",
    "network_error",
    "too_large",
    "lookup_error",
)
MAX_WORKERS = 4
MAX_PDF_BYTES = 100 << 20
SAVE_EVERY = 20
TITLE_MATCH = 0.5
PDF_ACCEPT = "application/pdf,*/*;q=0.8"
PDF_TIMEOUT = 60.0  # per socket read, not for the whole file
RCSB_ENTRY = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
OPENALEX_WORK = "https://api.openalex.org/works/"
UNPAYWALL = "https://api.unpaywall.org/v2/"
HOST_INTERVALS = {
    "data.rcsb.org": 0.1,
    "www.ebi.ac.uk": 0.2,
    "europepmc.org": 1.0,
    "api.openalex.org": 0.15,
    "api.unpaywall.org": 0.2,
    "pmc-oa-opendata.s3.amazonaws.com": 0.1,
}
PUBLISHER_INTERVAL = 1.0
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Paper:
    paper_id: str
    index: int
    pdb_id: str
    split: str


@dataclass(frozen=True, slots=True)
class PaperRecord:
    """One row of data/manifests/biovista_papers.csv. Every field is the CSV's text."""

    paper_id: str
    pdb_id: str
    split: str = ""
    structures: str = ""
    doi: str = ""
    doi_source: str = ""
    pmid: str = ""
    pmcid: str = ""
    title: str = ""
    year: str = ""
    journal: str = ""
    title_match: str = ""
    status: str = ""
    source: str = ""
    oa_version: str = ""
    pdf_url: str = ""
    sha256: str = ""
    bytes: str = ""
    retrieved_at: str = ""
    detail: str = ""


COLUMNS = tuple(field.name for field in fields(PaperRecord))


def read_papers(root: str | Path) -> tuple[Paper, ...]:
    """The 500 papers in file_names.txt, in index order, each tagged with its split."""
    root = Path(root)
    names = _read_names(root / PAPER_LIST)
    split_of = {name: split for split, path in SPLIT_LISTS.items() for name in _read_names(root / path)}
    papers = []
    for name in names:
        index, _, pdb_id = name.partition("_")
        if not (index.isdigit() and len(pdb_id) == 4 and pdb_id.isalnum()):
            raise ValueError(f"{root / PAPER_LIST}: {name!r} is not <index>_<pdb id>")
        papers.append(Paper(name, int(index), pdb_id.lower(), split_of.get(name, "")))
    if len({paper.pdb_id for paper in papers}) != len(papers):
        raise ValueError(f"{root / PAPER_LIST}: a PDB ID is listed twice")
    return tuple(sorted(papers, key=lambda paper: paper.index))


def count_structures(root: str | Path, paper_id: str) -> str:
    """Ground-truth structure rows in labels/<paper_id>_structure.csv, or "" if the file is missing."""
    path = Path(root) / LABELS / f"{paper_id}_structure.csv"
    if not path.exists():
        return ""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return str(sum(1 for _ in csv.DictReader(handle)))


def read_paper_manifest(path: str | Path) -> dict[str, PaperRecord]:
    """Rows of a papers CSV keyed by PDB ID; {} if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"{path}: columns must be {', '.join(COLUMNS)}")
        records = [_record(row, path, reader.line_num) for row in reader]
    return {record.pdb_id: record for record in records}


def write_paper_manifest(path: str | Path, records: Iterable[PaperRecord]) -> None:
    """Write rows sorted by BioVista index, LF line ends, replacing the file atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(records, key=lambda record: (_index(record.paper_id), record.pdb_id))
    partial = path.with_name(path.name + ".part")
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        for record in rows:
            writer.writerow(getattr(record, column) for column in COLUMNS)
    partial.replace(path)


def is_fetched(record: PaperRecord, pdf_dir: Path) -> bool:
    """True if the row says ok and its PDF is on disk with the recorded sha256."""
    pdf = pdf_dir / f"{record.pdb_id}.pdf"
    return record.status == "ok" and pdf.is_file() and sha256_file(pdf) == record.sha256


def new_client() -> PoliteClient:
    return PoliteClient(intervals=HOST_INTERVALS, default_interval=PUBLISHER_INTERVAL)


def fetch_papers(
    root: str | Path,
    manifest_path: str | Path,
    *,
    recovered_path: str | Path | None = None,
    limit: int | None = None,
    workers: int = MAX_WORKERS,
    client: PoliteClient | None = None,
    unpaywall_email: str | None = None,
    log: Callable[[str], None] = print,
) -> Counter[str]:
    """Fetch every listed paper's PDF into root/pdfs/<pdb id>.pdf and update the papers CSV.

    Papers whose row is ok and whose PDF still matches its sha256 are skipped, so a rerun only
    retries the rest. After the open-access pass, each paper still without a PDF that has a
    pinned copy in `recovered_path` gets that copy. Returns the status count over the papers
    processed (the first `limit`).
    """
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer")
    root, manifest_path = Path(root), Path(manifest_path)
    every_paper = read_papers(root)
    papers = every_paper[:limit]
    pinned = _pinned_sources(recovered_path, every_paper)
    pdf_dir = root / PDF_DIR
    records = read_paper_manifest(manifest_path)
    todo = [paper for paper in papers if not _skip(records.get(paper.pdb_id), pdf_dir, log)]
    fetcher = PaperFetcher(client or new_client(), pdf_dir, unpaywall_email=unpaywall_email)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        jobs = [pool.submit(fetcher.fetch, paper, count_structures(root, paper.paper_id)) for paper in todo]
        for done, job in enumerate(as_completed(jobs), start=1):
            record = job.result()
            records[record.pdb_id] = record
            log(f"biovista: [{done}/{len(todo)}] {_summary(record)}")
            if done % SAVE_EVERY == 0:
                write_paper_manifest(manifest_path, records.values())
        _recover(fetcher, papers, records, pinned, log)
    finally:
        # On an error or Ctrl-C, drop queued papers instead of fetching them all first; keep what is done.
        pool.shutdown(wait=True, cancel_futures=True)
        write_paper_manifest(manifest_path, records.values())
    return Counter(records[paper.pdb_id].status for paper in papers if paper.pdb_id in records)


class PaperFetcher:
    """Resolves one paper's identifiers and downloads its best open-access PDF."""

    def __init__(
        self,
        client: PoliteClient,
        pdf_dir: Path,
        *,
        unpaywall_email: str | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._client = client
        self._pdf_dir = pdf_dir
        self._email = unpaywall_email
        self._now = now

    def fetch(self, paper: Paper, structures: str = "") -> PaperRecord:
        record = PaperRecord(paper.paper_id, paper.pdb_id, paper.split, structures)
        try:
            citation = self._rcsb(paper.pdb_id)
        except LookupFailed as exc:
            return replace(record, status="lookup_error", detail=str(exc))
        notes = Notes()
        epmc = self._quietly(notes, "europepmc", self._europepmc, citation)
        doi, doi_source = _choose_doi(citation, epmc, notes)
        pmid = citation.pmid or (epmc.pmid if epmc else "")
        work = self._quietly(notes, "openalex", self._openalex, doi, pmid)
        pmcid = (epmc.pmcid if epmc else "") or (work.pmcid if work else "")
        record = replace(
            record,
            doi=doi,
            doi_source=doi_source,
            pmid=pmid,
            pmcid=pmcid,
            title=citation.title,
            year=citation.year,
            journal=citation.journal,
            title_match=_title_match(citation.title, work, epmc),
        )
        if not (doi or pmid or pmcid):
            return replace(record, status="no_doi", detail="; ".join([citation.problem, *notes]))
        if not doi:
            notes.append(citation.problem)
        candidates = [
            *((self._quietly(notes, "pmc", self._pmc, pmcid) or []) if pmcid else []),
            *(work.candidates if work else ()),
            *(oa.europepmc_candidates(epmc) if epmc else ()),
            *((self._quietly(notes, "unpaywall", self._unpaywall, doi) or []) if self._email and doi else []),
        ]
        return self._download_first(record, oa.order_candidates(candidates), notes, work)

    def _download_first(
        self,
        record: PaperRecord,
        candidates: list[oa.Candidate],
        notes: Notes,
        work: oa.OpenAlexWork | None,
    ) -> PaperRecord:
        attempts: list[tuple[oa.Candidate, str, str]] = []
        for candidate in candidates:
            outcome = self._try(record, candidate)
            if isinstance(outcome, PaperRecord):
                return outcome
            attempts.append((candidate, *outcome))
        if not attempts:
            oa_status = f" (OpenAlex oa_status: {work.oa_status})" if work and work.oa_status else ""
            notes.insert(0, f"no open-access PDF URL from PMC, OpenAlex or Europe PMC{oa_status}")
            # A failed lookup may have hidden a PDF, so this is not a verdict of "no open access".
            status = "lookup_error" if notes.failed else "no_oa"
            return replace(record, status=status, detail="; ".join(notes))
        first, status, _ = attempts[0]
        tried = [f"{c.source}/{c.version or 'unknown'} {c.url}: {reason}" for c, _, reason in attempts]
        return replace(record, status=status, pdf_url=first.url, detail="; ".join([*tried, *notes]))

    def _try(self, record: PaperRecord, candidate: oa.Candidate) -> PaperRecord | tuple[str, str]:
        """A finished ok record, or (status, reason) for this candidate's failure."""
        body = self._download(candidate.url)
        if isinstance(body, tuple):
            return body
        sha256 = _save(self._pdf_dir / f"{record.pdb_id}.pdf", body)
        return replace(
            record,
            status="ok",
            source=candidate.source,
            oa_version=candidate.version,
            pdf_url=candidate.url,
            sha256=sha256,
            bytes=str(len(body)),
            retrieved_at=self._timestamp(),
        )

    def fetch_recovered(self, record: PaperRecord, source: RecoveredSource) -> PaperRecord:
        """`record` made ok from the paper's pinned copy, or `record` with the reason it was not.

        A valid PDF already at pdfs/<pdb id>.pdf is used without downloading. A failed download
        keeps the row's open-access status and adds the reason to its detail.
        """
        destination = self._pdf_dir / f"{record.pdb_id}.pdf"
        body, origin = _read_valid_pdf(destination), "already in pdfs/"
        if body is None:
            body, origin = self._download(source.url), "downloaded from the pinned URL"
            if isinstance(body, tuple):
                reason = f"recovered:{source.source_host} {source.url}: {body[1]}"
                return replace(record, detail="; ".join(filter(None, [record.detail, reason])))
            _save(destination, body)
        sha256 = hashlib.sha256(body).hexdigest()
        return replace(
            record,
            status="ok",
            source=f"recovered:{source.source_host}",
            oa_version=source.version,
            pdf_url=source.url,
            sha256=sha256,
            bytes=str(len(body)),
            retrieved_at=self._timestamp(),
            detail="" if sha256 == source.sha256 else _mismatch(source, origin),
        )

    def _download(self, url: str) -> bytes | tuple[str, str]:
        """The body of a complete PDF at url, or (status, reason) for why there is none."""
        try:
            response = self._client.get(url, accept=PDF_ACCEPT, max_bytes=MAX_PDF_BYTES, timeout=PDF_TIMEOUT)
        except NetworkError as exc:
            return "network_error", str(exc)
        host = urllib.parse.urlsplit(response.url).hostname
        kind = oa.classify_payload(response.body, response.headers.get("content-type", ""))
        if response.truncated:
            return "too_large", f"larger than {MAX_PDF_BYTES:,} bytes"
        if not response.ok:
            if kind == "blocked" or "cf-mitigated" in response.headers:
                return "blocked", f"HTTP {response.status} bot challenge from {host}"
            return "http_error", f"HTTP {response.status} from {host}"
        if kind == "truncated":
            return "truncated_pdf", f"PDF from {host} has no %%EOF trailer"
        if kind != "pdf":
            status = {"html": "html_not_pdf", "blocked": "blocked"}.get(kind, "not_pdf")
            return status, f"{kind} from {host}"
        return response.body

    def _timestamp(self) -> str:
        return self._now().strftime("%Y-%m-%dT%H:%M:%SZ")

    def _rcsb(self, pdb_id: str) -> oa.Citation:
        response = self._get_json_response(RCSB_ENTRY.format(pdb_id=pdb_id.upper()), "RCSB")
        if response.status == 404:
            raise LookupFailed(f"RCSB has no entry {pdb_id.upper()}")
        return oa.parse_rcsb_entry(_json(response, "RCSB"))

    def _europepmc(self, citation: oa.Citation) -> oa.EuropePmcRecord | None:
        if citation.pmid:
            query = f"EXT_ID:{citation.pmid} AND SRC:MED"
        elif citation.doi:
            unquoted = citation.doi.replace('"', "")  # a quote would end the phrase early
            query = f'DOI:"{unquoted}"'
        else:
            return None
        params = urllib.parse.urlencode({"query": query, "resultType": "core", "format": "json", "pageSize": 1})
        response = self._get_json_response(EUROPEPMC_SEARCH + params, "Europe PMC")
        return oa.parse_europepmc_search(_json(response, "Europe PMC"))

    def _openalex(self, doi: str, pmid: str) -> oa.OpenAlexWork | None:
        keys = [f"doi:{urllib.parse.quote(doi, safe='/')}"] if doi else []
        keys += [f"pmid:{pmid}"] if pmid else []
        for key in keys:
            response = self._get_json_response(OPENALEX_WORK + key, "OpenAlex")
            if response.status != 404:
                return oa.parse_openalex_work(_json(response, "OpenAlex"))
        return None

    def _unpaywall(self, doi: str) -> list[oa.Candidate]:
        # The request URL carries the email, so no message below may include it.
        query = urllib.parse.urlencode({"email": self._email})
        try:
            response = self._client.get(f"{UNPAYWALL}{urllib.parse.quote(doi, safe='/')}?{query}")
        except NetworkError as exc:
            raise LookupFailed(f"Unpaywall lookup failed: {exc}") from exc
        if response.status == 404:
            return []
        if not response.ok:
            raise LookupFailed(f"Unpaywall lookup failed: HTTP {response.status}")
        return oa.unpaywall_candidates(_json(response, "Unpaywall"))

    def _pmc(self, pmcid: str) -> list[oa.Candidate]:
        listing_url = f"{oa.PMC_BUCKET_URL}?list-type=2&prefix={pmcid}."
        listing = self._get_json_response(listing_url, "PMC", accept="application/xml")
        try:
            key = oa.pmc_metadata_key(listing.body, pmcid)
        except ET.ParseError as exc:
            raise LookupFailed(f"PMC listing is not XML: {exc}") from exc
        if not key:
            return []
        return oa.pmc_candidates(_json(self._get_json_response(oa.PMC_BUCKET_URL + key, "PMC"), "PMC"))

    def _get_json_response(self, url: str, service: str, *, accept: str = "application/json") -> Response:
        try:
            response = self._client.get(url, accept=accept)
        except NetworkError as exc:
            raise LookupFailed(f"{service} lookup failed: {exc}") from exc
        if not response.ok and response.status != 404:
            raise LookupFailed(f"{service} lookup failed: HTTP {response.status}")
        return response

    @staticmethod
    def _quietly(notes: Notes, service: str, lookup: Callable[..., T], *args: object) -> T | None:
        """Run an optional lookup; on failure note it and carry on without that source."""
        try:
            return lookup(*args)
        except LookupFailed as exc:
            notes.append(f"{service}: {exc}")
            notes.failed = True
            return None


class Notes(list[str]):
    """Remarks for a row's detail column; `failed` is set once any lookup has failed."""

    failed: bool = False


class LookupFailed(RuntimeError):
    """A metadata API could not be reached or answered with an error."""


def unpaywall_email_from_env(environ: Mapping[str, str] = os.environ) -> str | None:
    """UNPAYWALL_EMAIL if set and plausible; Unpaywall is skipped otherwise."""
    email = environ.get("UNPAYWALL_EMAIL", "").strip()
    return email if "@" in email else None


def _choose_doi(citation: oa.Citation, epmc: oa.EuropePmcRecord | None, notes: list[str]) -> tuple[str, str]:
    """RCSB's DOI, unless the PubMed record of RCSB's PMID gives a different one, which wins."""
    pubmed_doi = epmc.doi if epmc and citation.pmid else ""
    if pubmed_doi and pubmed_doi != citation.doi:
        notes.append(f"RCSB DOI {citation.doi or 'missing'}; used PubMed's {pubmed_doi}")
        return pubmed_doi, "pubmed"
    return (citation.doi, "rcsb") if citation.doi else ("", "")


def _title_match(title: str, work: oa.OpenAlexWork | None, epmc: oa.EuropePmcRecord | None) -> str:
    other = (work.title if work else "") or (epmc.title if epmc else "")
    if not (title and other):
        return ""
    return "yes" if oa.title_similarity(title, other) >= TITLE_MATCH else "no"


def _pinned_sources(path: str | Path | None, papers: Iterable[Paper]) -> dict[str, RecoveredSource]:
    """The pinned copies in `path` ({} without one); each must name a listed paper."""
    if path is None:
        return {}
    pinned = read_recovered_sources(path)
    listed = {paper.pdb_id: paper.paper_id for paper in papers}
    for source in pinned.values():
        if listed.get(source.pdb_id) != source.paper_id:
            raise ValueError(f"{path}: {source.paper_id} is not a BioVista paper")
    return pinned


def _recover(
    fetcher: PaperFetcher,
    papers: Iterable[Paper],
    records: dict[str, PaperRecord],
    pinned: Mapping[str, RecoveredSource],
    log: Callable[[str], None],
) -> None:
    """Give each paper still without a PDF its pinned copy, one paper at a time."""
    todo = [
        (records[paper.pdb_id], pinned[paper.pdb_id])
        for paper in papers
        if paper.pdb_id in pinned and paper.pdb_id in records and records[paper.pdb_id].status != "ok"
    ]
    for done, (record, source) in enumerate(todo, start=1):
        result = fetcher.fetch_recovered(record, source)
        records[result.pdb_id] = result
        if result.status != "ok":
            # fetch_recovered appended the pinned copy's failure; _summary would cut it off.
            reason = result.detail.removeprefix(record.detail).removeprefix("; ")
            log(f"biovista: recovered [{done}/{len(todo)}] {result.paper_id} still {result.status}: {reason}")
            continue
        log(f"biovista: recovered [{done}/{len(todo)}] {_summary(result)}")
        if result.sha256 != source.sha256:
            log(f"biovista: warning: {result.paper_id}: {result.detail}")


def _read_valid_pdf(path: Path) -> bytes | None:
    """The bytes of path if it is a complete PDF no larger than a download may be, else None."""
    if not path.is_file() or path.stat().st_size > MAX_PDF_BYTES:
        return None
    body = path.read_bytes()
    return body if oa.classify_payload(body) == "pdf" else None


def _mismatch(source: RecoveredSource, origin: str) -> str:
    return f"sha256 mismatch: pinned copy is {source.sha256} ({source.bytes} bytes); kept the valid PDF {origin}"


def _skip(record: PaperRecord | None, pdf_dir: Path, log: Callable[[str], None]) -> bool:
    if record is not None and is_fetched(record, pdf_dir):
        log(f"biovista: {record.paper_id} already present, checksum ok")
        return True
    return False


def _save(destination: Path, body: bytes) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    try:
        partial.write_bytes(body)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return hashlib.sha256(body).hexdigest()


def _json(response: Response, service: str) -> dict:
    try:
        data = response.json()
    except (UnicodeDecodeError, ValueError) as exc:
        raise LookupFailed(f"{service} answered with invalid JSON") from exc
    if not isinstance(data, dict):
        raise LookupFailed(f"{service} answered with unexpected JSON")
    return data


def _record(row: dict[str | None, str | None], path: Path, line: int) -> PaperRecord:
    if None in row or None in row.values():
        raise ValueError(f"{path}: line {line}: expected {len(COLUMNS)} fields")
    record = PaperRecord(**row)  # type: ignore[arg-type]
    if record.status not in STATUSES:
        raise ValueError(f"{path}: line {line}: unknown status {record.status!r}")
    if record.status == "ok" and not record.sha256:
        raise ValueError(f"{path}: line {line}: an ok row needs a sha256")
    return record


def _read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _index(paper_id: str) -> int:
    head = paper_id.partition("_")[0]
    return int(head) if head.isdigit() else -1


def _summary(record: PaperRecord) -> str:
    if record.status == "ok":
        version = record.oa_version or "unknown version"
        return f"{record.paper_id} ok ({record.source}, {version}, {int(record.bytes):,} bytes)"
    return f"{record.paper_id} {record.status}: {record.detail[:160]}"
