"""Read citation and open-access records from RCSB, Europe PMC, OpenAlex, PMC and Unpaywall.

Every function here is pure: it takes a decoded API response and returns plain values, so the
network code in molscout.data.biovista stays thin and these rules are unit-tested offline.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

DOI = re.compile(r"10\.\d{4,9}/\S+")
DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:")
VERSIONS = ("publishedVersion", "acceptedVersion", "submittedVersion")
# PMC Article Datasets first: an official open-access bucket with no bot wall in front of it.
SOURCES = ("pmc", "openalex", "europepmc", "unpaywall")
PMC_BUCKET_URL = "https://pmc-oa-opendata.s3.amazonaws.com/"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
HTML_MARKERS = (b"<!doctype html", b"<html", b"<head", b"<body", b"<meta ", b"<script")
BLOCK_MARKERS = (
    b"just a moment",
    b"cf-chl",
    b"challenge-platform",
    b"cf-browser-verification",
    b"attention required! | cloudflare",
    b"captcha",
    b"are you a robot",
    b"verify you are human",
)
PDF_MAGIC = b"%PDF-"
PDF_MAGIC_WINDOW = 1024
PDF_TRAILER = b"%%EOF"
PDF_TRAILER_WINDOW = 2048
LEADING_JUNK = b"\xef\xbb\xbf \t\r\n\x00"


@dataclass(frozen=True, slots=True)
class Citation:
    """A PDB entry's primary citation. `problem` says why there is no DOI when there is none."""

    doi: str = ""
    pmid: str = ""
    title: str = ""
    journal: str = ""
    year: str = ""
    problem: str = ""


@dataclass(frozen=True, slots=True)
class EuropePmcRecord:
    pmid: str
    pmcid: str
    doi: str
    title: str
    manuscript: bool
    pdf_urls: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OpenAlexWork:
    doi: str
    pmid: str
    pmcid: str
    title: str
    oa_status: str
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True, slots=True)
class Candidate:
    """One direct PDF URL an API offered, with where it came from and which version it is."""

    url: str
    source: str
    version: str


def normalize_doi(value: object) -> str:
    """Lower-case DOI without a resolver prefix, or "" if value is not a DOI."""
    if not isinstance(value, str):
        return ""
    doi = value.strip()
    for prefix in DOI_PREFIXES:
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix) :]
    doi = doi.strip().lower()
    return doi if DOI.fullmatch(doi) else ""


def normalize_pmcid(value: object) -> str:
    """"PMC1234567" from "PMC1234567", "1234567" or a PMC article URL; "" otherwise."""
    if value is None:
        return ""
    text = str(value).strip().rstrip("/")
    tail = text.rsplit("/", 1)[-1].upper()
    digits = tail.removeprefix("PMC")
    return f"PMC{digits}" if digits.isdigit() else ""


def normalize_pmid(value: object) -> str:
    if value is None or isinstance(value, bool):
        return ""
    tail = str(value).strip().rstrip("/").rsplit("/", 1)[-1]
    return tail if tail.isdigit() and int(tail) > 0 else ""


def parse_rcsb_entry(entry: Mapping[str, object]) -> Citation:
    """The primary citation in an RCSB Data API entry (GET /rest/v1/core/entry/{id})."""
    citation = entry.get("rcsb_primary_citation")
    if not isinstance(citation, Mapping):
        return Citation(problem="RCSB entry has no primary citation")
    raw_doi = citation.get("pdbx_database_id_DOI")
    doi = normalize_doi(raw_doi)
    journal = str(citation.get("rcsb_journal_abbrev") or citation.get("journal_abbrev") or "").strip()
    year = citation.get("year")
    fields = {
        "doi": doi,
        "pmid": normalize_pmid(citation.get("pdbx_database_id_PubMed")),
        "title": " ".join(str(citation.get("title") or "").split()),
        "journal": journal,
        "year": str(year) if isinstance(year, int) else "",
    }
    if doi:
        return Citation(**fields)
    if raw_doi:
        problem = f"RCSB DOI {str(raw_doi).strip()!r} is not a valid DOI"
    elif journal.lower() in {"to be published", "", "tbp"}:
        problem = f"RCSB primary citation is unpublished (journal: {journal or 'none'})"
    else:
        problem = "RCSB primary citation has no DOI"
    return Citation(**fields, problem=problem)


def parse_europepmc_search(payload: Mapping[str, object]) -> EuropePmcRecord | None:
    """The first hit of a Europe PMC REST search (resultType=core), or None if nothing matched."""
    results = _items(_mapping(payload.get("resultList")).get("result"))
    hit = _mapping(results[0]) if results else {}
    if not hit:
        return None
    urls = (_mapping(item) for item in _items(_mapping(hit.get("fullTextUrlList")).get("fullTextUrl")))
    pdf_urls = tuple(
        str(item["url"])
        for item in urls
        if item.get("site") == "Europe_PMC"
        and item.get("documentStyle") == "pdf"
        and item.get("availabilityCode") in {"OA", "F"}
        and _is_web_url(item.get("url"))
    )
    manuscript = any(hit.get(flag) == "Y" for flag in ("authMan", "epmcAuthMan", "nihAuthMan"))
    return EuropePmcRecord(
        pmid=normalize_pmid(hit.get("pmid")),
        pmcid=normalize_pmcid(hit.get("pmcid")),
        doi=normalize_doi(hit.get("doi")),
        title=_one_line(hit.get("title")),
        manuscript=manuscript,
        pdf_urls=pdf_urls,
    )


def europepmc_candidates(record: EuropePmcRecord) -> list[Candidate]:
    version = "acceptedVersion" if record.manuscript else "publishedVersion"
    return [Candidate(url, "europepmc", version) for url in record.pdf_urls]


def parse_openalex_work(work: Mapping[str, object]) -> OpenAlexWork:
    """Identifiers and open-access PDF locations of an OpenAlex work (best_oa_location first)."""
    ids = _mapping(work.get("ids"))
    locations = (_mapping(loc) for loc in [work.get("best_oa_location"), *_items(work.get("locations"))])
    candidates = tuple(
        Candidate(str(loc["pdf_url"]), "openalex", _version(loc.get("version")))
        for loc in locations
        if loc.get("is_oa") is True and _is_web_url(loc.get("pdf_url"))
    )
    return OpenAlexWork(
        doi=normalize_doi(work.get("doi")),
        pmid=normalize_pmid(ids.get("pmid")),
        pmcid=normalize_pmcid(ids.get("pmcid")),
        title=_one_line(work.get("title") or work.get("display_name")),
        oa_status=str(_mapping(work.get("open_access")).get("oa_status") or ""),
        candidates=candidates,
    )


def unpaywall_candidates(payload: Mapping[str, object]) -> list[Candidate]:
    locations = (_mapping(loc) for loc in [payload.get("best_oa_location"), *_items(payload.get("oa_locations"))])
    return [
        Candidate(str(loc["url_for_pdf"]), "unpaywall", _version(loc.get("version")))
        for loc in locations
        if _is_web_url(loc.get("url_for_pdf"))
    ]


def pmc_metadata_key(listing_xml: bytes, pmcid: str) -> str:
    """Key of the newest `PMCn.v/PMCn.v.json` in a PMC Article Datasets S3 listing, or ""."""
    pattern = re.compile(rf"{re.escape(pmcid)}\.(\d+)/{re.escape(pmcid)}\.\1\.json")
    root = ET.fromstring(listing_xml)
    versions = []
    for key in root.iter(f"{S3_NS}Key"):
        match = pattern.fullmatch(key.text or "")
        if match:
            versions.append((int(match[1]), match[0]))
    return max(versions)[1] if versions else ""


def pmc_candidates(metadata: Mapping[str, object]) -> list[Candidate]:
    """The PDF in a PMC Article Datasets metadata JSON, as an https URL into the public bucket."""
    pdf_url = metadata.get("pdf_url")
    prefix = "s3://pmc-oa-opendata/"
    if not isinstance(pdf_url, str) or not pdf_url.startswith(prefix):
        return []
    key = pdf_url[len(prefix) :].split("?", 1)[0]
    version = "acceptedVersion" if metadata.get("is_manuscript") else "publishedVersion"
    return [Candidate(PMC_BUCKET_URL + key, "pmc", version)]


def order_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Published before accepted before submitted before unknown; then by SOURCES order.

    The sort is stable, so each API's own order (OpenAlex's best location first) is kept within
    a tie. A URL offered twice is kept once, at its best position.
    """
    ranked = sorted(candidates, key=lambda c: (_rank(VERSIONS, c.version), _rank(SOURCES, c.source)))
    seen: set[str] = set()
    unique = []
    for candidate in ranked:
        if candidate.url not in seen:
            seen.add(candidate.url)
            unique.append(candidate)
    return unique


def classify_payload(body: bytes, content_type: str = "") -> str:
    """"pdf", "truncated", "blocked" (a bot or captcha challenge), "html", "empty" or "not_pdf".

    A PDF starts with %PDF- once leading whitespace or a byte-order mark is skipped; a body served
    as application/pdf may also have up to 1024 bytes of junk first, as PDF readers allow. A PDF
    without the %%EOF trailer in its last 2048 bytes was cut short.
    """
    if not body.strip(LEADING_JUNK):
        return "empty"
    if _starts_as_pdf(body, content_type):
        return "pdf" if PDF_TRAILER in body[-PDF_TRAILER_WINDOW:] else "truncated"
    lowered = body[: 64 << 10].lower()
    if any(marker in lowered for marker in BLOCK_MARKERS):
        return "blocked"
    if any(marker in lowered[:PDF_MAGIC_WINDOW] for marker in HTML_MARKERS) or "html" in content_type.lower():
        return "html"
    return "not_pdf"


def _starts_as_pdf(body: bytes, content_type: str) -> bool:
    if body.lstrip(LEADING_JUNK).startswith(PDF_MAGIC):
        return True
    magic = body.find(PDF_MAGIC, 0, PDF_MAGIC_WINDOW)
    if magic < 0 or "pdf" not in content_type.lower():
        return False
    before = body[:magic].lower()
    return not any(marker in before for marker in HTML_MARKERS)


def title_similarity(a: str, b: str) -> float:
    """Jaccard overlap of the two titles' lower-case alphanumeric words (0 to 1)."""
    words_a, words_b = _words(a), _words(b)
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _version(value: object) -> str:
    return value if isinstance(value, str) and value in VERSIONS else ""


def _rank(order: tuple[str, ...], value: str) -> int:
    return order.index(value) if value in order else len(order)


def _is_web_url(url: object) -> bool:
    return isinstance(url, str) and url.startswith(("https://", "http://"))


def _mapping(value: object) -> Mapping[str, object]:
    """value if it is a JSON object, else {}: API fields can be null or of an unexpected type."""
    return value if isinstance(value, Mapping) else {}


def _items(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _one_line(value: object) -> str:
    return " ".join(str(value).split()) if isinstance(value, str) else ""
