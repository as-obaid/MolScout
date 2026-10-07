import hashlib
import json
from datetime import UTC, datetime

import pytest

from molscout.data import biovista
from molscout.data.biovista import (
    COLUMNS,
    Paper,
    PaperFetcher,
    PaperRecord,
    count_structures,
    fetch_papers,
    is_fetched,
    read_paper_manifest,
    read_papers,
    unpaywall_email_from_env,
    write_paper_manifest,
)
from molscout.data.httpclient import NetworkError, Response

PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
PDF_SHA = hashlib.sha256(PDF).hexdigest()
NOW = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)


def write_lists(root, names=("2_6v1c", "1_6v5l", "10_6ueg"), test=("1_6v5l", "10_6ueg"), valid=("2_6v1c",)):
    (root / "bioactivity_extraction" / "data").mkdir(parents=True, exist_ok=True)
    (root / "bioactivity_extraction" / "data" / "file_names.txt").write_text("\n".join(names))
    (root / "test_set_names.txt").write_text("\n".join(test) + "\n")
    (root / "valid_set_names.txt").write_text("\n".join(valid))


def test_read_papers_sorts_by_index_and_tags_splits(tmp_path):
    write_lists(tmp_path)
    assert read_papers(tmp_path) == (
        Paper("1_6v5l", 1, "6v5l", "test"),
        Paper("2_6v1c", 2, "6v1c", "valid"),
        Paper("10_6ueg", 10, "6ueg", "test"),
    )


@pytest.mark.parametrize("names", [("1_6v5l", "x_6v1c"), ("1_6v5l", "2_6v5"), ("1_6v5l", "2_6v5l")])
def test_read_papers_rejects_bad_lists(tmp_path, names):
    write_lists(tmp_path, names=names)
    with pytest.raises(ValueError, match="file_names.txt"):
        read_papers(tmp_path)


def test_count_structures(tmp_path):
    labels = tmp_path / "bioactivity_extraction" / "labels"
    labels.mkdir(parents=True)
    (labels / "1_6v5l_structure.csv").write_text("﻿smiles,ligand,backbone,groups\nCCO,1,NA,NA\nc1ccccc1,2,NA,NA\n")
    (labels / "3_6uyz_structure.csv").write_text("smiles,ligand,backbone,groups\n")
    assert count_structures(tmp_path, "1_6v5l") == "2"
    assert count_structures(tmp_path, "3_6uyz") == "0"
    assert count_structures(tmp_path, "9_9zzz") == ""


def test_manifest_round_trip_is_sorted_with_lf_endings(tmp_path):
    path = tmp_path / "biovista_papers.csv"
    records = [
        PaperRecord("10_6ueg", "6ueg", status="no_oa", detail='says "no", twice; fine'),
        PaperRecord("2_6v1c", "6v1c", title="Trefoil, factors", status="ok", sha256=PDF_SHA),
        PaperRecord("1_6v5l", "6v5l", status="blocked"),
    ]
    write_paper_manifest(path, records)
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert raw.splitlines()[0].decode() == ",".join(COLUMNS)
    assert [line.split(b",")[0] for line in raw.splitlines()[1:]] == [b"1_6v5l", b"2_6v1c", b"10_6ueg"]
    assert read_paper_manifest(path) == {record.pdb_id: record for record in records}
    write_paper_manifest(path, reversed(records))
    assert path.read_bytes() == raw


def test_read_paper_manifest_checks_columns(tmp_path):
    path = tmp_path / "papers.csv"
    path.write_text("pdb_id,status\n6v5l,ok\n")
    with pytest.raises(ValueError, match="columns"):
        read_paper_manifest(path)
    assert read_paper_manifest(tmp_path / "missing.csv") == {}


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (PaperRecord("1_6v5l", "6v5l", status="maybe"), "line 2: unknown status 'maybe'"),
        (PaperRecord("1_6v5l", "6v5l", status="ok"), "line 2: an ok row needs a sha256"),
    ],
)
def test_read_paper_manifest_checks_rows(tmp_path, record, message):
    path = tmp_path / "papers.csv"
    write_paper_manifest(path, [record])
    with pytest.raises(ValueError, match=message):
        read_paper_manifest(path)


def test_read_paper_manifest_rejects_short_rows(tmp_path):
    path = tmp_path / "papers.csv"
    path.write_text(",".join(COLUMNS) + "\n1_6v5l,6v5l,test\n")
    with pytest.raises(ValueError, match="line 2: expected 20 fields"):
        read_paper_manifest(path)


def test_is_fetched(tmp_path):
    record = PaperRecord("2_6v1c", "6v1c", status="ok", sha256=PDF_SHA)
    assert not is_fetched(record, tmp_path)
    (tmp_path / "6v1c.pdf").write_bytes(PDF)
    assert is_fetched(record, tmp_path)
    assert not is_fetched(PaperRecord("2_6v1c", "6v1c", status="blocked", sha256=PDF_SHA), tmp_path)
    (tmp_path / "6v1c.pdf").write_bytes(PDF + b"changed")
    assert not is_fetched(record, tmp_path)


def test_unpaywall_email_from_env():
    assert unpaywall_email_from_env({}) is None
    assert unpaywall_email_from_env({"UNPAYWALL_EMAIL": " not-an-email "}) is None
    assert unpaywall_email_from_env({"UNPAYWALL_EMAIL": "me@example.org"}) == "me@example.org"


def js(payload, status=200):
    return Response(status, "https://api", {"content-type": "application/json"}, json.dumps(payload).encode())


def pdf(url="https://pub/a.pdf"):
    return Response(200, url, {"content-type": "application/pdf"}, PDF)


def html(status=200, body=b"<!DOCTYPE html><html><body>Log in</body></html>", headers=None):
    return Response(status, "https://pub/landing", {"content-type": "text/html", **(headers or {})}, body)


class FakeClient:
    """Answers by URL prefix (first match wins); unknown URLs get a 404."""

    def __init__(self, routes):
        self.routes = routes
        self.urls = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        for prefix, answer in self.routes.items():
            if url.startswith(prefix):
                if isinstance(answer, BaseException):
                    raise answer
                return answer
        return Response(404, url, {}, b"")


RCSB_6V1C = "https://data.rcsb.org/rest/v1/core/entry/6V1C"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
OPENALEX_6V1C = "https://api.openalex.org/works/doi:10.1038/s41467-020-16223-7"
PMC_LIST = "https://pmc-oa-opendata.s3.amazonaws.com/?list-type=2&prefix=PMC7221086."


def rcsb(doi="10.1038/s41467-020-16223-7", pmid=32404934, journal="Nat Commun"):
    return js(
        {
            "rcsb_primary_citation": {
                "pdbx_database_id_DOI": doi,
                "pdbx_database_id_PubMed": pmid,
                "title": "Trefoil factors share a lectin activity that defines their role in mucus",
                "rcsb_journal_abbrev": journal,
                "year": 2020,
            }
        }
    )


def epmc(doi="10.1038/s41467-020-16223-7", pmcid=None, pdf_urls=()):
    urls = [{"availabilityCode": "OA", "documentStyle": "pdf", "site": "Europe_PMC", "url": u} for u in pdf_urls]
    hit = {"pmid": "32404934", "pmcid": pmcid, "doi": doi, "title": "Trefoil factors share a lectin activity.",
           "fullTextUrlList": {"fullTextUrl": urls}}
    return js({"resultList": {"result": [hit]}})


def openalex(*locations, oa_status="gold", pmcid=None):
    return js(
        {
            "doi": "https://doi.org/10.1038/s41467-020-16223-7",
            "title": "Trefoil factors share a lectin activity that defines their role in mucus",
            "ids": {"pmcid": pmcid} if pmcid else {},
            "open_access": {"oa_status": oa_status},
            "best_oa_location": locations[0] if locations else None,
            "locations": list(locations),
        }
    )


def loc(url, version="publishedVersion"):
    return {"is_oa": True, "pdf_url": url, "version": version}


PAPER = Paper("2_6v1c", 2, "6v1c", "valid")


def fetch(tmp_path, routes, **kwargs):
    client = FakeClient(routes)
    record = PaperFetcher(client, tmp_path / "pdfs", now=lambda: NOW, **kwargs).fetch(PAPER, "1")
    return record, client


def test_fetch_downloads_the_openalex_published_pdf(tmp_path):
    routes = {
        RCSB_6V1C: rcsb(),
        EPMC: epmc(),
        OPENALEX_6V1C: openalex(loc("https://repo/accepted.pdf", "acceptedVersion"), loc("https://pub/a.pdf")),
        "https://pub/a.pdf": pdf(),
    }
    record, client = fetch(tmp_path, routes)
    assert record == PaperRecord(
        paper_id="2_6v1c",
        pdb_id="6v1c",
        split="valid",
        structures="1",
        doi="10.1038/s41467-020-16223-7",
        doi_source="rcsb",
        pmid="32404934",
        title="Trefoil factors share a lectin activity that defines their role in mucus",
        year="2020",
        journal="Nat Commun",
        title_match="yes",
        status="ok",
        source="openalex",
        oa_version="publishedVersion",
        pdf_url="https://pub/a.pdf",
        sha256=PDF_SHA,
        bytes=str(len(PDF)),
        retrieved_at="2026-10-05T12:00:00Z",
    )
    assert (tmp_path / "pdfs" / "6v1c.pdf").read_bytes() == PDF
    assert "https://repo/accepted.pdf" not in client.urls


def test_fetch_prefers_the_pmc_copy_and_records_its_pmcid(tmp_path):
    routes = {
        RCSB_6V1C: rcsb(),
        EPMC: epmc(pmcid="PMC7221086", pdf_urls=["https://europepmc.org/articles/PMC7221086?pdf=render"]),
        OPENALEX_6V1C: openalex(loc("https://pub/a.pdf")),
        PMC_LIST: Response(200, PMC_LIST, {}, b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                           b"<Contents><Key>PMC7221086.1/PMC7221086.1.json</Key></Contents></ListBucketResult>"),
        "https://pmc-oa-opendata.s3.amazonaws.com/PMC7221086.1/PMC7221086.1.json": js(
            {"pdf_url": "s3://pmc-oa-opendata/PMC7221086.1/PMC7221086.1.pdf?md5=x", "is_manuscript": False}
        ),
        "https://pmc-oa-opendata.s3.amazonaws.com/PMC7221086.1/PMC7221086.1.pdf": pdf(),
    }
    record, client = fetch(tmp_path, routes)
    assert (record.status, record.source, record.pmcid) == ("ok", "pmc", "PMC7221086")
    assert "https://pub/a.pdf" not in client.urls


def test_fetch_corrects_a_truncated_rcsb_doi_from_pubmed(tmp_path):
    routes = {
        RCSB_6V1C: rcsb(doi="10.1038/s41467"),
        EPMC: epmc(),
        OPENALEX_6V1C: openalex(loc("https://pub/a.pdf")),
        "https://pub/a.pdf": pdf(),
    }
    record, _ = fetch(tmp_path, routes)
    assert (record.doi, record.doi_source, record.status) == ("10.1038/s41467-020-16223-7", "pubmed", "ok")


def test_fetch_falls_through_failures_and_keeps_the_first_failure_as_status(tmp_path):
    challenge = html(403, b"<html><title>Just a moment...</title></html>", {"cf-mitigated": "challenge"})
    routes = {
        RCSB_6V1C: rcsb(),
        EPMC: epmc(),
        OPENALEX_6V1C: openalex(loc("https://cf/a.pdf"), loc("https://login/a.pdf"), loc("https://gone/a.pdf"),
                                loc("https://slow/a.pdf")),
        "https://cf/": challenge,
        "https://login/": html(),
        "https://gone/": html(404),
        "https://slow/": NetworkError("slow: TimeoutError: timed out"),
    }
    record, _ = fetch(tmp_path, routes)
    assert (record.status, record.pdf_url, record.sha256) == ("blocked", "https://cf/a.pdf", "")
    assert "https://login/a.pdf: html from pub" in record.detail
    assert "HTTP 404" in record.detail and "TimeoutError" in record.detail
    assert not (tmp_path / "pdfs" / "6v1c.pdf").exists()


@pytest.mark.parametrize(
    ("answer", "status"),
    [
        (html(), "html_not_pdf"),
        (html(403), "http_error"),
        (Response(200, "https://pub/a.pdf", {}, b"PK\x03\x04"), "not_pdf"),
        (Response(200, "https://pub/a.pdf", {}, b"%PDF-1.4", truncated=True), "too_large"),
        (Response(200, "https://pub/a.pdf", {"content-type": "application/pdf"}, b"%PDF-1.4\n1 0 obj"), "truncated_pdf"),
    ],
)
def test_fetch_failure_statuses(tmp_path, answer, status):
    routes = {RCSB_6V1C: rcsb(), EPMC: epmc(), OPENALEX_6V1C: openalex(loc("https://pub/a.pdf")), "https://pub/": answer}
    assert fetch(tmp_path, routes)[0].status == status


def test_fetch_without_any_oa_pdf_is_no_oa(tmp_path):
    routes = {RCSB_6V1C: rcsb(), EPMC: epmc(), OPENALEX_6V1C: openalex(oa_status="closed")}
    record, _ = fetch(tmp_path, routes)
    assert record.status == "no_oa" and "oa_status: closed" in record.detail


def test_fetch_calls_it_a_lookup_error_when_a_failed_lookup_may_hide_a_pdf(tmp_path):
    routes = {RCSB_6V1C: rcsb(), EPMC: epmc(), OPENALEX_6V1C: js({}, status=503)}
    record, _ = fetch(tmp_path, routes)
    assert record.status == "lookup_error"
    assert "openalex: OpenAlex lookup failed: HTTP 503" in record.detail


def test_fetch_without_doi_or_pmid_is_no_doi(tmp_path):
    routes = {RCSB_6V1C: rcsb(doi=None, pmid=None, journal="To be published")}
    record, client = fetch(tmp_path, routes)
    assert (record.status, record.doi) == ("no_doi", "")
    assert "unpublished" in record.detail
    assert client.urls == [RCSB_6V1C]


def test_fetch_records_an_rcsb_failure(tmp_path):
    record, _ = fetch(tmp_path, {})
    assert (record.status, record.detail) == ("lookup_error", "RCSB has no entry 6V1C")
    record, _ = fetch(tmp_path, {RCSB_6V1C: js({}, status=500)})
    assert (record.status, record.detail) == ("lookup_error", "RCSB lookup failed: HTTP 500")


def test_fetch_carries_on_when_an_optional_lookup_fails(tmp_path):
    routes = {
        RCSB_6V1C: rcsb(),
        EPMC: NetworkError("www.ebi.ac.uk: TimeoutError"),
        OPENALEX_6V1C: openalex(loc("https://pub/a.pdf")),
        "https://pub/a.pdf": pdf(),
    }
    record, _ = fetch(tmp_path, routes)
    assert record.status == "ok"


def test_fetch_asks_unpaywall_only_with_an_email_and_never_records_it(tmp_path):
    routes = {
        RCSB_6V1C: rcsb(),
        EPMC: epmc(),
        OPENALEX_6V1C: openalex(),
        "https://api.unpaywall.org/v2/10.1038/s41467-020-16223-7?email=me%40example.org": js(
            {"best_oa_location": {"url_for_pdf": "https://repo/b.pdf", "version": "acceptedVersion"}, "oa_locations": []}
        ),
        "https://repo/b.pdf": html(),
    }
    record, client = fetch(tmp_path, routes)
    assert record.status == "no_oa"
    assert not any("unpaywall" in url for url in client.urls)
    record, client = fetch(tmp_path, routes, unpaywall_email="me@example.org")
    assert (record.status, record.pdf_url) == ("html_not_pdf", "https://repo/b.pdf")
    assert "example.org" not in "".join(getattr(record, column) for column in COLUMNS)


class RecordingFetcher:
    calls: list[str] = []

    def __init__(self, client, pdf_dir, **kwargs):
        self.pdf_dir = pdf_dir

    def fetch(self, paper, structures=""):
        RecordingFetcher.calls.append(paper.paper_id)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        (self.pdf_dir / f"{paper.pdb_id}.pdf").write_bytes(PDF)
        return PaperRecord(
            paper.paper_id, paper.pdb_id, paper.split, structures, status="ok", sha256=PDF_SHA, bytes=str(len(PDF))
        )


def test_fetch_papers_limits_skips_verified_pdfs_and_writes_the_csv(tmp_path, monkeypatch):
    write_lists(tmp_path)
    monkeypatch.setattr(biovista, "PaperFetcher", RecordingFetcher)
    RecordingFetcher.calls = []
    manifest = tmp_path / "manifests" / "biovista_papers.csv"
    counts = fetch_papers(tmp_path, manifest, limit=2, client=object(), log=lambda _: None)
    assert counts == {"ok": 2}
    assert sorted(RecordingFetcher.calls) == ["1_6v5l", "2_6v1c"]
    assert list(read_paper_manifest(manifest)) == ["6v5l", "6v1c"]

    RecordingFetcher.calls = []
    counts = fetch_papers(tmp_path, manifest, client=object(), log=lambda _: None)
    assert counts == {"ok": 3}
    assert RecordingFetcher.calls == ["10_6ueg"]
    assert list(read_paper_manifest(manifest)) == ["6v5l", "6v1c", "6ueg"]


def test_fetch_papers_retries_rows_that_failed(tmp_path, monkeypatch):
    write_lists(tmp_path, names=("1_6v5l",), test=("1_6v5l",), valid=())
    manifest = tmp_path / "biovista_papers.csv"
    write_paper_manifest(manifest, [PaperRecord("1_6v5l", "6v5l", status="blocked")])
    monkeypatch.setattr(biovista, "PaperFetcher", RecordingFetcher)
    RecordingFetcher.calls = []
    assert fetch_papers(tmp_path, manifest, client=object(), log=lambda _: None) == {"ok": 1}
    assert RecordingFetcher.calls == ["1_6v5l"]


@pytest.mark.parametrize(("kwargs", "message"), [({"workers": 5}, "workers"), ({"limit": 0}, "limit")])
def test_fetch_papers_checks_its_options(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        fetch_papers(tmp_path, tmp_path / "m.csv", **kwargs)


class FailingFetcher(RecordingFetcher):
    def fetch(self, paper, structures=""):
        if paper.paper_id == "2_6v1c":
            raise OSError("disk full")
        return super().fetch(paper, structures)


def test_fetch_papers_stops_on_an_unexpected_error_and_keeps_finished_rows(tmp_path, monkeypatch):
    write_lists(tmp_path)
    monkeypatch.setattr(biovista, "PaperFetcher", FailingFetcher)
    manifest = tmp_path / "biovista_papers.csv"
    with pytest.raises(OSError, match="disk full"):
        fetch_papers(tmp_path, manifest, workers=1, client=object(), log=lambda _: None)
    assert list(read_paper_manifest(manifest)) == ["6v5l"]
