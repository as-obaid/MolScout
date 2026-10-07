import csv
import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from molscout.data import biovista
from molscout.data.biovista import (
    PDF_ACCEPT,
    PaperFetcher,
    PaperRecord,
    fetch_papers,
    read_paper_manifest,
)
from molscout.data.httpclient import NetworkError, Response
from molscout.data.recovered import COLUMNS, RecoveredSource, read_recovered_sources, write_recovered_sources

REPO = Path(__file__).resolve().parents[1]
PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
# The same paper as a repository serves it later, with a regenerated cover page.
COVERED = b"%PDF-1.7\n%cover page generated 2026-10-06\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
NOW = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)
URL = "https://www.osti.gov/servlets/purl/1502197"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pinned(paper_id="2_6v1c", **changes):
    pdb_id = paper_id.partition("_")[2]
    base = RecoveredSource(
        paper_id=paper_id,
        pdb_id=pdb_id,
        doi="10.1021/acs.jmedchem.8b01593",
        url=f"https://www.osti.gov/servlets/purl/{pdb_id}",
        source_host="osti.gov",
        source_type="repository",
        version="acceptedVersion",
        sha256=sha(PDF),
        bytes=len(PDF),
        pages=42,
        title_check="match",
        notes="OSTI author manuscript, 42pp incl. SI",
    )
    return replace(base, **changes)


# --- the pinned list --------------------------------------------------------------------------


def test_recovered_list_round_trip_is_sorted_by_index_with_lf_endings(tmp_path):
    path = tmp_path / "biovista_recovered.csv"
    sources = [pinned("10_6ueg"), pinned("2_6v1c", notes='says "SI", twice; fine'), pinned("1_6v5l")]
    write_recovered_sources(path, sources)
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert raw.splitlines()[0].decode() == ",".join(COLUMNS)
    assert [line.split(b",")[0] for line in raw.splitlines()[1:]] == [b"1_6v5l", b"2_6v1c", b"10_6ueg"]
    assert read_recovered_sources(path) == {source.pdb_id: source for source in sources}
    write_recovered_sources(path, reversed(sources))
    assert path.read_bytes() == raw


def test_read_recovered_sources_without_a_file_is_empty(tmp_path):
    assert read_recovered_sources(tmp_path / "missing.csv") == {}


def test_read_recovered_sources_checks_columns(tmp_path):
    path = tmp_path / "recovered.csv"
    path.write_text("pdb_id,url\n6v1c,https://example.org/a.pdf\n")
    with pytest.raises(ValueError, match="columns"):
        read_recovered_sources(path)


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def row(**changes):
    source = pinned()
    return {column: str(getattr(source, column)) for column in COLUMNS} | changes


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"paper_id": "2_6v1d"}, "paper_id"),
        ({"paper_id": "x_6v1c"}, "paper_id"),
        ({"pdb_id": "6V1C", "paper_id": "2_6V1C"}, "paper_id"),
        ({"doi": "not a doi"}, "doi"),
        ({"url": "http://www.osti.gov/servlets/purl/1502197"}, "url"),
        ({"url": "https://"}, "url"),
        ({"url": "https://reader:secret@www.osti.gov/servlets/purl/1502197"}, "url"),
        ({"url": "https://www.osti.gov/servlets/purl/1502197 extra"}, "url"),
        ({"source_host": "OSTI gov"}, "source_host"),
        ({"source_host": "hal.science"}, "source_host"),
        ({"source_host": "sti.gov"}, "source_host"),
        ({"source_type": "shadow_library"}, "source_type"),
        ({"version": "final"}, "version"),
        ({"sha256": "ABC"}, "sha256"),
        ({"bytes": "0"}, "bytes"),
        ({"bytes": "12kb"}, "bytes"),
        ({"pages": "-3"}, "pages"),
        ({"title_check": "maybe"}, "title_check"),
    ],
)
def test_read_recovered_sources_rejects_bad_values(tmp_path, change, message):
    path = tmp_path / "recovered.csv"
    write_rows(path, [row(**change)])
    with pytest.raises(ValueError, match=f"line 2: {message}"):
        read_recovered_sources(path)


def test_read_recovered_sources_rejects_a_paper_listed_twice(tmp_path):
    path = tmp_path / "recovered.csv"
    write_rows(path, [row(), row(url="https://www.osti.gov/servlets/purl/1558314")])
    with pytest.raises(ValueError, match="line 3: 6v1c is listed twice"):
        read_recovered_sources(path)


def test_read_recovered_sources_rejects_short_rows(tmp_path):
    path = tmp_path / "recovered.csv"
    path.write_text(",".join(COLUMNS) + "\n2_6v1c,6v1c\n")
    with pytest.raises(ValueError, match=f"line 2: expected {len(COLUMNS)} fields"):
        read_recovered_sources(path)


def test_committed_recovered_list_matches_the_papers_manifest():
    sources = read_recovered_sources(REPO / "data" / "manifests" / "biovista_recovered.csv")
    papers = read_paper_manifest(REPO / "data" / "manifests" / "biovista_papers.csv")
    assert len(sources) == 24
    assert all(papers[pdb_id].paper_id == source.paper_id for pdb_id, source in sources.items())
    assert all(source.title_check == "match" for source in sources.values())


# --- one paper's pinned copy ------------------------------------------------------------------


class FakeClient:
    """Answers by exact URL; unknown URLs get a 404."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    @property
    def urls(self):
        return [url for url, _ in self.calls]

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        answer = self.routes.get(url, Response(404, url, {}, b""))
        if isinstance(answer, BaseException):
            raise answer
        return answer


def served(body, url=URL, status=200, content_type="application/pdf", headers=None):
    return Response(status, url, {"content-type": content_type, **(headers or {})}, body)


BLOCKED = PaperRecord(
    "2_6v1c",
    "6v1c",
    "valid",
    "17",
    doi="10.1021/acs.jmedchem.8b01593",
    title="A paper",
    status="blocked",
    pdf_url="https://pubs.acs.org/doi/pdf/10.1021/acs.jmedchem.8b01593",
    detail="openalex/publishedVersion https://pubs.acs.org/doi/pdf/10.1021/acs.jmedchem.8b01593: HTTP 403 bot "
    "challenge from pubs.acs.org",
)


def recover(tmp_path, routes, source=None):
    client = FakeClient(routes)
    fetcher = PaperFetcher(client, tmp_path / "pdfs", now=lambda: NOW)
    return fetcher.fetch_recovered(BLOCKED, source or pinned(url=URL)), client


def test_fetch_recovered_downloads_the_pinned_copy(tmp_path):
    record, client = recover(tmp_path, {URL: served(PDF)})
    assert record == replace(
        BLOCKED,
        status="ok",
        source="recovered:osti.gov",
        oa_version="acceptedVersion",
        pdf_url=URL,
        sha256=sha(PDF),
        bytes=str(len(PDF)),
        retrieved_at="2026-10-05T12:00:00Z",
        detail="",
    )
    assert (tmp_path / "pdfs" / "6v1c.pdf").read_bytes() == PDF
    [(url, kwargs)] = client.calls
    assert url == URL and kwargs["accept"] == PDF_ACCEPT and kwargs["max_bytes"] == biovista.MAX_PDF_BYTES


def test_fetch_recovered_uses_a_valid_local_copy_without_downloading(tmp_path):
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "pdfs" / "6v1c.pdf").write_bytes(PDF)
    record, client = recover(tmp_path, {URL: served(PDF)})
    assert (record.status, record.source, record.sha256, record.detail) == ("ok", "recovered:osti.gov", sha(PDF), "")
    assert client.calls == []


def test_fetch_recovered_replaces_an_invalid_local_copy(tmp_path):
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "pdfs" / "6v1c.pdf").write_bytes(b"%PDF-1.7\n1 0 obj\n")  # cut short: no %%EOF
    record, client = recover(tmp_path, {URL: served(PDF)})
    assert (record.status, record.sha256) == ("ok", sha(PDF))
    assert client.urls == [URL]
    assert (tmp_path / "pdfs" / "6v1c.pdf").read_bytes() == PDF


def test_fetch_recovered_keeps_a_downloaded_pdf_whose_sha256_differs_and_records_it(tmp_path):
    record, client = recover(tmp_path, {URL: served(COVERED)})
    assert (record.status, record.sha256, record.bytes) == ("ok", sha(COVERED), str(len(COVERED)))
    assert record.detail == (
        f"sha256 mismatch: pinned copy is {sha(PDF)} ({len(PDF)} bytes); kept the valid PDF downloaded from the "
        "pinned URL"
    )
    assert (tmp_path / "pdfs" / "6v1c.pdf").read_bytes() == COVERED and client.urls == [URL]


def test_fetch_recovered_keeps_a_valid_local_copy_whose_sha256_differs_and_records_it(tmp_path):
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "pdfs" / "6v1c.pdf").write_bytes(COVERED)
    record, client = recover(tmp_path, {URL: served(PDF)})  # the pinned bytes are on offer, but not fetched
    assert (record.status, record.sha256, record.bytes) == ("ok", sha(COVERED), str(len(COVERED)))
    assert record.detail == (
        f"sha256 mismatch: pinned copy is {sha(PDF)} ({len(PDF)} bytes); kept the valid PDF already in pdfs/"
    )
    assert (tmp_path / "pdfs" / "6v1c.pdf").read_bytes() == COVERED and client.calls == []


def test_fetch_recovered_downloads_instead_of_using_an_oversized_local_copy(tmp_path, monkeypatch):
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "pdfs" / "6v1c.pdf").write_bytes(COVERED)
    monkeypatch.setattr(biovista, "MAX_PDF_BYTES", len(PDF))
    record, client = recover(tmp_path, {URL: served(PDF)})
    assert (record.status, record.sha256, record.detail) == ("ok", sha(PDF), "")
    assert client.urls == [URL]


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        (served(b"<!DOCTYPE html><html><body>Item</body></html>", content_type="text/html"), "html from www.osti.gov"),
        (served(b"", status=404, content_type="text/html"), "HTTP 404 from www.osti.gov"),
        (
            served(b"<title>Just a moment...</title>", status=403, content_type="text/html",
                   headers={"cf-mitigated": "challenge"}),
            "HTTP 403 bot challenge from www.osti.gov",
        ),
        (served(b"%PDF-1.7\n1 0 obj\n"), "PDF from www.osti.gov has no %%EOF trailer"),
        (NetworkError("www.osti.gov: TimeoutError: timed out"), "www.osti.gov: TimeoutError: timed out"),
    ],
)
def test_fetch_recovered_failure_keeps_the_status_and_adds_the_reason(tmp_path, answer, reason):
    record, _ = recover(tmp_path, {URL: answer})
    assert record == replace(BLOCKED, detail=f"{BLOCKED.detail}; recovered:osti.gov {URL}: {reason}")
    assert not (tmp_path / "pdfs" / "6v1c.pdf").exists()


# --- the biovista step ------------------------------------------------------------------------


def write_lists(root, names=("2_6v1c", "1_6v5l", "10_6ueg")):
    (root / "bioactivity_extraction" / "data").mkdir(parents=True, exist_ok=True)
    (root / "bioactivity_extraction" / "data" / "file_names.txt").write_text("\n".join(names))
    (root / "test_set_names.txt").write_text("\n".join(names))
    (root / "valid_set_names.txt").write_text("")


def stub_open_access_pass(monkeypatch, pdf_dir, ok=()):
    """Stand in for the open-access pass: papers in `ok` get an OpenAlex PDF, the rest are no_oa."""
    calls = []

    def fetch(self, paper, structures=""):
        calls.append(paper.pdb_id)
        record = PaperRecord(paper.paper_id, paper.pdb_id, paper.split, structures)
        if paper.pdb_id not in ok:
            return replace(record, status="no_oa", detail="no open-access PDF URL from PMC, OpenAlex or Europe PMC")
        pdf_dir.mkdir(parents=True, exist_ok=True)
        (pdf_dir / f"{paper.pdb_id}.pdf").write_bytes(PDF)
        return replace(record, status="ok", source="openalex", oa_version="publishedVersion",
                       pdf_url="https://pub/a.pdf", sha256=sha(PDF), bytes=str(len(PDF)),
                       retrieved_at="2026-10-05T12:00:00Z")

    monkeypatch.setattr(biovista.PaperFetcher, "fetch", fetch)
    return calls


def run(tmp_path, client, sources, log=None, **kwargs):
    recovered = tmp_path / "manifests" / "biovista_recovered.csv"
    write_recovered_sources(recovered, sources)
    manifest = tmp_path / "manifests" / "biovista_papers.csv"
    counts = fetch_papers(tmp_path, manifest, recovered_path=recovered, client=client,
                          log=log or (lambda _: None), **kwargs)
    return counts, read_paper_manifest(manifest), manifest


def test_fetch_papers_falls_back_to_pinned_copies_only_for_papers_still_without_a_pdf(tmp_path, monkeypatch):
    write_lists(tmp_path)
    stub_open_access_pass(monkeypatch, tmp_path / "pdfs", ok={"6v1c"})
    sources = [pinned("2_6v1c"), pinned("1_6v5l")]
    client = FakeClient({source.url: served(PDF, url=source.url) for source in sources})
    counts, rows, _ = run(tmp_path, client, sources)
    assert counts == {"ok": 2, "no_oa": 1}
    assert rows["6v1c"].source == "openalex"
    assert (rows["6v5l"].status, rows["6v5l"].source, rows["6v5l"].oa_version) == (
        "ok", "recovered:osti.gov", "acceptedVersion")
    assert (rows["6ueg"].status, rows["6ueg"].source) == ("no_oa", "")
    assert client.urls == [sources[1].url]


def test_fetch_papers_recovers_only_within_the_limit(tmp_path, monkeypatch):
    write_lists(tmp_path)
    stub_open_access_pass(monkeypatch, tmp_path / "pdfs")
    sources = [pinned("10_6ueg")]
    client = FakeClient({sources[0].url: served(PDF, url=sources[0].url)})
    counts, rows, _ = run(tmp_path, client, sources, limit=2)
    assert counts == {"no_oa": 2}
    assert "6ueg" not in rows and client.calls == []


def test_fetch_papers_warns_about_a_sha256_mismatch(tmp_path, monkeypatch):
    write_lists(tmp_path, names=("1_6v5l",))
    stub_open_access_pass(monkeypatch, tmp_path / "pdfs")
    source = pinned("1_6v5l")
    lines = []
    counts, rows, _ = run(tmp_path, FakeClient({source.url: served(COVERED, url=source.url)}), [source], lines.append)
    assert counts == {"ok": 1}
    assert rows["6v5l"].detail.startswith("sha256 mismatch: pinned copy is ")
    assert any(line.startswith("biovista: warning: 1_6v5l: sha256 mismatch") for line in lines)


def test_fetch_papers_logs_why_a_pinned_copy_failed(tmp_path, monkeypatch):
    write_lists(tmp_path, names=("1_6v5l",))
    stub_open_access_pass(monkeypatch, tmp_path / "pdfs")
    source = pinned("1_6v5l")
    lines = []
    counts, _, _ = run(tmp_path, FakeClient({}), [source], lines.append)
    assert counts == {"no_oa": 1}
    reason = f"recovered:osti.gov {source.url}: HTTP 404 from www.osti.gov"
    assert f"biovista: recovered [1/1] 1_6v5l still no_oa: {reason}" in lines


def test_fetch_papers_rerun_is_idempotent(tmp_path, monkeypatch):
    write_lists(tmp_path, names=("2_6v1c", "1_6v5l", "10_6ueg", "11_6uyz"))
    calls = stub_open_access_pass(monkeypatch, tmp_path / "pdfs")
    sources = [pinned("2_6v1c"), pinned("1_6v5l"), pinned("11_6uyz")]
    routes = {
        # 6v1c's repository now adds a regenerated cover page, so its row records a mismatch.
        sources[0].url: served(COVERED, url=sources[0].url),
        sources[1].url: served(PDF, url=sources[1].url),
        # 6uyz's repository now answers with its landing page, so the paper stays without a PDF.
        sources[2].url: served(b"<html><body>Item page</body></html>", url=sources[2].url, content_type="text/html"),
    }
    client = FakeClient(routes)
    first_counts, rows, manifest = run(tmp_path, client, sources)
    first = manifest.read_bytes()
    assert first_counts == {"ok": 2, "no_oa": 2}
    assert rows["6uyz"].detail.endswith(f"; recovered:osti.gov {sources[2].url}: html from www.osti.gov")

    calls.clear()
    client.calls.clear()
    second_counts, _, _ = run(tmp_path, client, sources)
    assert second_counts == first_counts
    assert manifest.read_bytes() == first
    # Only the papers still without a PDF are tried again; the two recovered ones are skipped.
    assert sorted(calls) == ["6ueg", "6uyz"] and client.urls == [sources[2].url]


def test_fetch_papers_rerun_downloads_a_recovered_pdf_again_once_it_is_gone(tmp_path, monkeypatch):
    write_lists(tmp_path, names=("1_6v5l",))
    stub_open_access_pass(monkeypatch, tmp_path / "pdfs")
    source = pinned("1_6v5l")
    client = FakeClient({source.url: served(PDF, url=source.url)})
    run(tmp_path, client, [source])
    (tmp_path / "pdfs" / "6v5l.pdf").unlink()
    counts, rows, _ = run(tmp_path, client, [source])
    assert counts == {"ok": 1} and rows["6v5l"].source == "recovered:osti.gov"
    assert client.urls == [source.url, source.url]


@pytest.mark.parametrize("paper_id", ["7_1abc", "3_6v1c"])
def test_fetch_papers_rejects_a_pinned_copy_for_a_paper_not_in_biovista(tmp_path, monkeypatch, paper_id):
    write_lists(tmp_path)
    stub_open_access_pass(monkeypatch, tmp_path / "pdfs")
    with pytest.raises(ValueError, match=f"{paper_id} is not a BioVista paper"):
        run(tmp_path, FakeClient({}), [pinned(paper_id)])

