import pytest

from molscout.data.openaccess import (
    Candidate,
    Citation,
    classify_payload,
    europepmc_candidates,
    normalize_doi,
    normalize_pmcid,
    normalize_pmid,
    order_candidates,
    parse_europepmc_search,
    parse_openalex_work,
    parse_rcsb_entry,
    pmc_candidates,
    pmc_metadata_key,
    title_similarity,
    unpaywall_candidates,
)


@pytest.mark.parametrize(
    ("raw", "doi"),
    [
        ("10.1021/acs.jmedchem.9b01234", "10.1021/acs.jmedchem.9b01234"),
        (" 10.1038/S41467-020-16223-7 ", "10.1038/s41467-020-16223-7"),
        ("https://doi.org/10.1016/j.str.2019.11.019", "10.1016/j.str.2019.11.019"),
        ("doi:10.1128/AAC.01473-19", "10.1128/aac.01473-19"),
        ("", ""),
        ("To be published", ""),
        (None, ""),
        (12345, ""),
    ],
)
def test_normalize_doi(raw, doi):
    assert normalize_doi(raw) == doi


def test_normalize_ids():
    assert normalize_pmcid("PMC7221086") == "PMC7221086"
    assert normalize_pmcid("https://www.ncbi.nlm.nih.gov/pmc/articles/7221086/") == "PMC7221086"
    assert normalize_pmcid("pmc12") == "PMC12"
    assert normalize_pmcid(None) == normalize_pmcid("PMCabc") == ""
    assert normalize_pmid(30903639) == normalize_pmid("https://pubmed.ncbi.nlm.nih.gov/30903639") == "30903639"
    assert normalize_pmid(-1) == normalize_pmid(None) == normalize_pmid(True) == ""


RCSB_CITATION = {
    "id": "primary",
    "journal_abbrev": "Nat Commun",
    "rcsb_journal_abbrev": "Nat Commun",
    "pdbx_database_id_DOI": "10.1038/s41467-020-16223-7",
    "pdbx_database_id_PubMed": 32404934,
    "title": "Trefoil factors share a lectin activity\n that defines their role in mucus.",
    "year": 2020,
}


def test_parse_rcsb_entry():
    citation = parse_rcsb_entry({"rcsb_id": "6V1C", "rcsb_primary_citation": RCSB_CITATION})
    assert citation == Citation(
        doi="10.1038/s41467-020-16223-7",
        pmid="32404934",
        title="Trefoil factors share a lectin activity that defines their role in mucus.",
        journal="Nat Commun",
        year="2020",
    )


@pytest.mark.parametrize(
    ("citation", "problem"),
    [
        (None, "no primary citation"),
        ({**RCSB_CITATION, "pdbx_database_id_DOI": None, "rcsb_journal_abbrev": "To be published"}, "unpublished"),
        ({**RCSB_CITATION, "pdbx_database_id_DOI": None}, "has no DOI"),
        ({**RCSB_CITATION, "pdbx_database_id_DOI": "doi pending"}, "not a valid DOI"),
    ],
)
def test_parse_rcsb_entry_explains_a_missing_doi(citation, problem):
    entry = {} if citation is None else {"rcsb_primary_citation": citation}
    parsed = parse_rcsb_entry(entry)
    assert parsed.doi == ""
    assert problem in parsed.problem


def europepmc_hit(**changes):
    hit = {
        "pmid": "31685462",
        "pmcid": "PMC7187628",
        "doi": "10.1128/AAC.01473-19",
        "title": "Influence of the alpha-Methoxy Group.",
        "authMan": "N",
        "epmcAuthMan": "N",
        "nihAuthMan": "N",
        "fullTextUrlList": {
            "fullTextUrl": [
                {"availabilityCode": "S", "documentStyle": "doi", "site": "DOI", "url": "https://doi.org/x"},
                {"availabilityCode": "OA", "documentStyle": "html", "site": "Europe_PMC", "url": "https://e/h"},
                {"availabilityCode": "OA", "documentStyle": "pdf", "site": "Europe_PMC", "url": "https://e/a.pdf"},
                {"availabilityCode": "S", "documentStyle": "pdf", "site": "Europe_PMC", "url": "https://e/s.pdf"},
            ]
        },
    }
    return {"resultList": {"result": [{**hit, **changes}]}}


def test_parse_europepmc_search_keeps_only_free_europe_pmc_pdfs():
    record = parse_europepmc_search(europepmc_hit())
    assert (record.pmid, record.pmcid, record.doi) == ("31685462", "PMC7187628", "10.1128/aac.01473-19")
    assert record.pdf_urls == ("https://e/a.pdf",)
    assert europepmc_candidates(record) == [Candidate("https://e/a.pdf", "europepmc", "publishedVersion")]


def test_europepmc_author_manuscript_is_the_accepted_version():
    record = parse_europepmc_search(europepmc_hit(nihAuthMan="Y"))
    assert europepmc_candidates(record)[0].version == "acceptedVersion"


def test_parse_europepmc_search_with_no_hit():
    assert parse_europepmc_search({"hitCount": 0, "resultList": {"result": []}}) is None


def location(pdf_url, version="publishedVersion", is_oa=True):
    return {"is_oa": is_oa, "pdf_url": pdf_url, "landing_page_url": "https://landing", "version": version}


def test_parse_openalex_work_lists_oa_pdf_locations_best_first():
    work = parse_openalex_work(
        {
            "doi": "https://doi.org/10.1038/S41467-020-16223-7",
            "title": "Trefoil factors",
            "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/32404934", "pmcid": "https://www.ncbi.nlm.nih.gov/pmc/articles/7221086"},
            "open_access": {"is_oa": True, "oa_status": "gold"},
            "best_oa_location": location("https://publisher/best.pdf"),
            "locations": [
                location("https://repo/accepted.pdf", "acceptedVersion"),
                location("https://closed/x.pdf", is_oa=False),
                location(None),
                location("https://repo/odd.pdf", "weirdVersion"),
            ],
        }
    )
    assert (work.doi, work.pmid, work.pmcid, work.oa_status) == ("10.1038/s41467-020-16223-7", "32404934", "PMC7221086", "gold")
    assert work.candidates == (
        Candidate("https://publisher/best.pdf", "openalex", "publishedVersion"),
        Candidate("https://repo/accepted.pdf", "openalex", "acceptedVersion"),
        Candidate("https://repo/odd.pdf", "openalex", ""),
    )


def test_parse_openalex_work_without_oa():
    work = parse_openalex_work({"doi": None, "ids": {}, "best_oa_location": None, "locations": [location(None)]})
    assert work.candidates == ()
    assert work.doi == work.pmcid == ""


def test_unpaywall_candidates():
    payload = {
        "best_oa_location": {"url_for_pdf": "https://pub/a.pdf", "version": "publishedVersion"},
        "oa_locations": [
            {"url_for_pdf": None, "version": "acceptedVersion"},
            {"url_for_pdf": "https://repo/b.pdf", "version": "submittedVersion"},
        ],
    }
    assert unpaywall_candidates(payload) == [
        Candidate("https://pub/a.pdf", "unpaywall", "publishedVersion"),
        Candidate("https://repo/b.pdf", "unpaywall", "submittedVersion"),
    ]


S3_LISTING = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Name>pmc-oa-opendata</Name>
<Contents><Key>PMC123.1/PMC123.1.json</Key></Contents>
<Contents><Key>PMC123.1/PMC123.1.pdf</Key></Contents>
<Contents><Key>PMC123.2/PMC123.2.json</Key></Contents>
<Contents><Key>PMC1234.9/PMC1234.9.json</Key></Contents>
</ListBucketResult>"""


def test_pmc_metadata_key_takes_the_newest_version_of_that_pmcid():
    assert pmc_metadata_key(S3_LISTING, "PMC123") == "PMC123.2/PMC123.2.json"
    assert pmc_metadata_key(S3_LISTING, "PMC999") == ""


def test_pmc_candidates():
    metadata = {"pdf_url": "s3://pmc-oa-opendata/PMC123.2/PMC123.2.pdf?md5=abc", "is_manuscript": False}
    assert pmc_candidates(metadata) == [
        Candidate("https://pmc-oa-opendata.s3.amazonaws.com/PMC123.2/PMC123.2.pdf", "pmc", "publishedVersion")
    ]
    assert pmc_candidates({**metadata, "is_manuscript": True})[0].version == "acceptedVersion"
    assert pmc_candidates({"pdf_url": None}) == []


def test_order_candidates_prefers_published_then_source_order_and_drops_repeats():
    candidates = [
        Candidate("https://a/submitted.pdf", "openalex", "submittedVersion"),
        Candidate("https://a/unknown.pdf", "openalex", ""),
        Candidate("https://a/accepted.pdf", "openalex", "acceptedVersion"),
        Candidate("https://e/pub.pdf", "europepmc", "publishedVersion"),
        Candidate("https://a/pub.pdf", "openalex", "publishedVersion"),
        Candidate("https://a/pub2.pdf", "openalex", "publishedVersion"),
        Candidate("https://p/pub.pdf", "pmc", "publishedVersion"),
        Candidate("https://a/pub.pdf", "unpaywall", "publishedVersion"),
    ]
    assert [c.url for c in order_candidates(candidates)] == [
        "https://p/pub.pdf",
        "https://a/pub.pdf",
        "https://a/pub2.pdf",
        "https://e/pub.pdf",
        "https://a/accepted.pdf",
        "https://a/submitted.pdf",
        "https://a/unknown.pdf",
    ]


CLOUDFLARE = b"<!DOCTYPE html><html><head><title>Just a moment...</title></head><body>cf-chl</body></html>"
EOF = b"\ntrailer\n<<>>\nstartxref\n123\n%%EOF\n"


@pytest.mark.parametrize(
    ("body", "content_type", "kind"),
    [
        (b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj" + EOF, "application/pdf", "pdf"),
        (b"\xef\xbb\xbf\r\n%PDF-1.4\n" + EOF, "application/octet-stream", "pdf"),
        (b"%PDF-1.5 served as html" + EOF, "text/html", "pdf"),
        (b"junk before %PDF-1.4\n" + EOF, "application/pdf", "pdf"),
        (b"junk before %PDF-1.4\n" + EOF, "application/octet-stream", "not_pdf"),
        (b'{"error": "expected %PDF- here"}', "application/json", "not_pdf"),
        (b"%PDF-1.7\n1 0 obj\n<< /Length 9000 >>\nstream\nx", "application/pdf", "truncated"),
        (b"<!DOCTYPE html><html><body>Sign in to download</body></html>", "text/html", "html"),
        (b"<html><body>see %PDF- below</body></html>" + EOF, "application/pdf", "html"),
        (CLOUDFLARE, "text/html", "blocked"),
        (b"", "application/pdf", "empty"),
        (b"  \n\x00", "", "empty"),
        (b"PK\x03\x04zipfile", "application/zip", "not_pdf"),
        (b"plain text", "text/html; charset=utf-8", "html"),
    ],
)
def test_classify_payload(body, content_type, kind):
    assert classify_payload(body, content_type) == kind


def test_classify_payload_needs_the_magic_near_the_start():
    assert classify_payload(b"x" * 2000 + b"%PDF-1.4" + EOF, "application/pdf") == "not_pdf"


@pytest.mark.parametrize(
    "payload",
    [
        {"resultList": None},
        {"resultList": {"result": "oops"}},
        {"resultList": {"result": ["not a dict"]}},
    ],
)
def test_parse_europepmc_search_tolerates_odd_shapes(payload):
    assert parse_europepmc_search(payload) is None


def test_parse_europepmc_search_skips_odd_url_items():
    payload = europepmc_hit(fullTextUrlList={"fullTextUrl": ["a string", None, {"site": "Europe_PMC"}]})
    assert parse_europepmc_search(payload).pdf_urls == ()


def test_parse_openalex_work_tolerates_odd_shapes():
    work = parse_openalex_work({"ids": "x", "locations": "y", "best_oa_location": [1], "open_access": 3, "title": 7})
    assert (work.candidates, work.pmid, work.oa_status, work.title) == ((), "", "", "")
    assert unpaywall_candidates({"best_oa_location": "z", "oa_locations": None}) == []


def test_title_similarity():
    rcsb = "Multi target ensemble based virtual screening yields novel allosteric KRAS inhibitors"
    pubmed = "Multi-target, ensemble-based virtual screening yields novel allosteric KRAS inhibitors."
    assert title_similarity(rcsb, pubmed) == 1.0
    assert title_similarity(rcsb, "Synthesis and SAR of aryl thiazoles") < 0.2
    assert title_similarity("", pubmed) == 0.0
