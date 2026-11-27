"""The frozen BioVista development papers: selection, the committed manifest and the dev/heldout groups."""

import csv
from pathlib import Path
from types import MappingProxyType

import pytest

from molscout.data import biovista_dev as dev
from molscout.data import biovista_truth as bt

REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "biovista"
MANIFEST = FIXTURE / "manifest.csv"
# The 18 papers dropped for an unreadable label (test_public_data.DROPPED); none may be a dev paper.
DROPPED = {
    "38_6s9d", "50_6rtn", "67_6ra1", "73_6r5f", "77_6r49", "102_6ql3", "111_6qfx", "149_6prf", "167_6peu",
    "176_6p5w", "244_6nzg", "295_6n5c", "323_6mny", "427_6ic7", "448_6i18", "451_6i0c", "461_6hvw", "465_6hs0",
}


def synthetic_truth(papers: int = 30) -> bt.BioVistaTruth:
    """Papers 0..n-1: even ones mostly enumerated (table-heavy), odd ones mostly drawn."""
    kept, references, enumerated, drawn = [], {}, {}, {}
    for i in range(papers):
        paper_id = f"{i}_p{i:03d}"
        kept.append(bt.ScoredPaper(paper_id, f"p{i:03d}", "publishedVersion", "0" * 64, 4))
        table = i % 2 == 0
        enumerated[paper_id] = ("C",) * (3 if table else 1)
        drawn[paper_id] = ("N",) * (1 if table else 3)
        references[paper_id] = drawn[paper_id] + enumerated[paper_id]
    return bt.BioVistaTruth(
        papers=tuple(kept),
        references=MappingProxyType(references),
        drawn=MappingProxyType(drawn),
        enumerated=MappingProxyType(enumerated),
        unreadable=MappingProxyType({}),
        dropped=MappingProxyType({}),
        labels=4 * papers,
    )


def test_selection_is_seeded_stratified_and_in_paper_order():
    truth = synthetic_truth()
    first = dev.select_dev_papers(truth)
    assert first == dev.select_dev_papers(truth)
    assert len(first) == 2 * dev.PER_KIND
    assert sum(p.kind == "table" for p in first) == dev.PER_KIND
    assert all((p.kind == "table") == (int(p.paper_id.split("_")[0]) % 2 == 0) for p in first)
    order = [p.paper_id for p in truth.papers]
    assert [p.paper_id for p in first] == sorted((p.paper_id for p in first), key=order.index)
    assert dev.select_dev_papers(truth, seed=1) != first


def test_kind_is_table_only_when_most_labels_are_enumerated():
    assert dev.paper_kind(drawn=1, enumerated=2) == "table"
    assert dev.paper_kind(drawn=2, enumerated=2) == "drawn"
    assert dev.paper_kind(drawn=3, enumerated=0) == "drawn"


def test_counts_come_from_the_labels():
    paper = dev.select_dev_papers(synthetic_truth())[0]
    assert (paper.labels, paper.drawn + paper.enumerated) == (4, 4)


def test_too_few_papers_of_a_kind_is_an_error():
    with pytest.raises(ValueError, match="table"):
        dev.select_dev_papers(synthetic_truth(papers=12))


def test_write_then_read_round_trips(tmp_path):
    papers = dev.select_dev_papers(synthetic_truth())
    path = tmp_path / "dev.csv"
    dev.write_dev_manifest(path, papers)
    assert dev.read_dev_papers(path) == papers
    assert dev.dev_paper_ids(path) == frozenset(p.paper_id for p in papers)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rows: rows[:1] + rows[2:], "10 table"),
        (lambda rows: rows + [rows[1]], "more than once"),
        (lambda rows: [rows[0][:-1]] + rows[1:], "columns"),
        (lambda rows: [rows[0]] + [[*rows[1][:3], "x", *rows[1][4:]]] + rows[2:], "integer"),
        (lambda rows: [rows[0]] + [[*rows[1][:2], "image", *rows[1][3:]]] + rows[2:], "kind"),
        (lambda rows: [rows[0]] + [[*rows[1][:3], "99", *rows[1][4:]]] + rows[2:], "labels"),
    ],
)
def test_reading_a_bad_manifest_is_an_error(tmp_path, change, message):
    path = tmp_path / "dev.csv"
    dev.write_dev_manifest(path, dev.select_dev_papers(synthetic_truth()))
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(change(rows))
    with pytest.raises(ValueError, match=message):
        dev.read_dev_papers(path)


def test_committed_manifest_is_frozen_and_well_formed():
    papers = dev.read_dev_papers(REPO / dev.BIOVISTA_DEV_PATH)
    scored = {p.paper_id: p for p in bt.scored_papers(REPO / bt.BIOVISTA_PAPERS_PATH)}
    assert len(papers) == 20
    assert all(p.paper_id in scored and p.paper_id not in DROPPED for p in papers)
    assert all(scored[p.paper_id].pdb_id == p.pdb_id and scored[p.paper_id].structures == p.labels for p in papers)
    order = list(scored)
    assert [p.paper_id for p in papers] == sorted((p.paper_id for p in papers), key=order.index)


def test_committed_manifest_regenerates_from_the_labels():
    root = REPO / "data" / "raw" / "biovista"
    if not (root / bt.LABELS).is_dir():
        pytest.skip("run scripts/fetch_data.py biovista first")
    truth = bt.load_biovista_truth(root, REPO / bt.BIOVISTA_PAPERS_PATH)
    assert dev.select_dev_papers(truth) == dev.read_dev_papers(REPO / dev.BIOVISTA_DEV_PATH)


# The dev and heldout groups in BioVista truth


def test_groups_add_dev_and_heldout_when_dev_papers_are_given():
    truth = bt.load_biovista_truth(FIXTURE, MANIFEST, dev={"1_aaaa", "3_cccc"})
    assert truth.groups() == {
        "all": frozenset({"1_aaaa", "2_bbbb", "3_cccc"}),
        "without_submitted": frozenset({"1_aaaa", "3_cccc"}),
        "dev": frozenset({"1_aaaa", "3_cccc"}),
        "heldout": frozenset({"2_bbbb"}),
    }
    # 3_cccc has no drawn label, so the drawn view has no paper of it
    assert truth.drawn_groups()["dev"] == frozenset({"1_aaaa"})
    assert truth.drawn_groups()["heldout"] == frozenset({"2_bbbb"})


def test_without_dev_papers_the_groups_are_unchanged():
    truth = bt.load_biovista_truth(FIXTURE, MANIFEST)
    assert set(truth.groups()) == {"all", "without_submitted"}


def test_a_dev_paper_must_be_a_benchmark_paper():
    with pytest.raises(ValueError, match="6_ffff"):
        bt.load_biovista_truth(FIXTURE, MANIFEST, dev={"1_aaaa", "6_ffff"})
