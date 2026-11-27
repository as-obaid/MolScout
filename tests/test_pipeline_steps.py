"""The in-process steps' calls into the pipeline modules (with fake modules), inputs and export."""

import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from types import ModuleType
from typing import NamedTuple

import pytest
from pipeline_helpers import fake_config, make_papers

import molscout.pipeline
from molscout.pipeline import export, steps
from molscout.pipeline.inputs import Paper, resolve_papers


@dataclass
class Molecule:
    paper_id: str
    label: str | None
    smiles: str | None
    status: str = "review"
    reasons: list = field(default_factory=list)


def fake_module(monkeypatch, name, **functions):
    module = ModuleType(f"molscout.pipeline.{name}")
    module.__dict__.update(functions)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(molscout.pipeline, name, module, raising=False)
    return module


@pytest.fixture
def schema(monkeypatch):
    def read_jsonl(path, cls):
        return [cls(**json.loads(line)) for line in path.read_text().splitlines()] if path.is_file() else []

    def write_jsonl(path, records):
        path.write_text("".join(json.dumps(asdict(record)) + "\n" for record in records))

    return fake_module(monkeypatch, "schema", Molecule=Molecule, read_jsonl=read_jsonl, write_jsonl=write_jsonl)


def test_vote_calls_vote_paper_with_the_fixes_and_readers_and_returns_its_counts(tmp_path, monkeypatch):
    calls = []
    counts = {"in": 3, "out": 2, "dropped_no_usable_read": 1, "cores": 0}

    def vote_paper(paper_dir, *, fixes, readers, primary_reader):
        calls.append((paper_dir, fixes, readers, primary_reader))
        return counts

    fake_module(monkeypatch, "vote", vote_paper=vote_paper)
    config = fake_config(tmp_path, "readers: [molscribe, molglyph]\nprimary_reader: molscribe\n", fixes_off=("stereo",))
    assert steps.vote(Paper("p1", tmp_path / "p1.pdf"), tmp_path, config) == counts
    assert calls == [(tmp_path, config.fixes, ("molscribe", "molglyph"), "molscribe")] and not calls[0][1].stereo


def test_finalize_reads_the_joiners_molecules_and_writes_molecules_jsonl(tmp_path, monkeypatch, schema):
    def finalize(paper_dir, molecules, *, fixes):
        kept = [m for m in molecules if m.smiles != "C"]
        return kept, {"in": len(molecules), "out": len(kept), "dropped_junk": len(molecules) - len(kept)}

    fake_module(monkeypatch, "finalize", finalize=finalize)
    config = fake_config(tmp_path, "check_endpoint: false\n", joiner="32b", vllm_url="http://h:1/v1")
    schema.write_jsonl(tmp_path / "molecules_32b.jsonl", [Molecule("p1", "1", "CCO"), Molecule("p1", "2", "C")])
    counts = steps.finalize(Paper("p1", tmp_path / "p1.pdf"), tmp_path, config)
    assert counts == {"in": 2, "out": 1, "dropped_junk": 1}
    assert [json.loads(line)["label"] for line in (tmp_path / "molecules.jsonl").read_text().splitlines()] == ["1"]


@pytest.mark.parametrize("returns_counts", [False, True])
def test_merge_takes_either_return_shape(tmp_path, monkeypatch, schema, returns_counts):
    class MergeResult(NamedTuple):  # as merge.MergeResult
        molecules: list
        counts: dict

    def merge(rules, llm):
        molecules = [*rules, *llm]
        return MergeResult(molecules, {"in": len(molecules), "out": len(molecules)}) if returns_counts else molecules

    fake_module(monkeypatch, "merge", merge=merge)
    config = fake_config(tmp_path, "check_endpoint: false\n", joiner="both", vllm_url="http://h:1/v1")
    schema.write_jsonl(tmp_path / "molecules_rules.jsonl", [Molecule("p1", "1", "CCO")])
    schema.write_jsonl(tmp_path / "molecules_32b.jsonl", [Molecule("p1", "2", "CCN")])
    counts = steps.merge(Paper("p1", tmp_path / "p1.pdf"), tmp_path, config)
    assert counts == ({"in": 2, "out": 2} if returns_counts else None)
    assert len((tmp_path / "molecules_merged.jsonl").read_text().splitlines()) == 2


def test_link_writes_the_rules_molecules_and_returns_their_counts(tmp_path, monkeypatch, schema):
    def link_paper(paper_dir, opsin, fixes, pdf=None):
        return [Molecule("p1", "1", "CCO")], {"in": 1, "out": 1, "whole_compound_rows": 0}

    fake_module(monkeypatch, "link", link_paper=link_paper)
    fake_module(monkeypatch, "opsin", Opsin=lambda jar, java: None)
    config = fake_config(tmp_path)
    counts = steps.link(Paper("p1", tmp_path / "p1.pdf"), tmp_path, config)
    assert counts == {"in": 1, "out": 1, "whole_compound_rows": 0}
    lines = (tmp_path / config.output("link")).read_text().splitlines()
    assert [json.loads(line)["label"] for line in lines] == ["1"]


def test_link_needs_the_opsin_jar(tmp_path):
    config = fake_config(tmp_path)
    config.opsin_jar.unlink()
    with pytest.raises(FileNotFoundError, match="OPSIN"):
        steps.link(Paper("p1", tmp_path / "p1.pdf"), tmp_path, config)


# Inputs


def test_inputs_from_a_pdf_a_folder_and_a_csv(tmp_path):
    papers = make_papers(tmp_path, ("b", "a"))
    assert resolve_papers(papers[0].pdf) == (Paper("b", papers[0].pdf),)
    assert [p.paper_id for p in resolve_papers(tmp_path / "pdfs")] == ["a", "b"]
    listing = tmp_path / "papers.csv"
    listing.write_text("paper_id,pdf\n26_6st3,pdfs/b.pdf\n")
    assert resolve_papers(listing) == (Paper("26_6st3", (tmp_path / "pdfs" / "b.pdf").absolute()),)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("id,pdf\nx,a.pdf\n", "header"),
        ("paper_id,pdf\nx,missing.pdf\n", "not a file"),
        ("paper_id,pdf\n../x,pdfs/a.pdf\n", "letters"),
        ("paper_id,pdf\nx,pdfs/a.pdf\nx,pdfs/a.pdf\n", "more than once"),
        ("paper_id,pdf\n", "no papers"),
    ],
)
def test_bad_paper_lists(tmp_path, text, message):
    make_papers(tmp_path, ("a",))
    listing = tmp_path / "papers.csv"
    listing.write_text(text)
    with pytest.raises(ValueError, match=message):
        resolve_papers(listing)


def test_a_folder_is_read_recursively_only_on_request(tmp_path):
    make_papers(tmp_path, ("b",))
    nested = tmp_path / "pdfs" / "more"
    nested.mkdir()
    (nested / "a.PDF").write_bytes(b"%PDF-1.4 a")
    assert [p.paper_id for p in resolve_papers(tmp_path / "pdfs")] == ["b"]
    assert [p.paper_id for p in resolve_papers(tmp_path / "pdfs", recursive=True)] == ["b", "a"]
    (nested / "b.pdf").write_bytes(b"%PDF-1.4 b2")
    with pytest.raises(ValueError, match="more than once"):
        resolve_papers(tmp_path / "pdfs", recursive=True)


def test_an_empty_folder_or_other_file_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no PDF"):
        resolve_papers(tmp_path / "empty")
    with pytest.raises(ValueError, match="expected a PDF"):
        resolve_papers(tmp_path / "notes.txt")


# Export


def test_molecule_rows_and_predictions_skip_what_the_scorer_would_reject(tmp_path):
    config = fake_config(tmp_path, export="accepted+review")
    path = tmp_path / "molecules.jsonl"
    records = [
        {"paper_id": "p1", "label": "1a", "smiles": "CCO", "page": 2, "bbox": [1, 2, 3, 4], "source": "drawn",
         "status": "accepted", "confidence": 0.75},
        {"paper_id": "p1", "label": "1b", "smiles": "CCN", "page": 0, "bbox": [5, 2, 3, 4], "source": "table",
         "status": "review", "confidence": None},
        {"paper_id": "p1", "label": "1c", "smiles": "C", "page": None, "bbox": None, "source": "name",
         "status": "abstained", "confidence": float("nan")},
    ]
    import json

    path.write_text("".join(json.dumps(record, allow_nan=True) + "\n" for record in records))
    out = tmp_path / "molecules.csv"
    assert export.write_molecules_csv(out, [path]) == 3
    rows = list(csv.reader(out.open()))
    assert rows[0] == list(export.MOLECULE_COLUMNS)
    assert rows[1] == ["p1", "1a", "CCO", "2", "1.0,2.0,3.0,4.0", "drawn", "accepted", "0.75"]
    assert rows[3][-1] == ""
    predictions = export.predictions({"p1": path}, config, dataset="biovista", tool="MolScout x", seconds={"p1": 3.5})
    assert [(p.smiles, p.page, p.bbox, p.confidence) for p in predictions] == [
        ("CCO", 2, (1.0, 2.0, 3.0, 4.0), 0.75),
        ("CCN", None, None, None),
    ]
    assert {p.seconds for p in predictions} == {3.5}
    export.write_predictions_csv(tmp_path / "predictions.csv", predictions)
