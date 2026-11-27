"""reads.jsonl -> vote_paper -> link_and_build -> finalize on a synthetic three-page paper.

The reads are written with the reader workers' own helpers (`_readers_common.assign_slots`,
`read_record`, `merge_reads`), three readers per region. Page 2 holds an R-group core with R1 on
the ring and R2 on the amide N, a nearer core whose labels do not match the table, and the
table; one reader numbers the core's slots the other way. Row 3e's R1 is the code M0, the label of
a morpholine drawn below the table and read with a methyl at its attachment (the legends fix).
Page 3 holds two more tables. The core of the first is drawn with its slot as `*` by MolGlyph and
as an ethyl and an isopropyl by the others (the attachments fix); the readers of the second core
label its slots three ways, and only MolScribe's labels match the columns (the slot_labels fix).
Only MolGlyph gives compound 1's stereo (the partial_stereo fix). OPSIN is a lookup table.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
from pipeline_helpers import fake_config
from rdkit import Chem

from molscout.pipeline import steps
from molscout.pipeline.finalize import finalize
from molscout.pipeline.inputs import Paper
from molscout.pipeline.link import link_and_build
from molscout.pipeline.schema import FIXES, LayoutBlock, Molecule, PageRecord, Region, read_jsonl, write_jsonl
from molscout.pipeline.vote import vote_paper
from molscout.scoring.smiles import canonical_smiles, stereo_stripped_smiles

WORKERS = Path(__file__).resolve().parents[2] / "src" / "molscout" / "pipeline" / "workers"
READERS = ("molglyph", "molscribe", "molnextr")
PAPER_ID = "e2e_paper"

LEAD = "C[C@H](NC(=O)c1ccc(OC)cc1)c1ccccc1"  # compound 1, drawn and named
TOLUANILIDE = "Cc1ccc(C(=O)Nc2ccccc2)cc1"  # compound 2; molglyph misreads its aniline as a pyridine
MISREAD = "Cc1ccc(C(=O)Nc2ccccn2)cc1"
ROWS = {  # the table on page 2, built on the core R1-c1ccc(C(=O)N-R2)cc1
    "3a": canonical_smiles("CNC(=O)c1ccccc1"),
    "3b": canonical_smiles("CCNC(=O)c1ccc(F)cc1"),
    "3c": canonical_smiles("CNC(=O)c1ccc(OC)cc1"),
    "3d": canonical_smiles("CNC(=O)c1ccc(Cl)cc1"),
    "3e": canonical_smiles("CNC(=O)c1ccc(N2CCOCC2)cc1"),  # R1 = M0, the morpholin-4-yl drawn as p2-r3
    "7a": canonical_smiles("Cn1nccc1-c1ccccc1"),  # page 3, on the core Cn1nccc1-R
    "8a": canonical_smiles("CNS(=O)(=O)c1ccc(F)cc1"),  # page 3, on the core R1-c1ccc(S(=O)(=O)N-R2)cc1
}
NAMES = {
    "(S)-4-Methoxy-N-(1-phenylethyl)benzamide": LEAD,
    "N-Ethyl-4-fluorobenzamide": ROWS["3b"],
    "N-Benzylbenzamide": "O=C(NCc1ccccc1)c1ccccc1",
}


def _common():
    sys.path.insert(0, str(WORKERS))
    try:
        return importlib.import_module("_readers_common")
    finally:
        sys.path.remove(str(WORKERS))


common = _common()


class FakeOpsin:
    def to_smiles(self, names: list[str], radicals: bool = False) -> list[str | None]:
        return [NAMES.get(name) for name in names]


def text(page, index, bbox, body, kind="text"):
    return LayoutBlock(page=page, index=index, type=kind, bbox=bbox, text=body)


TABLE = (
    "<table><tr><th>Compd</th><th>R<sup>1</sup></th><th>R<sup>2</sup></th><th>IC<sub>50</sub> (nM)</th></tr>"
    "<tr><td>3a</td><td>H</td><td>Me</td><td>14</td></tr><tr><td>3b</td><td>F</td><td>Et</td><td>25</td></tr>"
    "<tr><td>3c</td><td>OMe</td><td>Me</td><td>30</td></tr><tr><td>3d</td><td>Cl</td><td>Me</td><td>NT</td></tr>"
    "<tr><td>3e</td><td>M0</td><td>Me</td><td>40</td></tr></table>"
)
PYRAZOLES = (
    "<table><tr><th>Compd</th><th>R</th><th>IC<sub>50</sub> (nM)</th></tr>"
    "<tr><td>7a</td><td>Ph</td><td>8</td></tr></table>"
)
SULFONAMIDES = (
    "<table><tr><th>Compd</th><th>R<sup>1</sup></th><th>R<sup>2</sup></th><th>IC<sub>50</sub> (nM)</th></tr>"
    "<tr><td>8a</td><td>F</td><td>Me</td><td>9</td></tr></table>"
)
LAYOUT = [
    text(1, 0, [72, 60, 540, 90], "Benzamide inhibitors of a model kinase", "title"),
    text(1, 1, [165, 232, 185, 244], "1"),
    text(1, 2, [415, 232, 435, 244], "2"),
    text(1, 3, [72, 380, 540, 440], "Compound 1 inhibited the kinase with an IC50 of 12 nM, while 2 was inactive."),
    text(1, 4, [365, 607, 385, 619], "5"),
    text(2, 0, [72, 40, 540, 60], "Table 1. Activity of benzamides 3a–d.", "caption"),
    LayoutBlock(2, 2, "table", [72, 215, 540, 400], "", html=TABLE),
    text(3, 0, [72, 60, 540, 80], "Experimental section", "title"),
    text(3, 1, [72, 90, 540, 120], "(S)-4-Methoxy-N-(1-phenylethyl)benzamide (1). White solid."),
    text(3, 2, [72, 130, 540, 160],
         "N-Ethyl-4-fluorobenzamide (3b). HRMS (ESI) calcd for C9H11FNO [M+H]+ 168.0825, found 168.0822."),
    text(3, 3, [72, 170, 540, 190], "N-Benzylbenzamide (6). Colourless oil."),
    text(2, 3, [135, 525, 165, 537], "M0"),
    LayoutBlock(3, 4, "table", [72, 300, 540, 340], "", html=PYRAZOLES),
    LayoutBlock(3, 5, "table", [72, 500, 540, 540], "", html=SULFONAMIDES),
]
REGIONS = [
    ("p1-r1", 1, [100, 120, 250, 225], True),
    ("p1-r2", 1, [350, 120, 500, 225], True),
    ("p1-r3", 1, [100, 500, 250, 600], True),
    ("p1-r4", 1, [300, 500, 450, 600], True),
    ("p1-r5", 1, [100, 650, 250, 720], False),
    ("p2-r1", 2, [210, 80, 390, 180], True),
    ("p2-r2", 2, [210, 190, 390, 212], True),
    ("p2-r3", 2, [100, 450, 200, 520], True),
    ("p3-r1", 3, [210, 220, 390, 290], True),
    ("p3-r2", 3, [210, 420, 390, 490], True),
]
PAGE_3_CORES = {  # per reader: the pyrazole core's reading, and the labels of the sulfonamide core's two slots
    "molglyph": ("[1*]c1ccnn1C", {1: "R1", 2: "R3"}),
    "molscribe": ("CC(C)c1ccnn1C", {1: "R1", 2: "R2"}),
    "molnextr": ("CCc1ccnn1C", {1: "Rf", 2: "R2"}),
}


def drawn_core(smiles: str, labels: dict[int, str]) -> tuple[str, dict[str, str]]:
    """A core as a reader worker writes it: dummies with aliases, numbered by `assign_slots`."""
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            Chem.SetAtomAlias(atom, labels[atom.GetIsotope()])
    return common.assign_slots(mol)


def plain(region_id, reader, smiles):
    return common.read_record(region_id, reader, *common.smiles_read(smiles))


def reads_of(reader: str, other_numbering: set[str]) -> list[dict]:
    if reader in other_numbering:  # a reader that numbers R2 first
        core = common.read_record("p2-r1", reader, "[*:2]c1ccc(C(=O)N[*:1])cc1", {"1": "R2", "2": "R1"})
    else:
        core = common.read_record("p2-r1", reader, *drawn_core("[1*]c1ccc(C(=O)N[2*])cc1", {1: "R1", 2: "R2"}))
    tie = {"molglyph": "CC(=O)Nc1ccc(O)cc1", "molscribe": "CC(=O)Nc1ccc(N)cc1", "molnextr": "CC(=O)Nc1ccc(Cl)cc1"}
    pyrazole, labels = PAGE_3_CORES[reader]
    return [
        plain("p1-r1", reader, LEAD if reader == "molglyph" else stereo_stripped_smiles(LEAD)),
        plain("p1-r2", reader, MISREAD if reader == "molglyph" else TOLUANILIDE),
        plain("p1-r3", reader, tie[reader]),
        plain("p1-r4", reader, "CCO"),
        core,
        common.read_record("p2-r2", reader, *drawn_core("[1*]c1ccc(C(=O)O[2*])cc1", {1: "R1", 2: "X"})),
        plain("p2-r3", reader, "CN1CCOCC1"),
        common.read_record("p3-r1", reader, *(drawn_core(pyrazole, {1: "R"}) if "*" in pyrazole
                                               else common.smiles_read(pyrazole))),
        common.read_record("p3-r2", reader, *drawn_core("[1*]c1ccc(S(=O)(=O)N[2*])cc1", labels)),
    ]


def make_paper(root: Path, other_numbering=frozenset({"molscribe"})) -> Path:
    paper_dir = root / PAPER_ID
    write_jsonl(paper_dir / "layout.jsonl", LAYOUT)
    write_jsonl(paper_dir / "pages.jsonl", [PageRecord(p, 612.0, 792.0, f"pages/{p}.png", 200, True) for p in (1, 2, 3)])
    write_jsonl(paper_dir / "regions.jsonl", [Region(rid, page, box, f"regions/{rid}.png", 0.01, ok)
                                              for rid, page, box, ok in REGIONS])
    for reader in READERS:
        common.merge_reads(paper_dir, reader, reads_of(reader, set(other_numbering)))
    return paper_dir


def run_chain(paper_dir: Path, fixes=None) -> tuple[list[Molecule], dict[str, int], dict[str, int]]:
    voted = vote_paper(paper_dir, fixes=fixes)
    molecules, counts = finalize(paper_dir, link_and_build(paper_dir, FakeOpsin(), fixes), fixes=fixes)
    return molecules, voted, counts


def of(molecules, source):
    return {m.label: m for m in molecules if m.source == source}


@pytest.fixture
def paper(tmp_path) -> Path:
    return make_paper(tmp_path)


def test_reads_are_what_the_reader_workers_write(paper):
    lines = [json.loads(line) for line in (paper / "reads.jsonl").read_text().splitlines()]
    assert len(lines) == 3 * 9 and {line["reader"] for line in lines} == set(READERS)
    cores = {line["reader"]: (line["smiles"], line["slots"]) for line in lines if line["region_id"] == "p2-r1"}
    assert cores["molglyph"] == cores["molnextr"] == (canonical_smiles("[*:1]c1ccc(C(=O)N[*:2])cc1"),
                                                      {"1": "R1", "2": "R2"})
    assert cores["molscribe"][1] == {"1": "R2", "2": "R1"}


@pytest.mark.parametrize("other_numbering", [{"molscribe"}, {"molglyph", "molnextr"}, set()])
def test_each_table_row_has_the_right_substituent_on_the_right_slot(tmp_path, other_numbering):
    paper_dir = make_paper(tmp_path, frozenset(other_numbering))
    molecules, voted, _ = run_chain(paper_dir)
    core = {v["region_id"]: v for v in map(json.loads, (paper_dir / "voted.jsonl").read_text().splitlines())}["p2-r1"]
    assert core["agree"] == 3
    rows = of(molecules, "table")
    assert {label: m.smiles for label, m in rows.items()} == ROWS
    assert [rows[label].status for label in ("3a", "3b", "3c", "3d", "7a", "8a")] == ["accepted"] * 6
    assert voted["cores"] == 3


def test_the_paper_end_to_end(paper):
    molecules, voted, counts = run_chain(paper)
    assert voted == {"in": 10, "out": 7, "cores": 3, "dropped_tie": 2, "dropped_not_structure": 1}
    drawn = of(molecules, "drawn")
    assert (drawn["1"].smiles, drawn["1"].status) == (canonical_smiles(LEAD), "accepted")
    assert "duplicate merged: name (accepted)" in drawn["1"].reasons
    assert (drawn["2"].smiles, drawn["2"].status) == (canonical_smiles(TOLUANILIDE), "accepted")
    assert (drawn["5"].smiles, drawn["5"].status, drawn["5"].reasons[-1]) == (None, "abstained", "junk: too few heavy atoms")
    unlabelled = [m for m in molecules if m.source == "drawn" and m.label is None]
    assert sorted(m.reasons[0] for m in unlabelled) == [
        "R-group core for table p2-t2", "R-group core for table p3-t4", "R-group core for table p3-t5",
        "legend fragment M0 for table p2-t2", "no vote: tie between 3 structures",
        "open R-group slots and no table uses this core",
    ]
    assert {m.status for m in unlabelled} == {"abstained"} and {m.smiles for m in unlabelled} == {None}
    rows = of(molecules, "table")
    assert "printed formula C9H10FNO matches" in rows["3b"].reasons
    assert "duplicate merged: name (accepted)" in rows["3b"].reasons
    assert "label 3b is in an activity table or sentence" in rows["3b"].reasons
    assert rows["3d"].status == "accepted" and not any("activity table" in r for r in rows["3d"].reasons)
    assert set(of(molecules, "name")) == {"6"} and of(molecules, "name")["6"].status == "review"
    assert counts["in"] == counts["out"] + counts["dropped_duplicate"] and counts["dropped_duplicate"] == 2


def outcome(molecules, counts):
    return sorted(json.dumps([m.label, m.smiles, m.source, m.status, m.reasons], sort_keys=True) for m in molecules), counts


@pytest.mark.parametrize("fix", FIXES)
def test_every_fix_changes_the_output(paper, fix):
    on = outcome(*run_chain(paper)[::2])
    off = outcome(*run_chain(paper, {fix: False})[::2])
    assert off != on


def test_what_each_fix_changes(paper):
    def run(fix):
        return run_chain(paper, {fix: False})[0]

    assert all(m.label is None for m in run("labels") if m.source == "drawn")
    assert of(run("relevance"), "drawn")["2"].status == "accepted"
    assert of(run("tables"), "table") == {}
    assert {m.status for m in of(run("core_check"), "table").values() if m.table_id == "p2-t2"} == {"abstained"}
    assert of(run("vote"), "drawn")["2"].smiles == canonical_smiles(MISREAD)
    assert of(run("stereo"), "drawn")["1"].smiles == stereo_stripped_smiles(LEAD)
    assert of(run("junk"), "drawn")["5"].smiles == "CCO"
    assert of(run("names"), "name") == {}
    assert not any("printed formula" in r for r in of(run("formulas"), "table")["3b"].reasons)
    assert of(run("legends"), "table")["3e"].reasons == ["cannot read R1 = 'M0'"]
    assert of(run("attachments"), "table")["7a"].smiles is None  # p3-r1 is a tie, with no core for the table
    assert of(run("slot_labels"), "table")["8a"].smiles == canonical_smiles("CNC(=O)c1ccc(F)cc1")  # page 2's core
    assert of(run("partial_stereo"), "drawn")["1"].smiles == stereo_stripped_smiles(LEAD)


def test_the_runner_steps_call_the_real_vote_and_finalize(tmp_path):
    paper_dir = make_paper(tmp_path / "out" / "papers")
    config = fake_config(tmp_path)
    paper = Paper(PAPER_ID, tmp_path / f"{PAPER_ID}.pdf")
    voted = steps.vote(paper, paper_dir, config)
    write_jsonl(paper_dir / config.output("link"), link_and_build(paper_dir, FakeOpsin(), config.fixes))
    counts = steps.finalize(paper, paper_dir, config)
    for step_counts in (voted, counts):
        dropped = sum(v for k, v in step_counts.items() if k == "dropped" or k.startswith("dropped_"))
        assert step_counts["in"] == step_counts["out"] + dropped
    molecules = read_jsonl(paper_dir / config.output("finalize"), Molecule)
    assert {m.label: m.smiles for m in molecules if m.source == "table"} == ROWS
