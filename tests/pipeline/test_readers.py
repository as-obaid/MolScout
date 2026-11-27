"""read_molglyph / read_molscribe / read_molnextr: R groups as [*:n] with a slots map, and the read loop."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
from rdkit import Chem

WORKERS = Path(__file__).resolve().parents[2] / "src" / "molscout" / "pipeline" / "workers"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "b2"


def load(name):
    """Load a worker by path, as its environment runs it (no molscout import); the shared helper is
    imported by name, so the workers and the tests share one copy of it."""
    sys.path.insert(0, str(WORKERS))
    try:
        if name.startswith("_"):
            return importlib.import_module(name)
        spec = importlib.util.spec_from_file_location(f"b2_{name}", WORKERS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(WORKERS))


common = load("_readers_common")
molglyph = load("read_molglyph")
molscribe = load("read_molscribe")
molnextr = load("read_molnextr")


def canon(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


# --- labels ------------------------------------------------------------------------------------


@pytest.mark.parametrize("raw, label", [
    ("R1", "R1"),
    ("[R1]", "R1"),  # MolScribe's atom symbol
    ("R[1]", "R1"),  # MolParser script
    ("R^1", "R1"),
    ("R_{1}", "R1"),
    ("R<sup>1</sup>", "R1"),
    ("R¹", "R1"),
    ("R₂", "R2"),
    (" R 3 ", "R3"),
    ("R′", "R'"),
    ("R″", "R''"),
    ("R1'", "R1'"),
    ("1*", "R1"),  # MolNexTR's R-group symbols
    ("a*", "Ra"),
    ("X", "X"),
    ("Ar", "Ar"),
    ("OMe", "OMe"),
    ("", "*"),
    (None, "*"),
])
def test_normalize_label(raw, label):
    assert common.normalize_label(raw) == label


def test_labels_sort_numbered_r_groups_first_then_names_then_bare_attachments():
    labels = ["X", "R10", "*", "R2", "Ar", "R1", "R", "R1'"]
    assert sorted(labels, key=common.label_key) == ["R1", "R1'", "R2", "R10", "Ar", "R", "X", "*"]


# --- slots -------------------------------------------------------------------------------------


def test_assign_slots_numbers_dummies_by_label_and_keeps_the_label_text():
    mol = Chem.MolFromSmiles("[2*]c1ccc(C(=O)[1*])cc1")
    smiles, slots = common.assign_slots(mol)
    assert smiles == canon("[*:2]c1ccc(C(=O)[*:1])cc1")
    assert slots == {"1": "R1", "2": "R2"}


def test_assign_slots_prefers_the_alias_over_the_isotope():
    mol = Chem.MolFromSmiles("[1*]CC[3*]")
    Chem.SetAtomAlias(mol.GetAtomWithIdx(0), "X")
    smiles, slots = common.assign_slots(mol)
    assert slots == {"1": "R3", "2": "X"}
    assert smiles == canon("[*:2]CC[*:1]")


def test_assign_slots_is_independent_of_atom_order():
    first = common.assign_slots(Chem.MolFromSmiles("[1*]C(=O)c1ccc(Cl)cc1"))
    second = common.assign_slots(Chem.MolFromSmiles("Clc1ccc(cc1)C([1*])=O"))
    assert first == second == (canon("[*:1]C(=O)c1ccc(Cl)cc1"), {"1": "R1"})


def test_assign_slots_gives_repeated_labels_their_own_slots():
    smiles, slots = common.assign_slots(Chem.MolFromSmiles("[1*]CC[1*]"))
    assert smiles == canon("[*:1]CC[*:2]")
    assert slots == {"1": "R1", "2": "R1"}


def test_assign_slots_without_dummies_is_the_canonical_smiles():
    assert common.assign_slots(Chem.MolFromSmiles("OCC")) == ("CCO", {})


def test_smiles_read_keeps_what_rdkit_cannot_parse():
    assert common.smiles_read("C1CC") == ("C1CC", {})
    assert common.smiles_read("[1*]CC*") == (canon("[*:1]CC[*:2]"), {"1": "R1", "2": "*"})


# --- MolScribe and MolNexTR (graph readers) ----------------------------------------------------


def expanded_mol(aliases):
    """A molecule as MolScribe leaves it after expanding abbreviations: unexpanded atoms keep their alias."""
    mol = Chem.RWMol(Chem.MolFromSmiles("c1ccccc1"))
    for ring_atom, (alias, isotope) in zip((0, 3), aliases):
        dummy = Chem.Atom(0)
        dummy.SetIsotope(isotope)
        if alias:
            Chem.SetAtomAlias(dummy, alias)
        index = mol.AddAtom(dummy)
        mol.AddBond(ring_atom, index, Chem.BondType.SINGLE)
    return mol.GetMol()


def test_graph_read_labels_r_groups_and_unexpanded_text_from_aliases():
    smiles, slots = common.graph_read("ignored", expanded_mol([("R1", 1), ("Boc2N", 0)]))
    assert smiles == canon("[*:1]c1ccc([*:2])cc1")
    assert slots == {"1": "R1", "2": "Boc2N"}


def test_graph_read_falls_back_to_isotopes_when_aliases_are_missing():
    assert common.graph_read("x", expanded_mol([(None, 2), (None, 0)]))[1] == {"1": "R2", "2": "*"}


def test_graph_read_keeps_the_readers_own_smiles_when_there_are_no_dummies():
    assert common.graph_read("OCC", Chem.MolFromSmiles("CCO")) == ("OCC", {})


def test_graph_read_without_a_molecule_uses_the_smiles_or_fails():
    assert common.graph_read("[1*]CC", None) == (canon("[*:1]CC"), {"1": "R1"})
    for nothing in ("<invalid>", "", None):
        with pytest.raises(common.NoMolecule):
            common.graph_read(nothing, None)


def test_capture_expansions_keeps_each_expanded_molecule():
    module = types.SimpleNamespace(_expand_functional_group=lambda mol, mappings, debug=False: (mol + "!", "M"))
    captured = common.capture_expansions(module)
    assert module._expand_functional_group("S", {}, False) == ("S!", "M")
    assert captured == ["M"]


# --- MolGlyph (E-SMILES captions) ------------------------------------------------------------


ABBREVS = {"OMe": "OC", "Ph": "c1ccccc1", "Ar": "c1ccccc1", "CF3": "C(F)(F)F"}


@pytest.mark.parametrize("caption, smiles, slots", [
    ("c1ccccc1<sep>", "c1ccccc1", {}),
    ("*c1ccc(*)cc1<sep><a>0:R[1]</a><a>5:X</a>", "[*:1]c1ccc([*:2])cc1", {"1": "R1", "2": "X"}),
    ("*c1ccccc1<sep><a>0:OMe</a>", "COc1ccccc1", {}),
    ("*C(=O)N*<sep><a>0:Ph</a><a>4:R[2]</a>", "O=C(N[*:1])c1ccccc1", {"1": "R2"}),
    ("*c1ccccc1<sep><a>0:Ar</a>", "[*:1]c1ccccc1", {"1": "Ar"}),  # Ar may be any aryl: never expanded
    ("C*C<sep><a>1:OMe</a>", "C[*:1]C", {"1": "OMe"}),  # two bonds: cannot take a group
    ("*CC<sep><a>0:<dum></a>", "[*:1]CC", {"1": "*"}),
    ("*CC<sep>", "[*:1]CC", {"1": "*"}),
    ("CC*<sep><a>0:R1</a>", "CC[*:1]", {"1": "*"}),  # the group names a carbon: ignored
    ("*CC*<sep><a>0:R1</a><a>3:R1</a>", "[*:1]CC[*:2]", {"1": "R1", "2": "R1"}),
    ("*c1ccccc1<sep><a>0:R'</a>", "[*:1]c1ccccc1", {"1": "R'"}),
    ("*CC(F)(F)F<sep><a>0:R^1</a>|Sg:n|", "[*:1]CC(F)(F)F", {"1": "R1"}),
])
def test_caption_to_read(caption, smiles, slots):
    assert molglyph.caption_to_read(caption, ABBREVS) == (canon(smiles), slots)


def test_caption_slots_do_not_depend_on_the_order_molglyph_wrote_the_atoms():
    first = molglyph.caption_to_read("*C(=O)c1ccc(*)cc1<sep><a>0:R2</a><a>7:R1</a>", ABBREVS)
    second = molglyph.caption_to_read("*c1ccc(C(*)=O)cc1<sep><a>0:R1</a><a>6:R2</a>", ABBREVS)
    assert first == second == (canon("[*:2]C(=O)c1ccc([*:1])cc1"), {"1": "R1", "2": "R2"})


def test_runaway_and_invalid_captions_are_kept_for_the_checks_to_reject():
    assert molglyph.caption_to_read("C" * 512, ABBREVS) == ("C" * 512, {})
    assert molglyph.caption_to_read("C1CC<sep>", ABBREVS) == ("C1CC", {})


@pytest.mark.parametrize("caption", ["*c1ccccc1<sep><r>0:R</r>", "C1CCCCC1<sep><c>0:Ar</c>"])
def test_ring_bound_groups_have_no_fixed_attachment(caption):
    with pytest.raises(common.NoMolecule, match="ring"):
        molglyph.caption_to_read(caption, ABBREVS)


def test_load_abbrevs_reads_biominers_table():
    table = molglyph.load_abbrevs(FIXTURES / "abbrevs.csv")
    assert table["OMe"] == "OC" and table["CH3"] == "C" and table["t-Bu"] == "C(C)(C)C"
    assert molglyph.caption_to_read("*c1ccccc1<sep><a>0:t-Bu</a>", table) == (canon("CC(C)(C)c1ccccc1"), {})


# --- the read loop -----------------------------------------------------------------------------


def make_regions(tmp_path, paper_id="p1"):
    paper_dir = tmp_path / paper_id
    (paper_dir / "regions").mkdir(parents=True)
    regions = []
    for n, is_structure in enumerate((True, False, True, True), 1):
        crop = f"regions/p1-r{n}.png"
        (paper_dir / crop).write_bytes(b"png")
        regions.append({"region_id": f"p1-r{n}", "page": 1, "bbox": [0, 0, 1, 1], "crop": crop,
                        "structure_score": 0.0, "is_structure": is_structure})
    common.write_jsonl(paper_dir / "regions.jsonl", regions)
    return common.Paper(paper_id, tmp_path / "x.pdf", paper_dir)


def fake_predict(crop):
    return {
        "p1-r1.png": lambda: ("[*:1]CC", {"1": "R1"}),
        "p1-r3.png": lambda: (_ for _ in ()).throw(ValueError("decoder blew up")),
        "p1-r4.png": lambda: (_ for _ in ()).throw(common.NoMolecule("ring-bound group")),
    }[crop.name]()


@pytest.fixture
def logger(tmp_path):
    log = common.open_log(tmp_path / "read.log", "read_test")
    yield log
    common.close_log(log)


def test_run_reader_reads_structures_only_and_records_errors(tmp_path, logger):
    paper = make_regions(tmp_path)

    assert common.run_reader("molscribe", fake_predict, [paper], logger) == 0

    assert read_jsonl(paper.paper_dir / "reads.jsonl") == [
        {"region_id": "p1-r1", "reader": "molscribe", "smiles": "[*:1]CC", "slots": {"1": "R1"}, "error": None},
        {"region_id": "p1-r3", "reader": "molscribe", "smiles": None, "slots": {},
         "error": "ValueError: decoder blew up"},
        {"region_id": "p1-r4", "reader": "molscribe", "smiles": None, "slots": {},
         "error": "NoMolecule: ring-bound group"},
    ]


def test_run_reader_replaces_only_its_own_reads(tmp_path, logger):
    paper = make_regions(tmp_path)
    common.write_jsonl(paper.paper_dir / "reads.jsonl", [
        {"region_id": "p1-r1", "reader": "molscribe", "smiles": "OLD", "slots": {}, "error": None},
        {"region_id": "p1-r1", "reader": "molglyph", "smiles": "CC", "slots": {}, "error": None},
    ])

    common.run_reader("molscribe", fake_predict, [paper], logger)

    reads = read_jsonl(paper.paper_dir / "reads.jsonl")
    assert [(r["reader"], r["region_id"]) for r in reads] == [
        ("molglyph", "p1-r1"), ("molscribe", "p1-r1"), ("molscribe", "p1-r3"), ("molscribe", "p1-r4"),
    ]
    assert "OLD" not in json.dumps(reads)


def test_run_reader_records_missing_crops_and_missing_regions(tmp_path, logger):
    paper = make_regions(tmp_path)
    (paper.paper_dir / "regions" / "p1-r1.png").unlink()
    empty = common.Paper("p2", tmp_path / "y.pdf", tmp_path / "p2")

    assert common.run_reader("molscribe", fake_predict, [paper, empty], logger) == 1

    first = read_jsonl(paper.paper_dir / "reads.jsonl")[0]
    assert first["smiles"] is None and first["error"] == "NoMolecule: crop not found: regions/p1-r1.png"
    (error,) = read_jsonl(empty.paper_dir / "errors.jsonl")
    assert error["stage"] == "read_molscribe" and "regions.jsonl" in error["error"]


def test_a_malformed_paper_fails_alone(tmp_path, logger):
    bad = make_regions(tmp_path, paper_id="bad")
    common.write_jsonl(bad.paper_dir / "regions.jsonl", [{"page": 1, "is_structure": True}])
    corrupt = make_regions(tmp_path, paper_id="corrupt")
    (corrupt.paper_dir / "reads.jsonl").write_text("{not json\n")
    good = make_regions(tmp_path, paper_id="good")

    assert common.run_reader("molscribe", fake_predict, [bad, corrupt, good], logger) == 2

    assert read_jsonl(bad.paper_dir / "errors.jsonl")[0]["stage"] == "read_molscribe"
    assert read_jsonl(corrupt.paper_dir / "errors.jsonl")[0]["stage"] == "read_molscribe"
    assert len(read_jsonl(good.paper_dir / "reads.jsonl")) == 3


def test_reads_merge_under_lockf_when_flock_is_refused(tmp_path, logger, monkeypatch):
    fcntl = pytest.importorskip("fcntl")
    paper = make_regions(tmp_path)

    def no_locks(handle, operation):
        raise OSError(37, "No locks available")

    monkeypatch.setattr(fcntl, "flock", no_locks)
    assert common.run_reader("molscribe", fake_predict, [paper], logger) == 0
    assert len(read_jsonl(paper.paper_dir / "reads.jsonl")) == 3


def test_run_reader_clears_its_stale_errors_and_keeps_other_stages(tmp_path, logger):
    paper = make_regions(tmp_path)
    common.append_error(paper.paper_dir, "read_molscribe", "p1", "an old failure")
    common.append_error(paper.paper_dir, "segment", "p1", "a segment failure")
    assert common.run_reader("molscribe", fake_predict, [paper], logger) == 0
    assert [e["stage"] for e in read_jsonl(paper.paper_dir / "errors.jsonl")] == ["segment"]


def reader_argv(tmp_path, papers):
    manifest = tmp_path / "work.jsonl"
    manifest.write_text("".join(json.dumps({"paper_id": p.paper_id, "pdf": str(p.pdf), "paper_dir": str(p.paper_dir)})
                                + "\n" for p in papers))
    return ["--manifest", str(manifest), "--log", str(tmp_path / "r.log"), "--checkpoint", "w.pt"]


def test_reader_main_exits_4_when_a_paper_failed(tmp_path, monkeypatch):
    good, empty = make_regions(tmp_path), common.Paper("p2", tmp_path / "y.pdf", tmp_path / "p2")
    monkeypatch.setattr(molscribe, "load_reader", lambda args, logger: fake_predict)
    assert molscribe.main(reader_argv(tmp_path, [good, empty])) == 4
    assert len(read_jsonl(good.paper_dir / "reads.jsonl")) == 3


def test_reader_main_exits_1_when_reads_cannot_be_locked(tmp_path, monkeypatch):
    paper = make_regions(tmp_path)

    def unlockable(paper_dir, reader, records):
        raise common.LockError("cannot lock reads.jsonl: flock failed and lockf failed")

    monkeypatch.setattr(molscribe, "load_reader", lambda args, logger: fake_predict)
    monkeypatch.setattr(common, "merge_reads", unlockable)
    assert molscribe.main(reader_argv(tmp_path, [paper])) == 1
    assert "lockf failed" in (tmp_path / "r.log").read_text()


@pytest.mark.parametrize("caption", ["", "<sep>", "  <sep><a>0:R1</a>"])
def test_empty_captions_give_no_molecule(caption):
    with pytest.raises(common.NoMolecule, match="empty"):
        molglyph.caption_to_read(caption, ABBREVS)


def test_an_empty_graph_gives_no_molecule():
    with pytest.raises(common.NoMolecule, match="empty"):
        common.graph_read("", Chem.RWMol().GetMol())


def test_run_reader_stops_when_the_runtime_looks_broken(tmp_path, logger, monkeypatch):
    paper = make_regions(tmp_path)
    monkeypatch.setattr(common, "MAX_CONSECUTIVE_FAILURES", 2)

    def broken(crop):
        raise RuntimeError("CUDA error")

    with pytest.raises(common.BrokenRuntime, match="in a row"):
        common.run_reader("molscribe", broken, [paper], logger)


def test_reader_main_exits_1_when_the_runtime_breaks(tmp_path, monkeypatch):
    paper = make_regions(tmp_path)
    manifest = tmp_path / "work.jsonl"
    manifest.write_text(json.dumps({"paper_id": "p1", "pdf": str(paper.pdf), "paper_dir": str(paper.paper_dir)}))
    monkeypatch.setattr(common, "MAX_CONSECUTIVE_FAILURES", 1)

    def broken(crop):
        raise RuntimeError("CUDA error")

    monkeypatch.setattr(molscribe, "load_reader", lambda args, logger: broken)
    argv = ["--manifest", str(manifest), "--log", str(tmp_path / "r.log"), "--checkpoint", "w.pt"]
    assert molscribe.main(argv) == 1
    assert "looks broken" in (tmp_path / "r.log").read_text()


def test_no_molecule_does_not_count_as_a_broken_runtime(tmp_path, logger, monkeypatch):
    paper = make_regions(tmp_path)
    monkeypatch.setattr(common, "MAX_CONSECUTIVE_FAILURES", 2)

    def nothing(crop):
        raise common.NoMolecule("empty")

    assert common.run_reader("molscribe", nothing, [paper], logger) == 0


# --- worker entry points ---------------------------------------------------------------------


@pytest.mark.parametrize("worker, reader", [(molglyph, "molglyph"), (molscribe, "molscribe"), (molnextr, "molnextr")])
def test_reader_main_writes_reads_with_a_loaded_model(worker, reader, tmp_path, monkeypatch):
    paper = make_regions(tmp_path)
    manifest = tmp_path / "work.jsonl"
    manifest.write_text(json.dumps({"paper_id": "p1", "pdf": str(paper.pdf), "paper_dir": str(paper.paper_dir)}))
    loaded = {}

    def fake_load(args, logger):
        loaded["args"] = args
        return lambda crop: ("CCO", {})

    monkeypatch.setattr(worker, "load_reader", fake_load)
    argv = ["--manifest", str(manifest), "--log", str(tmp_path / "r.log"), "--checkpoint", "w.pt", "--device", "cpu"]
    if worker is molglyph:
        argv += ["--abbrevs", str(FIXTURES / "abbrevs.csv")]

    assert worker.main(argv) == 0
    assert loaded["args"].device == "cpu"
    reads = read_jsonl(paper.paper_dir / "reads.jsonl")
    assert {r["reader"] for r in reads} == {reader} and len(reads) == 3


@pytest.mark.parametrize("worker", [molglyph, molscribe, molnextr])
def test_reader_main_exit_codes_for_setup_errors(worker, tmp_path, monkeypatch):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("[]\n")
    extra = ["--checkpoint", "w.pt"] + (["--abbrevs", str(FIXTURES / "abbrevs.csv")] if worker is molglyph else [])
    assert worker.main(["--manifest", str(bad), "--log", str(tmp_path / "a.log"), *extra]) == 2

    good = tmp_path / "work.jsonl"
    good.write_text(json.dumps({"paper_id": "p", "pdf": "p.pdf", "paper_dir": str(tmp_path / "p")}))

    def broken(args, logger):
        raise ImportError("no torch here")

    monkeypatch.setattr(worker, "load_reader", broken)
    assert worker.main(["--manifest", str(good), "--log", str(tmp_path / "b.log"), *extra]) == 1
    assert "no torch here" in (tmp_path / "b.log").read_text()


def test_molnextr_keeps_the_benchmarks_pad_to_square_switch():
    parser = common.worker_parser("x")
    molnextr.add_arguments(parser)
    args = parser.parse_args(["--manifest", "m", "--log", "l", "--checkpoint", "c", "--pad-to-square"])
    assert args.pad_to_square is True and args.device == "cuda"
