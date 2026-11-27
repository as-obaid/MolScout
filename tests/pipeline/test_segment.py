"""segment.py (decimer_ai env) with fake models, and the shared worker helpers it uses."""

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

WORKERS = Path(__file__).resolve().parents[2] / "src" / "molscout" / "pipeline" / "workers"


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
segment = load("segment")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write_manifest(path, papers):
    path.write_text("".join(json.dumps(p) + "\n" for p in papers))
    return path


# --- shared helpers -----------------------------------------------------------------------------


def test_px_to_points_scales_by_page_size_over_image_size():
    page = {"page": 1, "width_pt": 200.0, "height_pt": 100.0}
    # pdfium stretches the page over ceil(pt * 300 / 72) pixels, so 834 px span all 200 pt
    assert common.px_to_points([0, 0, 834, 417], page, (834, 417)) == [0.0, 0.0, 200.0, 100.0]
    assert common.px_to_points([417, 0, 834, 208.5], page, (834, 417)) == [100.0, 0.0, 200.0, 50.0]


def test_px_to_points_on_a_letter_page_at_300_dpi():
    page = {"page": 3, "width_pt": 612.0, "height_pt": 792.0}
    assert common.px_to_points([300, 600, 900, 1200], page, (2550, 3300)) == [72.0, 144.0, 216.0, 288.0]


def test_read_manifest_resolves_paths_and_rejects_bad_lines(tmp_path):
    good = write_manifest(tmp_path / "work.jsonl", [
        {"paper_id": "a", "pdf": "a.pdf", "paper_dir": str(tmp_path / "a")},
    ])
    (paper,) = common.read_manifest(good)
    assert paper.paper_id == "a"
    assert paper.pdf.is_absolute() and paper.paper_dir == tmp_path / "a"

    for bad in (
        "not json\n",
        '["a", "b", "c"]\n',
        '{"paper_id": "a", "pdf": "a.pdf"}\n',
        '{"paper_id": "", "pdf": "a.pdf", "paper_dir": "x"}\n',
        '{"paper_id": "a", "pdf": "a.pdf", "paper_dir": "x"}\n{"paper_id": "a", "pdf": "b.pdf", "paper_dir": "y"}\n',
    ):
        (tmp_path / "bad.jsonl").write_text(bad)
        with pytest.raises(common.ManifestError):
            common.read_manifest(tmp_path / "bad.jsonl")


def test_read_manifest_skips_blank_lines(tmp_path):
    path = tmp_path / "work.jsonl"
    path.write_text('\n{"paper_id": "a", "pdf": "a.pdf", "paper_dir": "x"}\n\n')
    assert [p.paper_id for p in common.read_manifest(path)] == ["a"]


def test_write_jsonl_is_atomic_and_round_trips(tmp_path):
    path = tmp_path / "d" / "x.jsonl"
    common.write_jsonl(path, [{"a": 1}, {"b": [1, 2]}])
    assert common.read_jsonl(path) == [{"a": 1}, {"b": [1, 2]}]
    assert not list(path.parent.glob("*.part"))
    common.write_jsonl(path, [])
    assert path.read_text() == ""


def test_append_error_adds_one_line_per_failure(tmp_path):
    common.append_error(tmp_path, "segment", "p1", "OSError: gone", page=3)
    common.append_error(tmp_path, "read_molscribe", "p1", "boom")
    assert read_jsonl(tmp_path / "errors.jsonl") == [
        {"stage": "segment", "paper_id": "p1", "error": "OSError: gone", "page": 3},
        {"stage": "read_molscribe", "paper_id": "p1", "error": "boom"},
    ]


# The Python of each worker's environment on Explorer; the shared _*.py helpers run in all of them.
WORKER_PYTHON = {
    "segment.py": (3, 9), "cells.py": (3, 9),  # decimer_ai
    "read_molglyph.py": (3, 10), "read_molscribe.py": (3, 10), "read_molnextr.py": (3, 10),
    "join_32b.py": (3, 10),  # biominer
    "mineru_parse.py": (3, 12), "ocr_labels.py": (3, 12),  # mineru
}


@pytest.mark.parametrize("path", sorted(WORKERS.glob("*.py")), ids=lambda path: path.name)
def test_worker_sources_parse_on_their_environments_python_and_never_import_molscout(path):
    assert path.name.startswith("_") or path.name in WORKER_PYTHON, f"{path.name}: add its environment's Python"
    source = path.read_text()
    tree = ast.parse(source, feature_version=WORKER_PYTHON.get(path.name, (3, 9)))
    imported = {
        (node.module or "") if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(name.split(".")[0] == "molscout" for name in imported), path.name
    assert "from __future__ import annotations" in source, path.name


# --- segment ------------------------------------------------------------------------------------


class FakeModels:
    """Stands in for DECIMER Segmentation and the classifier: boxes are (y0, x0, y1, x1) pixels."""

    threshold = 0.000089

    def __init__(self, pages, scores, fail_on=()):
        self.pages, self.scores, self.fail_on = pages, list(scores), set(fail_on)
        self.scored = []

    def segment(self, image):
        if image.name in self.fail_on:
            raise RuntimeError(f"cannot segment {image.name}")
        return (834, 417), [(f"crop-of-{image.stem}-{i}", box) for i, box in enumerate(self.pages[image.name])]

    def save(self, crop, path):
        path.write_text(crop)

    def score(self, path):
        self.scored.append(path.name)
        return self.scores.pop(0)


def make_paper(tmp_path, pages=2, paper_id="p1"):
    paper_dir = tmp_path / paper_id
    (paper_dir / "pages").mkdir(parents=True)
    records = []
    for n in range(1, pages + 1):
        (paper_dir / "pages" / f"p{n:03d}.png").write_bytes(b"png")
        records.append({"page": n, "width_pt": 200.0, "height_pt": 100.0, "image": f"pages/p{n:03d}.png",
                        "dpi": 300, "vector": True})
    common.write_jsonl(paper_dir / "pages.jsonl", records)
    return common.Paper(paper_id, tmp_path / f"{paper_id}.pdf", paper_dir)


def test_segment_paper_writes_regions_with_point_boxes_crops_and_scores(tmp_path):
    paper = make_paper(tmp_path)
    models = FakeModels(
        {"p001.png": [(0, 0, 417, 417), (0, 417, 208.5, 834)], "p002.png": [(104.25, 208.5, 312.75, 625.5)]},
        scores=[0.00001, 0.9, 0.000089],
    )

    records, failed_pages = segment.segment_paper(paper, models)

    assert records == read_jsonl(paper.paper_dir / "regions.jsonl") and failed_pages == 0
    assert [r["region_id"] for r in records] == ["p1-r1", "p1-r2", "p2-r1"]
    assert list(records[0]) == ["region_id", "page", "bbox", "crop", "structure_score", "is_structure"]
    assert records[0]["bbox"] == [0.0, 0.0, 100.0, 100.0]
    assert records[1]["bbox"] == [100.0, 0.0, 200.0, 50.0]
    assert records[2]["bbox"] == [50.0, 25.0, 150.0, 75.0]
    assert [r["crop"] for r in records] == ["regions/p1-r1.png", "regions/p1-r2.png", "regions/p2-r1.png"]
    assert (paper.paper_dir / "regions" / "p1-r2.png").read_text() == "crop-of-p001-1"
    # the classifier's score is low for structures; at or under the threshold is a structure
    assert [r["structure_score"] for r in records] == [0.00001, 0.9, 0.000089]
    assert [r["is_structure"] for r in records] == [True, False, True]


def test_segment_paper_clears_stale_crops_and_writes_empty_regions_for_no_drawings(tmp_path):
    paper = make_paper(tmp_path, pages=1)
    (paper.paper_dir / "regions").mkdir()
    (paper.paper_dir / "regions" / "p7-r3.png").write_text("old")
    (paper.paper_dir / "regions" / "notes.txt").write_text("keep")

    assert segment.segment_paper(paper, FakeModels({"p001.png": []}, scores=[])) == ([], 0)

    assert (paper.paper_dir / "regions.jsonl").read_text() == ""
    assert sorted(p.name for p in (paper.paper_dir / "regions").iterdir()) == ["notes.txt"]


def test_a_failing_page_is_logged_and_the_other_pages_are_kept(tmp_path):
    paper = make_paper(tmp_path, pages=2)
    models = FakeModels({"p002.png": [(0, 0, 10, 10)]}, scores=[0.0], fail_on={"p001.png"})

    records, failed_pages = segment.segment_paper(paper, models)

    assert [r["region_id"] for r in records] == ["p2-r1"] and failed_pages == 1
    (error,) = read_jsonl(paper.paper_dir / "errors.jsonl")
    assert error["stage"] == "segment" and error["page"] == 1
    assert error["error"] == "RuntimeError: cannot segment p001.png"


def test_a_score_that_is_not_a_number_is_not_a_structure(tmp_path):
    paper = make_paper(tmp_path, pages=1)
    records, _ = segment.segment_paper(paper, FakeModels({"p001.png": [(0, 0, 10, 10)]}, scores=[float("nan")]))
    assert records[0]["structure_score"] is None and records[0]["is_structure"] is False


def test_new_regions_remove_the_reads_and_votes_made_from_the_old_ones(tmp_path):
    paper = make_paper(tmp_path, pages=1)
    for name in ("reads.jsonl", "voted.jsonl", ".reads.lock", "layout.jsonl"):
        (paper.paper_dir / name).write_text("{}\n")

    segment.segment_paper(paper, FakeModels({"p001.png": [(0, 0, 10, 10)]}, scores=[0.0]))

    left = {p.name for p in paper.paper_dir.iterdir()}
    assert {"regions.jsonl", "layout.jsonl"} <= left and not left & {"reads.jsonl", "voted.jsonl", ".reads.lock"}


def run_logged(tmp_path, papers, models):
    logger = common.open_log(tmp_path / "segment.log", "segment")
    try:
        return segment.run(papers, models, logger)
    finally:
        common.close_log(logger)


def test_a_rerun_clears_old_segment_errors_and_a_failed_page_fails_the_paper(tmp_path):
    paper = make_paper(tmp_path, pages=2)
    common.append_error(paper.paper_dir, "segment", "p1", "an old failure")
    common.append_error(paper.paper_dir, "render", "p1", "a render failure")
    models = FakeModels({"p002.png": [(0, 0, 10, 10)]}, scores=[0.0], fail_on={"p001.png"})

    assert run_logged(tmp_path, [paper], models) == 1

    errors = read_jsonl(paper.paper_dir / "errors.jsonl")
    assert [(e["stage"], e.get("page")) for e in errors] == [("render", None), ("segment", 1)]


def test_a_paper_without_pages_loses_its_old_regions(tmp_path):
    paper = make_paper(tmp_path, pages=1)
    (paper.paper_dir / "pages.jsonl").unlink()
    (paper.paper_dir / "regions.jsonl").write_text("{}\n")
    assert run_logged(tmp_path, [paper], FakeModels({}, scores=[])) == 1
    assert not (paper.paper_dir / "regions.jsonl").exists()


def test_run_segments_every_paper_and_records_papers_without_pages(tmp_path):
    good = make_paper(tmp_path, pages=1, paper_id="good")
    missing = common.Paper("missing", tmp_path / "m.pdf", tmp_path / "missing")
    models = FakeModels({"p001.png": [(0, 0, 10, 10)]}, scores=[0.0])
    logger = common.open_log(tmp_path / "segment.log", "segment")
    try:
        failed = segment.run([good, missing], models, logger)
    finally:
        common.close_log(logger)

    assert failed == 1
    assert len(read_jsonl(good.paper_dir / "regions.jsonl")) == 1
    (error,) = read_jsonl(missing.paper_dir / "errors.jsonl")
    assert error["stage"] == "segment" and "pages.jsonl" in error["error"]
    assert not (missing.paper_dir / "regions.jsonl").exists()
    text = (tmp_path / "segment.log").read_text()
    assert "good: 1 regions, 1 structures" in text and "missing" in text


def test_main_exits_2_on_a_bad_manifest_and_1_when_models_do_not_load(tmp_path, monkeypatch):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{}\n")
    assert segment.main(["--manifest", str(bad), "--log", str(tmp_path / "a.log")]) == 2

    good = write_manifest(tmp_path / "work.jsonl", [{"paper_id": "a", "pdf": "a.pdf", "paper_dir": str(tmp_path)}])

    def broken(device, logger):
        raise ImportError("no tensorflow here")

    monkeypatch.setattr(segment, "load_models", broken)
    monkeypatch.chdir(tmp_path)
    assert segment.main(["--manifest", str(good), "--log", str(tmp_path / "b.log")]) == 1
    assert "no tensorflow here" in (tmp_path / "b.log").read_text()


def test_main_runs_with_loaded_models_from_a_scratch_folder(tmp_path, monkeypatch):
    paper = make_paper(tmp_path, pages=1, paper_id="a")
    manifest = write_manifest(tmp_path / "work.jsonl", [
        {"paper_id": "a", "pdf": str(paper.pdf), "paper_dir": str(paper.paper_dir)},
    ])
    seen = {}

    def fake_load(device, logger):
        seen["device"], seen["cwd"] = device, Path.cwd()
        return FakeModels({"p001.png": [(0, 0, 10, 10)]}, scores=[0.0])

    monkeypatch.setattr(segment, "load_models", fake_load)
    monkeypatch.chdir(tmp_path)
    code = segment.main(["--manifest", str(manifest), "--log", str(tmp_path / "s.log"), "--device", "cpu"])

    assert code == 0 and seen["device"] == "cpu"
    assert not (paper.paper_dir / "errors.jsonl").exists()
    assert seen["cwd"] != tmp_path  # the classifier writes caffeine_mod.png into the working directory
    assert len(read_jsonl(paper.paper_dir / "regions.jsonl")) == 1


def test_main_exits_4_when_a_paper_failed(tmp_path, monkeypatch):
    paper = make_paper(tmp_path, pages=1, paper_id="a")
    manifest = write_manifest(tmp_path / "work.jsonl", [
        {"paper_id": "a", "pdf": str(paper.pdf), "paper_dir": str(paper.paper_dir)},
        {"paper_id": "b", "pdf": "b.pdf", "paper_dir": str(tmp_path / "b")},
    ])
    monkeypatch.setattr(segment, "load_models", lambda device, logger: FakeModels({"p001.png": []}, scores=[]))
    monkeypatch.chdir(tmp_path)
    assert segment.main(["--manifest", str(manifest), "--log", str(tmp_path / "s.log")]) == 4
    assert (paper.paper_dir / "regions.jsonl").read_text() == ""


def test_main_exits_1_when_errors_cannot_be_locked(tmp_path, monkeypatch):
    paper = make_paper(tmp_path, pages=1, paper_id="a")
    manifest = write_manifest(tmp_path / "work.jsonl", [
        {"paper_id": "a", "pdf": str(paper.pdf), "paper_dir": str(paper.paper_dir)}])

    def unlockable(paper_dir, stage):
        raise common.LockError("cannot lock errors.jsonl: flock failed and lockf failed")

    monkeypatch.setattr(segment, "load_models", lambda device, logger: FakeModels({"p001.png": []}, scores=[]))
    monkeypatch.setattr(common, "clear_stage_errors", unlockable)
    monkeypatch.chdir(tmp_path)
    assert segment.main(["--manifest", str(manifest), "--log", str(tmp_path / "s.log")]) == 1
    assert "lockf failed" in (tmp_path / "s.log").read_text()
