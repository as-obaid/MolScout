"""ocr_labels.py (mineru env) with a fake OCR engine: figure areas, line boxes in points, counts and errors."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

WORKERS = Path(__file__).resolve().parents[2] / "src" / "molscout" / "pipeline" / "workers"
PAGE_PT = (612.0, 792.0)
PAGE_PX = (2550, 3300)  # 300 dpi: 612 * 300 / 72 by 792 * 300 / 72
SCALE = PAGE_PX[0] / PAGE_PT[0]


def load(name):
    """Load the worker by path, as its environment runs it (no molscout import)."""
    sys.path.insert(0, str(WORKERS))
    try:
        spec = importlib.util.spec_from_file_location(f"i1_{name}", WORKERS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # dataclasses look their module up while the class is made
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(WORKERS))


ocr = load("ocr_labels")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, records):
    Path(path).write_text("".join(json.dumps(record) + "\n" for record in records))


def px(points):
    return [round(value * SCALE) for value in points]


class FakeEngine:
    """Stands in for MinerU's PP-OCRv6: `lines` holds (page box in points, text, score) per page image name;
    `read` returns the lines whose centre lies in the asked pixel box, boxes in page pixels."""

    def __init__(self, lines, fail_on=()):
        self.lines, self.fail_on, self.calls = lines, set(fail_on), []

    def image_size(self, image):
        return PAGE_PX

    def read(self, image, box):
        self.calls.append((image.name, tuple(box)))
        if tuple(box) in self.fail_on:
            raise RuntimeError("onnx session failed")
        x0, y0, x1, y1 = box
        found = []
        for points, text, score in self.lines.get(image.name, []):
            bx = px(points)
            cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                found.append((bx, text, score))
        return found


def make_paper(tmp_path, paper_id="p1", layout=None, regions=None, pages=2):
    folder = tmp_path / paper_id
    folder.mkdir()
    write_jsonl(folder / "pages.jsonl", [
        {"page": n, "width_pt": PAGE_PT[0], "height_pt": PAGE_PT[1], "image": f"pages/p{n:03d}.png", "dpi": 300,
         "vector": True}
        for n in range(1, pages + 1)
    ])
    write_jsonl(folder / "layout.jsonl", layout if layout is not None else [
        {"page": 1, "index": 0, "type": "text", "bbox": [50, 50, 550, 90], "text": "Body", "html": None,
         "parent": None, "image": None},
        {"page": 1, "index": 1, "type": "image", "bbox": [100, 100, 400, 300], "text": "", "html": None,
         "parent": None, "image": "layout_images/p1-1.jpg"},
    ])
    write_jsonl(folder / "regions.jsonl", regions if regions is not None else [
        {"region_id": "p1-r1", "page": 1, "bbox": [120, 120, 200, 200], "crop": "regions/p1-r1.png",
         "structure_score": 0.0, "is_structure": True},
        {"region_id": "p2-r1", "page": 2, "bbox": [300, 400, 380, 460], "crop": "regions/p2-r1.png",
         "structure_score": 0.0, "is_structure": True},
        {"region_id": "p2-r2", "page": 2, "bbox": [50, 600, 90, 640], "crop": "regions/p2-r2.png",
         "structure_score": 0.9, "is_structure": False},
    ])
    return ocr.common.Paper(paper_id, tmp_path / f"{paper_id}.pdf", folder)


LINES = {
    "p001.png": [([150, 210, 170, 222], "12a", 0.98), ([140, 150, 150, 160], "N", 0.95)],
    "p002.png": [([330, 470, 345, 480], "(7)", 0.91)],
}


# --- areas --------------------------------------------------------------------------------------


def test_areas_are_image_blocks_with_a_margin_and_drawings_outside_them_padded(tmp_path):
    paper = make_paper(tmp_path)
    pages, layout, regions = (read_jsonl(paper.paper_dir / name) for name in ("pages.jsonl", "layout.jsonl",
                                                                               "regions.jsonl"))
    areas = ocr.figure_areas(pages, layout, regions)
    margin = ocr.IMAGE_MARGIN_PT
    left, top, right, bottom = ocr.REGION_PAD_PT
    assert [(a.page, a.source) for a in areas] == [(1, "image:p1-1"), (2, "region:p2-r1")]
    assert areas[0].bbox == (100 - margin, 100 - margin, 400 + margin, 300 + margin)
    assert areas[1].bbox == (300 - left, 400 - top, 380 + right, 460 + bottom)


def test_table_cell_drawings_are_not_read_for_labels(tmp_path):
    paper = make_paper(tmp_path)
    pages, layout, regions = (read_jsonl(paper.paper_dir / name) for name in ("pages.jsonl", "layout.jsonl",
                                                                               "regions.jsonl"))
    cell = {**regions[-1], "region_id": "p2-t3-r1-c1", "kind": "cell"}
    assert [a.source for a in ocr.figure_areas(pages, layout, [*regions, cell])] == ["image:p1-1", "region:p2-r1"]


def test_areas_are_clipped_to_the_page(tmp_path):
    layout = [{"page": 1, "index": 0, "type": "image", "bbox": [0, 5, 612, 790], "text": "", "html": None,
               "parent": None, "image": None}]
    paper = make_paper(tmp_path, layout=layout, regions=[], pages=1)
    pages = read_jsonl(paper.paper_dir / "pages.jsonl")
    (area,) = ocr.figure_areas(pages, layout, [])
    assert area.bbox == (0.0, 0.0, 612.0, 792.0)


def test_a_figure_on_a_page_that_was_not_rendered_is_an_error(tmp_path):
    regions = [{"region_id": "p3-r1", "page": 3, "bbox": [1, 1, 2, 2], "crop": "c", "structure_score": 0.0,
                "is_structure": True}]
    with pytest.raises(ValueError, match="page 3"):
        ocr.figure_areas([{"page": 1, "width_pt": 612, "height_pt": 792, "image": "x.png"}], [], regions)


# --- one paper ------------------------------------------------------------------------------------


def test_lines_are_written_in_points_with_their_area(tmp_path):
    paper = make_paper(tmp_path)
    failed = ocr.ocr_paper(paper, FakeEngine(LINES))
    assert failed == 0
    lines = read_jsonl(paper.paper_dir / "ocr.jsonl")
    assert [(r["page"], r["text"], r["source_block"]) for r in lines] == [
        (1, "N", "image:p1-1"), (1, "12a", "image:p1-1"), (2, "(7)", "region:p2-r1"),
    ]
    assert set(lines[0]) == {"page", "bbox", "text", "score", "source_block"}
    assert lines[1]["bbox"] == pytest.approx([150, 210, 170, 222], abs=0.3)
    assert lines[1]["score"] == pytest.approx(0.98)


def test_counts_are_written_for_the_runner(tmp_path):
    paper = make_paper(tmp_path)
    ocr.ocr_paper(paper, FakeEngine({"p001.png": LINES["p001.png"]}))
    counts = json.loads((paper.paper_dir / ".counts" / "ocr_labels.json").read_text())
    assert counts == {"in": 2, "out": 1, "dropped_no_text": 1, "dropped_failed": 0, "lines": 2,
                      "lines_duplicate": 0}


def test_a_line_seen_from_two_overlapping_areas_is_written_once(tmp_path):
    layout = [
        {"page": 1, "index": 0, "type": "image", "bbox": [100, 100, 300, 300], "text": "", "html": None,
         "parent": None, "image": None},
        {"page": 1, "index": 1, "type": "image", "bbox": [200, 200, 400, 400], "text": "", "html": None,
         "parent": None, "image": None},
    ]
    paper = make_paper(tmp_path, layout=layout, regions=[], pages=1)
    ocr.ocr_paper(paper, FakeEngine({"p001.png": [([240, 240, 260, 250], "5", 0.9)]}))
    assert [r["source_block"] for r in read_jsonl(paper.paper_dir / "ocr.jsonl")] == ["image:p1-0"]
    counts = json.loads((paper.paper_dir / ".counts" / "ocr_labels.json").read_text())
    assert counts["lines"] == 1 and counts["lines_duplicate"] == 1 and counts["out"] == 2


def test_an_area_that_fails_is_an_error_and_the_others_are_kept(tmp_path):
    paper = make_paper(tmp_path)
    area = ocr.figure_areas(*(read_jsonl(paper.paper_dir / n) for n in ("pages.jsonl", "layout.jsonl",
                                                                         "regions.jsonl")))[1]
    box = ocr.to_pixels(area.bbox, {"width_pt": PAGE_PT[0], "height_pt": PAGE_PT[1]}, PAGE_PX)
    failed = ocr.ocr_paper(paper, FakeEngine(LINES, fail_on=[tuple(box)]))
    assert failed == 1
    assert [r["text"] for r in read_jsonl(paper.paper_dir / "ocr.jsonl")] == ["N", "12a"]
    (error,) = read_jsonl(paper.paper_dir / "errors.jsonl")
    assert error["stage"] == "ocr_labels" and error["area"] == "region:p2-r1" and "onnx" in error["error"]
    counts = json.loads((paper.paper_dir / ".counts" / "ocr_labels.json").read_text())
    assert counts["in"] == counts["out"] + counts["dropped_no_text"] + counts["dropped_failed"]
    assert counts["dropped_failed"] == 1


def test_a_paper_without_figures_gets_an_empty_file(tmp_path):
    paper = make_paper(tmp_path, layout=[], regions=[])
    assert ocr.ocr_paper(paper, FakeEngine({})) == 0
    assert (paper.paper_dir / "ocr.jsonl").read_text() == ""


# --- the command line -----------------------------------------------------------------------------


def run_main(tmp_path, papers, engine, *extra):
    manifest = tmp_path / "work.jsonl"
    write_jsonl(manifest, [{"paper_id": p.paper_id, "pdf": str(p.pdf), "paper_dir": str(p.paper_dir)} for p in papers])
    return ocr.main(["--manifest", str(manifest), "--log", str(tmp_path / "ocr.log"), "--device", "cpu", *extra],
                    engine=engine)


def test_main_runs_every_paper_and_exits_0(tmp_path):
    papers = [make_paper(tmp_path, "a"), make_paper(tmp_path, "b")]
    assert run_main(tmp_path, papers, FakeEngine(LINES)) == 0
    assert all((p.paper_dir / "ocr.jsonl").is_file() for p in papers)


def test_a_paper_that_fails_is_isolated_and_the_exit_is_partial(tmp_path):
    good, bad = make_paper(tmp_path, "a"), make_paper(tmp_path, "b")
    (bad.paper_dir / "pages.jsonl").unlink()
    assert run_main(tmp_path, [bad, good], FakeEngine(LINES)) == ocr.common.PARTIAL
    assert (good.paper_dir / "ocr.jsonl").is_file() and not (bad.paper_dir / "ocr.jsonl").exists()
    (error,) = read_jsonl(bad.paper_dir / "errors.jsonl")
    assert error["stage"] == "ocr_labels" and "pages.jsonl" in error["error"]


def test_a_rerun_clears_the_stage_errors_and_keeps_other_stages(tmp_path):
    paper = make_paper(tmp_path)
    write_jsonl(paper.paper_dir / "errors.jsonl", [
        {"stage": "ocr_labels", "paper_id": "p1", "error": "old"},
        {"stage": "segment", "paper_id": "p1", "error": "kept"},
    ])
    assert run_main(tmp_path, [paper], FakeEngine(LINES)) == 0
    assert read_jsonl(paper.paper_dir / "errors.jsonl") == [{"stage": "segment", "paper_id": "p1", "error": "kept"}]


def test_a_rerun_removes_the_old_output_before_it_fails(tmp_path):
    paper = make_paper(tmp_path)
    (paper.paper_dir / "ocr.jsonl").write_text('{"page": 1}\n')
    (paper.paper_dir / "regions.jsonl").unlink()
    assert run_main(tmp_path, [paper], FakeEngine(LINES)) == ocr.common.PARTIAL
    assert not (paper.paper_dir / "ocr.jsonl").exists()
    assert not (paper.paper_dir / ".counts" / "ocr_labels.json").exists()


def test_a_bad_manifest_exits_2(tmp_path):
    manifest = tmp_path / "work.jsonl"
    manifest.write_text("not json\n")
    assert ocr.main(["--manifest", str(manifest), "--log", str(tmp_path / "l.log"), "--device", "cpu"],
                    engine=FakeEngine({})) == 2


def test_an_unknown_device_is_a_bad_argument(tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        ocr.main(["--manifest", "m", "--log", str(tmp_path / "l.log"), "--device", "tpu"], engine=FakeEngine({}))
    assert exit_info.value.code == 2


def test_an_engine_that_cannot_load_exits_1(tmp_path, monkeypatch):
    def broken(device, logger):
        raise ocr.SetupError("models missing")

    monkeypatch.setattr(ocr, "load_engine", broken)
    paper = make_paper(tmp_path)
    assert run_main(tmp_path, [paper], None) == 1
    assert "models missing" in (tmp_path / "ocr.log").read_text()


def test_a_broken_runtime_stops_the_run_with_1(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr.common, "MAX_CONSECUTIVE_FAILURES", 2)

    class Broken(FakeEngine):
        def read(self, image, box):
            raise RuntimeError("onnxruntime crashed")

    papers = [make_paper(tmp_path, "a"), make_paper(tmp_path, "b")]
    assert run_main(tmp_path, papers, Broken({})) == 1


def test_cpu_device_asks_mineru_for_its_cpu_session():
    assert ocr.engine_environment("cpu")["MINERU_MODEL_ONNX_DEVICE"] == "cpu"
    assert ocr.engine_environment("cpu")["MINERU_MODEL_SOURCE"] == "local"
    assert "MINERU_MODEL_ONNX_DEVICE" not in ocr.engine_environment("gpu")


# --- the worker runs in the mineru environment -------------------------------------------------


def test_the_worker_never_imports_molscout_and_parses_on_python_3_12():
    source = (WORKERS / "ocr_labels.py").read_text()
    tree = ast.parse(source, feature_version=(3, 12))
    imported = {
        (node.module or "") if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(name.split(".")[0] == "molscout" for name in imported)
    top_level = {
        alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for alias in node.names
    } | {(node.module or "").split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert top_level <= set(sys.stdlib_module_names) | {"__future__", "_readers_common"}, top_level
