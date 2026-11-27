"""Tests for the MinerU parse worker (src/molscout/pipeline/workers/mineru_parse.py).

The worker runs in the mineru environment and never imports molscout, so it is loaded from its
file path. The fixture is MinerU 4.2.3's basic-tier middle_json.json for BioVista 26_6st3
(PMC7496690, open access), trimmed to pages 1 and 3: two figures with captions and Table 1, an
R-group table whose drawn substituents MinerU keeps as <img> cells.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "src" / "molscout" / "pipeline" / "workers" / "mineru_parse.py"
FIXTURE = Path(__file__).parent / "fixtures" / "b1" / "26_6st3_basic_middle.json"
INTERFACE_TYPES = {"text", "title", "table", "image", "caption", "footnote", "equation"}
CROP_RE = re.compile(r"images/[A-Za-z0-9_.-]+\.jpg")
FIELDS = {"page", "index", "type", "bbox", "text", "html", "parent", "image"}


def _load_worker():
    spec = importlib.util.spec_from_file_location("mineru_parse", WORKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mineru_parse"] = module
    spec.loader.exec_module(module)
    return module


mp = _load_worker()


@pytest.fixture
def middle() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def complete(middle) -> dict:
    """The fixture as a full 4-page parse: pages 2 and 4 added back without blocks."""
    full = copy.deepcopy(middle)
    full["pages"] = sorted(full["pages"] + [{"page_idx": 1, "blocks": []}, {"page_idx": 3, "blocks": []}],
                           key=lambda page: page["page_idx"])
    return full


@pytest.fixture
def layout(middle):
    records, images = mp.convert_middle(middle)
    return records, images


def _by_type(records, kind):
    return [r for r in records if r["type"] == kind]


def _find(records, prefix):
    hits = [r for r in records if (r["text"] or "").startswith(prefix)]
    assert len(hits) == 1, f"expected one record starting with {prefix!r}, got {len(hits)}"
    return hits[0]


# --- pure helpers -------------------------------------------------------------------------------


def test_to_points_scales_by_page_size_and_rounds():
    assert mp.to_points([0.5, 0.25, 1.0, 0.5], 600, 800) == [300.0, 200.0, 600.0, 400.0]
    assert mp.to_points([0.511, 0.136, 0.922, 0.655], 595, 842) == [304.05, 114.51, 548.59, 551.51]
    with pytest.raises(ValueError, match="no bbox"):
        mp.to_points(None, 595, 842)


@pytest.mark.parametrize(
    ("latex", "text"),
    [
        (r"\mathsf { I C } _ { 5 0 } \mu \mathsf { M }", "IC50 μM"),
        (r"0 . 3 9 6   \pm   0 . 1 8 0", "0.396 ± 0.180"),
        (r"{ \bf 7 }", "7"),
        (r"1 C _ { 5 0 }", "1C50"),
        (r"\mathrm { p K } _ { \mathrm { i } } = 7 . 2", "pKi = 7.2"),
        (r"\alpha", "α"),
        (r"x \times 1 0 ^ { - 3 }", "x × 10-3"),
    ],
)
def test_latex_to_text_gives_plain_searchable_text(latex, text):
    assert mp.latex_to_text(latex) == text


def test_spans_to_text_joins_text_equations_and_links():
    spans = [
        {"type": "text", "content": "Compound "},
        {"type": "equation_inline", "content": r"{ \bf 8 }"},
        {"type": "text", "content": " (IC"},
        {"type": "text", "content": "50", "styles": ["subscript"]},
        {"type": "text", "content": ") see "},
        {"type": "hyperlink", "url": "https://x.org", "content": [{"type": "text", "content": "ref"}]},
        {"type": "code_inline", "content": "c1ccccc1"},
    ]
    assert mp.spans_to_text(spans) == "Compound 8 (IC50) see refc1ccccc1"


# --- conversion of the recorded fixture ----------------------------------------------------------


def test_records_have_exactly_the_interface_fields_and_types(layout):
    records, _ = layout
    assert records
    for record in records:
        assert set(record) == FIELDS
        assert record["type"] in INTERFACE_TYPES


def test_pages_are_one_based_and_indexes_follow_reading_order(layout):
    records, _ = layout
    assert {r["page"] for r in records} == {1, 3}
    for page in (1, 3):
        indexes = [r["index"] for r in records if r["page"] == page]
        assert indexes == list(range(len(indexes)))
    # page 1 starts with the article title, then the authors, then the abstract
    first = [r for r in records if r["page"] == 1][:2]
    assert first[0]["type"] == "title" and first[0]["text"].startswith("Structure-Activity Relationship")
    assert first[1]["type"] == "text" and first[1]["text"].startswith("James P. Holt-Martyn")


def test_page_furniture_is_dropped(layout):
    records, _ = layout
    texts = [r["text"] or "" for r in records]
    assert not any("CHEMMEDCHEM Communications" in t for t in texts)  # header
    assert not any(t.startswith("ChemMedChem 2020, 15, 270") for t in texts)  # footer
    assert "270" not in texts and "272" not in texts  # page numbers


def test_page_footnotes_are_footnotes_without_parent(layout):
    records, _ = layout
    footnote = _find(records, "[a] Dr. J. P. Holt-Martyn")
    assert footnote["type"] == "footnote" and footnote["parent"] is None


def test_boxes_are_pdf_points_from_the_page_sizes_in_the_file(layout):
    records, _ = layout
    (table,) = _by_type(records, "table")
    assert table["page"] == 3
    assert table["bbox"] == [304.05, 114.51, 548.59, 551.51]  # [0.511, 0.136, 0.922, 0.655] on 595 x 842 pt


def test_figure_captions_link_to_their_figures(layout):
    records, _ = layout
    figure = _find(records, "Figure 1. HIF prolyl hydroxylases")
    assert figure["type"] == "caption"
    parent = next(r for r in records if r["page"] == 1 and r["index"] == figure["parent"])
    assert parent["type"] == "image"
    assert parent["bbox"] == mp.to_points([0.518, 0.345, 0.913, 0.509], 595, 842)
    # a one-letter panel caption placed above its image still links to that image
    panel = next(r for r in records if r["type"] == "caption" and r["text"] == "B")
    panel_parent = next(r for r in records if r["page"] == 1 and r["index"] == panel["parent"])
    assert panel_parent["type"] == "image" and panel_parent["index"] != parent["index"]


def test_table_caption_left_as_text_by_mineru_is_linked_to_the_table(layout):
    records, _ = layout
    caption = _find(records, "Table 1. Studies on 4-hydroxypyrimidine")
    (table,) = _by_type(records, "table")
    assert caption["type"] == "caption"
    assert caption["parent"] == table["index"]
    assert caption["index"] == table["index"] - 1


def test_body_text_that_mentions_a_table_is_not_made_a_caption(layout):
    records, _ = layout
    mention = _find(records, "8–18 were tested for inhibition of other 2OG oxygenases (Table 1)")
    assert mention["type"] == "text" and mention["parent"] is None


def test_table_keeps_drawn_substituents_as_image_cells(layout):
    records, images = layout
    (table,) = _by_type(records, "table")
    html = table["html"]
    assert html.startswith("<table>") and html.endswith("</table>")
    srcs = mp.IMG_SRC_RE.findall(html)
    assert len(srcs) == 9  # the core above the header and 8 drawn R groups
    assert all(src.startswith("layout_images/p3-") for src in srcs)
    assert "images/page_" not in html
    for src in srcs:
        assert images[src].startswith("images/page_2_table_image_7_")
    assert table["image"] == "layout_images/p3-" + str(table["index"]) + ".jpg"
    assert images[table["image"]] == "images/page_2_table_7.jpg"


def test_table_html_equations_become_plain_text(layout):
    records, _ = layout
    (table,) = _by_type(records, "table")
    assert "<eq>" not in table["html"]
    assert "<td>0.396 ± 0.180</td>" in table["html"]
    assert "IC50 μM" in table["html"]


def test_running_text_inline_equations_become_text(layout):
    records, _ = layout
    text = _find(records, "levels of inhibition to 8.")
    assert "(IC50 0.213" in text["text"]
    assert "\\mathsf" not in text["text"]


def test_images_and_tables_have_crops_and_other_blocks_do_not(layout):
    records, images = layout
    for record in records:
        if record["type"] in ("image", "table"):
            assert record["image"] in images
            assert record["text"] is None
        else:
            assert record["image"] is None
        if record["type"] != "table":
            assert record["html"] is None
    assert len(_by_type(records, "image")) == 2


def test_conversion_does_not_modify_its_input(middle):
    before = copy.deepcopy(middle)
    mp.convert_middle(middle)
    assert middle == before


# --- synthetic shapes MinerU produces on other papers ---------------------------------------------


def _doc(blocks, page_idx=0, size=(600, 800)):
    return {
        "schema": "docvortex.middle",
        "schema_version": "2.0",
        "metadata": {"file_suffix": "pdf", "producer": {"name": "mineru", "version": "4.2.3"}},
        "extensions": {
            "docvortex_layout": {
                "version": 1,
                "pages": [{"page_idx": page_idx, "width_pt": size[0], "height_pt": size[1]}],
            }
        },
        "pages": [{"page_idx": page_idx, "blocks": blocks}],
    }


def _text(kind, index, words, bbox=(0.1, 0.1, 0.9, 0.2)):
    return {"type": kind, "index": index, "bbox": list(bbox), "content": [{"type": "text", "content": words}]}


def test_native_table_caption_and_footnote_children_link_to_the_table():
    table = {
        "type": "table",
        "index": 3,
        "bbox": [0.1, 0.3, 0.9, 0.6],
        "content": [
            _text("table_caption", 2, "Table 2. Kinase data", (0.1, 0.25, 0.9, 0.29)),
            {"type": "table_body", "index": 3, "bbox": [0.1, 0.3, 0.9, 0.6],
             "image_path": "images/t.jpg", "content": "<table><tr><td>1</td></tr></table>"},
            _text("table_footnote", 4, "[a] n.d., not determined", (0.1, 0.61, 0.9, 0.63)),
        ],
    }
    records, images = mp.convert_middle(_doc([_text("text", 1, "Intro"), table]))
    kinds = [(r["index"], r["type"], r["parent"]) for r in records]
    assert kinds == [(0, "text", None), (1, "caption", 2), (2, "table", None), (3, "footnote", 2)]
    assert records[2]["html"] == "<table><tr><td>1</td></tr></table>"
    assert images == {"layout_images/p1-2.jpg": "images/t.jpg"}


def test_table_caption_typed_as_a_title_is_linked_but_a_section_title_is_not():
    table = {
        "type": "table",
        "index": 2,
        "bbox": [0.1, 0.3, 0.9, 0.6],
        "content": [{"type": "table_body", "index": 2, "bbox": [0.1, 0.3, 0.9, 0.6], "content": "<table></table>"}],
    }
    heading = _text("paragraph_title", 0, "Tables and figures")
    caption = _text("paragraph_title", 1, "Table 1. Inhibition of PHD2 by compounds 8-10")
    heading["level"] = caption["level"] = 2
    records, _ = mp.convert_middle(_doc([heading, caption, table]))
    assert [(r["type"], r["parent"]) for r in records] == [("title", None), ("caption", 2), ("table", None)]


def test_charts_are_images_and_lists_flatten_to_text_items():
    chart = {
        "type": "chart",
        "index": 0,
        "bbox": [0.1, 0.1, 0.5, 0.3],
        "content": [
            {"type": "chart_body", "index": 0, "bbox": [0.1, 0.1, 0.5, 0.3], "image_path": "images/c.jpg", "content": ""},
            _text("chart_caption", 1, "Figure 3. Dose response", (0.1, 0.31, 0.5, 0.33)),
        ],
    }
    listing = {
        "type": "list",
        "index": 2,
        "bbox": [0.1, 0.4, 0.9, 0.6],
        "content": [_text("text", 3, "12a: 4-chloro-N-methylbenzamide"), _text("text", 4, "12b: N-methylbenzamide")],
    }
    equation = {"type": "equation", "index": 5, "bbox": [0.1, 0.7, 0.9, 0.75], "content": r"y = a x + b"}
    records, _ = mp.convert_middle(_doc([chart, listing, equation]))
    assert [(r["type"], r["text"], r["parent"]) for r in records] == [
        ("image", None, None),
        ("caption", "Figure 3. Dose response", 0),
        ("text", "12a: 4-chloro-N-methylbenzamide", None),
        ("text", "12b: N-methylbenzamide", None),
        ("equation", "y = a x + b", None),
    ]


def test_blocks_without_index_keep_document_order():
    blocks = [_text("text", None, "first"), _text("paragraph_title", None, "Results"), _text("text", None, "third")]
    blocks[1]["level"] = 2
    records, _ = mp.convert_middle(_doc(blocks))
    assert [(r["index"], r["type"], r["text"]) for r in records] == [
        (0, "text", "first"), (1, "title", "Results"), (2, "text", "third"),
    ]


def test_unknown_block_type_is_an_error_not_a_silent_drop():
    with pytest.raises(ValueError, match="unknown MinerU block type 'sidebar'"):
        mp.convert_middle(_doc([_text("sidebar", 0, "?")]))


def test_missing_page_size_is_an_error():
    doc = _doc([_text("text", 0, "x")])
    doc["extensions"]["docvortex_layout"]["pages"] = []
    with pytest.raises(ValueError, match="no page size for page 1"):
        mp.convert_middle(doc)


def test_a_block_without_a_box_is_an_error():  # layout.jsonl's bbox is never null
    block = _text("text", 0, "x")
    del block["bbox"]
    with pytest.raises(ValueError, match="no bbox"):
        mp.convert_middle(_doc([block]))


# --- fail loudly ----------------------------------------------------------------------------------


A4 = (595.276, 841.89)  # pdfium's size of an A4 page; MinerU records it rounded, 595 x 842


def test_problems_flags_an_empty_parse_of_a_non_empty_pdf():
    doc = _doc([])
    assert mp.parse_problems(doc, [], [A4] * 4) == [("empty_parse", "MinerU returned no blocks for a 4-page PDF")]


def test_problems_flags_pages_mineru_did_not_return(middle, layout):
    records, _ = layout
    problems = mp.parse_problems(middle, records, [A4] * 4)
    assert problems == [("pages_missing", "MinerU returned 2 of 4 pages; missing: 2, 4")]


def test_problems_is_empty_for_a_complete_parse(complete, layout):
    records, _ = layout
    assert mp.parse_problems(complete, records, [A4] * 4) == []


def test_problems_flags_pages_whose_size_differs_from_the_pdfs(complete, layout):
    records, _ = layout
    letter, turned = (612.0, 792.0), (841.89, 595.276)
    problems = mp.parse_problems(complete, records, [letter, A4, turned, A4])
    assert problems == [("page_size_mismatch", "MinerU's page sizes differ from the PDF's, so its boxes would be "
                                               "misplaced: page 1 595 x 842 pt (PDF 612 x 792), page 3 595 x 842 pt "
                                               "(PDF 842 x 595)")]


# --- worker: per-paper IO and CLI ------------------------------------------------------------------


class FakeEngine:
    """Stands in for MinerU: writes the fixture and empty crop files, or fails as told."""

    def __init__(self, middle: dict | None, pages: int = 4, error: Exception | None = None, size=A4):
        self.middle, self.pages, self.error, self.size, self.calls = middle, pages, error, size, []

    def page_sizes(self, pdf: Path) -> list[tuple[float, float]]:
        return [self.size] * self.pages

    def parse(self, pdf: Path, out_dir: Path) -> None:
        self.calls.append(pdf)
        if self.error:
            raise self.error
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "middle_json.json").write_text(json.dumps(self.middle), encoding="utf-8")
        for path in set(CROP_RE.findall(json.dumps(self.middle))):
            target = out_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"\xff\xd8 jpeg " + path.encode())


def _paper(tmp_path, paper_id="26_6st3"):
    pdf = tmp_path / f"{paper_id}.pdf"
    pdf.write_bytes(b"%PDF-1.7 placeholder")
    return {"paper_id": paper_id, "pdf": str(pdf), "paper_dir": str(tmp_path / "papers" / paper_id)}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_process_paper_writes_layout_and_copies_crops(tmp_path, complete):
    paper = _paper(tmp_path)
    engine = FakeEngine(complete, pages=4)
    result = mp.process_paper(paper, engine)
    paper_dir = Path(paper["paper_dir"])
    records = _read_jsonl(paper_dir / "layout.jsonl")
    assert result.status == "ok" and result.blocks == len(records)
    assert (paper_dir / "mineru" / "middle_json.json").is_file()
    referenced = {r["image"] for r in records if r["image"]}
    referenced |= {src for r in records if r["html"] for src in mp.IMG_SRC_RE.findall(r["html"])}
    assert len(referenced) == 12  # 2 figures, the table crop and 9 cell drawings
    for rel in referenced:
        copied = paper_dir / rel
        assert copied.is_file() and copied.read_bytes().startswith(b"\xff\xd8")
    assert not (paper_dir / "errors.jsonl").exists()


def test_process_paper_records_missing_pages_but_keeps_the_layout(tmp_path, middle):
    paper = _paper(tmp_path)
    result = mp.process_paper(paper, FakeEngine(middle, pages=4))
    paper_dir = Path(paper["paper_dir"])
    assert result.status == "error"
    assert (paper_dir / "layout.jsonl").is_file()
    (error,) = _read_jsonl(paper_dir / "errors.jsonl")
    assert error["stage"] == "parse" and error["paper_id"] == "26_6st3" and error["error"] == "pages_missing"


def test_process_paper_empty_parse_writes_no_layout(tmp_path):
    paper = _paper(tmp_path)
    result = mp.process_paper(paper, FakeEngine(_doc([]), pages=3))
    paper_dir = Path(paper["paper_dir"])
    assert result.status == "error"
    assert not (paper_dir / "layout.jsonl").exists()
    (error,) = _read_jsonl(paper_dir / "errors.jsonl")
    assert error["error"] == "empty_parse"


def test_process_paper_records_a_mineru_failure(tmp_path):
    paper = _paper(tmp_path)
    result = mp.process_paper(paper, FakeEngine(None, error=RuntimeError("CUDA out of memory")))
    paper_dir = Path(paper["paper_dir"])
    assert result.status == "error"
    assert not (paper_dir / "layout.jsonl").exists()
    (error,) = _read_jsonl(paper_dir / "errors.jsonl")
    assert error["error"] == "mineru_failed" and "CUDA out of memory" in error["message"]


def test_process_paper_records_a_missing_pdf(tmp_path, middle):
    paper = _paper(tmp_path)
    Path(paper["pdf"]).unlink()
    engine = FakeEngine(middle)
    result = mp.process_paper(paper, engine)
    assert result.status == "error" and engine.calls == []
    (error,) = _read_jsonl(Path(paper["paper_dir"]) / "errors.jsonl")
    assert error["error"] == "pdf_missing"


def test_process_paper_records_a_missing_crop(tmp_path, complete):
    class LosesOneCrop(FakeEngine):
        def parse(self, pdf, out_dir):
            super().parse(pdf, out_dir)
            (out_dir / "images" / "page_2_table_image_7_5.jpg").unlink()

    paper = _paper(tmp_path)
    result = mp.process_paper(paper, LosesOneCrop(complete, pages=4))
    paper_dir = Path(paper["paper_dir"])
    assert result.status == "error"
    assert (paper_dir / "layout.jsonl").is_file()
    (error,) = _read_jsonl(paper_dir / "errors.jsonl")
    assert error["error"] == "image_missing" and "page_2_table_image_7_5.jpg" in error["message"]


def test_process_paper_skips_a_finished_paper_unless_forced(tmp_path, complete):
    paper = _paper(tmp_path)
    engine = FakeEngine(complete, pages=4)
    mp.process_paper(paper, engine)
    assert mp.process_paper(paper, engine).status == "skipped"
    assert len(engine.calls) == 1
    assert mp.process_paper(paper, engine, force=True).status == "ok"
    assert len(engine.calls) == 2


def _manifest(tmp_path, papers):
    path = tmp_path / "work.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in papers), encoding="utf-8")
    return path


def test_process_paper_records_page_sizes_that_differ_but_keeps_the_layout(tmp_path, complete):
    paper = _paper(tmp_path)
    result = mp.process_paper(paper, FakeEngine(complete, pages=4, size=(612.0, 792.0)))
    paper_dir = Path(paper["paper_dir"])
    assert result.status == "error" and (paper_dir / "layout.jsonl").is_file()
    (error,) = _read_jsonl(paper_dir / "errors.jsonl")
    assert error["error"] == "page_size_mismatch" and "PDF 612 x 792" in error["message"]


def test_process_paper_clears_its_old_errors_and_keeps_lines_it_cannot_read(tmp_path, complete):
    paper = _paper(tmp_path)
    paper_dir = Path(paper["paper_dir"])
    paper_dir.mkdir(parents=True)
    (paper_dir / "errors.jsonl").write_text('not json\n{"stage": "parse", "error": "old"}\n{"stage": "render"}\n')
    assert mp.process_paper(paper, FakeEngine(complete, pages=4)).status == "ok"
    assert (paper_dir / "errors.jsonl").read_text().splitlines() == ["not json", '{"stage": "render"}']


def test_process_paper_records_a_layout_it_cannot_write(tmp_path, complete, monkeypatch):
    def full_disk(images, raw_dir, paper_dir):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(mp, "_copy_images", full_disk)
    paper = _paper(tmp_path)
    result = mp.process_paper(paper, FakeEngine(complete, pages=4))
    assert result.status == "error" and not (Path(paper["paper_dir"]) / "layout.jsonl").exists()
    (error,) = _read_jsonl(Path(paper["paper_dir"]) / "errors.jsonl")
    assert error["error"] == "write_failed" and "No space left" in error["message"]


def test_main_keeps_going_after_an_unexpected_failure_and_exits_4(tmp_path, complete, monkeypatch):
    crash, good = _paper(tmp_path, "crash"), _paper(tmp_path, "26_6st3")
    real, calls = mp.convert_middle, []

    def convert(middle):  # the first paper meets an error nobody expected
        calls.append(middle)
        if len(calls) == 1:
            raise ZeroDivisionError("division by zero")
        return real(middle)

    monkeypatch.setattr(mp, "convert_middle", convert)
    log = tmp_path / "stage.log"
    code = mp.main(["--manifest", str(_manifest(tmp_path, [crash, good])), "--log", str(log)],
                   engine=FakeEngine(complete, pages=4))
    assert code == 4
    (error,) = _read_jsonl(Path(crash["paper_dir"]) / "errors.jsonl")
    assert error["error"] == "internal" and "ZeroDivisionError" in error["message"]
    assert (Path(good["paper_dir"]) / "layout.jsonl").is_file()
    assert "1 ok, 0 skipped, 1 error" in log.read_text(encoding="utf-8")


def test_main_processes_every_paper_and_exits_4_on_paper_errors(tmp_path, complete):
    good = _paper(tmp_path, "26_6st3")
    bad = _paper(tmp_path, "broken")
    Path(bad["pdf"]).unlink()
    manifest = _manifest(tmp_path, [good, bad])
    log = tmp_path / "stage.log"
    code = mp.main(["--manifest", str(manifest), "--log", str(log)], engine=FakeEngine(complete, pages=4))
    assert code == 4
    assert (Path(good["paper_dir"]) / "layout.jsonl").is_file()
    assert (Path(bad["paper_dir"]) / "errors.jsonl").is_file()
    text = log.read_text(encoding="utf-8")
    assert "26_6st3 ok" in text and "broken error pdf_missing" in text
    assert "1 ok, 0 skipped, 1 error" in text


@pytest.mark.parametrize(
    "lines",
    [
        None,  # manifest file missing
        ["{not json"],
        [json.dumps({"paper_id": "x", "pdf": "x.pdf"})],  # no paper_dir
    ],
)
def test_main_exits_non_zero_on_a_bad_manifest(tmp_path, middle, lines):
    manifest = tmp_path / "work.jsonl"
    if lines is not None:
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log = tmp_path / "stage.log"
    code = mp.main(["--manifest", str(manifest), "--log", str(log)], engine=FakeEngine(middle))
    assert code == 2
    assert "fatal" in log.read_text(encoding="utf-8")


def test_main_exits_non_zero_when_mineru_cannot_start(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise mp.SetupError("models missing: MinerU-4_models_onnx")

    monkeypatch.setattr(mp, "MineruEngine", broken)
    manifest = _manifest(tmp_path, [_paper(tmp_path)])
    log = tmp_path / "stage.log"
    assert mp.main(["--manifest", str(manifest), "--log", str(log), "--tier", "basic"]) == 1
    assert "models missing" in log.read_text(encoding="utf-8")


def test_the_environment_installs_only_packages_checked_by_hash():
    env = ROOT / "envs" / "mineru"
    assert "--require-hashes" in (env / "setup.sh").read_text(encoding="utf-8")
    text = (env / "requirements.txt").read_text(encoding="utf-8")
    entries = re.split(r"\n(?=[A-Za-z0-9])", text)  # the header, then one entry per pinned package
    pins = [entry for entry in entries if re.match(r"[A-Za-z0-9_.\-]+==\S+ \\\n", entry)]
    assert len(pins) == len(entries) - 1 > 200 and all("--hash=sha256:" in entry for entry in pins)
    assert any(entry.startswith("mineru==4.2.3") for entry in pins)
    assert any(entry.startswith("vllm==0.19.1") for entry in pins)


def test_engine_environment_is_local_and_telemetry_free(monkeypatch):
    for key in ("MINERU_TELEMETRY_DISABLED", "MINERU_MODEL_SOURCE", "MINERU_MODEL_VLM_ENGINE"):
        monkeypatch.delenv(key, raising=False)
    env = mp.engine_environment(backend="vllm", small_backend=None)
    assert env["MINERU_TELEMETRY_DISABLED"] == "1"
    assert env["MINERU_MODEL_SOURCE"] == "local"
    assert env["MINERU_MODEL_VLM_ENGINE"] == "vllm"
    assert "MINERU_MODEL_SMALL_BACKEND" not in env
    assert "MINERU_API_KEY" not in env


def test_worker_never_imports_molscout_or_remote_parsing():
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported |= {f"{node.module}.{alias.name}" for alias in node.names}
    assert not any(name == "molscout" or name.startswith("molscout.") for name in imported)
    assert not any("MinerUApiParser" in name or "api_client" in name for name in imported)
    assert "--remote" not in WORKER.read_text(encoding="utf-8")
