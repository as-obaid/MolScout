"""render.py: every PDF page at 300 dpi on white, pages.jsonl with sizes in points and a vector flag."""

import json
from pathlib import Path

import pytest

pdfium = pytest.importorskip("pypdfium2")
from PIL import Image  # noqa: E402

from molscout.pipeline import render  # noqa: E402

LINE = b"0 0 0 RG 2 w 10 10 m 190 90 l S"
TEXT = b"BT /F1 24 Tf 20 40 Td (Hi) Tj ET"
IMAGE = b"q 200 0 0 100 0 0 cm /Im1 Do Q"
HIDDEN_TEXT = b"BT 3 Tr /F1 12 Tf 10 10 Td (ocr) Tj ET"


def make_pdf(pages):
    """A minimal PDF; pages are (width, height, content stream, /Rotate, has image) in points."""
    objects = {3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"}
    kids = []
    for width, height, content, rotate, image in pages:
        number = max(objects) + 1
        objects[number] = b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content)
        resources = b"/Font << /F1 3 0 R >>"
        if image:
            pixels = bytes([0, 128, 255, 64, 200, 30, 90, 10])
            objects[number + 1] = (
                b"<< /Type /XObject /Subtype /Image /Width 4 /Height 2 /ColorSpace /DeviceGray "
                b"/BitsPerComponent 8 /Length %d >>\nstream\n%s\nendstream" % (len(pixels), pixels)
            )
            resources += b" /XObject << /Im1 %d 0 R >>" % (number + 1)
        page = max(objects) + 1
        turn = b" /Rotate %d" % rotate if rotate else b""
        objects[page] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d]%s /Resources << %s >> /Contents %d 0 R >>"
            % (width, height, turn, resources, number)
        )
        kids.append(page)
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(b"%d 0 R" % k for k in kids), len(kids))
    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += b"%d 0 obj\n%s\nendobj\n" % (number, objects[number])
    start = len(out)
    size = max(objects) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    out += b"".join(b"%010d 00000 n \n" % offsets[n] for n in range(1, size))
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, start)
    return bytes(out)


@pytest.fixture
def pdf(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(make_pdf([
        (200, 100, LINE + b" " + TEXT, 0, False),  # vector drawing and text
        (200, 100, IMAGE, 0, True),  # a scan: one full-page image
        (200, 100, IMAGE + b" " + HIDDEN_TEXT, 0, True),  # a scan with an invisible OCR text layer
        (200, 100, LINE, 90, False),  # turned a quarter: shown 100 wide, 200 high
    ]))
    return path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_render_paper_writes_one_png_per_page_and_pages_jsonl(pdf, tmp_path):
    paper_dir = tmp_path / "papers" / "p1"
    records = render.render_paper(pdf, paper_dir)

    assert records == read_jsonl(paper_dir / "pages.jsonl")
    assert [r["page"] for r in records] == [1, 2, 3, 4]
    assert [r["image"] for r in records] == [f"pages/p00{n}.png" for n in (1, 2, 3, 4)]
    assert list(records[0]) == ["page", "width_pt", "height_pt", "image", "dpi", "vector"]
    assert all(r["dpi"] == 300 for r in records)
    assert all((paper_dir / r["image"]).is_file() for r in records)


def test_sizes_are_points_of_the_page_as_shown(pdf, tmp_path):
    records = render.render_paper(pdf, tmp_path / "out")

    assert (records[0]["width_pt"], records[0]["height_pt"]) == (200.0, 100.0)
    assert (records[3]["width_pt"], records[3]["height_pt"]) == (100.0, 200.0)
    with Image.open(tmp_path / "out" / records[3]["image"]) as image:
        assert image.size == (417, 834)  # ceil(100 * 300 / 72), ceil(200 * 300 / 72)


def test_pages_are_rgb_on_white_at_300_dpi(pdf, tmp_path):
    records = render.render_paper(pdf, tmp_path / "out")

    with Image.open(tmp_path / "out" / records[0]["image"]) as image:
        assert image.mode == "RGB"
        assert image.size == (834, 417)
        assert image.getpixel((833, 0)) == (255, 255, 255)  # nothing drawn there: white, not transparent
        assert image.convert("L").getextrema()[0] < 64  # the line and the text are drawn


def test_vector_flag_separates_drawn_pages_from_scans(pdf, tmp_path):
    records = render.render_paper(pdf, tmp_path / "out")

    assert [r["vector"] for r in records] == [True, False, False, True]


def test_dpi_is_configurable(pdf, tmp_path):
    records = render.render_paper(pdf, tmp_path / "out", dpi=72)

    assert records[0]["dpi"] == 72
    with Image.open(tmp_path / "out" / records[0]["image"]) as image:
        assert image.size == (200, 100)


def test_rerender_replaces_old_pages(pdf, tmp_path):
    paper_dir = tmp_path / "out"
    (paper_dir / "pages").mkdir(parents=True)
    (paper_dir / "pages" / "p009.png").write_bytes(b"stale")

    render.render_paper(pdf, paper_dir)

    assert sorted(p.name for p in (paper_dir / "pages").iterdir()) == [f"p00{n}.png" for n in (1, 2, 3, 4)]


def test_huge_pages_are_rendered_below_the_pixel_cap(tmp_path, monkeypatch):
    path = tmp_path / "poster.pdf"
    path.write_bytes(make_pdf([(2000, 1000, LINE, 0, False)]))
    monkeypatch.setattr(render, "MAX_PIXELS", 1_000_000)

    record = render.render_paper(path, tmp_path / "out")[0]

    with Image.open(tmp_path / "out" / record["image"]) as image:
        assert image.size[0] * image.size[1] <= 1_000_000
    assert record["dpi"] < 300
    assert record["width_pt"] == 2000.0


def test_main_renders_every_paper_in_the_manifest_and_logs_failures(pdf, tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    manifest = tmp_path / "work.jsonl"
    lines = [
        {"paper_id": "good", "pdf": str(pdf), "paper_dir": str(tmp_path / "papers" / "good")},
        {"paper_id": "bad", "pdf": str(broken), "paper_dir": str(tmp_path / "papers" / "bad")},
    ]
    manifest.write_text("".join(json.dumps(line) + "\n" for line in lines))
    log = tmp_path / "render.log"

    code = render.main(["--manifest", str(manifest), "--log", str(log)])

    assert code == 4
    assert len(read_jsonl(tmp_path / "papers" / "good" / "pages.jsonl")) == 4
    assert not (tmp_path / "papers" / "bad" / "pages.jsonl").exists()
    errors = read_jsonl(tmp_path / "papers" / "bad" / "errors.jsonl")
    assert errors[0]["stage"] == "render" and errors[0]["paper_id"] == "bad"
    assert "PdfiumError" in errors[0]["error"]
    assert "bad" in log.read_text()


def test_main_rejects_a_malformed_manifest(tmp_path):
    manifest = tmp_path / "work.jsonl"
    manifest.write_text('{"paper_id": "x"}\n')

    assert render.main(["--manifest", str(manifest), "--log", str(tmp_path / "log")]) == 2


def test_new_pages_remove_the_regions_reads_and_votes_made_from_the_old_ones(pdf, tmp_path):
    paper_dir = tmp_path / "out"
    paper_dir.mkdir()
    for name in ("regions.jsonl", "reads.jsonl", "voted.jsonl", ".reads.lock", "layout.jsonl"):
        (paper_dir / name).write_text("{}\n")

    render.render_paper(pdf, paper_dir)

    left = {p.name for p in paper_dir.iterdir()}
    assert {"pages.jsonl", "layout.jsonl"} <= left
    assert not left & {"regions.jsonl", "reads.jsonl", "voted.jsonl", ".reads.lock"}


def test_a_rerun_clears_old_render_errors(pdf, tmp_path):
    paper_dir = tmp_path / "papers" / "good"
    paper_dir.mkdir(parents=True)
    (paper_dir / "errors.jsonl").write_text(
        '{"stage": "render", "paper_id": "good", "error": "old"}\n{"stage": "parse", "paper_id": "good", "error": "x"}\n')
    manifest = tmp_path / "work.jsonl"
    manifest.write_text(json.dumps({"paper_id": "good", "pdf": str(pdf), "paper_dir": str(paper_dir)}) + "\n")

    assert render.main(["--manifest", str(manifest), "--log", str(tmp_path / "render.log")]) == 0
    assert [e["stage"] for e in read_jsonl(paper_dir / "errors.jsonl")] == ["parse"]
