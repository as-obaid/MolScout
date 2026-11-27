"""Render every page of a paper's PDF with pypdfium2. Runs in the molscout environment.

Each page is drawn at 300 dpi on white (form fields and annotations included, no alpha) and saved
as <paper_dir>/pages/p001.png, p002.png, ...; <paper_dir>/pages.jsonl gets one record per page:

    {"page": 1, "width_pt": 612.0, "height_pt": 792.0, "image": "pages/p001.png", "dpi": 300,
     "vector": true}

Sizes are in points as the page is shown (after /Rotate), so a box in image pixels converts to
points by width_pt / image width (pdfium stretches the page over ceil(points * dpi / 72) pixels).
vector is true when the page has text that is drawn (not the invisible OCR layer of a scan) or
vector paths or shadings, and false when it holds only images, such as a scanned page.
A page larger than MAX_PIXELS at 300 dpi is drawn at the dpi that fits, recorded in its "dpi".

New pages remove what was made from the old ones (regions, reads, votes) and the paper's old render
errors. As a stage: python -m molscout.pipeline.render --manifest work.jsonl --log render.log; a
paper whose PDF cannot be rendered is a line in its errors.jsonl and gets no pages.jsonl. Exit
status 0, 4 when some papers failed, 1 when files cannot be locked, 2 on a bad manifest.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

from molscout.pipeline.workers import _readers_common as common

STAGE = "render"
DPI = 300
MAX_PIXELS = 60_000_000  # about seven Letter pages at 300 dpi; a poster-sized page is drawn coarser
WHITE = (255, 255, 255, 255)
TEXT, PATH, SHADING = 1, 2, 4  # FPDF_PAGEOBJ_*
INVISIBLE_TEXT = {3, 7}  # FPDF_TEXTRENDERMODE_INVISIBLE, _CLIP: text that paints nothing


def render_paper(pdf, paper_dir, dpi: int = DPI) -> list[dict]:
    """Render every page of pdf into paper_dir; write and return the page records."""
    import pypdfium2 as pdfium

    paper_dir = Path(paper_dir)
    pages_dir = paper_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    common.clear_stage_errors(paper_dir, STAGE)
    common.remove_outputs(paper_dir, ("pages.jsonl", *common.AFTER_PAGES))
    for stale in pages_dir.glob("p*.png"):
        stale.unlink()
    document = pdfium.PdfDocument(str(pdf))
    try:
        document.init_forms()  # before any page is loaded, so form fields are drawn
        records = [_render_page(document, index, paper_dir, dpi) for index in range(len(document))]
    finally:
        document.close()
    common.write_jsonl(paper_dir / "pages.jsonl", records)
    return records


def _render_page(document, index: int, paper_dir: Path, dpi: int) -> dict:
    page = document[index]
    try:
        width, height = page.get_size()
        scale = dpi / 72
        while math.ceil(width * scale) * math.ceil(height * scale) > MAX_PIXELS:
            scale = min(scale * 0.99, math.sqrt(MAX_PIXELS / (width * height)))
        image = page.render(scale=scale, fill_color=WHITE, may_draw_forms=True).to_pil().convert("RGB")
        name = f"pages/p{index + 1:03d}.png"
        image.save(paper_dir / name, format="PNG", compress_level=3)
        return {
            "page": index + 1,
            "width_pt": round(width, 3),
            "height_pt": round(height, 3),
            "image": name,
            "dpi": dpi if scale == dpi / 72 else round(scale * 72, 2),
            "vector": is_vector(page),
        }
    finally:
        page.close()


def is_vector(page) -> bool:
    """True when the page draws text or vector graphics, False when it holds only images."""
    import pypdfium2.raw as pdfium_c

    for obj in page.get_objects():
        if obj.type in (PATH, SHADING):
            return True
        if obj.type == TEXT and pdfium_c.FPDFTextObj_GetTextRenderMode(obj.raw) not in INVISIBLE_TEXT:
            return True
    return False


def run(papers: list[common.Paper], logger, dpi: int = DPI) -> int:
    """Render every paper; return how many failed. LockError stops the run."""
    failed = 0
    for paper in papers:
        start = time.perf_counter()
        try:
            records = render_paper(paper.pdf, paper.paper_dir, dpi)
        except common.LockError:
            raise
        except Exception as exc:
            failed += 1
            common.append_error(paper.paper_dir, STAGE, paper.paper_id, common.describe(exc))
            logger.error("%s: %s", paper.paper_id, common.describe(exc))
            continue
        logger.info("%s: %d pages, %d vector, %.1f s", paper.paper_id, len(records),
                    sum(r["vector"] for r in records), time.perf_counter() - start)
    logger.info("%d papers, %d failed", len(papers), failed)
    return failed


def main(argv=None) -> int:
    parser = common.worker_parser(__doc__.split("\n\n")[0])
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args(argv)
    logger = common.open_log(args.log, STAGE)
    try:
        try:
            papers = common.read_manifest(args.manifest)
        except common.ManifestError as exc:
            logger.error("bad manifest: %s", exc)
            return 2
        try:
            failed = run(papers, logger, args.dpi)
        except common.LockError as exc:
            logger.error("%s", exc)
            return 1
        return common.PARTIAL if failed else 0
    finally:
        common.close_log(logger)


if __name__ == "__main__":
    sys.exit(main())
