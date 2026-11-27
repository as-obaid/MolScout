"""Label OCR: read the text printed in and around the figures, which MinerU's layout text does not cover.

Runs in the mineru environment as a script; it imports the standard library, the shared worker helper and,
when the engine loads, MinerU 4.2.3's own PP-OCRv6 det and rec ONNX models from the store
(MINERU_MODEL_SOURCE=local, nothing downloaded, telemetry off). Never imports molscout.

    python ocr_labels.py --manifest work.jsonl --log stage.log --device cpu|gpu

For each paper it reads pages.jsonl (render), layout.jsonl (parse) and regions.jsonl (segment) and reads one
area at a time from the 300 dpi page images: every MinerU image block with IMAGE_MARGIN_PT around it, and every
structure drawing outside all image blocks padded by REGION_PAD_PT (left, top, right, bottom; the label of a
drawing usually sits under it). It writes <paper_dir>/ocr.jsonl, one text line per line:

    {"page": 2, "bbox": [x0, y0, x1, y1], "text": "12a", "score": 0.98, "source_block": "image:p2-7"}

bbox is in PDF points, origin top-left, pages 1-based, as regions.jsonl; source_block names the area that
read it, "image:p<page>-<layout index>" or "region:<region_id>". A line two overlapping areas both read
(boxes overlapping by DUPLICATE_IOU) is written once. Its counts go to <paper_dir>/.counts/ocr_labels.json:
in (areas) = out (areas with text) + dropped_no_text + dropped_failed, plus lines and lines_duplicate.

A rerun first clears the paper's ocr_labels errors, its ocr.jsonl and its counts. An area whose OCR raises is
a line in errors.jsonl and the other areas are kept; a paper whose inputs cannot be read gets no ocr.jsonl.
MAX_CONSECUTIVE_FAILURES areas failing in a row stop the run (the runtime looks broken). Exit status 0; 4 when
some papers or areas failed; 1 when MinerU's OCR cannot load, the runtime breaks or files cannot be locked;
2 on a bad manifest or bad arguments. --device cpu asks MinerU for a CPU session; gpu lets MinerU choose.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _readers_common as common  # noqa: E402

STAGE = "ocr_labels"
OUTPUT = "ocr.jsonl"
COUNTS = Path(".counts") / f"{STAGE}.json"
MINERU_VERSION = "4.2.3"
IMAGE_MARGIN_PT = 12.0
REGION_PAD_PT = (30.0, 10.0, 30.0, 50.0)  # left, top, right, bottom
DUPLICATE_IOU = 0.5


class SetupError(RuntimeError):
    """MinerU's OCR cannot run at all (not installed, wrong version, models missing)."""


@dataclass(frozen=True)
class Area:
    """One part of a page to read: a box in points and the block or drawing it comes from."""

    page: int
    bbox: tuple[float, float, float, float]
    source: str


# --- areas and boxes (pure) -----------------------------------------------------------------------


def _clip(box, page: dict) -> tuple[float, float, float, float]:
    width, height = float(page["width_pt"]), float(page["height_pt"])
    x0, y0, x1, y1 = box
    return (max(0.0, x0), max(0.0, y0), min(width, x1), min(height, y1))


def _centre_inside(box, outer) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _page_of(record: dict, pages: dict) -> dict:
    page = pages.get(record["page"])
    if page is None:
        raise ValueError(f"a figure is on page {record['page']}, which pages.jsonl does not have")
    return page


def figure_areas(pages: list, layout: list, regions: list) -> list[Area]:
    """MinerU's image blocks with a margin, then the structure drawings outside every image block, padded."""
    by_page = {page["page"]: page for page in pages}
    images = [block for block in layout if block.get("type") == "image"]
    areas = []
    for block in images:
        x0, y0, x1, y1 = block["bbox"]
        m = IMAGE_MARGIN_PT
        box = _clip((x0 - m, y0 - m, x1 + m, y1 + m), _page_of(block, by_page))
        areas.append(Area(block["page"], box, f"image:p{block['page']}-{block['index']}"))
    left, top, right, bottom = REGION_PAD_PT
    for region in regions:
        if not region.get("is_structure") or region.get("kind") == "cell":  # cells: the cells stage's table drawings
            continue
        covered = any(b["page"] == region["page"] and _centre_inside(region["bbox"], b["bbox"]) for b in images)
        if covered:
            continue
        x0, y0, x1, y1 = region["bbox"]
        box = _clip((x0 - left, y0 - top, x1 + right, y1 + bottom), _page_of(region, by_page))
        areas.append(Area(region["page"], box, f"region:{region['region_id']}"))
    return areas


def to_pixels(box, page: dict, image_size) -> list[int]:
    """A box in points to whole page-image pixels (the page is stretched over the whole image)."""
    width, height = image_size
    sx, sy = width / float(page["width_pt"]), height / float(page["height_pt"])
    x0, y0, x1, y1 = box
    return [max(0, int(x0 * sx)), max(0, int(y0 * sy)), min(width, round(x1 * sx)), min(height, round(y1 * sy))]


def _iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def without_duplicates(lines: list[dict]) -> list[dict]:
    """Lines in page, top and left order; a line whose box overlaps a kept one on its page is dropped."""
    kept: list[dict] = []
    for line in lines:
        if not any(k["page"] == line["page"] and _iou(k["bbox"], line["bbox"]) >= DUPLICATE_IOU for k in kept):
            kept.append(line)
    return sorted(kept, key=lambda r: (r["page"], r["bbox"][1], r["bbox"][0]))


# --- one paper ------------------------------------------------------------------------------------


def read_area(engine, paper: common.Paper, page: dict, area: Area) -> list[dict]:
    """The text lines the engine finds in one area, boxes in points."""
    image = paper.paper_dir / page["image"]
    size = engine.image_size(image)
    found = engine.read(image, to_pixels(area.bbox, page, size))
    lines = []
    for box, text, score in found:
        text = " ".join(str(text).split())
        if text:
            lines.append({"page": area.page, "bbox": common.px_to_points(box, page, size), "text": text,
                          "score": round(float(score), 4), "source_block": area.source})
    return lines


def _read_areas(engine, paper: common.Paper, pages: dict, areas: list[Area], streak: list) -> tuple:
    """(lines, counts) of every area; a failing area is an error line, too many in a row BrokenRuntime."""
    lines: list[dict] = []
    counts = Counter({"in": len(areas), "out": 0, "dropped_no_text": 0, "dropped_failed": 0})
    for area in areas:
        try:
            found = read_area(engine, paper, pages[area.page], area)
            streak[0] = 0
        except Exception as exc:
            streak[0] += 1
            if streak[0] >= common.MAX_CONSECUTIVE_FAILURES:
                raise common.BrokenRuntime(f"{streak[0]} areas failed in a row, the last: {common.describe(exc)}") \
                    from exc
            counts["dropped_failed"] += 1
            common.append_error(paper.paper_dir, STAGE, paper.paper_id, common.describe(exc), area=area.source)
            continue
        counts["out" if found else "dropped_no_text"] += 1
        lines.extend(found)
    return lines, counts


def ocr_paper(paper: common.Paper, engine, streak: list | None = None) -> int:
    """OCR one paper's figure areas into ocr.jsonl and its counts; return how many areas failed."""
    folder = paper.paper_dir
    common.remove_outputs(folder, (OUTPUT, COUNTS))
    pages = common.read_jsonl(folder / "pages.jsonl")
    areas = figure_areas(pages, common.read_jsonl(folder / "layout.jsonl"), common.read_jsonl(folder / "regions.jsonl"))
    lines, counts = _read_areas(engine, paper, {p["page"]: p for p in pages}, areas, streak or [0])
    kept = without_duplicates(lines)
    counts["lines"], counts["lines_duplicate"] = len(kept), len(lines) - len(kept)
    common.write_jsonl(folder / OUTPUT, kept)
    (folder / COUNTS).parent.mkdir(exist_ok=True)
    (folder / COUNTS).write_text(json.dumps(dict(counts)) + "\n", encoding="utf-8")
    return counts["dropped_failed"]


def run(papers: list[common.Paper], engine, logger) -> int:
    """OCR every paper; return how many failed, as a whole or in some area. BrokenRuntime and LockError stop."""
    failed, streak = 0, [0]
    for paper in papers:
        start = time.perf_counter()
        try:
            common.clear_stage_errors(paper.paper_dir, STAGE)
            failed_areas = ocr_paper(paper, engine, streak)
        except (common.BrokenRuntime, common.LockError):
            raise
        except Exception as exc:
            failed += 1
            common.append_error(paper.paper_dir, STAGE, paper.paper_id, common.describe(exc))
            logger.error("%s: %s", paper.paper_id, common.describe(exc))
            continue
        failed += failed_areas > 0
        counts = json.loads((paper.paper_dir / COUNTS).read_text(encoding="utf-8"))
        logger.info("%s: %d areas, %d with text, %d failed, %d lines, %.1f s", paper.paper_id, counts["in"],
                    counts["out"], failed_areas, counts["lines"], time.perf_counter() - start)
    logger.info("%d papers, %d failed", len(papers), failed)
    return failed


# --- MinerU's OCR -------------------------------------------------------------------------------------


def engine_environment(device: str) -> dict[str, str]:
    """Environment for MinerU: local models only, no telemetry; a CPU ONNX session for --device cpu."""
    env = {"MINERU_TELEMETRY_DISABLED": "1", "MINERU_MODEL_SOURCE": "local"}
    if device == "cpu":
        env["MINERU_MODEL_ONNX_DEVICE"] = "cpu"
    return env


class MineruOcr:
    """MinerU 4.2.3's PP-OCRv6 (tiny det, small rec, ONNX), loaded once; page images are read once each."""

    def __init__(self, device: str):
        os.environ.update(engine_environment(device))
        os.environ.pop("MINERU_API_KEY", None)
        try:
            import numpy as np
            from mineru.model.download import model_path_exists
            from mineru.model.ocr.pp_ocr_v6_onnx import PPOCRv6ONNX
            from mineru.model.registry import MINERU_4_MODELS_ONNX
            from mineru.version import __version__
            from PIL import Image
        except ImportError as exc:
            raise SetupError(f"cannot import MinerU's OCR: {exc}") from exc
        if __version__ != MINERU_VERSION:
            raise SetupError(f"MinerU {__version__} installed, {MINERU_VERSION} expected")
        det, rec = MINERU_4_MODELS_ONNX.ocr_det, MINERU_4_MODELS_ONNX.ocr_rec
        missing = [str(path.local_path()) for path in (det, rec) if not model_path_exists(path)]
        if missing:
            raise SetupError(f"OCR models missing: {', '.join(missing)}; run envs/mineru/setup.sh")
        self._model = PPOCRv6ONNX(det_model_path=str(det.local_path()), rec_model_path=str(rec.local_path()))
        self._np, self._image = np, Image
        self._cached: tuple[Path, object] | None = None

    def _page(self, image: Path):
        if self._cached is None or self._cached[0] != image:
            with self._image.open(image) as handle:
                rgb = self._np.asarray(handle.convert("RGB"))
            self._cached = (image, self._np.ascontiguousarray(rgb[:, :, ::-1]))  # BGR, as MinerU reads images
        return self._cached[1]

    def image_size(self, image: Path) -> tuple[int, int]:
        page = self._page(image)
        return page.shape[1], page.shape[0]

    def read(self, image: Path, box) -> list[tuple[list[float], str, float]]:
        """(box in page pixels, text, score) of every line MinerU's OCR finds in the pixel box."""
        x0, y0, x1, y1 = box
        crop = self._page(image)[y0:y1, x0:x1]
        if crop.size == 0:
            return []
        found = self._model.ocr(self._np.ascontiguousarray(crop))[0] or []
        lines = []
        for quad, (text, score) in found:
            xs, ys = [point[0] for point in quad], [point[1] for point in quad]
            lines.append(([x0 + min(xs), y0 + min(ys), x0 + max(xs), y0 + max(ys)], text, score))
        return lines


def load_engine(device: str, logger) -> MineruOcr:
    engine = MineruOcr(device)
    logger.info("MinerU %s PP-OCRv6 loaded, device %s", MINERU_VERSION, device)
    return engine


def main(argv=None, engine=None) -> int:
    parser = common.worker_parser(__doc__.split("\n\n")[0])
    parser.add_argument("--device", choices=("gpu", "cpu"), required=True)
    args = parser.parse_args(argv)
    logger = common.open_log(args.log.resolve(), STAGE)
    try:
        try:
            papers = common.read_manifest(args.manifest)
        except common.ManifestError as exc:
            logger.error("bad manifest: %s", exc)
            return 2
        if engine is None:
            try:
                engine = load_engine(args.device, logger)
            except Exception as exc:
                logger.error("cannot load MinerU's OCR: %s", common.describe(exc))
                return 1
        try:
            failed = run(papers, engine, logger)
        except (common.BrokenRuntime, common.LockError) as exc:
            logger.error("%s", exc)
            return 1
        return common.PARTIAL if failed else 0
    finally:
        common.close_log(logger)


if __name__ == "__main__":
    sys.exit(main())
