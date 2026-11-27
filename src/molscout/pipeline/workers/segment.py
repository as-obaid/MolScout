"""Find drawings on rendered pages and gate them. Runs in the decimer_ai environment (Python 3.9).

For each paper of the manifest, reads <paper_dir>/pages.jsonl (written by render.py) and, for each
page image, runs DECIMER Segmentation 1.3.0 the way DECIMER.ai does (Mask R-CNN, mask expansion,
segments in reading order; segment_chemical_structures called function by function to keep the
boxes, as benchmarks/tools/complete_systems/decimer_ai/run.py does). Each segment is saved as
<paper_dir>/regions/p<page>-r<k>.png at the page's 300 dpi, scored by the DECIMER Image
Classifier, and written to <paper_dir>/regions.jsonl:

    {"region_id": "p3-r2", "page": 3, "bbox": [x0, y0, x1, y1], "crop": "regions/p3-r2.png",
     "structure_score": 1.2e-06, "is_structure": true}

bbox is in PDF points (origin top-left), from the pixel box and the page size in pages.jsonl.
structure_score is the classifier's raw score: low means structure, and a segment is a
structure when the score is at or under the classifier's default threshold (0.000089), as in
DecimerImageClassifier.is_chemical_structure. Segments that are not structures stay in
regions.jsonl with is_structure false; the runner drops them before reading.

A page that fails is a line in <paper_dir>/errors.jsonl and the other pages are kept, but the paper
counts as failed; a paper without pages.jsonl is a line in errors.jsonl and gets no regions.jsonl.
New regions reuse region ids, so the reads and votes of the old ones are removed first, and a rerun
first clears the paper's old segment errors. Exit status 0, 4 when some papers failed, 1 when the
models do not load or files cannot be locked, 2 on a bad manifest.
"""

from __future__ import annotations

import ctypes
import glob
import inspect
import os
import re
import sys
import sysconfig
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _readers_common as common  # noqa: E402

STAGE = "segment"
CROP = re.compile(r"p\d+-r\d+\.png")
LIB_ORDER = ("cuda_runtime", "cublas", "cufft", "curand", "cusparse", "cusolver", "cudnn")


def segment_paper(paper: common.Paper, models) -> tuple[list[dict], int]:
    """Segment, save and score every page of a paper; write its regions and return them with the
    number of pages that failed (each one a line in errors.jsonl).

    models has segment(image) -> ((width, height), [(segment, (y0, x0, y1, x1)), ...]),
    save(segment, path), score(path) -> float and threshold.
    """
    common.remove_outputs(paper.paper_dir, (common.REGIONS, *common.AFTER_REGIONS))
    pages = common.read_jsonl(paper.paper_dir / "pages.jsonl")
    crops = paper.paper_dir / "regions"
    crops.mkdir(parents=True, exist_ok=True)
    for stale in crops.iterdir():
        if CROP.fullmatch(stale.name):
            stale.unlink()
    records, failed_pages = [], 0
    for page in pages:
        try:
            records.extend(segment_page(paper, page, models))
        except Exception as exc:
            failed_pages += 1
            common.append_error(paper.paper_dir, STAGE, paper.paper_id, common.describe(exc), page=page["page"])
    common.write_jsonl(paper.paper_dir / common.REGIONS, records)
    return records, failed_pages


def segment_page(paper: common.Paper, page: dict, models) -> list[dict]:
    size, segments = models.segment(paper.paper_dir / page["image"])
    records = []
    for k, (segment, (y0, x0, y1, x1)) in enumerate(segments, 1):
        region_id = f"p{page['page']}-r{k}"
        crop = f"regions/{region_id}.png"
        models.save(segment, paper.paper_dir / crop)
        score = common.finite_or_none(models.score(paper.paper_dir / crop))
        records.append({
            "region_id": region_id,
            "page": page["page"],
            "bbox": common.px_to_points([x0, y0, x1, y1], page, size),
            "crop": crop,
            "structure_score": score,
            "is_structure": score is not None and score <= models.threshold,
        })
    return records


def run(papers: list[common.Paper], models, logger) -> int:
    """Segment every paper; return how many failed, as a whole or on some page. LockError stops the run."""
    failed = 0
    for paper in papers:
        start = time.perf_counter()
        try:
            common.clear_stage_errors(paper.paper_dir, STAGE)
            records, failed_pages = segment_paper(paper, models)
        except common.LockError:
            raise
        except Exception as exc:
            failed += 1
            common.append_error(paper.paper_dir, STAGE, paper.paper_id, common.describe(exc))
            logger.error("%s: %s", paper.paper_id, common.describe(exc))
            continue
        failed += failed_pages > 0
        kept = sum(r["is_structure"] for r in records)
        logger.info("%s: %d regions, %d structures, %d pages failed, %.1f s", paper.paper_id, len(records), kept,
                    failed_pages, time.perf_counter() - start)
    logger.info("%d papers, %d failed", len(papers), failed)
    return failed


def preload_cuda_libraries(logger) -> None:
    """Load the CUDA 11 libraries of the env's nvidia-*-cu11 packages by path: TensorFlow 2.7 asks
    for CUDA 11 sonames, which Explorer's CUDA >= 12 modules do not provide (as decimer_ai/run.py)."""
    root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    for part in LIB_ORDER:
        for path in sorted(glob.glob(str(root / part / "lib" / "lib*.so*"))):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                logger.warning("preload %s: %s", path, exc)


class DecimerModels:
    """DECIMER Segmentation 1.3.0 and the DECIMER Image Classifier, loaded once for all papers."""

    def __init__(self, device: str, logger):
        preload_cuda_libraries(logger)
        import tensorflow as tf

        # Listing GPUs initialises CUDA before the DECIMER packages set CUDA_VISIBLE_DEVICES=0 on
        # import, so the GPU SLURM assigned is the one used.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpus = tf.config.list_physical_devices("GPU")
        if device == "cpu":
            tf.config.set_visible_devices([], "GPU")
        elif not gpus:
            raise RuntimeError("TensorFlow sees no GPU; pass --device cpu to run on CPU")
        import cv2
        from decimer_image_classifier import DecimerImageClassifier
        from decimer_segmentation import decimer_segmentation

        self._cv2, self._segmentation = cv2, decimer_segmentation
        self._classifier = DecimerImageClassifier()
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
        self.threshold = inspect.signature(self._classifier.is_chemical_structure).parameters["threshold"].default
        logger.info("TensorFlow %s, GPUs %s, running on %s, classifier threshold %s",
                    tf.__version__, [g.name for g in gpus], device, self.threshold)

    def segment(self, image: Path):
        """segment_chemical_structures(page, expand=True) of DECIMER Segmentation 1.3.0, with boxes."""
        page = self._cv2.imread(str(image))  # BGR, as DECIMER.ai's segmentation server reads a page
        if page is None:
            raise OSError(f"cannot read page image {image}")
        masks = self._segmentation.get_expanded_masks(page)
        segments, boxes = self._segmentation.apply_masks(page, masks)
        if len(segments) > 0:
            segments, boxes = self._segmentation.sort_segments_bboxes(segments, boxes)
        kept = [(s, b) for s, b in zip(segments, boxes) if s.shape[0] > 0 and s.shape[1] > 0]
        return (page.shape[1], page.shape[0]), kept

    def save(self, segment, path: Path) -> None:
        if not self._cv2.imwrite(str(path), segment):
            raise OSError(f"cannot write {path}")

    def score(self, path: Path) -> float:
        return float(self._classifier.get_classifier_score(str(path)))


def load_models(device: str, logger) -> DecimerModels:
    return DecimerModels(device, logger)


def main(argv=None) -> int:
    parser = common.worker_parser(__doc__.split("\n\n")[0])
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args(argv)
    logger = common.open_log(args.log.resolve(), STAGE)
    previous = os.getcwd()
    # The classifier saves "caffeine_mod.png" into the working directory for every RGBA image, and
    # segments are RGBA: work from a scratch folder so nothing lands in the repository.
    scratch = tempfile.TemporaryDirectory(prefix="molscout_segment_")
    try:
        try:
            papers = common.read_manifest(args.manifest)
        except common.ManifestError as exc:
            logger.error("bad manifest: %s", exc)
            return 2
        os.chdir(scratch.name)
        try:
            models = load_models(args.device, logger)
        except Exception as exc:
            logger.error("cannot load DECIMER Segmentation and the classifier: %s", common.describe(exc))
            return 1
        try:
            failed = run(papers, models, logger)
        except common.LockError as exc:
            logger.error("%s", exc)
            return 1
        return common.PARTIAL if failed else 0
    finally:
        os.chdir(previous)
        scratch.cleanup()
        common.close_log(logger)


if __name__ == "__main__":
    sys.exit(main())
