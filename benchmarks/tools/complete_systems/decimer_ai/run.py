"""DECIMER.ai on whole PDFs, written as predictions.csv. Runs in the decimer_ai environment.

DECIMER.ai (github.com/OBrink/DECIMER.ai) is a web app; this reproduces its pipeline without the
web server, from the app's own scripts (app/Python/ at the pinned commit):

1. convert_pdf_to_images.py: pdf2image.convert_from_path(pdf, 300), one page image per page. The app
   stops after 10 pages; here every page is read.
2. decimer_segmentation_server.py: segment_chemical_structures_from_file(page), that is DECIMER
   Segmentation (Mask R-CNN) with mask expansion on, segments in reading order, each saved as PNG.
   The package function drops the boxes, so this calls the same functions it does, in the same
   order, keeping the boxes (segment_chemical_structures, decimer_segmentation.py).
3. decimer_predictor_server.py: DECIMER.predict_SMILES(segment PNG), the DECIMER Image Transformer.
   The app reads only the first 20 segments; here every segment is read.
4. decimer_classifier_server.py: DecimerImageClassifier().is_chemical_structure(segment PNG) with
   its default threshold. The app does not drop a segment the classifier rejects: it prints the
   SMILES with a warning ("We are not sure if this is a chemical structure"). So every SMILES is
   output, and --drop-non-structures keeps only the accepted segments. The classifier verdict of
   each output row's verdict is printed on stderr, one line per row, no SMILES:
   "<paper> page <n> segment <i> classifier_score <s> verdict <bool>".

A segment's box is in 300-DPI page pixels and is written in PDF points (pixels * 72 / 300). The
app reports no confidence. A segment the Transformer cannot read (nothing left after its
preprocessing) gets no molecule. The environment's CUDA 11 pip libraries are preloaded because
TensorFlow 2.7 wants CUDA 11 sonames, which Explorer's modules do not provide. An exception
other than that ValueError on a segment (or any other error on a page) fails the whole paper.
"""

import ctypes
import glob
import inspect
import os
import sys
import sysconfig
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_runner import Molecule, base_parser, read_papers, run_papers  # noqa: E402

DPI = 300  # convert_pdf_to_images.py
LIB_ORDER = ("cuda_runtime", "cublas", "cufft", "curand", "cusparse", "cusolver", "cudnn")


def preload_cuda_libraries():
    """Load the CUDA 11 libraries of the pip nvidia-*-cu11 packages by path, so that TensorFlow 2.7's
    dlopen of libcudart.so.11.0 and the like finds them already loaded."""
    root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    for part in LIB_ORDER:
        for path in sorted(glob.glob(str(root / part / "lib" / "lib*.so*"))):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                print(f"preload {path}: {exc}", file=sys.stderr)


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--poppler-path", type=Path, required=True, help="folder with pdftoppm and pdfinfo")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument(
        "--drop-non-structures", action="store_true",
        help="output only segments the DECIMER Image Classifier accepts (the app outputs all of them)",
    )
    args = parser.parse_args()

    # The classifier saves "caffeine_mod.png" into the working directory for every RGBA image (a
    # leftover of its example in decimer_image_classifier.py), and segments are RGBA. Work in a
    # temporary folder so that nothing lands in the repository or an upstream clone.
    args.papers, args.output = args.papers.resolve(), args.output.resolve()
    if args.resume:
        args.resume = args.resume.resolve()
    scratch = tempfile.TemporaryDirectory(prefix="decimer_ai_cwd_")
    os.chdir(scratch.name)

    preload_cuda_libraries()
    import cv2
    import numpy as np
    import tensorflow as tf
    from pdf2image import convert_from_path, pdfinfo_from_path

    # Listing GPUs initialises CUDA before the DECIMER packages set CUDA_VISIBLE_DEVICES=0 on
    # import, so the GPU SLURM assigned is the one used.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = tf.config.list_physical_devices("GPU")
    if args.device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    elif not gpus:
        parser.error("TensorFlow sees no GPU; pass --device cpu to run on CPU")

    from decimer_segmentation import decimer_segmentation as segmentation
    from decimer_image_classifier import DecimerImageClassifier
    from DECIMER import predict_SMILES

    classifier = DecimerImageClassifier()
    if visible is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    threshold = inspect.signature(classifier.is_chemical_structure).parameters["threshold"].default
    print(f"TensorFlow {tf.__version__}, GPUs {[g.name for g in gpus]}, running on {args.device}, "
          f"classifier threshold {threshold}", file=sys.stderr)

    def segment_page(page):
        """segment_chemical_structures(page) of DECIMER Segmentation 1.3.0 (expand=True), with boxes."""
        masks = segmentation.get_expanded_masks(page)
        segments, boxes = segmentation.apply_masks(page, masks)
        if len(segments) > 0:
            segments, boxes = segmentation.sort_segments_bboxes(segments, boxes)
        return [(s, b) for s, b in zip(segments, boxes) if s.shape[0] > 0 and s.shape[1] > 0]

    def predict(paper):
        pages = int(pdfinfo_from_path(str(paper.pdf), poppler_path=str(args.poppler_path))["Pages"])
        molecules = []
        with tempfile.TemporaryDirectory(prefix="decimer_ai_") as work:
            for number in range(1, pages + 1):
                # The app saves each page as PNG and the segmentation server reads it with cv2.imread,
                # so the page is a BGR array.
                image = convert_from_path(str(paper.pdf), DPI, first_page=number, last_page=number,
                                          poppler_path=str(args.poppler_path))[0]
                page = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
                for index, (segment, (y0, x0, y1, x1)) in enumerate(segment_page(page)):
                    path = os.path.join(work, f"{number}_{index}.png")
                    cv2.imwrite(path, segment)
                    score = float(classifier.get_classifier_score(path))
                    is_structure = score <= threshold
                    try:
                        smiles = predict_SMILES(path)
                    except ValueError as exc:  # a blank segment: no drawing left after DECIMER's preprocessing
                        print(f"{paper.paper_id} page {number} segment {index}: no drawing left ({exc})",
                              file=sys.stderr)
                        continue
                    if not smiles or (args.drop_non_structures and not is_structure):
                        continue
                    scale = 72.0 / DPI
                    box = (x0 * scale, y0 * scale, x1 * scale, y1 * scale)
                    print(f"{paper.paper_id} page {number} segment {index} classifier_score {score} "
                          f"verdict {bool(is_structure)}", file=sys.stderr)
                    molecules.append(Molecule(smiles, number, box, None))
        return molecules

    errors = run_papers(predict, args)
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        print(f"GPU memory peak {info['peak'] / 2**20:.0f} MiB", file=sys.stderr)
    except Exception as exc:
        print(f"GPU memory peak not available: {exc}", file=sys.stderr)
    return 1 if errors == len(read_papers(args.papers)) else 0


if __name__ == "__main__":
    sys.exit(main())
