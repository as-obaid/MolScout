"""OpenChemIE on whole PDFs, written as predictions.csv. Runs in the openchemie environment.

The default call for all molecules in a PDF is OpenChemIE.extract_molecules_from_figures_in_pdf
(README, "Extracting Molecule Information From PDFs"): PubLayNet EfficientDet finds the figures
and tables on every page, MolDet finds the molecule boxes in each, MolScribe reads each box.

Boxes. The call returns each molecule's box as fractions (0..1) of its figure image, and the
figure's page only as a 0-based index. The figure's own box is not in the result, so this file
wraps extract_figures_from_pdf of the model instance to keep the figure list the call builds, and
leaves the call itself untouched. interface/tableextractor.py gives a figure's box as
(x1, H - y2, x2, H - y1) in PDF points with the origin at the bottom left (H = mediabox top; the
page image is rendered at 200 dpi and scaled by 72/200); a table that holds a figure carries a
box padded by 20 points on the left and right (the image is the unpadded block). Both are undone
here and the molecule box is mapped through the figure box to page points, origin top left.
Assumes the page's mediabox origin is (0,0) with no /Rotate and no smaller cropbox.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_runner import Molecule, base_parser, read_papers, run_papers  # noqa: E402

TABLE_PAD = 20.0  # tableextractor.extract_table_information pads a table's box by this on each side


def page_tops(pdf):
    """Mediabox top (PDF points) of every page, the H of tableextractor's flip; [] when unreadable."""
    from PyPDF2 import PdfReader

    try:
        return [float(page.mediabox.upper_left[1]) for page in PdfReader(str(pdf)).pages]
    except Exception as exc:
        print(f"{pdf}: cannot read page sizes ({type(exc).__name__}: {exc}); no boxes", file=sys.stderr)
        return []


def figure_box(figure, tops):
    """A figure's box as (x0, y0, x1, y1) in page points, origin top left; None when it cannot be told."""
    page, box = figure["page"], figure["figure"]["bbox"]
    if len(box) != 4 or page >= len(tops):
        return None
    pad = TABLE_PAD if figure["table"]["bbox"] else 0.0
    # box is (x1, top - y_bottom, x2, top - y_top) with the y values in top-left page points
    x0, x1 = box[0] + pad, box[2] - pad
    y0, y1 = tops[page] - box[3], tops[page] - box[1]
    return (x0, y0, x1, y1) if x0 <= x1 and y0 <= y1 else None


def molecule_box(fraction, figure):
    """A molecule box given as fractions of its figure, in page points (clipped to the figure)."""
    fx0, fy0, fx1, fy1 = figure
    fractions = [min(max(float(v), 0.0), 1.0) for v in fraction]
    x0, x1 = sorted((fx0 + fractions[0] * (fx1 - fx0), fx0 + fractions[2] * (fx1 - fx0)))
    y0, y1 = sorted((fy0 + fractions[1] * (fy1 - fy0), fy0 + fractions[3] * (fy1 - fy0)))
    return (x0, y0, x1, y1)


def molecules_of(figures, results, tops):
    """One Molecule per molecule MolScribe read, from the figures the call found and its results."""
    out = []
    for figure, result in zip(figures, results):
        page = int(result["page"]) + 1  # the call reports 0-based pages
        box = figure_box(figure, tops)
        for molecule in result["molecules"]:
            fraction = molecule.get("bbox")
            where = molecule_box(fraction, box) if box is not None and fraction is not None else None
            out.append(Molecule(molecule.get("smiles"), page, where, None))
    return out


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poppler-bin", type=Path, required=True, help="folder with pdftoppm (conda poppler)")
    parser.add_argument("--batch-size", type=int, default=16, help="OpenChemIE's default")
    args = parser.parse_args()

    os.environ["PATH"] = f"{args.poppler_bin}{os.pathsep}{os.environ['PATH']}"
    import pdftotext  # noqa: F401  first: it loads the libstdc++ that poppler needs, before torch loads an older one
    import torch
    from openchemie import OpenChemIE

    model = OpenChemIE(device=torch.device(args.device))
    kept = []
    extract_figures = model.extract_figures_from_pdf

    def keep_figures(*a, **kw):
        figures = extract_figures(*a, **kw)
        kept.append(figures)
        return figures

    model.extract_figures_from_pdf = keep_figures  # the call below runs unchanged and builds this list

    def predict(paper):
        kept.clear()
        with tempfile.TemporaryDirectory(prefix="openchemie-") as work:
            tempfile.tempdir = work  # pdf2image's page renders go here, not into the clone
            try:
                results = model.extract_molecules_from_figures_in_pdf(str(paper.pdf), batch_size=args.batch_size)
            finally:
                tempfile.tempdir = None
        return molecules_of(kept[-1] if kept else [], results, page_tops(paper.pdf))

    failed = run_papers(predict, args)
    if args.device.startswith("cuda"):
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB allocated, "
              f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved", file=sys.stderr)
    return 1 if failed == len(read_papers(args.papers)) else 0


if __name__ == "__main__":
    sys.exit(main())
