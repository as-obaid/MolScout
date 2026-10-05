"""OCSRGlyph on a folder of crop images, written as predictions.csv. Runs in glyph's uv environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from crop_runner import base_parser, list_images, run_crops  # noqa: E402


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    # glyph's own USPTO evaluation runs in fp32; its predictor defaults to fp16 on CUDA.
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    args = parser.parse_args()

    from glyph.ocsr.predict import OCSRPredictor

    model = OCSRPredictor(args.checkpoint, device=args.device, precision=args.precision)

    def predict(image: Path):
        # predict() keeps glyph's default cleanup (drop [H]/[HH] fragments, RDKit canonical
        # form), as in its published evaluation. OCSRGlyph gives no confidence.
        return model.predict(image), None

    errors = run_crops(predict, args)
    return 1 if errors == len(list_images(args.images)) else 0


if __name__ == "__main__":
    sys.exit(main())
