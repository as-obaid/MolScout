"""MolScribe on a folder of crop images, written as predictions.csv. Runs in the molscribe environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from crop_runner import base_parser, list_images, run_crops  # noqa: E402


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from molscribe import MolScribe

    model = MolScribe(str(args.checkpoint), device=torch.device(args.device))

    def predict(image: Path):
        output = model.predict_image_file(str(image), return_confidence=True)
        return output["smiles"], output.get("confidence")

    errors = run_crops(predict, args)
    return 1 if errors == len(list_images(args.images)) else 0


if __name__ == "__main__":
    sys.exit(main())
