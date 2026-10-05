"""MolNexTR on a folder of crop images, written as predictions.csv. Runs in the molnextr environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from crop_runner import base_parser, list_images, run_crops  # noqa: E402


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--pad-to-square", action="store_true",
        help="pad each crop to a square before resizing; the class API never does (upstream PR #13)",
    )
    args = parser.parse_args()

    import functools
    import itertools

    import MolNexTR.chemical as chemical
    import MolNexTR.model as nextr_model
    import torch
    # The class is MolNexTR.model.molnextr; `from MolNexTR import molnextr` gives the module.
    from MolNexTR.model import molnextr

    # predict_images converts each graph with the default of 16 workers, forking a new Pool per
    # image. Convert in-process instead: same function, same order. That branch uses itertools,
    # which chemical.py never imports.
    chemical.itertools = itertools
    nextr_model.convert_graph_to_smiles = functools.partial(
        chemical.convert_graph_to_smiles, num_workers=1)

    model = molnextr(str(args.checkpoint), torch.device(args.device))
    if args.pad_to_square:
        from MolNexTR.dataset import get_transforms

        size = next(t.height for t in model.transform.transforms if type(t).__name__ == "Resize")
        # get_transforms turns PadToSquare on only for these test-file names.
        model.transform = get_transforms(size, "real/acs.csv", augment=False)

    # predict_final_results drops the decoder's overall_score (MolScribe's confidence), so keep
    # the last decoder output. The SMILES path is unchanged.
    decoded = []
    decode = model.decoder.decode

    def decode_and_keep(*a, **kw):
        out = decode(*a, **kw)
        decoded[:] = out
        return out

    model.decoder.decode = decode_and_keep

    def predict(image: Path):
        decoded.clear()
        output = model.predict_final_results(str(image), return_confidence=True)
        confidence = decoded[0].get("overall_score") if len(decoded) == 1 else None
        return output["predicted_smiles"], confidence

    errors = run_crops(predict, args)
    return 1 if errors == len(list_images(args.images)) else 0


if __name__ == "__main__":
    sys.exit(main())
