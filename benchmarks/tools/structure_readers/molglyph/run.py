"""MolGlyph on a folder of crop images, written as predictions.csv. Runs in the molglyph environment.

MolGlyph writes a MolParser E-SMILES caption, "<smiles><sep><groups>", not a SMILES. It is turned
into SMILES exactly as BioMiner does (BioMiner/interface.py, run_ocsr_global_batch), with
BioMiner's own molglyph_utils.py from the pinned clone.
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from crop_runner import base_parser, list_images, run_crops  # noqa: E402


def load_utils(path):
    """Load molglyph_utils.py by path; importing BioMiner.commons pulls in the whole pipeline."""
    sys.dont_write_bytecode = True  # leave the pinned clone untouched
    spec = importlib.util.spec_from_file_location("molglyph_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def caption_to_smiles(caption, utils):
    """BioMiner's conversion of one caption to SMILES; may raise where BioMiner catches and gives ""."""
    smiles = utils.get_refactor(utils.Translator.refactor(caption))
    core, groups = caption.split("<sep>")[0], caption.split("<sep>")[1]
    if "<r>" in groups or "<c>" in groups:
        return ""  # ring or circle groups: BioMiner gives None
    return utils.get_new_smiles(smiles, core, groups) or ""


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--utils", type=Path, required=True, help="BioMiner/commons/molglyph_utils.py")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from molscribe import MolGlyph

    utils = load_utils(args.utils)
    model = MolGlyph(str(args.checkpoint), device=torch.device(args.device))

    def predict(image: Path):
        # return_confidence=True fails in MolGlyph (its tokenizer gives no atom indices).
        caption = model.predict_image_file(str(image), return_confidence=False)
        if "<sep>" not in caption:
            # Decoding repeated a token up to the 512-token limit; BioMiner gives "".
            print(f"{image.name}: no <sep> in a {len(caption)}-character caption", file=sys.stderr)
            return "", None
        try:
            smiles = caption_to_smiles(caption, utils)
        except Exception as exc:  # BioMiner catches every conversion error and gives "" (interface.py:698-714)
            print(f"{image.name}: conversion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return "", None
        return smiles, None

    errors = run_crops(predict, args)
    return 1 if errors == len(list_images(args.images)) else 0


if __name__ == "__main__":
    sys.exit(main())
