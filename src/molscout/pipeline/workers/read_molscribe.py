"""Read gated crops with MolScribe 1.1.1. Runs in the molscribe environment.

MolScribe predicts atoms and bonds; a drawn label becomes a dummy atom with that label as its
molfile alias (R1..R12, R, Ra..Rd, X, Y, Z, Q, A, E, Ar also get isotope n for Rn). It then
expands abbreviations and condensed formulas (OMe, CO2Et, ...) and leaves R groups and anything
it cannot expand as "*" or "[n*]" in its SMILES, which drops the label text. Here the molecule
it builds after that expansion is kept (its dummy atoms still carry their alias) and written
with every dummy as [*:n] and the label in slots ("1" -> "R1"), numbered as in _readers_common.
A molecule without dummy atoms keeps MolScribe's own SMILES. "<invalid>" gives no molecule.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _readers_common as common  # noqa: E402

READER = "molscribe"


def add_arguments(parser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True, help="swin_base_char_aux_1m680k.pth")
    parser.add_argument("--device", default="cuda")


def load_reader(args, logger):
    """MolScribe loaded as benchmarks/tools/structure_readers/molscribe/run.py loads it."""
    import molscribe.chemistry as chemistry
    import torch
    from molscribe import MolScribe

    # num_workers=1 (the default) converts graphs in-process, so the wrapper sees each conversion.
    expanded = common.capture_expansions(chemistry)
    model = MolScribe(str(args.checkpoint), device=torch.device(args.device))
    logger.info("MolScribe %s on %s", args.checkpoint, args.device)

    def predict(crop: Path):
        expanded.clear()
        output = model.predict_image_file(str(crop), return_confidence=True)
        mol = expanded[0] if len(expanded) == 1 else None
        return common.graph_read(output["smiles"], mol)

    return predict


def main(argv=None) -> int:
    return common.reader_main(READER, __doc__.split("\n\n")[0], add_arguments, load_reader, argv)


if __name__ == "__main__":
    sys.exit(main())
