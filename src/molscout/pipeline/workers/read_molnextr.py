"""Read gated crops with MolNexTR 1.0.2. Runs in the molnextr environment.

MolNexTR builds molecules as MolScribe does (its chemical.py is MolScribe's, with a longer list
of R-group symbols: Ar1.., R', EWG, Nu, and "1*".."12*" for R1..R12): a drawn label is a dummy
atom with the label as alias, abbreviations are expanded, R groups stay "*" or "[n*]". As in
read_molscribe.py, the molecule after expansion is kept and every dummy is written as [*:n]
with its label in slots, numbered as in _readers_common; without dummies MolNexTR's own SMILES
is kept. Loading follows benchmarks/tools/structure_readers/molnextr/run.py: graphs converted
in-process (no forked pool per image) and, with --pad-to-square, crops padded to a square
before resizing (upstream PR #13; the benchmark config passes it).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _readers_common as common  # noqa: E402

READER = "molnextr"


def add_arguments(parser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True, help="molnextr_best.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pad-to-square", action="store_true",
                        help="pad each crop to a square before resizing; the class API never does")


def load_reader(args, logger):
    import functools
    import itertools

    import MolNexTR.chemical as chemical
    import MolNexTR.model as nextr_model
    import torch
    from MolNexTR.model import molnextr

    # predict_images converts each graph with a new 16-worker Pool per image; convert in-process
    # instead (same function, same order). That branch uses itertools, which chemical.py never imports.
    chemical.itertools = itertools
    nextr_model.convert_graph_to_smiles = functools.partial(chemical.convert_graph_to_smiles, num_workers=1)
    expanded = common.capture_expansions(chemical)

    model = molnextr(str(args.checkpoint), torch.device(args.device))
    if args.pad_to_square:
        from MolNexTR.dataset import get_transforms

        size = next(t.height for t in model.transform.transforms if type(t).__name__ == "Resize")
        # get_transforms turns PadToSquare on only for these test-file names.
        model.transform = get_transforms(size, "real/acs.csv", augment=False)
    logger.info("MolNexTR %s on %s, pad to square %s", args.checkpoint, args.device, args.pad_to_square)

    def predict(crop: Path):
        expanded.clear()
        output = model.predict_final_results(str(crop), return_confidence=True)
        mol = expanded[0] if len(expanded) == 1 else None
        return common.graph_read(output["predicted_smiles"], mol)

    return predict


def main(argv=None) -> int:
    return common.reader_main(READER, __doc__.split("\n\n")[0], add_arguments, load_reader, argv)


if __name__ == "__main__":
    sys.exit(main())
