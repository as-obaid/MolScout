"""Write, or check, data/manifests/biovista_dev.csv: the 20 BioVista development papers.

    python scripts/make_biovista_dev.py            # write the manifest from data/raw/biovista
    python scripts/make_biovista_dev.py --check    # exit 1 if the committed manifest differs

10 table-heavy and 10 drawn-heavy benchmark papers, sampled with seed 6630 (molscout.data.biovista_dev).
The manifest was frozen before any MolScout BioVista run; --check guards it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from molscout.data.biovista_dev import BIOVISTA_DEV_PATH, read_dev_papers, select_dev_papers, write_dev_manifest
from molscout.data.biovista_truth import BIOVISTA_PAPERS_PATH, load_biovista_truth

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write or check the BioVista development-paper manifest.")
    parser.add_argument("--root", type=Path, default=REPO / "data" / "raw" / "biovista", help="BioVista dataset root")
    parser.add_argument("--papers", type=Path, default=REPO / BIOVISTA_PAPERS_PATH, help="biovista_papers.csv")
    parser.add_argument("--output", type=Path, default=REPO / BIOVISTA_DEV_PATH)
    parser.add_argument("--check", action="store_true", help="compare with the committed manifest; write nothing")
    args = parser.parse_args(argv)
    try:
        papers = select_dev_papers(load_biovista_truth(args.root, args.papers))
        if args.check:
            if read_dev_papers(args.output) != papers:
                print(f"make_biovista_dev: {args.output} differs from the selection", file=sys.stderr)
                return 1
            print(f"{args.output} matches the selection")
            return 0
        write_dev_manifest(args.output, papers)
    except (OSError, ValueError) as exc:
        print(f"make_biovista_dev: error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}: {', '.join(paper.paper_id for paper in papers)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
