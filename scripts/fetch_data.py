"""Fetch the public benchmark datasets into data/raw/, verified against data/manifests/.

    python scripts/fetch_data.py                       # every dataset
    python scripts/fetch_data.py uspto jpo             # chosen datasets
    python scripts/fetch_data.py biovista              # BioVista labels, then open-access PDFs
    python scripts/fetch_data.py biovista --limit 5    # a trial run on the first 5 papers

BioVista ships labels only. After its labels are verified, each paper's PDF is looked up by
PDB ID (RCSB, Europe PMC, OpenAlex; Unpaywall too if UNPAYWALL_EMAIL is set), saved to
data/raw/biovista/pdfs/<pdb id>.pdf and recorded in data/manifests/biovista_papers.csv.
A paper still without a PDF then gets its pinned copy from data/manifests/biovista_recovered.csv,
if it has one (a valid PDF already in pdfs/ is used without downloading). Rerunning skips PDFs
already present with their recorded sha256.

Run inside the MolScout environment (uv run python scripts/fetch_data.py ...).
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from collections import Counter
from pathlib import Path

from molscout.data.biovista import MAX_WORKERS, fetch_papers, unpaywall_email_from_env
from molscout.data.fetch import ChecksumError, fetch_dataset, load_manifests
from molscout.data.molrecbench import export_images

REPO = Path(__file__).resolve().parents[1]
PAPERS_CSV = "biovista_papers.csv"
RECOVERED_CSV = "biovista_recovered.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch public benchmark datasets and verify checksums.")
    parser.add_argument("datasets", nargs="*", help="datasets to fetch (default: all)")
    parser.add_argument("--manifests", type=Path, default=REPO / "data" / "manifests")
    parser.add_argument("--root", type=Path, default=REPO / "data" / "raw")
    parser.add_argument("--limit", type=_positive_int, help="BioVista: fetch PDFs for the first N papers only")
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        choices=range(1, MAX_WORKERS + 1),
        metavar=f"{{1..{MAX_WORKERS}}}",
        help=f"BioVista: papers fetched at once (default {MAX_WORKERS})",
    )
    args = parser.parse_args(argv)
    manifests = load_manifests(args.manifests)
    names = args.datasets or sorted(manifests)
    unknown = sorted(set(names) - set(manifests))
    if unknown:
        parser.error(f"no manifest for {', '.join(unknown)}; available: {', '.join(sorted(manifests))}")
    try:
        for name in names:
            target = fetch_dataset(manifests[name], args.root)
            if name == "molrecbench_wild":
                print(f"molrecbench_wild: exported {export_images(target)} images to {target / 'images'}")
            if name == "biovista":
                counts = fetch_papers(
                    target,
                    args.manifests / PAPERS_CSV,
                    recovered_path=args.manifests / RECOVERED_CSV,
                    limit=args.limit,
                    workers=args.workers,
                    unpaywall_email=unpaywall_email_from_env(),
                )
                print(f"biovista: {_tally(counts)}; PDFs in {target / 'pdfs'}, rows in {args.manifests / PAPERS_CSV}")
    except (ChecksumError, OSError, ValueError, tarfile.TarError) as exc:
        print(f"fetch_data: error: {exc}", file=sys.stderr)
        return 1
    return 0


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _tally(counts: Counter[str]) -> str:
    total = sum(counts.values())
    parts = ", ".join(f"{status} {n}" for status, n in sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    return f"{total} papers: {parts}" if total else "0 papers"


if __name__ == "__main__":
    sys.exit(main())
