"""Fetch the public benchmark datasets into data/raw/, verified against data/manifests/.

    python scripts/fetch_data.py              # every dataset
    python scripts/fetch_data.py uspto jpo    # chosen datasets

Run inside the MolScout environment (pip install -e .).
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

from molscout.data.fetch import ChecksumError, fetch_dataset, load_manifests

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch public benchmark datasets and verify checksums.")
    parser.add_argument("datasets", nargs="*", help="datasets to fetch (default: all)")
    parser.add_argument("--manifests", type=Path, default=REPO / "data" / "manifests")
    parser.add_argument("--root", type=Path, default=REPO / "data" / "raw")
    args = parser.parse_args(argv)
    manifests = load_manifests(args.manifests)
    names = args.datasets or sorted(manifests)
    unknown = sorted(set(names) - set(manifests))
    if unknown:
        parser.error(f"no manifest for {', '.join(unknown)}; available: {', '.join(sorted(manifests))}")
    try:
        for name in names:
            fetch_dataset(manifests[name], args.root)
    except (ChecksumError, OSError, tarfile.TarError) as exc:
        print(f"fetch_data: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
