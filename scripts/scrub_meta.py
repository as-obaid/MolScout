"""Scrub meta.json files in place: paths written from their roots, no host names or SLURM job IDs.

    python scripts/scrub_meta.py benchmarks/results/*/meta.json
    python scripts/scrub_meta.py --check benchmarks/results/*/meta.json
    python scripts/scrub_meta.py --repo /scratch/<user>/molscout --store /scratch/<user>/molscout-store \
        --home /home/<user> --user <user> benchmarks/results/*/meta.json

The roots default to this machine's: the repository this script is in, MOLSCOUT_STORE, HOME and USER. Pass the
cluster's when the files were made there. The harness already writes new runs scrubbed; this is for older ones.
Only meta.json changes: scores.json, predictions.csv and the other result files are never read or written. A file
the scrub leaves as it is stays byte for byte. --check writes nothing and exits 1 when a file needs a scrub.
See molscout.bench.scrub for the rules.

Run inside the MolScout environment (pip install -e .).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from molscout.bench.scrub import Places, scrub_meta

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scrub meta.json files of the paths, hosts and jobs of their machine.")
    parser.add_argument("files", nargs="+", type=Path, help="meta.json files")
    parser.add_argument("--repo", help="repository root the runs ran in (default: this repository)")
    parser.add_argument("--store", help="MOLSCOUT_STORE the runs used (default: $MOLSCOUT_STORE)")
    parser.add_argument("--home", help="home folder of the account (default: $HOME)")
    parser.add_argument("--user", help="account name (default: $USER)")
    parser.add_argument("--check", action="store_true", help="write nothing; exit 1 if any file needs a scrub")
    args = parser.parse_args(argv)
    given = {field: getattr(args, field) for field in ("repo", "store", "home", "user") if getattr(args, field)}
    places = dataclasses.replace(Places.here(REPO), **given)

    changed: list[tuple[Path, dict]] = []
    for path in args.files:
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError("not a JSON object")
            scrubbed = scrub_meta(meta, places)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            print(f"scrub_meta: error: {path}: {exc}", file=sys.stderr)
            return 1  # nothing is written unless every file can be scrubbed
        if scrubbed != meta:
            changed.append((path, scrubbed))

    if args.check:
        for path, _ in changed:
            print(f"needs a scrub: {path}")
        return 1 if changed else 0
    for path, scrubbed in changed:
        path.write_text(json.dumps(scrubbed, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"scrubbed {path}")
    print(f"{len(changed)} of {len(args.files)} files scrubbed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
