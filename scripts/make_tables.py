"""Fill the Type 1 tables in docs/Benchmarking.md from benchmarks/results/.

    python scripts/make_tables.py
    python scripts/make_tables.py --results benchmarks/results --doc docs/Benchmarking.md

Run inside the MolScout environment (pip install -e .).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from molscout.tables import fill_type1, git_warnings, load_runs, missing_runs

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fill the Type 1 tables from benchmark results.")
    parser.add_argument("--results", type=Path, default=REPO / "benchmarks" / "results")
    parser.add_argument("--doc", type=Path, default=REPO / "docs" / "Benchmarking.md")
    args = parser.parse_args(argv)
    try:
        runs = load_runs(args.results)
        old = args.doc.read_text(encoding="utf-8")
        new = fill_type1(old, runs)
    except (OSError, ValueError, KeyError) as exc:
        print(f"make_tables: error: {exc}", file=sys.stderr)
        return 1
    print(f"{len(runs)} runs found")
    for warning in git_warnings(runs):
        print(f"make_tables: warning: {warning}", file=sys.stderr)
    missing = missing_runs(runs)
    if missing:
        print(f"missing: {', '.join(missing)}")
    if new != old:
        args.doc.write_text(new, encoding="utf-8")
        print(f"wrote {args.doc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
