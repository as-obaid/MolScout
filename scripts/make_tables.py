"""Fill the Type 1 and Type 2 tables in docs/Benchmarking.md from benchmarks/results/.

    python scripts/make_tables.py
    python scripts/make_tables.py --results benchmarks/results --doc docs/Benchmarking.md

Run inside the MolScout environment (pip install -e .).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from molscout import tables_papers
from molscout.tables import fill_type1, git_warnings, load_runs, missing_runs

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fill the Type 1 and Type 2 tables from benchmark results.")
    parser.add_argument("--results", type=Path, default=REPO / "benchmarks" / "results")
    parser.add_argument("--doc", type=Path, default=REPO / "docs" / "Benchmarking.md")
    args = parser.parse_args(argv)
    try:
        runs = load_runs(args.results)
        old = args.doc.read_text(encoding="utf-8")
        paper_runs = tables_papers.load_paper_runs(args.results)
        new = tables_papers.fill_type2(fill_type1(old, runs), paper_runs)
    except (OSError, ValueError, KeyError) as exc:
        print(f"make_tables: error: {exc}", file=sys.stderr)
        return 1
    print(f"{len(runs)} runs found, {len(paper_runs)} complete-system runs found")
    for warning in (*git_warnings(runs), *tables_papers.git_warnings(paper_runs)):
        print(f"make_tables: warning: {warning}", file=sys.stderr)
    for label, missing in (("missing", missing_runs(runs)), ("missing Type 2", tables_papers.missing_paper_runs(paper_runs))):
        if missing:
            print(f"{label}: {', '.join(missing)}")
    if new != old:
        args.doc.write_text(new, encoding="utf-8")
        print(f"wrote {args.doc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
