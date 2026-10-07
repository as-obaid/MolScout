"""A stand-in whole-PDF reader for the harness tests; it runs through the real benchmarks/tools/paper_runner.py.

--answers is a JSON object of paper ID to a list of [smiles, page, bbox]; papers it does not name get no molecules.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def tools_folder() -> Path:
    """The nearest benchmarks/tools above this file: the repository's, or the test workspace's copy."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "benchmarks" / "tools" / "paper_runner.py").is_file():
            return parent / "benchmarks" / "tools"
    raise SystemExit("fake paper tool: benchmarks/tools/paper_runner.py not found above " + __file__)


sys.path.insert(0, str(tools_folder()))

import paper_runner  # noqa: E402

RECORDED_VARIABLES = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONNOUSERSITE", "PYTHONUNBUFFERED")


def main() -> int:
    parser = paper_runner.base_parser(__doc__)
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--crash", default="", help="comma-separated paper IDs whose predict call raises")
    parser.add_argument("--fail", action="store_true", help="exit with status 3 before reading anything")
    parser.add_argument("--drop-timing", action="store_true", help="delete the timing file, then exit 0")
    parser.add_argument("--record", type=Path, help="write argv, cwd and some environment variables here as JSON")
    parser.add_argument("--predicted", type=Path, help="append the ID of each paper predicted to this file")
    parser.add_argument(
        "--stop-once",
        type=Path,
        metavar="MARKER",
        help="if MARKER does not exist: create it and exit with status 3 once one paper is done",
    )
    args = parser.parse_args()

    if args.record:
        recorded = {name: os.environ.get(name) for name in RECORDED_VARIABLES}
        args.record.write_text(json.dumps({"argv": sys.argv, "cwd": os.getcwd(), "env": recorded}), encoding="utf-8")
    if args.fail:
        print("fake paper tool: failing on purpose", file=sys.stderr)
        return 3
    answers = json.loads(args.answers.read_text(encoding="utf-8")) if args.answers else {}
    crash = {paper for paper in args.crash.split(",") if paper}
    stop = args.stop_once is not None and not args.stop_once.exists()
    seen = []

    def predict(paper):
        if stop and len(seen) == 1:
            args.stop_once.write_text("stopped\n", encoding="utf-8")
            raise SystemExit(3)
        seen.append(paper.paper_id)
        if args.predicted:
            with args.predicted.open("a", encoding="utf-8") as handle:
                handle.write(paper.paper_id + "\n")
        if paper.paper_id in crash:
            raise ValueError("fake crash")
        return [tuple(molecule) for molecule in answers.get(paper.paper_id, [])]

    paper_runner.run_papers(predict, args, warmup=False)
    if args.drop_timing:
        paper_runner.timing_path(args.output).unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
