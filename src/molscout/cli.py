"""Command line: `molscout score` turns a predictions.csv into scores.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from molscout.data.internal import GROUND_TRUTH_PATH, SPLIT_PATH
from molscout.runs import score_run
from molscout.scoring import write_scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="molscout")
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="score a predictions.csv and write scores.json")
    score.add_argument("predictions", type=Path, help="predictions.csv from one tool on one dataset")
    score.add_argument("-o", "--output", type=Path, required=True, help="scores.json to write")
    score.add_argument("--dataset", help="dataset name; needed only when predictions.csv has no rows")
    score.add_argument(
        "--references",
        type=Path,
        help="crop datasets: directory of reference .mol/.sdf files, or the MolRecBench-Wild root",
    )
    score.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_PATH, help="internal: ground-truth CSV")
    score.add_argument("--split", type=Path, default=SPLIT_PATH, help="internal: dev/test split manifest")
    args = parser.parse_args(argv)
    try:
        write_scores(args.output, _score(args))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"molscout score: error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}")
    return 0


def _score(args: argparse.Namespace) -> dict[str, object]:
    return score_run(
        args.predictions,
        dataset=args.dataset,
        references=args.references,
        ground_truth=args.ground_truth,
        split=args.split,
    )
