"""Command line: `molscout score` writes scores.json for a predictions.csv; `molscout bench` runs and scores a tool;
`molscout is-current` says whether a run's results are current."""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from molscout.bench import BenchError, Terminated
from molscout.bench.current import check_current
from molscout.bench.harness import run_benchmark
from molscout.data.internal import GROUND_TRUTH_PATH, SPLIT_PATH
from molscout.data.biovista_truth import BIOVISTA_PAPERS_PATH
from molscout.runs import read_paper_seconds, score_run
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
        help="crop datasets: directory of reference .mol/.sdf files, or the MolRecBench-Wild root; "
        "BioVista: the dataset root (data/raw/biovista)",
    )
    score.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_PATH, help="internal: ground-truth CSV")
    score.add_argument("--split", type=Path, default=SPLIT_PATH, help="internal: dev/test split manifest")
    score.add_argument(
        "--papers",
        type=Path,
        default=BIOVISTA_PAPERS_PATH,
        help="BioVista: frozen paper manifest; papers with status ok and structures are scored, "
        "except those with an unreadable label, which are dropped and listed in scores.json",
    )
    score.add_argument("--paper-seconds", type=Path, help="paper datasets: a run's timing.json, for seconds per paper")
    bench = commands.add_parser("bench", help="run one tool on one crop dataset, then score and record the run")
    bench.add_argument("config", type=Path, help="benchmarks/configs/<tool>__<dataset>.yaml")
    bench.add_argument(
        "--repo-root", type=Path, help="where relative paths in the config start (default: the current directory)"
    )
    bench.add_argument("--results-root", type=Path, help="default: <repo root>/benchmarks/results")
    current = commands.add_parser(
        "is-current",
        help="whether each run's results are current: made from the code the run has at HEAD (its tool folder, "
        "its config, the paper manifest it reads and the harness), with none of it uncommitted; exit 0 only when "
        "all are",
        description="Prints one line per config: RUN current|blocked|not-current CODE WHY, where CODE is the run's "
        "code fingerprint at HEAD (- when it cannot be computed). A blocked run's code has uncommitted changes or "
        "lies outside the repository, so running it could not make it current. Exits 0 when every run is "
        "current, 1 when some is not, and 2 when the check itself fails.",
    )
    current.add_argument("configs", type=Path, nargs="+", metavar="config", help="benchmarks/configs/<tool>__<dataset>.yaml")
    current.add_argument("--repo-root", type=Path, help="the repository root (default: the current directory)")
    current.add_argument("--results-root", type=Path, help="default: <repo root>/benchmarks/results")
    args = parser.parse_args(argv)
    if args.command == "bench":
        return _bench(args)
    if args.command == "is-current":
        return _is_current(args)
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
        papers=args.papers,
        paper_seconds=None if args.paper_seconds is None else read_paper_seconds(args.paper_seconds),
    )


def _is_current(args: argparse.Namespace) -> int:
    repo_root = args.repo_root or Path.cwd()
    results_root = args.results_root or repo_root / "benchmarks" / "results"
    try:
        verdicts = [check_current(config, repo_root=repo_root, results_root=results_root) for config in args.configs]
    except Exception as exc:  # a bug, not a verdict: exit 2, so callers can tell it from "not current"
        print(f"molscout is-current: error: {exc}", file=sys.stderr)
        return 2
    for verdict in verdicts:
        print(verdict.line(), flush=True)
    return 0 if all(verdict.current for verdict in verdicts) else 1


def _bench(args: argparse.Namespace) -> int:
    repo_root = args.repo_root or Path.cwd()
    results_root = args.results_root or repo_root / "benchmarks" / "results"
    try:
        folder = run_benchmark(args.config, repo_root=repo_root, results_root=results_root)
    except (BenchError, RuntimeError, ValueError, OSError) as exc:
        print(f"molscout bench: error: {exc}", file=sys.stderr)
        return 1
    except Terminated as exc:
        print(f"molscout bench: error: {exc}", file=sys.stderr)
        return 128 + signal.SIGTERM
    print(f"wrote {folder}")
    return 0
