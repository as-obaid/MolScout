"""Publish a benchmark in benchmarks/results/ to Weights & Biases.

    python scripts/wandb_upload.py --entity ENTITY --project PROJECT
    python scripts/wandb_upload.py --entity ENTITY --project PROJECT --offline --images-per-dataset 0
    python scripts/wandb_upload.py --benchmark complete-systems --entity ENTITY --project PROJECT

--benchmark structure-readers (the default) publishes the crop datasets; complete-systems publishes the
BioVista and Internal paper runs (Internal as metrics and per-paper counts only).

Needs the report extra (pip install -e ".[report]"), W&B credentials (wandb login), and the datasets'
references (and, for structure-readers, images) under data/raw/; complete-systems also needs
data/manifests/biovista_papers.csv, and data/internal/ for the Internal runs' ground truth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCHMARKS = ("structure-readers", "complete-systems")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish a benchmark to W&B.")
    parser.add_argument(
        "--benchmark",
        choices=BENCHMARKS,
        default="structure-readers",
        help="structure-readers: the crop datasets; complete-systems: the BioVista and Internal paper runs",
    )
    parser.add_argument(
        "--results", type=Path, default=REPO / "benchmarks" / "results", help="folder of <tool>__<dataset>/ runs"
    )
    parser.add_argument("--entity", required=True, help="W&B entity (user or team)")
    parser.add_argument("--project", required=True, help="W&B project")
    parser.add_argument("--offline", action="store_true", help="log to ./wandb only; upload later with `wandb sync`")
    parser.add_argument(
        "--images-per-dataset",
        type=int,
        default=200,
        help="structure-readers only: crops per dataset in the failures table (0 leaves it out)",
    )
    parser.add_argument(
        "--allow-inconsistent",
        action="store_true",
        help="development only: publish runs from several commits, with uncommitted code or different references",
    )
    parser.add_argument(
        "--repo-root", type=Path, default=REPO, help="folder whose data/raw/ holds references and images"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.images_per_dataset < 0:
        parser.error("--images-per-dataset must be 0 or more")
    try:
        if args.benchmark == "complete-systems":
            from molscout.report.wandb_papers import prepare_papers as prepare
            from molscout.report.wandb_papers import publish_papers as publish
        else:
            from molscout.report.wandb_publish import prepare, publish
    except ImportError as exc:
        print(f'wandb_upload: error: {exc}; install the report extra: pip install -e ".[report]"', file=sys.stderr)
        return 1
    try:
        benchmark = prepare(args.results, args.repo_root, allow_inconsistent=args.allow_inconsistent, warn=_warn)
        print(f"{len(benchmark.runs)} runs: {', '.join(run.name for run in benchmark.runs)}")
        options = {"images_per_dataset": args.images_per_dataset} if args.benchmark == "structure-readers" else {}
        publish(
            benchmark,
            entity=args.entity,
            project=args.project,
            repo_root=args.repo_root,
            offline=args.offline,
            **options,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"wandb_upload: error: {exc}", file=sys.stderr)
        return 1
    return 0


def _warn(message: str) -> None:
    print(f"wandb_upload: warning: {message}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
