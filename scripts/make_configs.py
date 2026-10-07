"""Write benchmarks/configs/<tool>__<dataset>.yaml from tools' tool.yaml.

A structure reader gets the five Type 1 crop configs; a complete system gets the two Type 2 paper configs.

    python scripts/make_configs.py                                    # every structure reader and complete system
    python scripts/make_configs.py benchmarks/tools/structure_readers/molscribe/tool.yaml

Run inside the MolScout environment (pip install -e .). Edit tool.yaml, not the generated configs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from molscout.bench.configs import make_configs

REPO = Path(__file__).resolve().parents[1]
TOOLS = (Path("benchmarks/tools/structure_readers"), Path("benchmarks/tools/complete_systems"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate benchmark configs from tool.yaml files.")
    parser.add_argument("tool_yaml", nargs="*", type=Path, help="default: every tool.yaml under benchmarks/tools/structure_readers and complete_systems")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--out", type=Path, help="output folder (default: <repo root>/benchmarks/configs)")
    args = parser.parse_args(argv)
    tool_yamls = args.tool_yaml or sorted(
        path for folder in TOOLS for path in (args.repo_root / folder).glob("*/tool.yaml")
    )
    if not tool_yamls:
        parser.error(f"no tool.yaml files under {args.repo_root / 'benchmarks' / 'tools'}")
    out = args.out or args.repo_root / "benchmarks" / "configs"
    try:
        for tool_yaml in tool_yamls:
            for path in make_configs(tool_yaml, out, repo_root=args.repo_root):
                print(f"wrote {path}")
    except (ValueError, OSError) as exc:
        print(f"make_configs: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
