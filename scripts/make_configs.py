"""Write benchmarks/configs/<tool>__<dataset>.yaml for the five Type 1 crop datasets from tools' tool.yaml.

    python scripts/make_configs.py                                    # every structure reader
    python scripts/make_configs.py benchmarks/tools/structure_readers/molscribe/tool.yaml

Run inside the MolScout environment (pip install -e .). Edit tool.yaml, not the generated configs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from molscout.bench.configs import make_configs

REPO = Path(__file__).resolve().parents[1]
TOOLS = Path("benchmarks") / "tools" / "structure_readers"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate benchmark configs from tool.yaml files.")
    parser.add_argument("tool_yaml", nargs="*", type=Path, help=f"default: every {TOOLS}/*/tool.yaml")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--out", type=Path, help="output folder (default: <repo root>/benchmarks/configs)")
    args = parser.parse_args(argv)
    tool_yamls = args.tool_yaml or sorted((args.repo_root / TOOLS).glob("*/tool.yaml"))
    if not tool_yamls:
        parser.error(f"no tool.yaml files under {args.repo_root / TOOLS}")
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
