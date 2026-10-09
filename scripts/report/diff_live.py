"""Diff the spec build_report.py would save against a raw spec saved by raw_spec.py.

    python scripts/report/diff_live.py LIVE.json [--stage type1|full]

Blocks are aligned first, so an inserted section shows as inserted blocks rather than as every later block
changed; changed blocks are then diffed field by field. Fields W&B regenerates (ids, refs) are left out.
"""

import difflib
import json
import sys
from pathlib import Path

live_path = Path(sys.argv.pop(1))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report  # noqa: E402  (reads --stage from sys.argv)
import spec_tools  # noqa: E402


def label(block: dict) -> str:
    text = json.dumps(block.get("children", ""), ensure_ascii=False)
    if block.get("type") == "panel-grid":
        panels = block["metadata"]["panelBankSectionConfig"]["panels"]
        text = ", ".join(p["viewType"] + str(p["config"].get("mediaKeys", "")) for p in panels) or "runs table"
    return f"{block.get('type')}: {text[:110]}"


def main() -> None:
    live = spec_tools.strip_volatile(json.loads(live_path.read_text(encoding="utf-8"))["spec"])
    _, spec = build_report.build_spec()
    new = spec_tools.strip_volatile(spec)
    print("live blocks", len(live["blocks"]), "new blocks", len(new["blocks"]))
    top = {key: value for key, value in live.items() if key != "blocks"}
    for line in spec_tools.diff(top, {key: value for key, value in new.items() if key != "blocks"}):
        print("top", line)
    a = [json.dumps(block, sort_keys=True) for block in live["blocks"]]
    b = [json.dumps(block, sort_keys=True) for block in new["blocks"]]
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        print(f"{op.upper()} live[{i1}:{i2}] new[{j1}:{j2}]")
        if op == "replace" and i2 - i1 == j2 - j1:
            for i, j in zip(range(i1, i2), range(j1, j2)):
                print(f"  block live {i} / new {j}: {label(live['blocks'][i])}")
                for line in spec_tools.diff(live["blocks"][i], new["blocks"][j]):
                    print("    " + line)
            continue
        for i in range(i1, i2):
            print(f"  - live {i}: {label(live['blocks'][i])}")
        for j in range(j1, j2):
            print(f"  + new {j}: {label(new['blocks'][j])}")
    print("diff done")


if __name__ == "__main__":
    main()
