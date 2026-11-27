"""Checks and fixes on a W&B report spec (the JSON the report's GraphQL view holds); no W&B calls here.

build_report.py fixes the spec before saving; diff_live.py compares two specs; the tests check run-set filters.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

# Fields W&B or wandb-workspaces regenerate on every build; a diff leaves them out.
VOLATILE = frozenset({"id", "ref", "panelRefs", "openRunSet", "openViz", "__id__", "runSetRef"})


def panel_grids(spec: dict) -> Iterator[dict]:
    for block in spec["blocks"]:
        if block.get("type") == "panel-grid":
            yield block


def runset_filters(runset: dict) -> list[tuple[str, str, Any]]:
    """A run set's enabled filters as (key name, operator, value), e.g. ("displayName", "=", "summary")."""
    found = []
    for item in runset.get("filters", {}).get("filters", []):
        if "key" in item and not item.get("disabled"):
            found.append((item["key"]["name"], item["op"], item.get("value")))
        # filterV1 nests the conditions in an AND group.
        for inner in item.get("filters", []):
            if "key" in inner and not inner.get("disabled"):
                found.append((inner["key"]["name"], inner["op"], inner.get("value")))
    return found


def runset_problems(spec: dict, allowed_names: set[str] | frozenset[str]) -> list[str]:
    """Run sets that could pick up a run they should not: every grid that shows panels must name one run.

    A grid with panels reads its figures and tables from the runs its run set selects, so a filter on job type
    alone would mix a new summary run into the old panels. A grid without panels is a runs table and filters on
    job type and group instead.
    """
    problems = []
    for index, block in enumerate(spec["blocks"]):
        if block.get("type") != "panel-grid":
            continue
        panels = block["metadata"]["panelBankSectionConfig"]["panels"]
        for runset in block["metadata"].get("runSets", []):
            filters = runset_filters(runset)
            names = [value for key, op, value in filters if key == "displayName" and op == "="]
            if panels and (len(names) != 1 or names[0] not in allowed_names):
                problems.append(f"block {index} ({runset.get('name')}): panels without a run-name filter: {filters}")
            if not panels and not any(key == "group" for key, _, _ in filters):
                problems.append(f"block {index} ({runset.get('name')}): runs table without a group filter")
    return problems


def fix_table_types(spec: dict) -> int:
    """Name each Weave table's summary key in its run type; wandb-workspaces writes a generic `table` key.

    The panel's expression picks the logged key (e.g. `failures`) while the embedded type still says
    `table`, which can leave the table loading forever. Returns the number of panels fixed.
    """
    fixed = 0
    for block in panel_grids(spec):
        for panel in block["metadata"]["panelBankSectionConfig"]["panels"]:
            if panel.get("viewType") != "Weave":
                continue
            pick = panel["config"]["panel2Config"]["exp"]["fromOp"]
            key = pick["inputs"]["key"]["val"]
            members = pick["inputs"]["obj"]["type"]["value"]["objectType"]["value"]["members"]
            for member in members:
                types = member.get("propertyTypes", {})
                if "table" in types and key not in types:
                    types[key] = types.pop("table")
                    fixed += 1
    return fixed


def strip_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_volatile(item) for key, item in value.items() if key not in VOLATILE}
    if isinstance(value, list):
        return [strip_volatile(item) for item in value]
    return value


def diff(a: Any, b: Any, path: str = "") -> list[str]:
    """Where two specs differ, one line per difference (values cut to 150 characters)."""
    if type(a) is not type(b):
        return [f"TYPE {path} {str(a)[:120]} | {str(b)[:120]}"]
    lines = []
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a:
                lines.append(f"ONLY NEW  {path}/{key} {str(b[key])[:120]}")
            elif key not in b:
                lines.append(f"ONLY LIVE {path}/{key} {str(a[key])[:120]}")
            else:
                lines += diff(a[key], b[key], f"{path}/{key}")
    elif isinstance(a, list):
        if len(a) != len(b):
            lines.append(f"LEN {path} {len(a)} {len(b)}")
        for index, (x, y) in enumerate(zip(a, b)):
            lines += diff(x, y, f"{path}[{index}]")
    elif a != b:
        lines.append(f"VALUE {path} {repr(a)[:150]} | {repr(b)[:150]}")
    return lines
