"""Save the report's raw saved spec (what the W&B UI renders), read with the GraphQL view_report query.

    python scripts/report/raw_spec.py OUT.json

wr.Report.from_url drops fields (it reports width "readable" and loses panel settings), so read-backs and backups
use this instead. Needs the report-builder extra and W&B credentials.
"""

import json
import sys
from pathlib import Path

from wandb_workspaces.reports.v2 import gql
from wandb_workspaces.reports.v2.interface import _get_api, execute_graphql

REPORT_ID = "VmlldzoxODA1ODcwOQ=="


def fetch(report_id: str = REPORT_ID) -> dict:
    view = execute_graphql(_get_api(), gql.view_report, {"reportId": report_id})["view"]
    return {"view": {key: value for key, value in view.items() if key != "spec"}, "spec": json.loads(view["spec"])}


def main() -> None:
    saved = fetch()
    Path(sys.argv[1]).write_text(json.dumps(saved, indent=1), encoding="utf-8")
    view, spec = saved["view"], saved["spec"]
    print(view["displayName"], view["updatedAt"], "width:", spec.get("width"), "blocks:", len(spec["blocks"]))


if __name__ == "__main__":
    main()
