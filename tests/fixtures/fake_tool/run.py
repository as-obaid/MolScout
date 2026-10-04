"""A stand-in structure reader for the harness tests: one predictions.csv row per image, standard library only.

Like benchmarks/tools/crop_runner.py it writes predictions.errors.json beside predictions.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

COLUMNS = ("dataset", "item_id", "smiles", "page", "bbox", "confidence", "tool", "seconds")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp"}
RECORDED_VARIABLES = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONNOUSERSITE", "PYTHONUNBUFFERED", "FAKE_SETTING")
SECONDS = 0.25


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--answers", type=Path, help="JSON object of crop ID to SMILES; other crops get C")
    parser.add_argument("--skip", default="", help="comma-separated crop IDs to leave out")
    parser.add_argument("--fail", action="store_true", help="exit with status 3 before writing anything")
    parser.add_argument("--bad-row", action="store_true", help="write seconds = -1 on the first row")
    parser.add_argument("--tool-column", help="write this in the tool column instead of --tool")
    parser.add_argument("--record", type=Path, help="write argv, cwd, pid and some environment variables here as JSON")
    parser.add_argument("--sleep", type=float, default=0.0, help="sleep this many seconds before writing anything")
    parser.add_argument(
        "--crash", default="", help="comma-separated crop IDs that 'raise': an empty SMILES and an entry in the errors file"
    )
    parser.add_argument("--no-errors-file", action="store_true", help="write no predictions.errors.json")
    parser.add_argument("--errors-text", help="write this as predictions.errors.json instead")
    args = parser.parse_args()

    if args.record:
        recorded = {name: os.environ.get(name) for name in RECORDED_VARIABLES}
        write_atomically(args.record, json.dumps({"argv": sys.argv, "cwd": os.getcwd(), "pid": os.getpid(), "env": recorded}))
    if args.fail:
        print("fake tool: failing on purpose", file=sys.stderr)
        return 3
    time.sleep(args.sleep)

    answers = json.loads(args.answers.read_text()) if args.answers else {}
    skip = {item for item in args.skip.split(",") if item}
    crash = [item for item in args.crash.split(",") if item]
    images = sorted(
        path
        for path in args.images.iterdir()
        if not path.name.startswith(".") and path.suffix.lower() in IMAGE_SUFFIXES
    )
    tool = args.tool_column or args.tool
    rows = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for image in images:
            if image.stem in skip:
                continue
            seconds = -1 if args.bad_row and rows == 0 else SECONDS
            smiles = "" if image.stem in crash else answers.get(image.stem, "C")
            writer.writerow([args.dataset, image.stem, smiles, "", "", 0.9, tool, seconds])
            rows += 1
    errors = {item: "ValueError: fake crash" for item in crash}
    if args.errors_text is not None:
        write_atomically(errors_path(args.output), args.errors_text)
    elif not args.no_errors_file:
        write_atomically(errors_path(args.output), json.dumps({"images": len(images), "failed": len(errors), "errors": errors}))
    print(f"fake tool: wrote {rows} rows")
    return 0


def errors_path(output: Path) -> Path:
    return output.with_name(output.stem + ".errors.json")


def write_atomically(path: Path, text: str) -> None:
    part = Path(f"{path}.part")
    part.write_text(text, encoding="utf-8")
    os.replace(part, path)


if __name__ == "__main__":
    sys.exit(main())
