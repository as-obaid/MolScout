"""Shared loop for the run.py of every structure reader: images in, predictions.csv out.

Beside predictions.csv it writes predictions.errors.json, the images whose predict call raised.
Standard library only, Python 3.8+, so tool environments can import it without molscout.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from time import perf_counter

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp"}
COLUMNS = ("dataset", "item_id", "smiles", "page", "bbox", "confidence", "tool", "seconds")
PROGRESS_EVERY = 500
MAX_CONSECUTIVE_FAILURES = 25


def base_parser(description):
    """Argument parser with the four arguments the harness always passes."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--images", type=Path, required=True, help="folder of crop images")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tool", required=True, help='"<name> <version>"')
    parser.add_argument("--output", type=Path, required=True, help="predictions.csv to write")
    return parser


def list_images(directory):
    """Image files in a folder, sorted by name; hidden files are skipped."""
    found = [
        p
        for p in Path(directory).iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(found, key=lambda p: p.name)


def errors_path(output):
    """The errors file beside an output file: predictions.csv -> predictions.errors.json."""
    output = Path(output)
    return output.with_name(output.stem + ".errors.json")


def run_crops(predict, args, *, warmup=True):
    """Run predict(image) -> (smiles, confidence | None) on every image; return how many raised.

    An image that raises gets an empty SMILES and an entry in the errors file. After
    MAX_CONSECUTIVE_FAILURES in a row the runtime is taken to be broken: RuntimeError, and
    neither file is written.
    """
    images = list_images(args.images)
    output = Path(args.output)
    part = Path(f"{output}.part")
    errors_file = errors_path(output)
    errors_part = Path(f"{errors_file}.part")
    if warmup and images:
        try:
            predict(images[0])
        except Exception as exc:
            print(f"warmup on {images[0].name} failed: {_describe(exc)}", file=sys.stderr)
    errors = {}
    in_a_row = 0
    try:
        with part.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            for done, image in enumerate(images, 1):
                start = perf_counter()
                try:
                    smiles, confidence = predict(image)
                    in_a_row = 0
                except Exception as exc:
                    smiles, confidence = "", None
                    errors[image.stem] = _describe(exc)
                    in_a_row += 1
                    print(f"{image.name}: {errors[image.stem]}", file=sys.stderr)
                    if in_a_row >= MAX_CONSECUTIVE_FAILURES:
                        raise RuntimeError(
                            "{} images failed in a row, the last {}: {}; stopping, since the tool's runtime "
                            "looks broken".format(in_a_row, image.name, errors[image.stem])
                        ) from exc
                seconds = max(0.0, perf_counter() - start)
                writer.writerow(
                    [args.dataset, image.stem, "" if smiles is None else str(smiles), "", "",
                     _confidence(confidence), args.tool, repr(seconds)]
                )
                if done % PROGRESS_EVERY == 0:
                    print(f"{done}/{len(images)} images", file=sys.stderr)
        with errors_part.open("w", encoding="utf-8") as handle:
            json.dump({"images": len(images), "failed": len(errors), "errors": errors}, handle, indent=2)
            handle.write("\n")
        os.replace(errors_part, errors_file)
        os.replace(part, output)
    except BaseException:
        for path in (part, errors_part):
            if path.exists():
                path.unlink()
        raise
    print(f"{len(images)} images, {len(errors)} failed", file=sys.stderr)
    return len(errors)


def _describe(exc):
    return f"{type(exc).__name__}: {exc}"


def _confidence(value):
    if value is None:
        return ""
    value = float(value)
    return repr(value) if math.isfinite(value) else ""
