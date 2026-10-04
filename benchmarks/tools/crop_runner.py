"""Shared loop for the run.py of every structure reader: images in, predictions.csv out.

Standard library only, Python 3.8+, so tool environments can import it without molscout.
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from time import perf_counter

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp"}
COLUMNS = ("dataset", "item_id", "smiles", "page", "bbox", "confidence", "tool", "seconds")
PROGRESS_EVERY = 500


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


def run_crops(predict, args, *, warmup=True):
    """Run predict(image) -> (smiles, confidence | None) on every image; return how many raised."""
    images = list_images(args.images)
    output = Path(args.output)
    part = Path(f"{output}.part")
    if warmup and images:
        try:
            predict(images[0])
        except Exception:
            pass
    errors = 0
    try:
        with part.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            for done, image in enumerate(images, 1):
                start = perf_counter()
                try:
                    smiles, confidence = predict(image)
                except Exception as exc:
                    smiles, confidence = "", None
                    errors += 1
                    print(f"{image.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                seconds = max(0.0, perf_counter() - start)
                writer.writerow(
                    [args.dataset, image.stem, "" if smiles is None else str(smiles), "", "",
                     _confidence(confidence), args.tool, repr(seconds)]
                )
                if done % PROGRESS_EVERY == 0:
                    print(f"{done}/{len(images)} images", file=sys.stderr)
        os.replace(part, output)
    except BaseException:
        if part.exists():
            part.unlink()
        raise
    print(f"{len(images)} images, {errors} failed", file=sys.stderr)
    return errors


def _confidence(value):
    if value is None:
        return ""
    value = float(value)
    return repr(value) if math.isfinite(value) else ""
