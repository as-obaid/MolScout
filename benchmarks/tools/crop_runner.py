"""Shared loop for the run.py of every structure reader: images in, predictions.csv out.

Beside predictions.csv it writes predictions.errors.json, the images whose predict call raised.
With --resume it also keeps each finished row in a checkpoint, so a stopped run can carry on.
Standard library only, Python 3.8+, so tool environments can import it without molscout.
"""

import argparse
import csv
import io
import json
import math
import os
import signal
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
    parser.add_argument(
        "--resume", type=Path, metavar="CHECKPOINT",
        help="CSV of finished rows: images with a row in it are skipped, and each new row is appended to it",
    )
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

    With --resume, images that already have a row in the checkpoint are not run again. Each new
    row is appended to it as soon as its image is done, and its error to <checkpoint
    stem>.errors.jsonl; failures in a row are held back until an image succeeds, so the images
    behind a stop for failures run again. The output holds every row, in image order, and the
    errors file every error. SIGTERM exits with status 143. The checkpoint is never deleted.
    """
    images = list_images(args.images)
    checkpoint = getattr(args, "resume", None)
    if not checkpoint:
        return _run(predict, args, images, {}, {}, lambda row, error: None, warmup)
    checkpoint = Path(checkpoint)
    previous = signal.signal(signal.SIGTERM, _exit_143)
    try:
        done, errors = _resume(checkpoint, images, args.images)
        with checkpoint.open("a", newline="", encoding="utf-8") as rows, _log_path(checkpoint).open("a") as log:
            writer = csv.writer(rows)

            def append(row, error):
                if error is not None:  # the error first: the row is what marks the image done
                    log.write(json.dumps({"item_id": row[1], "error": error}) + "\n")
                    log.flush()
                writer.writerow(row)
                rows.flush()

            return _run(predict, args, images, done, errors, append, warmup)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)


def _run(predict, args, images, done, errors, append, warmup):
    """run_crops after the rows already done (crop ID -> row) and their errors are known."""
    output = Path(args.output)
    part = Path(f"{output}.part")
    if done:
        print(f"resuming: {len(done)} of {len(images)} images already done", file=sys.stderr)
    first = next((image for image in images if image.stem not in done), None)
    if warmup and first is not None:
        try:
            predict(first)
        except Exception as exc:
            print(f"warmup on {first.name} failed: {_describe(exc)}", file=sys.stderr)
    errors = dict(errors)
    streak = []  # the latest failures in a row, (row, error): appended only once an image succeeds
    try:
        with part.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            for count, image in enumerate(images, 1):
                row = done.get(image.stem)
                if row is None:
                    row, exc = _predict(predict, image, args)
                    error = None if exc is None else _describe(exc)
                    streak.append((row, error))
                    if exc is None:
                        for finished in streak:
                            append(*finished)
                        streak = []
                    else:
                        errors[image.stem] = error
                        print(f"{image.name}: {error}", file=sys.stderr)
                        if len(streak) >= MAX_CONSECUTIVE_FAILURES:
                            raise RuntimeError(
                                "{} images failed in a row, the last {}: {}; stopping, since the tool's runtime "
                                "looks broken".format(len(streak), image.name, error)
                            ) from exc
                writer.writerow(row)
                if count % PROGRESS_EVERY == 0:
                    print(f"{count}/{len(images)} images", file=sys.stderr)
        for finished in streak:
            append(*finished)
        ordered = {image.stem: errors[image.stem] for image in images if image.stem in errors}
        report = {"images": len(images), "failed": len(ordered), "errors": ordered}
        _replace(errors_path(output), json.dumps(report, indent=2) + "\n")
        os.replace(part, output)
    except BaseException:
        if part.exists():
            part.unlink()
        raise
    print(f"{len(images)} images, {len(ordered)} failed", file=sys.stderr)
    return len(ordered)


def _predict(predict, image, args):
    """The row for one image, and the exception its predict call raised (None when it did not)."""
    start = perf_counter()
    try:
        smiles, confidence = predict(image)
        exc = None
    except Exception as raised:
        smiles, confidence, exc = "", None, raised
    seconds = max(0.0, perf_counter() - start)
    smiles = "" if smiles is None else str(smiles)
    return [args.dataset, image.stem, smiles, "", "", _confidence(confidence), args.tool, repr(seconds)], exc


def _resume(checkpoint, images, folder):
    """The rows already in a checkpoint (crop ID -> row) and their logged errors.

    A last row or error line that a kill cut short is dropped, and so is an error whose row was
    never written; both files are rewritten to match, header first.
    """
    stems = {image.stem for image in images}
    done = {}
    for row in _read_rows(checkpoint):
        if len(row) != len(COLUMNS):
            raise ValueError(f"{checkpoint}: a row has {len(row)} fields instead of {len(COLUMNS)}: {row!r}")
        if row[1] not in stems:
            raise ValueError(f"{checkpoint} has a row for {row[1]!r}, which has no image in {folder}")
        done[row[1]] = row
    log = _log_path(checkpoint)
    lines = log.read_text(encoding="utf-8").split("\n")[:-1] if log.exists() else []
    errors = {}
    for line in lines:  # split leaves "" after the last newline, or a line cut short: [:-1] drops it
        entry = json.loads(line)
        if entry["item_id"] in done:
            errors[entry["item_id"]] = entry["error"]
    table = io.StringIO()
    csv.writer(table).writerows([COLUMNS, *done.values()])
    _replace(checkpoint, table.getvalue())
    _replace(log, "".join(json.dumps({"item_id": item, "error": error}) + "\n" for item, error in errors.items()))
    return done, errors


def _read_rows(checkpoint):
    """The rows after a checkpoint's header, without a last row that a kill cut short."""
    if not checkpoint.exists():
        return []
    with checkpoint.open(newline="", encoding="utf-8", errors="surrogateescape") as handle:
        text = handle.read()
    source = io.StringIO(text, newline="")
    rows = []
    try:
        rows.extend(csv.reader(source, strict=True))
    except csv.Error as exc:  # at the very end it is a cut inside a quoted field, and that row is not in rows
        if source.read():
            raise ValueError(f"{checkpoint}: {exc}") from None
    else:
        if not text.endswith("\n"):  # every finished row ends with a newline
            rows = rows[:-1]
    if rows and rows[0] != list(COLUMNS):
        raise ValueError(f"{checkpoint} is not a predictions checkpoint; its header is {rows[0]!r}")
    return rows[1:]


def _log_path(checkpoint):
    """The errors log beside a checkpoint: predictions.csv -> predictions.errors.jsonl."""
    return checkpoint.with_name(checkpoint.stem + ".errors.jsonl")


def _replace(path, text):
    """Write text to path through a .part file, so a stop leaves either the old file or the new one."""
    part = Path(f"{path}.part")
    try:
        with part.open("w", newline="", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(part, path)
    except BaseException:
        if part.exists():
            part.unlink()
        raise


def _exit_143(signum, frame):
    """SIGTERM (scancel, a time limit) unwinds through finally clauses and exits with status 143."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # a second SIGTERM must not cut the cleanup short
    sys.exit(143)


def _describe(exc):
    return f"{type(exc).__name__}: {exc}"


def _confidence(value):
    if value is None:
        return ""
    value = float(value)
    return repr(value) if math.isfinite(value) else ""
