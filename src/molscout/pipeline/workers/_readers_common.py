"""Helpers shared by the drawing-stage workers (segment.py, read_*.py) and render.py.

Standard library only at import time and Python 3.9+, so every tool environment can import it
without molscout. The slot functions import RDKit when called; every reader environment has it.

Paths in the per-paper records (page images, crops) are relative to the paper folder. A
per-paper failure is one line in <paper_dir>/errors.jsonl: {"stage", "paper_id", "error", ...}; a
stage clears its own lines when it starts on a paper again, so a rerun leaves no stale errors. Files
that several processes rewrite (reads.jsonl, errors.jsonl) are changed under a file lock: flock, else
lockf (the POSIX lock NFS serves), else LockError, never unlocked. A stage that rewrites its output
removes the files made from the old one (AFTER_PAGES, AFTER_REGIONS). Exit status of every worker: 0,
4 when some papers failed (PARTIAL), 1 for a model or setup failure, 2 for a bad manifest.

R groups: a reader's dummy atoms become atom-mapped dummies [*:n], one per atom, and `slots`
maps "n" to the drawn label ("1" -> "R1", "2" -> "X"). Slots are numbered by label (R1, R2, ...
R10 by number, then other labels alphabetically, bare attachment points "*" last) and, for a
repeated label, by the atom's place in the canonical SMILES, so that two readers that read the
same drawing give the same SMILES.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, NamedTuple

MAX_CONSECUTIVE_FAILURES = 25  # reader exceptions in a row before the runtime is taken to be broken
READS = "reads.jsonl"
REGIONS = "regions.jsonl"
ERRORS = "errors.jsonl"
READS_LOCK, ERRORS_LOCK = ".reads.lock", ".errors.lock"
AFTER_REGIONS = (READS, READS_LOCK, "voted.jsonl")  # made from regions.jsonl, stale when it is rewritten
AFTER_PAGES = (REGIONS, *AFTER_REGIONS)  # made from pages.jsonl
PARTIAL = 4  # exit status when some papers failed (their errors are in errors.jsonl)


class Paper(NamedTuple):
    paper_id: str
    pdf: Path
    paper_dir: Path


class ManifestError(ValueError):
    """The work manifest is not one JSON object per line with paper_id, pdf and paper_dir."""


class NoMolecule(Exception):
    """A reader produced no molecule for a crop (not a crash, so it never stops a run)."""


class BrokenRuntime(RuntimeError):
    """Too many reader exceptions in a row: the model's runtime (GPU, CUDA) looks broken."""


class LockError(OSError):
    """Neither flock nor lockf can lock a file: running on would lose other processes' records."""


# --- manifest, records, errors, log -----------------------------------------------------------


def worker_parser(description: str) -> argparse.ArgumentParser:
    """The two arguments every stage worker takes."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="work.jsonl: one {paper_id, pdf, paper_dir} object per line")
    parser.add_argument("--log", type=Path, required=True, help="stage log, appended to")
    return parser


def read_manifest(path) -> list[Paper]:
    """The papers of a work manifest, with absolute paths; ManifestError on any bad line."""
    papers, seen = [], set()
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ManifestError(f"{path}: {exc}") from exc
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{path}:{number}: not JSON ({exc})") from exc
        if not isinstance(entry, dict):
            raise ManifestError(f"{path}:{number}: not a JSON object")
        values = [entry.get(key) for key in Paper._fields]
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ManifestError(f"{path}:{number}: needs non-empty strings {', '.join(Paper._fields)}")
        if values[0] in seen:
            raise ManifestError(f"{path}:{number}: paper {values[0]!r} appears twice")
        seen.add(values[0])
        papers.append(Paper(values[0], Path(values[1]).resolve(), Path(values[2]).resolve()))
    return papers


def read_jsonl(path) -> list[dict]:
    """The records of a JSONL file (FileNotFoundError when it is missing)."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, records) -> None:
    """Write records through a .part file, so a stop leaves either the old file or the new one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    try:
        with part.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(part, path)
    except BaseException:
        if part.exists():
            part.unlink()
        raise


@contextmanager
def file_lock(path):
    """Hold an exclusive lock on path (created if missing): flock, else lockf; LockError when neither works."""
    try:
        import fcntl
    except ImportError as exc:
        raise LockError(f"cannot lock {path}: no fcntl on this platform") from exc
    with Path(path).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            lock = fcntl.flock
        except OSError as refused:
            try:
                fcntl.lockf(handle, fcntl.LOCK_EX)
                lock = fcntl.lockf
            except OSError as exc:
                raise LockError(f"cannot lock {path}: flock failed ({refused}) and lockf failed ({exc})") from exc
        try:
            yield
        finally:
            lock(handle, fcntl.LOCK_UN)


def append_error(paper_dir, stage: str, paper_id: str, error: str, **context) -> None:
    """Add one per-paper failure to <paper_dir>/errors.jsonl."""
    paper_dir = Path(paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)
    record = {"stage": stage, "paper_id": paper_id, "error": error, **context}
    with file_lock(paper_dir / ERRORS_LOCK), (paper_dir / ERRORS).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _stage_of(line: str):
    try:
        entry = json.loads(line)
    except ValueError:
        return None
    return entry.get("stage") if isinstance(entry, dict) else None


def replace_stage_errors(paper_dir, stage: str, records=()) -> None:
    """Replace stage's lines of errors.jsonl with records; other stages' lines, and lines that are not JSON
    objects, stay as they are. The file is removed when nothing is left."""
    paper_dir = Path(paper_dir)
    path = paper_dir / ERRORS
    if not records and not path.exists():
        return
    paper_dir.mkdir(parents=True, exist_ok=True)
    with file_lock(paper_dir / ERRORS_LOCK):
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        kept = [line for line in lines if line.strip() and _stage_of(line) != stage]
        kept += [json.dumps(record, ensure_ascii=False) for record in records]
        if not kept:
            path.unlink()
            return
        part = path.with_name(path.name + ".part")
        part.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
        os.replace(part, path)


def clear_stage_errors(paper_dir, stage: str) -> None:
    """Remove stage's lines from errors.jsonl, before the stage works on the paper again."""
    replace_stage_errors(paper_dir, stage, ())


def remove_outputs(paper_dir, names) -> None:
    """Delete the paper's files that a rewritten stage output has made stale."""
    for name in names:
        (Path(paper_dir) / name).unlink(missing_ok=True)


def describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def open_log(path, stage: str) -> logging.Logger:
    """A logger that appends to the stage log and echoes to stderr."""
    logger = logging.getLogger(f"molscout.{stage}.{id(path)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    close_log(logger)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(f"%(asctime)s {stage} %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stderr)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def close_log(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def px_to_points(box, page: dict, image_size) -> list[float]:
    """[x0, y0, x1, y1] in page-image pixels to PDF points, origin top-left.

    The page is stretched over the whole image (pdfium renders ceil(points * dpi / 72) pixels),
    so the scale is the page size over the image size, not 72 / dpi.
    """
    width, height = image_size
    sx, sy = page["width_pt"] / width, page["height_pt"] / height
    x0, y0, x1, y1 = box
    return [round(x0 * sx, 2), round(y0 * sy, 2), round(x1 * sx, 2), round(y1 * sy, 2)]


# --- labels and slots ----------------------------------------------------------------------


_SCRIPT_CHARS = {
    **{ord(c): str(i) for i, c in enumerate("⁰¹²³⁴⁵⁶⁷⁸⁹")},
    **{ord(c): str(i) for i, c in enumerate("₀₁₂₃₄₅₆₇₈₉")},
    ord("′"): "'", ord("’"): "'", ord("‘"): "'", ord("`"): "'", ord("ʹ"): "'", ord("″"): "''",
    ord("ᵃ"): "a", ord("ᵇ"): "b", ord("ᶜ"): "c", ord("ᵈ"): "d", ord("ₐ"): "a",
}
_MARKUP = re.compile(r"</?\s*su[bp]\s*>", re.IGNORECASE)
_SCRIPT_MARKS = re.compile(r"[\[\]{}^_\s]")
_STARRED = re.compile(r"(\d+|[a-z])\*")  # MolNexTR's R-group symbols: "1*" is R1, "a*" is Ra
_NUMBERED_R = re.compile(r"R(\d+)(.*)")


def normalize_label(text) -> str:
    """One spelling per drawn label: "[R1]", "R[1]", "R^1", "R<sup>1</sup>", "R¹" -> "R1"; "" -> "*"."""
    if text is None:
        return "*"
    label = _MARKUP.sub("", str(text)).translate(_SCRIPT_CHARS)
    label = _SCRIPT_MARKS.sub("", label)
    starred = _STARRED.fullmatch(label)
    if starred:
        label = "R" + starred.group(1)
    return label or "*"


def label_key(label: str):
    """Sort key: R1 < R2 < R10 < other labels (alphabetical) < "*"."""
    numbered = _NUMBERED_R.fullmatch(label)
    if numbered:
        return (0, int(numbered.group(1)), numbered.group(2))
    if label == "*":
        return (2, 0, "")
    return (1, 0, label)


def dummy_label(atom) -> str:
    """A dummy atom's label: its alias (molfile "A" line, MolScribe's symbol), else R<isotope>, else "*"."""
    if atom.HasProp("molFileAlias") and atom.GetProp("molFileAlias").strip():
        return normalize_label(atom.GetProp("molFileAlias"))
    if atom.GetIsotope() > 0:
        return f"R{atom.GetIsotope()}"
    return "*"


def assign_slots(mol) -> tuple[str, dict[str, str]]:
    """SMILES of mol with every dummy atom as [*:n], and the slots map {"n": label}."""
    from rdkit import Chem

    work = Chem.RWMol(mol)
    dummies = [atom.GetIdx() for atom in work.GetAtoms() if atom.GetAtomicNum() == 0]
    labels = {index: dummy_label(work.GetAtomWithIdx(index)) for index in dummies}
    for index in dummies:
        work.GetAtomWithIdx(index).SetIsotope(0)
        work.GetAtomWithIdx(index).SetAtomMapNum(0)
    if not dummies:
        return Chem.MolToSmiles(work), {}
    Chem.MolToSmiles(work)  # sets the canonical output order, the tie-break for repeated labels
    order = [int(i) for i in re.findall(r"\d+", work.GetProp("_smilesAtomOutputOrder"))]
    place = {index: position for position, index in enumerate(order)}
    ranked = sorted(dummies, key=lambda index: (label_key(labels[index]), place.get(index, len(place))))
    slots = {}
    for number, index in enumerate(ranked, 1):
        work.GetAtomWithIdx(index).SetAtomMapNum(number)
        slots[str(number)] = labels[index]
    return Chem.MolToSmiles(work), slots


def smiles_read(smiles: str) -> tuple[str, dict[str, str]]:
    """A reader's SMILES with slots; text RDKit cannot parse is kept as it is, for the checks to reject."""
    from rdkit import Chem, RDLogger

    if not smiles or not smiles.strip():
        raise NoMolecule("empty SMILES")
    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromSmiles(smiles)
    finally:
        RDLogger.EnableLog("rdApp.*")
    if mol is None:
        return smiles, {}
    return assign_slots(mol)


def graph_read(smiles, mol) -> tuple[str, dict[str, str]]:
    """MolScribe's or MolNexTR's output as (smiles, slots).

    mol is the molecule after the reader expanded abbreviations, with its unexpanded dummy atoms
    still carrying their alias; None when the reader stopped before that step. Without dummy
    atoms the reader's own SMILES is returned unchanged.
    """
    if mol is None:
        if not smiles or smiles == "<invalid>":
            raise NoMolecule(f"no molecule ({smiles or 'empty'})")
        return smiles_read(smiles)
    if mol.GetNumAtoms() == 0:
        raise NoMolecule("empty molecule")
    if not any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        return smiles, {}
    return assign_slots(mol)


def capture_expansions(module, name: str = "_expand_functional_group") -> list:
    """Wrap module.<name>, which returns (smiles, mol), so that each mol it returns is kept.

    MolScribe and MolNexTR look the function up in their chemistry module at call time, and
    convert graphs in-process when num_workers is 1, so the wrapper sees every conversion.
    """
    original = getattr(module, name)
    captured = []

    def expand_and_keep(*args, **kwargs):
        smiles, mol = original(*args, **kwargs)
        captured.append(mol)
        return smiles, mol

    setattr(module, name, expand_and_keep)
    return captured


# --- the read loop -------------------------------------------------------------------------


def structure_regions(paper_dir) -> list[dict]:
    """The regions of a paper that the gate kept (is_structure), in regions.jsonl order."""
    regions = [region for region in read_jsonl(Path(paper_dir) / REGIONS) if region.get("is_structure")]
    for region in regions:
        if not (isinstance(region.get("region_id"), str) and isinstance(region.get("crop"), str)):
            raise ValueError(f"{REGIONS}: a region needs region_id and crop, got {region!r}")
    return regions


def read_record(region_id: str, reader: str, smiles=None, slots=None, error=None) -> dict:
    return {"region_id": region_id, "reader": reader, "smiles": smiles, "slots": slots or {}, "error": error}


def merge_reads(paper_dir, reader: str, records: list[dict]) -> None:
    """Replace this reader's records in <paper_dir>/reads.jsonl, keeping the other readers', under the
    paper's reads lock, so readers running at once do not lose reads."""
    paper_dir = Path(paper_dir)
    path = paper_dir / READS
    with file_lock(paper_dir / READS_LOCK):
        others = [r for r in read_jsonl(path) if r.get("reader") != reader] if path.exists() else []
        merged = others + list(records)
        merged.sort(key=lambda r: r.get("reader") or "")  # stable: each reader keeps its region order
        write_jsonl(path, merged)


def run_reader(reader: str, predict: Callable[[Path], tuple], papers: list[Paper], logger) -> int:
    """Read every gated crop of every paper with predict(crop) -> (smiles, slots); return papers failed.

    A crop whose predict call raises gets smiles None and the error; NoMolecule is a reading
    with no molecule, any other exception counts towards MAX_CONSECUTIVE_FAILURES, after which
    the runtime is taken to be broken (BrokenRuntime). A paper whose regions or reads cannot be
    read or written is an error in its errors.jsonl, and the other papers go on. LockError stops
    the run: without the lock other readers' reads would be lost.
    """
    stage = f"read_{reader}"
    failed, streak = 0, [0]
    for paper in papers:
        start = time.perf_counter()
        try:
            clear_stage_errors(paper.paper_dir, stage)
            records = _read_paper(reader, predict, paper, streak)
            merge_reads(paper.paper_dir, reader, records)
        except (BrokenRuntime, LockError):
            raise
        except Exception as exc:
            failed += 1
            append_error(paper.paper_dir, stage, paper.paper_id, describe(exc))
            logger.error("%s: %s", paper.paper_id, describe(exc))
            continue
        logger.info("%s: %d crops read, %d without a molecule, %.1f s", paper.paper_id, len(records),
                    sum(r["error"] is not None for r in records), time.perf_counter() - start)
    logger.info("%d papers, %d failed", len(papers), failed)
    return failed


def _read_paper(reader: str, predict, paper: Paper, streak: list) -> list[dict]:
    """The read records of one paper; streak[0] counts reader exceptions in a row across papers."""
    records = []
    for region in structure_regions(paper.paper_dir):
        crop = paper.paper_dir / region["crop"]
        try:
            if not crop.is_file():
                raise NoMolecule(f"crop not found: {region['crop']}")
            smiles, slots = predict(crop)
            streak[0] = 0
            records.append(read_record(region["region_id"], reader, smiles, slots))
        except Exception as exc:
            error = describe(exc)
            if not isinstance(exc, NoMolecule):
                streak[0] += 1
                if streak[0] >= MAX_CONSECUTIVE_FAILURES:
                    raise BrokenRuntime(f"{streak[0]} crops failed in a row, the last {crop}: {error}; "
                                        "stopping, since the reader's runtime looks broken") from exc
            records.append(read_record(region["region_id"], reader, error=error))
    return records


def reader_main(reader: str, description: str, add_arguments, load_reader, argv=None) -> int:
    """Entry point of read_<reader>.py: exit 2 on a bad manifest, 1 when the model does not load, the
    runtime breaks or files cannot be locked, 4 when some papers failed (their errors are in each
    paper's errors.jsonl), else 0."""
    parser = worker_parser(description)
    add_arguments(parser)
    args = parser.parse_args(argv)
    logger = open_log(args.log, f"read_{reader}")
    try:
        try:
            papers = read_manifest(args.manifest)
        except ManifestError as exc:
            logger.error("bad manifest: %s", exc)
            return 2
        try:
            predict = load_reader(args, logger)
        except Exception as exc:
            logger.error("cannot load %s: %s", reader, describe(exc))
            return 1
        try:
            failed = run_reader(reader, predict, papers, logger)
        except (BrokenRuntime, LockError) as exc:
            logger.error("%s", exc)
            return 1
        return PARTIAL if failed else 0
    finally:
        close_log(logger)


def finite_or_none(value):
    """A float that JSON can hold, or None."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None
