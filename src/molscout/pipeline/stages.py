"""What each stage reads and writes, counted from the JSONL files in a paper folder.

The runner counts every stage's records in and out from the files themselves (dictionaries, not the schema
classes), so a stage that drops a paper's records is seen whatever wrote them. A stage's key, which marks it
done for a paper, hashes its settings, its code, the keys of the stages it reads from and, for the stages that
read the PDF, the PDF's sha256.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

from molscout.pipeline.config import PACKAGE_DIR, PLACEHOLDER, PipelineConfig

ORCHESTRATION = frozenset({"__init__.py", "config.py", "inputs.py", "runner.py", "stages.py", "render.py"})
BLOCK_TABLE = "table"
BLOCK_IMAGE = "image"
NOT_STRUCTURE = "not a structure"  # vote.NOT_STRUCTURE: the reason of a region the gate rejected
STRUCTURE_FLAG = "is_structure"
NOT_KEPT = "abstained"


def read_records(path: Path) -> list[dict[str, Any]]:
    """The JSON objects of a JSONL file, one per non-blank line; [] when the file does not exist."""
    if not path.is_file():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {number} is not JSON: {exc}") from None
        if not isinstance(record, dict):
            raise ValueError(f"{path}: line {number} is not a JSON object")
        records.append(record)
    return records


def joiner_output(config: PipelineConfig) -> str:
    """The molecules file finalize reads: the rules, the 32B or the merged molecules."""
    return config.output({"rules": "link", "32b": "join_32b", "both": "merge"}[config.joiner])


def dependencies(stage: str, config: PipelineConfig) -> tuple[str, ...]:
    """The stages whose output a stage reads (its key includes theirs)."""
    if stage in ("render", "parse"):
        return ()
    if stage == "segment":
        return ("render",)
    if stage in ("ocr_labels", "cells"):  # cells adds the table-cell drawings to segment's regions
        return ("render", "parse", "segment")
    if stage.startswith("read_"):
        return ("segment", "cells") if config.cells else ("segment",)
    if stage == "vote":
        return tuple(config.reader_stages())
    if stage == "link":
        return ("vote", "parse", *config.optional_stages())
    if stage == "join_32b":
        return ("vote", "parse", "link") if config.joiner == "both" else ("vote", "parse")
    if stage == "merge":
        return ("link", "join_32b")
    if stage == "finalize":
        return ({"rules": "link", "32b": "join_32b", "both": "merge"}[config.joiner], "parse")
    if stage == "export":
        return ("finalize",)
    raise ValueError(f"unknown stage {stage!r}")


def is_upstream(stage: str) -> bool:
    """Render, parse, segment, label OCR, cells and the readers: what a shared folder can hold for several
    configurations."""
    return stage in ("render", "parse", "segment", "ocr_labels", "cells") or stage.startswith("read_")


def stage_keys(config: PipelineConfig, pdf_sha256: str) -> dict[str, str]:
    """Every stage's key for one paper, in stage order."""
    keys: dict[str, str] = {}
    for stage in config.stage_names():
        parts = {
            "stage": stage,
            "settings": _settings(stage, config),
            "code": _code(stage, config),
            "after": [keys[name] for name in dependencies(stage, config)],
        }
        if stage in ("render", "parse"):
            parts["pdf"] = pdf_sha256
        keys[stage] = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()
    return keys


def upstream_key(keys: Mapping[str, str]) -> str:
    return hashlib.sha256("".join(keys[stage] for stage in keys if is_upstream(stage)).encode()).hexdigest()


def records_in(stage: str, folder: Path, config: PipelineConfig) -> int:
    """How many records a stage was given for a paper."""
    if stage == "render":
        return 1  # the PDF
    if stage in ("parse", "segment"):
        return len(read_records(folder / config.output("render")))
    if stage == "ocr_labels":
        return figure_areas(folder, config)
    if stage == "cells":  # the images MinerU kept in table cells
        layout = read_records(folder / config.output("parse"))
        return sum((block.get("html") or "").count("<img") for block in layout if block.get("type") == BLOCK_TABLE)
    if stage.startswith("read_"):
        regions = read_records(folder / config.output("segment"))
        return sum(bool(region.get(STRUCTURE_FLAG, True)) for region in regions)
    if stage == "vote":
        return len({read["region_id"] for read in _good_reads(folder, config) if "region_id" in read})
    if stage in ("link", "join_32b"):
        # Every voted region but those the gate rejected (link writes nothing for them), and every table.
        tables = [block for block in read_records(folder / config.output("parse")) if block.get("type") == BLOCK_TABLE]
        votes = read_records(folder / config.output("vote"))
        return sum(record.get("reason") != NOT_STRUCTURE for record in votes) + len(tables)
    if stage == "merge":
        return len(read_records(folder / config.output("link"))) + len(read_records(folder / config.output("join_32b")))
    if stage == "finalize":
        return len(read_records(folder / joiner_output(config)))
    if stage == "export":
        return len(read_records(folder / config.output("finalize")))
    raise ValueError(f"unknown stage {stage!r}")


def figure_areas(folder: Path, config: PipelineConfig) -> int:
    """The areas label OCR reads for a paper, as workers/ocr_labels.py picks them: MinerU's image blocks, and the
    structure drawings (not table cells) whose centre is outside every image block."""
    images = [block for block in read_records(folder / config.output("parse")) if block.get("type") == BLOCK_IMAGE]
    drawings = [r for r in read_records(folder / config.output("segment"))
                if r.get(STRUCTURE_FLAG) and r.get("kind") != "cell"]
    outside = [
        r for r in drawings
        if not any(b.get("page") == r.get("page") and _centred_in(r["bbox"], b["bbox"]) for b in images)
    ]
    return len(images) + len(outside)


def _centred_in(box: list[float], outer: list[float]) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def records_out(stage: str, folder: Path, config: PipelineConfig) -> int:
    """How many records a stage wrote for a paper (for a reader: its readable SMILES)."""
    if stage.startswith("read_"):
        reader = stage.removeprefix("read_")
        return sum(read.get("reader") == reader for read in _good_reads(folder, config, stage))
    if stage == "vote":  # one record per region; a failed vote has no SMILES
        return sum(bool(record.get("smiles")) for record in read_records(folder / config.output("vote")))
    if stage == "finalize":
        return sum(m.get("status") != NOT_KEPT for m in read_records(folder / config.output("finalize")))
    if stage == "export":
        molecules = read_records(folder / config.output("finalize"))
        return sum(m.get("status") in config.export_statuses for m in molecules)
    return len(read_records(folder / config.output(stage)))


def funnel(folder: Path, config: PipelineConfig) -> dict[str, int]:
    """A paper's counts along the pipeline, for the W&B funnel: pages to exported molecules."""
    layout = read_records(folder / config.output("parse"))
    regions = read_records(folder / config.output("segment"))
    votes = read_records(folder / config.output("vote"))
    voted = [record for record in votes if record.get("smiles")]
    molecules = read_records(folder / config.output("finalize"))
    counts = {
        "pages": len(read_records(folder / config.output("render"))),
        "layout_blocks": len(layout),
        "tables": sum(block.get("type") == BLOCK_TABLE for block in layout),
        "regions": len(regions),
        "structures": sum(bool(region.get(STRUCTURE_FLAG, True)) for region in regions),
        "cells": sum(region.get("kind") == "cell" for region in regions),
    }
    reads = _good_reads(folder, config)
    for reader in config.readers:
        counts[f"reads_{reader}"] = sum(read.get("reader") == reader for read in reads)
    counts["voted"] = len(voted)
    counts["vote_failed"] = sum(not r.get("smiles") and r.get("reason") != NOT_STRUCTURE for r in votes)
    for agree in range(len(config.readers) + 1):
        counts[f"voted_agree_{agree}"] = sum(record.get("agree") == agree for record in voted)
    counts["ocr_lines"] = len(read_records(folder / config.output("ocr_labels")))
    counts["molecules"] = len(molecules)
    counts["drawn_labelled"] = sum(m.get("source") == "drawn" and bool(m.get("label")) for m in molecules)
    for source in ("drawn", "table", "name", "32b"):
        counts[f"source_{source}"] = sum(m.get("source") == source for m in molecules)
    for status in ("accepted", "review", "abstained"):
        counts[f"status_{status}"] = sum(m.get("status") == status for m in molecules)
    counts["exported"] = sum(m.get("status") in config.export_statuses for m in molecules)
    return counts


def _good_reads(folder: Path, config: PipelineConfig, stage: str | None = None) -> list[dict[str, Any]]:
    """Reads with a SMILES and no error, from the reader stages' files (one stage's file, or all of them)."""
    files = dict.fromkeys(config.output(name) for name in ([stage] if stage else config.reader_stages()))
    reads = [read for name in files for read in read_records(folder / name)]
    return [read for read in reads if read.get("smiles") and not read.get("error")]


def _settings(stage: str, config: PipelineConfig) -> dict[str, Any]:
    """The settings that can change a stage's output (never the vLLM URL: it changes with the job)."""
    if stage in config.workers:
        spec, args = config.workers[stage], config.template_args(stage)
        used = {name for arg in args for name in PLACEHOLDER.findall(arg)} - {"vllm_url"}
        values = {name: _placeholder_value(name, config) for name in sorted(used)}
        return {
            "python": str(spec.python),
            "args": args,
            "values": values,
            "output": config.output(stage),
            "env": dict(sorted(spec.env.items())),
        }
    if stage == "render":
        return {"dpi": config.dpi}
    if stage == "link":
        found = config.opsin_jar_file()
        jar = None if found is None else found.name
        return {"fixes": config.fixes.as_dict(), "opsin_jar": jar, "java": config.java, "thresholds": dict(config.thresholds)}
    if stage in ("vote", "finalize"):
        return {"fixes": config.fixes.as_dict(), "readers": list(config.readers)}
    if stage == "export":
        return {"export": config.export}
    return {}


def _placeholder_value(name: str, config: PipelineConfig) -> object:
    if name.startswith("thresholds."):
        return config.thresholds.get(name.split(".", 1)[1])
    if name == "opsin_jar":
        found = config.opsin_jar_file()
        return None if found is None else found.name
    return str(getattr(config, name, None))


def _code(stage: str, config: PipelineConfig) -> str:
    if stage in config.workers:  # the script and the helpers every worker may import (workers/_*.py)
        worker = config.workers[stage].worker
        helpers = sorted(worker.parent.glob("_*.py"))
        return hashlib.sha256("".join(_file_sha256(path) for path in [worker, *helpers]).encode()).hexdigest()
    if stage == "render":
        return _file_sha256(PACKAGE_DIR / "render.py")
    return _steps_sha256()


@cache
def _steps_sha256() -> str:
    """The pipeline modules the in-process stages run (not the orchestration), hashed together."""
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        if path.name not in ORCHESTRATION:
            digest.update(f"{path.name}\0{_file_sha256(path)}\n".encode())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "missing"
