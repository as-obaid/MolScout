"""MolRecBench-Wild: crop images and CARBON graph labels, stored together in Parquet shards.

The dataset root (`data/raw/molrecbench_wild/`) holds the shards in `data/`. `export_images`
writes each crop's image to `images/<crop ID>.png`, where structure readers find it; the crop
ID is the sample ID without its `.jpg`. References come from the labels (`molscout.data.carbon`).
"""

from __future__ import annotations

import hashlib
import os
from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image

from molscout.data import carbon
from molscout.data.molfiles import Reference
from molscout.hashing import sha256_file

SHARDS = "data/*.parquet"
IMAGES = "images"
LABEL_COLUMNS = (
    "symbols",
    "charges",
    "radicals",
    "valences",
    "isotopes",
    "attach_points",
    "coords",
    "bonds",
    "brackets",
)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def crop_id(sample_id: str) -> str:
    """The sample ID without its image extension; the exported image has the same stem."""
    if sample_id.lower().endswith(IMAGE_SUFFIXES):
        return sample_id.rsplit(".", 1)[0]
    return sample_id


def shard_paths(root: str | Path) -> list[Path]:
    """The Parquet shards under the dataset root, sorted by name."""
    paths = sorted(Path(root).glob(SHARDS))
    if not paths:
        raise FileNotFoundError(f"no {SHARDS} parquet shards under {root}; run scripts/fetch_data.py molrecbench_wild")
    return paths


def read_labels(root: str | Path) -> dict[str, dict[str, Any]]:
    """Every sample's CARBON graph, keyed by crop ID."""
    labels: dict[str, dict[str, Any]] = {}
    for path in shard_paths(root):
        for record in pq.read_table(path, columns=["id", *LABEL_COLUMNS]).to_pylist():
            item = crop_id(record.pop("id"))
            if item in labels:
                raise ValueError(f"crop {item!r} appears twice in {root}")
            labels[item] = record
    return labels


def read_sample_labels(root: str | Path) -> dict[str, dict[str, Any]]:
    """Every crop's `evaluation_subset` (A, B or C) and `hardcase_label` tuple, keyed by crop ID."""
    samples: dict[str, dict[str, Any]] = {}
    for path in shard_paths(root):
        table = pq.read_table(path, columns=["id", "evaluation_subset", "hardcase_label"])
        for record in table.to_pylist():
            item = crop_id(record["id"])
            if item in samples:
                raise ValueError(f"crop {item!r} appears twice in {root}")
            samples[item] = {
                "evaluation_subset": record["evaluation_subset"],
                "hardcase_label": tuple(record["hardcase_label"] or ()),
            }
    return samples


def load_references(root: str | Path) -> dict[str, Reference]:
    """Reference SMILES for every crop, or the reason it is left out."""
    templates = carbon.load_templates()
    return {item: carbon.carbon_reference(item, record, templates) for item, record in read_labels(root).items()}


def reference_set_sha256(root: str | Path) -> str:
    """One checksum over the shards and the abbreviation table, as sha256sum lines."""
    root = Path(root)
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in shard_paths(root)]
    table = resources.files("molscout.data").joinpath(carbon.TEMPLATES_FILE).read_bytes()
    lines.append(f"{hashlib.sha256(table).hexdigest()}  {carbon.TEMPLATES_FILE}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def export_images(root: str | Path, out_dir: str | Path | None = None) -> int:
    """Write each crop's image as `<crop ID>.png`; images already there are kept. Returns the number written."""
    root = Path(root)
    out = Path(out_dir) if out_dir is not None else root / IMAGES
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in shard_paths(root):
        for batch in pq.ParquetFile(path).iter_batches(columns=["id", "image"], batch_size=256):
            for sample_id, image in zip(batch.column("id").to_pylist(), batch.column("image").to_pylist()):
                target = out / f"{crop_id(sample_id)}.png"
                if target.exists():
                    continue
                _write_png(image["bytes"], target)
                written += 1
    return written


def _write_png(data: bytes, target: Path) -> None:
    part = target.with_name(target.name + ".part")
    with Image.open(BytesIO(data)) as image:
        converted = image if image.mode in {"RGB", "L"} else image.convert("RGB")
        converted.save(part, format="PNG")
    os.replace(part, target)
