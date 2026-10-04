"""Download the public datasets listed in data/manifests/ and verify every file's checksum."""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from molscout.datasets import DATASETS
from molscout.hashing import sha256_file

CHUNK_BYTES = 1 << 20
MANIFEST_KEYS = ("dataset", "description", "source", "revision", "citation", "license", "items", "files")
FILE_KEYS = ("url", "path", "sha256", "size_bytes", "extract")
SHA256 = re.compile(r"[0-9a-f]{64}")


class ChecksumError(RuntimeError):
    """A downloaded file does not match the checksum in its manifest."""


@dataclass(frozen=True, slots=True)
class ManifestFile:
    url: str
    path: str
    sha256: str
    size_bytes: int
    extract: bool


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset: str
    description: str
    source: str
    revision: str
    citation: str
    license: str
    items: int
    files: tuple[ManifestFile, ...]


def load_manifest(path: str | Path) -> DatasetManifest:
    """Read and validate one dataset manifest (JSON)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    _check_keys(data, MANIFEST_KEYS, str(path))
    if data["dataset"] not in DATASETS:
        raise ValueError(f"{path}: unknown dataset {data['dataset']!r}")
    if data["dataset"] != path.stem:
        raise ValueError(f"{path}: file name must be {data['dataset']}.json")
    if not _positive_int(data["items"]):
        raise ValueError(f"{path}: items must be a positive integer")
    if not data["files"]:
        raise ValueError(f"{path}: no files")
    files = tuple(_manifest_file(entry, f"{path}: file {i}") for i, entry in enumerate(data["files"]))
    fields = {key: data[key] for key in MANIFEST_KEYS if key != "files"}
    return DatasetManifest(**fields, files=files)


def load_manifests(directory: str | Path) -> dict[str, DatasetManifest]:
    """Every *.json manifest in a directory, keyed by dataset."""
    manifests = (load_manifest(path) for path in sorted(Path(directory).glob("*.json")))
    return {manifest.dataset: manifest for manifest in manifests}


def download(url: str, destination: Path, *, sha256: str, timeout: float = 60.0) -> None:
    """Stream url to a .part file and move it into place only if its sha256 matches."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, partial.open("wb") as out:
            while chunk := response.read(CHUNK_BYTES):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != sha256:
            raise ChecksumError(f"{url}: sha256 {digest.hexdigest()} does not match the manifest's {sha256}")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def extract_archive(archive: Path, destination: Path) -> None:
    """Unpack a tar archive. The "data" filter refuses absolute paths, '..' and outside links."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter="data")


def fetch_dataset(manifest: DatasetManifest, root: str | Path, *, log: Callable[[str], None] = print) -> Path:
    """Download, verify and unpack one dataset into root/<dataset>/; skip files already verified."""
    target = Path(root) / manifest.dataset
    for entry in manifest.files:
        destination = target / entry.path
        if destination.exists() and sha256_file(destination) == entry.sha256:
            log(f"{manifest.dataset}: {entry.path} already present, checksum ok")
        else:
            log(f"{manifest.dataset}: downloading {entry.path} ({entry.size_bytes:,} bytes)")
            download(entry.url, destination, sha256=entry.sha256)
        if entry.extract:
            extract_archive(destination, target)
    return target


def _check_keys(data: object, expected: tuple[str, ...], where: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a JSON object")
    missing = [key for key in expected if key not in data]
    unknown = sorted(set(data) - set(expected))
    if missing:
        raise ValueError(f"{where}: missing key(s): {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{where}: unknown key(s): {', '.join(unknown)}")


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _manifest_file(entry: object, where: str) -> ManifestFile:
    _check_keys(entry, FILE_KEYS, where)
    assert isinstance(entry, dict)
    if not str(entry["url"]).startswith("https://"):
        raise ValueError(f"{where}: url must use https")
    relative = PurePosixPath(str(entry["path"]))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"{where}: path must be relative and stay inside the dataset folder")
    if not isinstance(entry["sha256"], str) or not SHA256.fullmatch(entry["sha256"]):
        raise ValueError(f"{where}: sha256 must be 64 lowercase hex characters")
    if not _positive_int(entry["size_bytes"]):
        raise ValueError(f"{where}: size_bytes must be a positive integer")
    if not isinstance(entry["extract"], bool):
        raise ValueError(f"{where}: extract must be true or false")
    return ManifestFile(entry["url"], str(relative), entry["sha256"], entry["size_bytes"], entry["extract"])
