"""sha256 of files, for manifests and for pinning a run's inputs."""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK_BYTES = 1 << 20


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    """A file's sha256, or for a folder the sha256 of its sorted `<sha256>  <relative path>` lines.

    A folder's lines cover every file under it, hidden ones included, sorted by relative path.
    """
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(f"no file or folder at {path}")
    files = sorted((file.relative_to(path).as_posix(), file) for file in path.rglob("*") if file.is_file())
    lines = "".join(f"{sha256_file(file)}  {relative}\n" for relative, file in files)
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()
