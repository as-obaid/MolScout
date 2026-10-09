"""Check that every module under the given folders has current bytecode; setup.sh runs it after compileall.

A .pyc in __pycache__ records its source's mtime and size. When they differ (the source was touched
after compiling), Python compiles the module again at every import and tries to write a new .pyc.
On /scratch a new file took 0.4 s from a compute node, which made vLLM's first start take over
30 min, so setup.sh compiles after it refreshes the mtimes and this check fails if any .pyc is stale.
Files without a .pyc are counted but allowed: a few package files are not Python 3.10 source.

Usage: <env python> bytecode.py FOLDER...   (exit 1 if any .pyc is stale)
"""

import importlib.util
import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor

TAG = sys.implementation.cache_tag  # e.g. cpython-310


def sources(folders):
    """Every .py file under the folders, each once (stdlib and site-packages overlap)."""
    seen = set()
    stack = [os.path.realpath(folder) for folder in folders]
    while stack:
        top = stack.pop()
        try:
            entries = list(os.scandir(top))
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                if entry.name != "__pycache__":
                    stack.append(entry.path)
            elif entry.name.endswith(".py") and entry.path not in seen:
                seen.add(entry.path)
    return sorted(seen)


def state(path):
    """'ok', 'stale' or 'missing' for one source file's __pycache__ .pyc."""
    folder, name = os.path.split(path)
    cached = os.path.join(folder, "__pycache__", f"{name[:-3]}.{TAG}.pyc")
    try:
        source = os.stat(path)
        with open(cached, "rb") as handle:
            header = handle.read(16)
    except OSError:
        return "missing"
    if len(header) < 16 or header[:4] != importlib.util.MAGIC_NUMBER:
        return "stale"
    flags, mtime, size = struct.unpack("<III", header[4:16])
    if flags:  # hash-based .pyc: valid whatever the source's mtime
        return "ok"
    current = mtime == int(source.st_mtime) & 0xFFFFFFFF and size == source.st_size & 0xFFFFFFFF
    return "ok" if current else "stale"


def main():
    folders = sys.argv[1:]
    if not folders:
        sys.exit(__doc__)
    paths = sources(folders)
    with ThreadPoolExecutor(32) as pool:  # one stat and one read per file: latency-bound on /scratch
        states = list(pool.map(state, paths, chunksize=64))
    stale = [path for path, value in zip(paths, states) if value == "stale"]
    print(f"bytecode under {' '.join(folders)}: {len(paths)} modules, {states.count('ok')} current, "
          f"{len(stale)} stale, {states.count('missing')} without a .pyc (not Python 3.10 source)")
    if stale:
        print("stale .pyc files, for example:", *stale[:5], sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
