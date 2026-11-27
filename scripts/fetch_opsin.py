"""Download the pinned OPSIN command-line jar and verify its sha256.

    uv run python scripts/fetch_opsin.py              # into the default folder
    uv run python scripts/fetch_opsin.py --dest DIR   # into DIR

The default folder is $MOLSCOUT_STORE/models/opsin/ when MOLSCOUT_STORE is set, else
~/.cache/molscout/opsin/; molscout.pipeline.opsin.find_jar() looks there. A jar already in
place with the pinned sha256 is kept. Prints the jar's path. OPSIN needs Java 8 or newer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from molscout.data.fetch import ChecksumError, download
from molscout.hashing import sha256_file
from molscout.pipeline.opsin import default_jar_dir

OPSIN_VERSION = "2.10.0"
JAR_NAME = f"opsin-cli-{OPSIN_VERSION}-jar-with-dependencies.jar"
JAR_URL = f"https://github.com/dan2097/opsin/releases/download/opsin-{OPSIN_VERSION}/{JAR_NAME}"
JAR_SHA256 = "300f6635384eb0971c5c9814cadd8de7f68c12d8e646650e1253a4056d8caf45"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Download the OPSIN {OPSIN_VERSION} CLI jar.")
    parser.add_argument("--dest", type=Path, default=None, help=f"folder for the jar (default: {default_jar_dir()})")
    args = parser.parse_args(argv)
    jar = (args.dest or default_jar_dir()) / JAR_NAME
    if not (jar.is_file() and sha256_file(jar) == JAR_SHA256):
        try:
            download(JAR_URL, jar, sha256=JAR_SHA256)
        except ChecksumError as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
    print(jar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
