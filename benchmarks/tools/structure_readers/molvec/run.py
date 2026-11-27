"""MolVec on a folder of crop images, written as predictions.csv. Runs in the molvec environment.

One JVM (MolvecBatch) stays up for the whole run. A timeout, a crash, an out-of-memory error or
a reply that does not echo the request path stops it, and the next image starts a new one.
RDKit turns each molfile into SMILES; no molfile or no atoms is an empty prediction.
"""

import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from crop_runner import base_parser, list_images, run_crops  # noqa: E402

_UNESCAPE = {"\\": "\\", "n": "\n", "r": "\r", "t": "\t"}


def unescape(text):
    return re.sub(r"\\(.)", lambda m: _UNESCAPE.get(m.group(1), m.group(1)), text)


class MolvecJVM:
    """One MolvecBatch process: a path goes in on stdin, one escaped line comes back."""

    def __init__(self, command, timeout):
        self.command = command
        self.timeout = timeout
        self.process = None
        self.reader = None
        self.lines = None

    def _start(self):
        self.process = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
        )
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self._read, args=(self.process, self.lines), daemon=True)
        self.reader.start()

    @staticmethod
    def _read(process, lines):
        try:
            for line in process.stdout:
                lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            lines.put(None)

    def stop(self):
        if self.process is None:
            return
        process, self.process = self.process, None
        process.kill()
        process.wait()
        self.reader.join(timeout=5)
        for stream in (process.stdin, process.stdout):
            try:
                stream.close()
            except OSError:
                pass

    def ocr(self, image):
        """MolVec's molfile for one image ("" when it gives none); raises on any JVM fault."""
        path = str(Path(image).resolve())
        if "\n" in path or "\r" in path:
            raise ValueError("image path contains a line break")
        if self.process is None or self.process.poll() is not None:
            self.stop()
            self._start()
        try:
            self.process.stdin.write(path + "\n")
            self.process.stdin.flush()
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty:
            self.stop()
            raise TimeoutError(f"MolVec took over {self.timeout:g} s; JVM stopped") from None
        except BrokenPipeError:
            line = None
        if line is None:
            code = self.process.poll()
            self.stop()
            raise RuntimeError(f"MolVec JVM exited (status {code})")
        fields = line.rstrip("\n").split("\t", 2)
        if len(fields) != 3 or fields[0] not in ("OK", "ERR") or unescape(fields[1]) != path:
            self.stop()
            raise RuntimeError(f"MolVec reply does not match the request; JVM stopped: {line[:200]!r}")
        status, _, payload = fields
        message = unescape(payload)
        if status == "ERR":
            if "OutOfMemoryError" in message:
                self.stop()
            raise RuntimeError(f"MolVec: {message}")
        return message


def molfile_to_smiles(molfile):
    """(SMILES, sanitized). No molfile or no atoms gives "". An unsanitizable molfile keeps its
    raw SMILES so it scores as invalid."""
    from rdkit import Chem

    if not molfile.strip():
        return "", True
    raw = Chem.MolFromMolBlock(molfile, sanitize=False)
    if raw is None:
        raise ValueError("RDKit could not parse the molfile")
    if raw.GetNumAtoms() == 0:
        return "", True
    mol = Chem.MolFromMolBlock(molfile)
    if mol is not None:
        return Chem.MolToSmiles(mol), True
    raw.UpdatePropertyCache(strict=False)
    return Chem.MolToSmiles(raw), False


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--java", type=Path, required=True, help="java executable (JDK 17)")
    parser.add_argument("--lib", type=Path, required=True, help="folder of MolVec jars")
    parser.add_argument("--classes", type=Path, required=True, help="folder with MolvecBatch.class")
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds per image")
    parser.add_argument("--heap", default="4g", help="JVM maximum heap (-Xmx)")
    args = parser.parse_args()

    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    # SIGTERM (scancel, time limit) unwinds through finally: the JVM stops, the temp folder goes.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    # MolVec writes resized copies of small images to java.io.tmpdir and never deletes them.
    tmp = tempfile.mkdtemp(prefix="molvec-", dir=os.environ.get("TMPDIR") or None)
    command = [
        str(args.java), f"-Xmx{args.heap}", "-Djava.awt.headless=true", f"-Djava.io.tmpdir={tmp}",
        "-cp", f"{args.classes}:{args.lib}/*", "MolvecBatch",
    ]
    jvm = MolvecJVM(command, args.timeout)

    def predict(image: Path):
        smiles, sanitized = molfile_to_smiles(jvm.ocr(image))
        if not sanitized:
            print(f"{image.name}: RDKit could not sanitize the molfile; kept the raw SMILES",
                  file=sys.stderr)
        return smiles, None

    try:
        errors = run_crops(predict, args)
    finally:
        jvm.stop()
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if errors == len(list_images(args.images)) else 0


if __name__ == "__main__":
    sys.exit(main())
