"""Chemical names to SMILES with the OPSIN command-line jar.

Each batch of names runs through one `java -jar` process: OPSIN reads one name per line
on stdin and writes `SMILES<TAB>name` per line on stdout, with an empty SMILES for a name
it cannot parse. Results are canonicalized with RDKit and cached per name and mode.

With `radicals=True`, OPSIN reads substituent names as radicals and writes the free
valences as `*` (flags `-r -w`): "4-fluorophenyl" gives `*c1ccc(F)cc1` and "methylene"
gives `*C*`. Scripts/fetch_opsin.py downloads the pinned jar into `default_jar_dir()`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from molscout.scoring.smiles import canonical_smiles

JAR_ENV = "MOLSCOUT_OPSIN_JAR"
JAR_PATTERN = re.compile(r"opsin-cli-(\d+(?:\.\d+)*)-jar-with-dependencies\.jar")
TIMEOUT_SECONDS = 600.0
# Java 18+ reads stdin as UTF-8 by default but writes stdout in the locale's charset unless told.
JAVA_OPTIONS = ("-Dfile.encoding=UTF-8", "-Dstdout.encoding=UTF-8", "-Dsun.stdout.encoding=UTF-8")


class OpsinError(RuntimeError):
    """OPSIN could not run, or its output did not line up with the names sent."""


class Opsin:
    """Name-to-SMILES conversion through one OPSIN process per batch, cached per instance."""

    def __init__(self, jar: Path, java: str = "java", timeout: float = TIMEOUT_SECONDS) -> None:
        self.jar = Path(jar)
        if not self.jar.is_file():
            raise OpsinError(f"OPSIN jar not found: {self.jar}; run scripts/fetch_opsin.py")
        resolved = shutil.which(java)
        if resolved is None:
            raise OpsinError(f"Java not found ({java!r}); OPSIN needs Java 8 or newer on PATH")
        self.java = resolved
        self.timeout = timeout
        self._cache: dict[tuple[str, bool], str | None] = {}

    def to_smiles(self, names: list[str], radicals: bool = False) -> list[str | None]:
        """Canonical SMILES for each name, in order; None where OPSIN or RDKit fails.

        Whitespace runs inside a name become single spaces; blank names give None.
        """
        cleaned = [_clean(name) for name in names]
        todo = list(dict.fromkeys(name for name in cleaned if name and (name, radicals) not in self._cache))
        if todo:
            for name, smiles in zip(todo, self._run(todo, radicals), strict=True):
                self._cache[(name, radicals)] = smiles
        return [self._cache[(name, radicals)] if name else None for name in cleaned]

    def _run(self, names: list[str], radicals: bool) -> list[str | None]:
        command = [self.java, *JAVA_OPTIONS, "-jar", str(self.jar), "-n"]
        if radicals:
            command += ["-r", "-w"]
        try:
            done = subprocess.run(
                command,
                input="".join(f"{name}\n" for name in names),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as err:
            raise OpsinError(f"OPSIN did not finish {len(names)} names in {self.timeout:.0f} s") from err
        if done.returncode != 0:
            raise OpsinError(f"OPSIN exited with {done.returncode}: {done.stderr.strip()[-500:]}")
        return _parse_output(done.stdout, names)


def default_jar_dir() -> Path:
    """`$MOLSCOUT_STORE/models/opsin` when MOLSCOUT_STORE is set, else `~/.cache/molscout/opsin`."""
    store = os.environ.get("MOLSCOUT_STORE")
    if store:
        return Path(store) / "models" / "opsin"
    return Path.home() / ".cache" / "molscout" / "opsin"


def find_jar() -> Path | None:
    """The jar named by $MOLSCOUT_OPSIN_JAR, else the newest OPSIN CLI jar in `default_jar_dir()`."""
    override = os.environ.get(JAR_ENV)
    if override:
        return Path(override)
    folder = default_jar_dir()
    if not folder.is_dir():
        return None
    jars = [(_version(match.group(1)), path) for path in folder.iterdir() if (match := JAR_PATTERN.fullmatch(path.name))]
    return max(jars)[1] if jars else None


def _parse_output(stdout: str, names: list[str]) -> list[str | None]:
    lines = stdout.splitlines()
    if len(lines) != len(names):
        raise OpsinError(f"OPSIN answered {len(lines)} of {len(names)} names")
    results = []
    for line, name in zip(lines, names, strict=True):
        smiles, _, echoed = line.partition("\t")
        if echoed != name:
            raise OpsinError(f"OPSIN output out of step: expected {name!r}, got {echoed!r}")
        results.append(canonical_smiles(smiles) if smiles else None)
    return results


def _clean(name: str) -> str:
    return " ".join(name.split())


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))
