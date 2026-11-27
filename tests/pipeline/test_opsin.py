import importlib.util
import json
import shutil
import stat
import sys
from pathlib import Path

import pytest

from molscout.pipeline.opsin import Opsin, OpsinError, default_jar_dir, find_jar

REPO = Path(__file__).resolve().parents[2]

# Stands in for `java -jar opsin.jar`: answers like the OPSIN CLI (`SMILES<TAB>name`, an empty
# SMILES for a name it cannot parse) and appends its arguments to calls.jsonl.
FAKE_OPSIN = """
import json, sys
from pathlib import Path
NAMES = {"ethanol": "C(C)O", "benzene": "C1=CC=CC=C1"}
RADICALS = {"4-fluorophenyl": "*C1=CC=C(C=C1)F", "methylene": "*C*"}
here = Path(__file__).parent
with (here / "calls.jsonl").open("a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
mode = (here / "mode").read_text().strip() if (here / "mode").exists() else "ok"
if mode == "crash":
    sys.stderr.write("Exception in thread main\\n")
    sys.exit(1)
table = dict(NAMES)
if "-r" in sys.argv and "-w" in sys.argv:
    table.update(RADICALS)
lines = [line.rstrip("\\n") for line in sys.stdin]
if mode == "short":
    lines = lines[:-1]
for name in lines:
    sys.stdout.write(table.get(name, "") + "\\t" + name + "\\n")
"""


@pytest.fixture
def fake(tmp_path):
    (tmp_path / "fake_opsin.py").write_text(FAKE_OPSIN)
    java = tmp_path / "java"
    # A shell wrapper, because a shebang line cannot hold an interpreter path with spaces.
    java.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{tmp_path / "fake_opsin.py"}" "$@"\n')
    java.chmod(java.stat().st_mode | stat.S_IXUSR)
    jar = tmp_path / "opsin-cli-9.9.9-jar-with-dependencies.jar"
    jar.write_bytes(b"not a real jar")
    return Opsin(jar, java=str(java)), tmp_path


def _calls(folder: Path) -> list[list[str]]:
    path = folder / "calls.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_names_become_canonical_smiles_in_input_order(fake):
    opsin, _ = fake
    assert opsin.to_smiles(["benzene", "nonsense", "ethanol"]) == ["c1ccccc1", None, "CCO"]


def test_one_process_per_batch_and_cached_afterwards(fake):
    opsin, folder = fake
    opsin.to_smiles(["benzene", "ethanol", "benzene"])
    opsin.to_smiles(["ethanol", "benzene"])
    assert len(_calls(folder)) == 1


def test_failures_are_cached_too(fake):
    opsin, folder = fake
    assert opsin.to_smiles(["nonsense"]) == [None]
    assert opsin.to_smiles(["nonsense"]) == [None]
    assert len(_calls(folder)) == 1


def test_radical_mode_passes_radical_and_wildcard_flags(fake):
    opsin, folder = fake
    assert opsin.to_smiles(["4-fluorophenyl", "methylene"], radicals=True) == ["*c1ccc(F)cc1", "*C*"]
    args = _calls(folder)[0]
    assert "-r" in args and "-w" in args


def test_radical_and_plain_results_are_cached_separately(fake):
    opsin, folder = fake
    assert opsin.to_smiles(["4-fluorophenyl"]) == [None]
    assert opsin.to_smiles(["4-fluorophenyl"], radicals=True) == ["*c1ccc(F)cc1"]
    assert len(_calls(folder)) == 2


def test_blank_names_give_none_without_a_process(fake):
    opsin, folder = fake
    assert opsin.to_smiles(["", "   "]) == [None, None]
    assert _calls(folder) == []


def test_whitespace_inside_names_is_collapsed(fake):
    opsin, _ = fake
    assert opsin.to_smiles([" ethanol\n", "eth\tanol"]) == ["CCO", None]


def test_crash_raises_with_stderr(fake):
    opsin, folder = fake
    (folder / "mode").write_text("crash")
    with pytest.raises(OpsinError, match="Exception in thread main"):
        opsin.to_smiles(["benzene"])


def test_missing_output_lines_raise(fake):
    opsin, folder = fake
    (folder / "mode").write_text("short")
    with pytest.raises(OpsinError, match="1 of 2"):
        opsin.to_smiles(["benzene", "ethanol"])


def test_missing_jar_raises(tmp_path):
    with pytest.raises(OpsinError, match="fetch_opsin.py"):
        Opsin(tmp_path / "absent.jar")


def test_missing_java_raises(tmp_path):
    jar = tmp_path / "opsin.jar"
    jar.write_bytes(b"")
    with pytest.raises(OpsinError, match="Java"):
        Opsin(jar, java="no-such-java-binary")


def test_default_jar_dir_prefers_molscout_store(monkeypatch, tmp_path):
    monkeypatch.setenv("MOLSCOUT_STORE", str(tmp_path))
    assert default_jar_dir() == tmp_path / "models" / "opsin"
    monkeypatch.delenv("MOLSCOUT_STORE")
    assert default_jar_dir() == Path.home() / ".cache" / "molscout" / "opsin"


def test_find_jar_uses_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MOLSCOUT_OPSIN_JAR", str(tmp_path / "x.jar"))
    assert find_jar() == tmp_path / "x.jar"


def test_find_jar_picks_highest_version_in_default_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("MOLSCOUT_OPSIN_JAR", raising=False)
    monkeypatch.setenv("MOLSCOUT_STORE", str(tmp_path))
    folder = tmp_path / "models" / "opsin"
    folder.mkdir(parents=True)
    assert find_jar() is None
    for version in ("2.9.0", "2.10.0"):
        (folder / f"opsin-cli-{version}-jar-with-dependencies.jar").write_bytes(b"")
    assert find_jar() == folder / "opsin-cli-2.10.0-jar-with-dependencies.jar"


# --- scripts/fetch_opsin.py ---------------------------------------------------------------


def _fetch_script():
    spec = importlib.util.spec_from_file_location("fetch_opsin", REPO / "scripts" / "fetch_opsin.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fetch_script_pins_a_release_and_checksum():
    script = _fetch_script()
    assert script.JAR_NAME == f"opsin-cli-{script.OPSIN_VERSION}-jar-with-dependencies.jar"
    assert script.JAR_URL.endswith(f"/opsin-{script.OPSIN_VERSION}/{script.JAR_NAME}")
    assert len(script.JAR_SHA256) == 64


def test_fetch_script_downloads_into_dest(monkeypatch, tmp_path, capsys):
    script = _fetch_script()
    calls = []
    monkeypatch.setattr(script, "download", lambda url, dest, sha256: calls.append((url, dest, sha256)))
    assert script.main(["--dest", str(tmp_path)]) == 0
    assert calls == [(script.JAR_URL, tmp_path / script.JAR_NAME, script.JAR_SHA256)]
    assert capsys.readouterr().out.strip() == str(tmp_path / script.JAR_NAME)


def test_fetch_script_keeps_a_verified_jar(monkeypatch, tmp_path):
    script = _fetch_script()
    (tmp_path / script.JAR_NAME).write_bytes(b"jar")
    monkeypatch.setattr(script, "sha256_file", lambda path: script.JAR_SHA256)
    monkeypatch.setattr(script, "download", lambda *a, **k: pytest.fail("downloaded again"))
    assert script.main(["--dest", str(tmp_path)]) == 0


def test_fetch_script_defaults_to_the_shared_jar_dir(monkeypatch, tmp_path):
    script = _fetch_script()
    monkeypatch.setenv("MOLSCOUT_STORE", str(tmp_path))
    calls = []
    monkeypatch.setattr(script, "download", lambda url, dest, sha256: calls.append(dest))
    script.main([])
    assert calls == [tmp_path / "models" / "opsin" / script.JAR_NAME]


def test_fetch_script_reports_checksum_failure(monkeypatch, tmp_path, capsys):
    script = _fetch_script()

    def bad_download(url, dest, sha256):
        raise script.ChecksumError("sha256 mismatch")

    monkeypatch.setattr(script, "download", bad_download)
    assert script.main(["--dest", str(tmp_path)]) == 1
    assert "sha256 mismatch" in capsys.readouterr().err


# --- the real OPSIN jar (skipped without Java or the jar) ---------------------------------

REAL_JAR = find_jar()
needs_opsin = pytest.mark.skipif(
    shutil.which("java") is None or REAL_JAR is None or not REAL_JAR.is_file(),
    reason="needs Java and the OPSIN jar (scripts/fetch_opsin.py)",
)


@pytest.fixture(scope="module")
def real_opsin():
    return Opsin(REAL_JAR)


@needs_opsin
def test_real_opsin_names(real_opsin):
    assert real_opsin.to_smiles(["ethanol", "not a chemical name", "4–fluorophenol"]) == ["CCO", None, "Oc1ccc(F)cc1"]


@needs_opsin
def test_real_opsin_substituent_names_get_wildcards(real_opsin):
    got = real_opsin.to_smiles(["4-fluorophenyl", "pyridin-2-yl", "morpholin-4-yl", "methylene"], radicals=True)
    assert got == ["*c1ccc(F)cc1", "*c1ccccn1", "*N1CCOCC1", "*C*"]
