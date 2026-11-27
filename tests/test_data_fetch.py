import hashlib
import importlib.util
import io
import json
import tarfile
import urllib.error
from collections import Counter
from email.message import Message
from pathlib import Path

import pytest

from molscout.data import fetch
from molscout.data.fetch import (
    ChecksumError,
    DatasetManifest,
    ManifestFile,
    download,
    extract_archive,
    fetch_dataset,
    load_manifest,
    load_manifests,
    sha256_file,
)
from molscout.data.httpclient import USER_AGENT

REPO = Path(__file__).resolve().parents[1]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_tar(path: Path, members: dict[str, bytes]) -> bytes:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path.read_bytes()


def local_manifest(source: Path, data: bytes, *, extract: bool) -> DatasetManifest:
    entry = ManifestFile(source.as_uri(), source.name, sha(data), len(data), extract)
    return DatasetManifest("uspto", "test", "local", "r1", "c", "l", 1, (entry,))


def test_sha256_file(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == sha(b"abc")


def test_download_verifies_and_keeps_the_file(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    destination = tmp_path / "out" / "file.bin"
    download(source.as_uri(), destination, sha256=sha(b"payload"))
    assert destination.read_bytes() == b"payload"
    assert not destination.with_name("file.bin.part").exists()


def test_download_rejects_bad_checksum_and_leaves_nothing(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"tampered")
    destination = tmp_path / "file.bin"
    with pytest.raises(ChecksumError, match="does not match"):
        download(source.as_uri(), destination, sha256=sha(b"payload"))
    assert not destination.exists()
    assert not destination.with_name("file.bin.part").exists()


class FlakyUrlopen:
    """Fails with the given errors first, then serves payload."""

    def __init__(self, payload, *errors):
        self.payload = payload
        self.errors = list(errors)
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        if self.errors:
            raise self.errors.pop(0)
        return io.BytesIO(self.payload)


def test_download_retries_server_errors_and_sends_a_user_agent(tmp_path, monkeypatch):
    error = urllib.error.HTTPError("https://drive.example/a", 503, "busy", Message(), io.BytesIO(b"<html>503</html>"))
    opener = FlakyUrlopen(b"payload", error, urllib.error.URLError("reset"))
    monkeypatch.setattr(fetch.urllib.request, "urlopen", opener)
    sleeps = []
    download("https://drive.example/a", tmp_path / "a.bin", sha256=sha(b"payload"), sleep=sleeps.append)
    assert (tmp_path / "a.bin").read_bytes() == b"payload"
    assert sleeps == [1.0, 2.0]
    assert opener.requests[0].get_header("User-agent") == USER_AGENT


def test_download_does_not_retry_client_errors(tmp_path, monkeypatch):
    error = urllib.error.HTTPError("https://x/a", 404, "missing", Message(), io.BytesIO(b""))
    monkeypatch.setattr(fetch.urllib.request, "urlopen", FlakyUrlopen(b"", error))
    with pytest.raises(urllib.error.HTTPError):
        download("https://x/a", tmp_path / "a.bin", sha256=sha(b""), sleep=lambda _: None)


def test_download_gives_up_on_network_errors_with_an_oserror(tmp_path, monkeypatch):
    errors = [urllib.error.URLError("reset") for _ in range(2)]
    monkeypatch.setattr(fetch.urllib.request, "urlopen", FlakyUrlopen(b"", *errors))
    with pytest.raises(OSError, match="reset"):
        download("https://x/a", tmp_path / "a.bin", sha256=sha(b""), attempts=2, sleep=lambda _: None)


def test_extract_archive(tmp_path):
    archive = tmp_path / "a.tar.gz"
    make_tar(archive, {"USPTO_mol_ref/c1.MOL": b"m"})
    extract_archive(archive, tmp_path / "out")
    assert (tmp_path / "out" / "USPTO_mol_ref" / "c1.MOL").read_bytes() == b"m"


def test_extract_refuses_path_traversal(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    make_tar(archive, {"../evil.txt": b"x"})
    with pytest.raises(tarfile.FilterError):
        extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_fetch_dataset_downloads_verifies_and_extracts(tmp_path):
    source = tmp_path / "USPTO_mol_ref.tar.gz"
    data = make_tar(source, {"USPTO_mol_ref/c1.MOL": b"m"})
    target = fetch_dataset(local_manifest(source, data, extract=True), tmp_path / "raw", log=lambda _: None)
    assert target == tmp_path / "raw" / "uspto"
    assert (target / "USPTO_mol_ref.tar.gz").read_bytes() == data
    assert (target / "USPTO_mol_ref" / "c1.MOL").read_bytes() == b"m"


def test_fetch_dataset_unpacks_an_archive_next_to_itself(tmp_path):
    source = tmp_path / "labels.tar.gz"
    data = make_tar(source, {"labels/1_6v5l_structure.csv": b"smiles"})
    entry = ManifestFile(source.as_uri(), "bioactivity_extraction/labels.tar.gz", sha(data), len(data), True)
    manifest = DatasetManifest("biovista", "test", "local", "r1", "c", "l", 1, (entry,))
    target = fetch_dataset(manifest, tmp_path / "raw", log=lambda _: None)
    assert (target / "bioactivity_extraction" / "labels" / "1_6v5l_structure.csv").read_bytes() == b"smiles"


def test_fetch_dataset_skips_files_already_verified(tmp_path, monkeypatch):
    source = tmp_path / "shard.parquet"
    source.write_bytes(b"rows")
    manifest = local_manifest(source, b"rows", extract=False)
    fetch_dataset(manifest, tmp_path / "raw", log=lambda _: None)

    def no_download(*args, **kwargs):
        raise AssertionError("verified file was downloaded again")

    monkeypatch.setattr(fetch, "download", no_download)
    fetch_dataset(manifest, tmp_path / "raw", log=lambda _: None)


def test_fetch_dataset_replaces_a_corrupt_file(tmp_path):
    source = tmp_path / "shard.parquet"
    source.write_bytes(b"rows")
    corrupt = tmp_path / "raw" / "uspto" / "shard.parquet"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"half")
    fetch_dataset(local_manifest(source, b"rows", extract=False), tmp_path / "raw", log=lambda _: None)
    assert corrupt.read_bytes() == b"rows"


GOOD_FILE = {
    "url": "https://example.org/a.tar.gz",
    "path": "a.tar.gz",
    "sha256": "0" * 64,
    "size_bytes": 10,
    "extract": True,
}
GOOD = {
    "dataset": "uspto",
    "description": "d",
    "source": "s",
    "revision": "r",
    "citation": "c",
    "license": "l",
    "items": 1,
    "files": [GOOD_FILE],
}


def write_manifest(tmp_path, data, name="uspto.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_load_manifest(tmp_path):
    manifest = load_manifest(write_manifest(tmp_path, GOOD))
    assert manifest.dataset == "uspto"
    assert manifest.files == (ManifestFile("https://example.org/a.tar.gz", "a.tar.gz", "0" * 64, 10, True),)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"files": [{**GOOD_FILE, "url": "http://example.org/a"}]}, "https"),
        ({"files": [{**GOOD_FILE, "sha256": "abc"}]}, "sha256"),
        ({"files": [{**GOOD_FILE, "path": "/etc/passwd"}]}, "relative"),
        ({"files": [{**GOOD_FILE, "path": "../a.tar.gz"}]}, "relative"),
        ({"files": [{**GOOD_FILE, "size_bytes": 0}]}, "size_bytes"),
        ({"files": [{**GOOD_FILE, "extract": "yes"}]}, "extract"),
        ({"files": []}, "no files"),
        ({"items": 0}, "items"),
        ({"dataset": "nope"}, "unknown dataset"),
        ({"extra": 1}, "unknown key"),
    ],
)
def test_load_manifest_rejects_bad_entries(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        load_manifest(write_manifest(tmp_path, {**GOOD, **change}))


def test_load_manifest_needs_every_key(tmp_path):
    data = {k: v for k, v in GOOD.items() if k != "revision"}
    with pytest.raises(ValueError, match="missing key"):
        load_manifest(write_manifest(tmp_path, data))


def test_manifest_name_must_match_dataset(tmp_path):
    with pytest.raises(ValueError, match="file name"):
        load_manifest(write_manifest(tmp_path, GOOD, name="uob.json"))


DRIVE = "https://drive.usercontent.google.com/download?id="


def test_committed_manifests():
    manifests = load_manifests(REPO / "data" / "manifests")
    assert {name: m.items for name, m in manifests.items()} == {
        "uspto": 5719,
        "uob": 5740,
        "jpo": 450,
        "clef": 992,
        "molrecbench_wild": 5024,
        "biovista": 500,
    }
    for manifest in manifests.values():
        # Google Drive links have no revision; the file ID and sha256 pin them instead.
        assert all(manifest.revision in f.url for f in manifest.files if not f.url.startswith(DRIVE))


def test_biovista_manifest_pins_its_label_archives_and_paper_list():
    manifest = load_manifest(REPO / "data" / "manifests" / "biovista.json")
    paths = {f.path: f for f in manifest.files}
    assert {path for path, f in paths.items() if f.url.startswith(DRIVE)} == {
        "bioactivity_extraction/bioactivity_extraction_normalization.tar.gz",
        "molecule_detection.tar.gz",
        "ocsr.tar.gz",
    }
    assert all(f.extract for f in paths.values() if f.url.startswith(DRIVE))
    assert {"bioactivity_extraction/data/file_names.txt", "test_set_names.txt", "valid_set_names.txt"} <= set(paths)


def load_script():
    spec = importlib.util.spec_from_file_location("fetch_data", REPO / "scripts" / "fetch_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_rejects_unknown_dataset():
    with pytest.raises(SystemExit) as info:
        load_script().main(["nope"])
    assert info.value.code == 2


def test_script_fetches_named_datasets(tmp_path, monkeypatch):
    script = load_script()
    fetched = []
    monkeypatch.setattr(script, "fetch_dataset", lambda manifest, root: fetched.append((manifest.dataset, root)))
    assert script.main(["jpo", "clef", "--root", str(tmp_path)]) == 0
    assert fetched == [("jpo", tmp_path), ("clef", tmp_path)]


def test_script_reports_fetch_errors(monkeypatch, capsys):
    script = load_script()

    def fail(manifest, root):
        raise ChecksumError("sha256 mismatch for JPO.tar.gz")

    monkeypatch.setattr(script, "fetch_dataset", fail)
    assert script.main(["jpo"]) == 1
    assert "sha256 mismatch for JPO.tar.gz" in capsys.readouterr().err


def test_script_fetches_biovista_papers_after_its_labels(tmp_path, monkeypatch, capsys):
    script = load_script()
    calls = []
    monkeypatch.setattr(script, "fetch_dataset", lambda manifest, root: root / manifest.dataset)
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)

    def fake_fetch_papers(target, manifest_csv, **kwargs):
        calls.append((target, manifest_csv, kwargs))
        return Counter({"ok": 3, "no_oa": 2})

    monkeypatch.setattr(script, "fetch_papers", fake_fetch_papers)
    assert script.main(["biovista", "--root", str(tmp_path), "--limit", "5", "--workers", "2"]) == 0
    target, manifest_csv, kwargs = calls[0]
    assert target == tmp_path / "biovista"
    assert manifest_csv == REPO / "data" / "manifests" / "biovista_papers.csv"
    assert kwargs == {
        "recovered_path": REPO / "data" / "manifests" / "biovista_recovered.csv",
        "limit": 5,
        "workers": 2,
        "unpaywall_email": None,
    }
    assert "biovista: 5 papers: ok 3, no_oa 2" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["biovista", "--limit", "0"], ["biovista", "--workers", "5"]])
def test_script_rejects_bad_biovista_options(argv):
    with pytest.raises(SystemExit) as info:
        load_script().main(argv)
    assert info.value.code == 2
