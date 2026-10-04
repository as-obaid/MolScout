import hashlib
import importlib.util
import io
import json
import tarfile
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


def test_committed_manifests():
    manifests = load_manifests(REPO / "data" / "manifests")
    assert {name: m.items for name, m in manifests.items()} == {
        "uspto": 5719,
        "uob": 5740,
        "jpo": 450,
        "clef": 992,
        "molrecbench_wild": 5024,
    }
    for manifest in manifests.values():
        assert all(manifest.revision in f.url for f in manifest.files)


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
