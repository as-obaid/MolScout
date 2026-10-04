"""The benchmark harness runs a fake tool on the hand-made crops, then checks, scores and records the run.

Images are empty files named after the hand-made crop references (c1 to c7). With the
hand-made answers below, scoring gives the same counts as tests/test_cli.py: accuracy 3/6,
stereo-stripped 4/6 and valid output 4/6, with c5 excluded and c6 answered with an empty SMILES.
"""

import hashlib
import json
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from molscout.bench import BenchError
from molscout.bench.harness import run_benchmark
from molscout.cli import main
from molscout.runs import score_run

FIXTURES = Path(__file__).parent / "fixtures"
REFERENCES = FIXTURES / "handmade" / "crops_references"
FAKE_TOOL = FIXTURES / "fake_tool"
ANSWERS = {"c1": "OCC", "c2": "C[C@@H](N)C(=O)O", "c3": "C1=CC=CC=C1", "c4": "*c1ccccc1", "c6": "", "c7": "C1CC"}
FILES = ["config.yaml", "meta.json", "predictions.csv", "scores.json"]
GIT = ("git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false")


class Workspace:
    """A repository root with the fake tool, crop images and references, plus a checkpoint store."""

    def __init__(self, root: Path, store: Path) -> None:
        self.root = root
        self.store = store
        self.results = root / "results"
        self.images = root / "data" / "images"
        self.references = root / "data" / "refs"
        self.run_dir = root / "tools" / "fake"
        self.checkpoint = store / "model.pt"
        self.answers = root / "answers.json"

    def config(self, *args: str, sha256: str | None = None, env: dict | None = None) -> Path:
        data = {
            "tool": "fake",
            "name": "Fake",
            "version": "1.0 (test)",
            "dataset": "uspto",
            "images": "data/images",
            "references": "data/refs",
            "run_dir": "tools/fake",
            "python": sys.executable,
            "args": list(args),
            "checkpoints": [{"path": "${STORE}/model.pt", "sha256": sha256 or sha(self.checkpoint)}],
            "env": env or {},
        }
        path = self.root / "configs" / "fake__uspto.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text("# test config\n" + yaml.safe_dump(data, sort_keys=False))
        return path

    def run(self, *args: str, **kwargs) -> Path:
        return run_benchmark(self.config(*args, **kwargs), repo_root=self.root, results_root=self.results)

    def leftovers(self) -> list[str]:
        if not self.results.exists():
            return []
        return sorted(p.name for p in self.results.iterdir() if p.name.startswith("."))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def ws(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "repo", tmp_path / "store")
    shutil.copytree(REFERENCES, workspace.references)
    workspace.images.mkdir(parents=True)
    for reference in REFERENCES.iterdir():
        (workspace.images / f"{reference.stem}.png").touch()
    shutil.copytree(FAKE_TOOL, workspace.run_dir, ignore=shutil.ignore_patterns("__pycache__"))
    workspace.store.mkdir()
    workspace.checkpoint.write_bytes(b"weights")
    workspace.answers.write_text(json.dumps(ANSWERS))
    monkeypatch.setenv("STORE", str(workspace.store))
    return workspace


def test_writes_the_four_files_and_scores_like_molscout_score(ws, monkeypatch):
    folder = ws.run("--answers", str(ws.answers))
    assert folder == ws.results / "fake__uspto"
    assert sorted(p.name for p in folder.iterdir()) == FILES
    assert ws.leftovers() == []
    assert (folder / "config.yaml").read_text() == (ws.root / "configs" / "fake__uspto.yaml").read_text()
    report = json.loads((folder / "scores.json").read_text())
    scores = report["scores"]
    assert (scores["accuracy"]["successes"], scores["accuracy"]["trials"]) == (3, 6)
    assert scores["accuracy_stereo_stripped"]["successes"] == 4
    assert scores["valid_output_rate"]["successes"] == 4
    assert scores["items_excluded"] == ["c5"]
    assert scores["items_without_prediction"] == 0
    # scores.json is what `molscout score` writes when run from the repository root.
    monkeypatch.chdir(ws.root)
    assert report == score_run(Path("results/fake__uspto/predictions.csv"), references=Path("data/refs"))


def test_meta_records_commit_environment_hardware_and_timing(ws, monkeypatch):
    ws.config("--answers", str(ws.answers))
    subprocess.run([*GIT, "init", "-q", str(ws.root)], check=True)
    subprocess.run([*GIT, "-C", str(ws.root), "add", "tools", "configs"], check=True)
    subprocess.run([*GIT, "-C", str(ws.root), "commit", "-q", "-m", "test"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(ws.root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "gpu")

    meta = json.loads((ws.run("--answers", str(ws.answers)) / "meta.json").read_text())

    assert meta["run"] == "fake__uspto"
    assert meta["dataset"] == "uspto"
    assert meta["items"] == 7
    assert meta["tool"] == {
        "tool": "fake",
        "name": "Fake",
        "version": "1.0 (test)",
        "checkpoints": [{"path": str(ws.checkpoint), "sha256": sha(ws.checkpoint)}],
    }
    assert meta["git"] == {"commit": commit, "dirty": False, "dirty_paths": []}
    environment = meta["environment"]
    assert environment["python"] == sys.executable
    assert environment["python_version"] == platform.python_version()
    assert any(line.lower().startswith("rdkit==") for line in environment["packages"])
    assert environment["conda_meta"] is None
    assert len(environment["sha256"]) == 64
    hardware = meta["hardware"]
    assert hardware["host"] == socket.gethostname()
    assert hardware["cpus_available"] >= 1
    assert isinstance(hardware["gpus"], list)
    assert meta["slurm"]["job"] == "123"
    assert meta["slurm"]["partition"] == "gpu"
    timing = meta["timing"]
    assert timing["started_utc"] <= timing["finished_utc"]
    assert timing["started_utc"].endswith("Z")
    assert timing["wall_seconds"] >= timing["tool_seconds"] + timing["scoring_seconds"]
    assert timing["item_seconds_total"] == pytest.approx(7 * 0.25)
    assert meta["command"][:9] == [
        sys.executable,
        "run.py",
        "--images",
        str(ws.images),
        "--dataset",
        "uspto",
        "--tool",
        "Fake 1.0 (test)",
        "--output",
    ]


def test_meta_lists_uncommitted_changes_to_the_tool(ws):
    subprocess.run([*GIT, "init", "-q", str(ws.root)], check=True)
    subprocess.run([*GIT, "-C", str(ws.root), "add", "tools"], check=True)
    subprocess.run([*GIT, "-C", str(ws.root), "commit", "-q", "-m", "test"], check=True)
    with (ws.run_dir / "run.py").open("a") as handle:
        handle.write("# changed\n")
    meta = json.loads((ws.run() / "meta.json").read_text())
    assert meta["git"]["dirty"] is True
    assert sorted(meta["git"]["dirty_paths"]) == ["configs/fake__uspto.yaml", "tools/fake/run.py"]


def test_meta_git_is_none_outside_a_repository(ws):
    meta = json.loads((ws.run() / "meta.json").read_text())
    assert meta["git"] == {"commit": None, "dirty": None, "dirty_paths": None}


def test_tool_runs_in_its_folder_with_a_clean_environment(ws, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/nowhere")
    monkeypatch.setenv("VIRTUAL_ENV", "/nowhere")
    record = ws.root / "record.json"
    ws.run("--record", str(record), env={"FAKE_SETTING": "on"})
    recorded = json.loads(record.read_text())
    assert Path(recorded["cwd"]).resolve() == ws.run_dir.resolve()
    assert recorded["env"] == {
        "PYTHONPATH": None,
        "PYTHONHOME": None,
        "VIRTUAL_ENV": None,
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "FAKE_SETTING": "on",
    }


def test_tool_failure_is_reported_and_nothing_is_written(ws):
    with pytest.raises(BenchError, match="exited with status 3"):
        ws.run("--fail")
    assert not (ws.results / "fake__uspto").exists()
    assert ws.leftovers() == []


def test_invalid_predictions_are_reported_with_line_numbers(ws):
    with pytest.raises(BenchError, match=r"(?s)line 2: seconds must be >= 0"):
        ws.run("--bad-row")
    assert not (ws.results / "fake__uspto").exists()
    assert ws.leftovers() == []


def test_missing_images_fail_the_run(ws):
    with pytest.raises(BenchError, match=r"misses 2 of 7 images.*c3, c5"):
        ws.run("--skip", "c3,c5")
    assert not (ws.results / "fake__uspto").exists()
    assert ws.leftovers() == []


def test_rows_must_name_the_configs_tool(ws):
    with pytest.raises(BenchError, match="Other 2.0"):
        ws.run("--tool-column", "Other 2.0")


def test_checkpoint_hash_mismatch_stops_before_tool_runs(ws):
    record = ws.root / "record.json"
    with pytest.raises(BenchError, match=rf"{re.escape(str(ws.checkpoint))}.*sha256 {sha(ws.checkpoint)}.*{'0' * 64}"):
        ws.run("--record", str(record), sha256="0" * 64)
    assert not record.exists()
    assert not ws.results.exists()


def test_missing_checkpoint_is_named(ws):
    ws.checkpoint.unlink()
    with pytest.raises(BenchError, match="checkpoint not found"):
        ws.run(sha256="0" * 64)


def test_empty_images_folder_stops_the_run(ws):
    for image in ws.images.iterdir():
        image.unlink()
    (ws.images / ".hidden.png").touch()
    with pytest.raises(BenchError, match="no images"):
        ws.run()


def test_unreadable_environment_lock_stops_the_run(ws, tmp_path):
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\nexit 1\n")
    python.chmod(0o755)
    config = ws.config()
    config.write_text(config.read_text().replace(sys.executable, str(python)))
    with pytest.raises(BenchError, match="environment lock"):
        run_benchmark(config, repo_root=ws.root, results_root=ws.results)
    assert not ws.results.exists()


def test_failed_run_keeps_previous_results(ws):
    folder = ws.run("--answers", str(ws.answers))
    before = {name: (folder / name).read_bytes() for name in FILES}
    with pytest.raises(BenchError):
        ws.run("--skip", "c1")
    assert {name: (folder / name).read_bytes() for name in FILES} == before
    assert ws.leftovers() == []


def test_rerun_replaces_results(ws):
    folder = ws.run()
    assert json.loads((folder / "scores.json").read_text())["scores"]["accuracy"]["successes"] == 0
    assert ws.run("--answers", str(ws.answers)) == folder
    assert json.loads((folder / "scores.json").read_text())["scores"]["accuracy"]["successes"] == 3
    assert sorted(p.name for p in folder.iterdir()) == FILES
    assert ws.leftovers() == []


def test_cli_bench_writes_the_results_folder(ws, capsys):
    config = ws.config("--answers", str(ws.answers))
    args = ["bench", str(config), "--repo-root", str(ws.root), "--results-root", str(ws.results)]
    assert main(args) == 0
    assert capsys.readouterr().out.splitlines()[-1] == f"wrote {ws.results / 'fake__uspto'}"


def test_cli_bench_defaults_to_the_current_directory(ws, monkeypatch, capsys):
    config = ws.config()
    monkeypatch.chdir(ws.root)
    assert main(["bench", str(config.relative_to(ws.root))]) == 0
    assert (ws.root / "benchmarks" / "results" / "fake__uspto" / "scores.json").exists()


def test_cli_bench_reports_errors(ws, capsys):
    config = ws.config("--fail")
    assert main(["bench", str(config), "--repo-root", str(ws.root), "--results-root", str(ws.results)]) == 1
    assert "molscout bench: error:" in capsys.readouterr().err


def test_sha256_tree_of_a_file_is_its_sha256(tmp_path):
    from molscout.hashing import sha256_tree

    path = tmp_path / "model.pt"
    path.write_bytes(b"weights")
    assert sha256_tree(path) == hashlib.sha256(b"weights").hexdigest()


def test_sha256_tree_of_a_folder_hashes_sorted_relative_path_lines(tmp_path):
    from molscout.hashing import sha256_tree

    folder = tmp_path / "models"
    (folder / "sub").mkdir(parents=True)
    files = {"b.bin": b"b", ".hidden": b"h", "sub/a.bin": b"a", "a.bin": b"z"}
    for name, data in files.items():
        (folder / name).write_bytes(data)
    lines = "".join(f"{hashlib.sha256(files[name]).hexdigest()}  {name}\n" for name in sorted(files))
    assert sha256_tree(folder) == hashlib.sha256(lines.encode()).hexdigest()


def test_sha256_tree_of_a_missing_path_is_an_error(tmp_path):
    from molscout.hashing import sha256_tree

    with pytest.raises(FileNotFoundError, match="missing"):
        sha256_tree(tmp_path / "missing")
