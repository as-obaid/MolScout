"""The benchmark harness runs a fake tool on the hand-made crops, then checks, scores and records the run.

Images are empty files named after the hand-made crop references (c1 to c7). With the
hand-made answers below, scoring gives the same counts as tests/test_cli.py: accuracy 3/6,
stereo-stripped 4/6 and valid output 4/6, with c5 excluded and c6 answered with an empty SMILES.
"""

import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from molscout.bench import BenchError, Terminated
from molscout.bench.harness import run_benchmark
from molscout.cli import main
from molscout.hashing import sha256_tree
from molscout.runs import score_run

FIXTURES = Path(__file__).parent / "fixtures"
REFERENCES = FIXTURES / "handmade" / "crops_references"
FAKE_TOOL = FIXTURES / "fake_tool"
ANSWERS = {"c1": "OCC", "c2": "C[C@@H](N)C(=O)O", "c3": "C1=CC=CC=C1", "c4": "*c1ccccc1", "c6": "", "c7": "C1CC"}
FILES = ["config.yaml", "errors.json", "meta.json", "predictions.csv", "scores.json"]
RECORDED_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "PYSTOW_HOME",
    "MOLSCOUT_STORE",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_PARTITION",
)
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
    (workspace.root / "pyproject.toml").write_text('[project]\nname = "molscout"\n')
    workspace.images.mkdir(parents=True)
    for reference in REFERENCES.iterdir():
        (workspace.images / f"{reference.stem}.png").touch()
    shutil.copytree(FAKE_TOOL, workspace.run_dir, ignore=shutil.ignore_patterns("__pycache__"))
    workspace.store.mkdir()
    workspace.checkpoint.write_bytes(b"weights")
    workspace.answers.write_text(json.dumps(ANSWERS))
    monkeypatch.setenv("STORE", str(workspace.store))
    return workspace


def test_writes_the_five_files_and_scores_like_molscout_score(ws, monkeypatch):
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
    assert json.loads((folder / "errors.json").read_text()) == {"images": 7, "failed": 0, "errors": {}}
    # scores.json is what `molscout score` writes when run from the repository root.
    monkeypatch.chdir(ws.root)
    assert report == score_run(Path("results/fake__uspto/predictions.csv"), references=Path("data/refs"))


def test_meta_records_commit_environment_hardware_and_timing(ws, monkeypatch):
    ws.config("--answers", str(ws.answers))
    subprocess.run([*GIT, "init", "-q", str(ws.root)], check=True)
    subprocess.run([*GIT, "-C", str(ws.root), "add", "tools", "configs", "pyproject.toml"], check=True)
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
    assert meta["inputs"] == {"images": str(ws.images), "images_sha256": sha256_tree(ws.images)}
    assert meta["tool_errors"] == 0
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
    subprocess.run([*GIT, "-C", str(ws.root), "add", "tools", "pyproject.toml"], check=True)
    subprocess.run([*GIT, "-C", str(ws.root), "commit", "-q", "-m", "test"], check=True)
    with (ws.run_dir / "run.py").open("a") as handle:
        handle.write("# changed\n")
    meta = json.loads((ws.run() / "meta.json").read_text())
    assert meta["git"]["dirty"] is True
    assert sorted(meta["git"]["dirty_paths"]) == ["configs/fake__uspto.yaml", "tools/fake/run.py"]


def test_meta_lists_uncommitted_changes_to_the_crop_loop_and_the_slurm_script(ws):
    shared = [ws.root / "benchmarks" / "tools" / "crop_runner.py", ws.root / "benchmarks" / "slurm" / "run.sbatch"]
    for path in shared:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# shared\n")
    ws.config()
    subprocess.run([*GIT, "init", "-q", str(ws.root)], check=True)
    subprocess.run([*GIT, "-C", str(ws.root), "add", "."], check=True)
    subprocess.run([*GIT, "-C", str(ws.root), "commit", "-q", "-m", "test"], check=True)
    for path in shared:
        path.write_text("# timing changed\n")
    meta = json.loads((ws.run() / "meta.json").read_text())
    assert meta["git"]["dirty_paths"] == ["benchmarks/slurm/run.sbatch", "benchmarks/tools/crop_runner.py"]


def test_meta_records_allow_listed_variables_and_never_secrets(ws, monkeypatch):
    for name in RECORDED_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_TOKEN", "hf-token-from-the-shell")
    monkeypatch.setenv("UNLISTED_SETTING", "unlisted-value")
    env = {
        "FAKE_SETTING": "on",
        "HF_HOME": "/models/hf",
        "MY_API_KEY": "api-key-from-the-config",
        "DB_PASSWORD": "password-from-the-config",
        "client_secret": "secret-from-the-config",
        "GH_TOKEN": "token-from-the-config",
    }
    text = (ws.run(env=env) / "meta.json").read_text()
    assert json.loads(text)["environment"]["variables"] == {
        **dict.fromkeys(RECORDED_VARIABLES),
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "HF_HOME": "/models/hf",
        "FAKE_SETTING": "on",
    }
    for value in ("from-the-shell", "from-the-config", "unlisted-value"):
        assert value not in text


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


def test_references_without_an_image_fail_the_run(ws):
    (ws.images / "c3.png").unlink()
    (ws.images / "c1.png").unlink()
    with pytest.raises(
        BenchError, match=rf"2 scored reference\(s\) have no image in {re.escape(str(ws.images))}, e\.g\. c1, c3"
    ):
        ws.run()
    assert not (ws.results / "fake__uspto").exists()
    assert ws.leftovers() == []


def test_repo_root_must_hold_pyproject_toml(ws):
    (ws.root / "pyproject.toml").unlink()
    with pytest.raises(BenchError, match=rf"{re.escape(str(ws.root))} is not the MolScout repository root"):
        ws.run()
    assert not ws.results.exists()


def test_tool_errors_are_kept_as_errors_json_and_counted_in_meta(ws):
    folder = ws.run("--answers", str(ws.answers), "--crash", "c2,c4")
    assert sorted(p.name for p in folder.iterdir()) == FILES
    assert json.loads((folder / "errors.json").read_text()) == {
        "images": 7,
        "failed": 2,
        "errors": {"c2": "ValueError: fake crash", "c4": "ValueError: fake crash"},
    }
    assert json.loads((folder / "meta.json").read_text())["tool_errors"] == 2
    assert ws.leftovers() == []


def test_a_run_without_an_errors_file_still_works(ws):
    folder = ws.run("--no-errors-file")
    assert sorted(p.name for p in folder.iterdir()) == [name for name in FILES if name != "errors.json"]
    assert json.loads((folder / "meta.json").read_text())["tool_errors"] is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{", "predictions.errors.json is not valid JSON"),
        ("[]", "must be a JSON object with images, failed and errors"),
        ('{"images": 7, "failed": "1", "errors": {}}', "must be a JSON object with images, failed and errors"),
        ('{"images": 7, "failed": 1, "errors": {}}', "failed is 1, but errors lists 0 crop"),
        ('{"images": 6, "failed": 0, "errors": {}}', "images is 6, but the images folder holds 7"),
        ('{"images": 7, "failed": 2, "errors": {"c1": "E: x", "zz": "E: y"}}', r"1 crop\(s\) with no image.*zz"),
    ],
)
def test_a_bad_errors_file_fails_the_run(ws, text, message):
    with pytest.raises(BenchError, match=message):
        ws.run("--errors-text", text)
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


def test_sigterm_stops_the_tool_removes_staging_and_keeps_earlier_results(ws):
    record = ws.root / "record.json"
    log = ws.root / "harness.log"

    def interrupted() -> None:
        config = ws.config("--record", str(record), "--sleep", "120")
        record.unlink(missing_ok=True)
        command = [sys.executable, "-m", "molscout", "bench", str(config), "--repo-root", str(ws.root)]
        with log.open("w") as handle:
            harness = subprocess.Popen([*command, "--results-root", str(ws.results)], stdout=handle, stderr=handle)
        tool = None
        try:
            deadline = time.monotonic() + 120
            while not record.exists():  # the fake tool writes its record once it runs
                assert harness.poll() is None, log.read_text()
                assert time.monotonic() < deadline, "the fake tool never started"
                time.sleep(0.05)
            tool = json.loads(record.read_text())["pid"]
            harness.send_signal(signal.SIGTERM)
            harness.wait(timeout=60)
            tool_stopped = process_gone(tool)
        finally:
            if harness.poll() is None:
                harness.kill()
                harness.wait()
            if tool is not None and not process_gone(tool, wait=0):
                os.kill(tool, signal.SIGKILL)
        output = log.read_text()
        assert harness.returncode == 128 + signal.SIGTERM, output
        assert "molscout bench: error:" in output
        assert "SIGTERM" in output
        assert tool_stopped
        assert ws.leftovers() == []

    interrupted()
    assert not (ws.results / "fake__uspto").exists()
    folder = ws.run("--answers", str(ws.answers))
    before = {name: (folder / name).read_bytes() for name in FILES}
    interrupted()
    assert {name: (folder / name).read_bytes() for name in FILES} == before


def process_gone(pid: int, wait: float = 5.0) -> bool:
    deadline = time.monotonic() + wait
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def test_sigterm_handler_is_restored_after_a_run(ws):
    def handler(signum, frame):
        pass

    previous = signal.signal(signal.SIGTERM, handler)
    try:
        ws.run()
        assert signal.getsignal(signal.SIGTERM) is handler
        with pytest.raises(BenchError):
            ws.run("--fail")
        assert signal.getsignal(signal.SIGTERM) is handler
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_sigterm_during_the_final_swap_waits_until_the_new_results_are_in_place(ws, monkeypatch):
    folder = ws.run()
    rename = Path.rename

    def rename_then_sigterm(self, target):
        moved = rename(self, target)
        if self == folder:  # the earlier result has just been moved aside
            os.kill(os.getpid(), signal.SIGTERM)
        return moved

    def no_handler(signum, frame):
        raise AssertionError("the harness did not install its SIGTERM handler")

    previous = signal.signal(signal.SIGTERM, no_handler)
    monkeypatch.setattr(Path, "rename", rename_then_sigterm)
    try:
        with pytest.raises(Terminated):
            ws.run("--answers", str(ws.answers))
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert json.loads((folder / "scores.json").read_text())["scores"]["accuracy"]["successes"] == 3
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


def test_cli_bench_reports_runtime_errors_cleanly(ws, monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise RuntimeError("CUDA driver too old")

    monkeypatch.setattr("molscout.cli.run_benchmark", broken)
    assert main(["bench", str(ws.config()), "--repo-root", str(ws.root)]) == 1
    assert capsys.readouterr().err == "molscout bench: error: CUDA driver too old\n"


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
