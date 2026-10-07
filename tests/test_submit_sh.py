"""benchmarks/slurm/submit.sh fills each H200 partition and the CPU queue with workers; fake sbatch and squeue."""

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmarks" / "slurm" / "submit.sh"
GPU_ORDER = [  # longest first: crops × seconds per crop on an H200
    "decimer__uob",
    "decimer__uspto",
    "decimer__molrecbench_wild",
    "molglyph__uob",
    "molglyph__uspto",
    "molscribe__uob",
    "molscribe__uspto",
    "molglyph__molrecbench_wild",
    "molnextr__uob",
    "molnextr__uspto",
    "molscribe__molrecbench_wild",
    "molnextr__molrecbench_wild",
    "ocsrglyph__uob",
    "ocsrglyph__uspto",
    "ocsrglyph__molrecbench_wild",
    "decimer__clef",
    "molglyph__clef",
    "molscribe__clef",
    "molnextr__clef",
    "decimer__jpo",
    "molglyph__jpo",
    "molscribe__jpo",
    "molnextr__jpo",
    "ocsrglyph__clef",
    "ocsrglyph__jpo",
]
MOLVEC_ORDER = ["molvec__uob", "molvec__uspto", "molvec__molrecbench_wild", "molvec__clef", "molvec__jpo"]
WORKERS = [  # partition, workers, time limit, GRES, configs
    ("gpu", 4, "08:00:00", "gpu:h200:1", GPU_ORDER),
    ("gpu-short", 2, "02:00:00", "gpu:h200:1", GPU_ORDER),
    ("gpu-interactive", 2, "02:00:00", "gpu:h200:1", GPU_ORDER),
    ("sharing", 2, "01:00:00", "gpu:h200:1", GPU_ORDER),
    ("short", 5, "24:00:00", "none", MOLVEC_ORDER),
]
FAKE_SBATCH = """#!/usr/bin/env python3
import json, os, pathlib, sys
log = pathlib.Path(os.environ["FAKE"]) / "sbatch.log"
calls = log.read_text().splitlines() if log.exists() else []
log.write_text("".join(line + "\\n" for line in [*calls, json.dumps(sys.argv[1:])]))
if "--partition=sharing" in sys.argv:
    sys.exit("sbatch: error: QOSMaxSubmitJobPerUserLimit")
print(1000 + len(calls))
"""
FAKE_SQUEUE = """#!/bin/bash
# squeue -h -u USER -n NAME -o %i: the job IDs in $FAKE/queued-NAME
while [ $# -gt 0 ]; do [ "$1" = -n ] && name=$2; shift; done
cat "$FAKE/queued-$name" 2>/dev/null || true
"""


@pytest.fixture
def fake(tmp_path):
    for name, text in {"sbatch": FAKE_SBATCH, "squeue": FAKE_SQUEUE}.items():
        (tmp_path / name).write_text(text)
        (tmp_path / name).chmod(0o755)
    return tmp_path


def submit(fake: Path, *args: str, cwd: Path = REPO) -> subprocess.CompletedProcess:
    env = {**os.environ, "PATH": f"{fake}{os.pathsep}{os.environ['PATH']}", "FAKE": str(fake)}
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=cwd, env=env, capture_output=True, text=True)


def expected_call(partition: str, limit: str, gres: str, configs: list[str], name: str = "", extra: str = "") -> list[str]:
    options = f"--partition={partition} --gres={gres} --time={limit} --signal=B:USR1@180 --job-name={name or 'molscout-' + partition}"
    options += extra
    options += " --output=benchmarks/slurm/logs/%x_%j.out --error=benchmarks/slurm/logs/%x_%j.err"
    paths = [f"benchmarks/configs/{run}.yaml" for run in configs]
    return ["--parsable", f"--export=ALL,MOLSCOUT_RESUBMIT={options}", *options.split(), "benchmarks/slurm/run.sbatch", *paths]


def commit_all(root: Path) -> None:
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "init"], cwd=root, check=True)


def commit_all_again(root: Path) -> None:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "more"], cwd=root, check=True)


def calls(fake: Path) -> list[list[str]]:
    return [json.loads(line) for line in (fake / "sbatch.log").read_text().splitlines()]


def test_submit_fills_each_partition_to_its_limit_with_the_longest_configs_first(fake, tmp_path):
    result = submit(fake, "structure-readers", cwd=type_repo(tmp_path))
    assert result.returncode == 0, result.stderr
    expected = [expected_call(p, limit, gres, configs) for p, workers, limit, gres, configs in WORKERS for _ in range(workers)]
    assert calls(fake) == expected
    type1 = [p.stem for p in (REPO / "benchmarks" / "configs").glob("*.yaml") if "complete_systems" not in p.read_text()]
    assert sorted(GPU_ORDER + MOLVEC_ORDER) == sorted(type1)
    assert "gpu: submitted 1000 1001 1002 1003" in result.stdout
    assert "short: submitted 1010 1011 1012 1013 1014" in result.stdout
    assert "sharing: sbatch: error: QOSMaxSubmitJobPerUserLimit" in result.stderr  # a refusal does not stop the rest


def test_submit_only_tops_up_partitions_that_already_have_workers(fake, tmp_path):
    (fake / "queued-molscout-gpu").write_text("11\n12\n13\n")
    (fake / "queued-molscout-short").write_text("".join(f"{job}\n" for job in range(20, 25)))
    result = submit(fake, "structure-readers", cwd=type_repo(tmp_path))
    assert result.returncode == 0, result.stderr
    partitions = [call[2] for call in calls(fake)]
    assert partitions.count("--partition=gpu") == 1
    assert partitions.count("--partition=short") == 0
    assert "gpu: 3 already queued; submitted 1000" in result.stdout
    assert "short: 5 already queued; submitted none" in result.stdout


def test_submit_must_run_from_the_repository_root(fake, tmp_path):
    result = submit(fake, "structure-readers", cwd=tmp_path)
    assert result.returncode == 2
    assert "repository root" in result.stderr
    assert not (fake / "sbatch.log").exists()


def test_no_argument_is_a_usage_error(fake):
    result = submit(fake)
    assert result.returncode == 2
    assert "structure-readers" in result.stderr and "complete-systems" in result.stderr
    assert not (fake / "sbatch.log").exists()


def test_an_unknown_argument_is_a_usage_error(fake):
    assert submit(fake, "everything").returncode == 2
    assert not (fake / "sbatch.log").exists()


TYPE2_ORDER = {  # internal first, then BioVista; longest first within each (the placeholder rates in submit.sh)
    "biominer": ["biominer__internal", "biominer__biovista"],
    "others": ["decimer_ai__internal", "openchemie__internal", "decimer_ai__biovista", "openchemie__biovista"],
}
BIOMINER = " --cpus-per-task=16 --mem=192G --dependency=singleton"
TYPE2_WORKERS = [  # partition, workers, time limit, GRES, job name, extra options, configs
    ("gpu", 1, "08:00:00", "gpu:h200:1", "molscout2-biominer", BIOMINER, TYPE2_ORDER["biominer"]),
    ("gpu", 3, "08:00:00", "gpu:h200:1", "molscout2-gpu-1gpu", "", TYPE2_ORDER["others"]),
    ("gpu-short", 2, "02:00:00", "gpu:h200:1", "molscout2-gpu-short-1gpu", "", TYPE2_ORDER["others"]),
    ("gpu-interactive", 2, "02:00:00", "gpu:h200:1", "molscout2-gpu-interactive-1gpu", "", TYPE2_ORDER["others"]),
]


def type_repo(tmp_path: Path) -> Path:
    """A repository root with fake configs of both types and the real submit.sh and run.sbatch."""
    root = tmp_path / "repo"
    (root / "benchmarks" / "configs").mkdir(parents=True)
    (root / "benchmarks" / "slurm").mkdir()
    (root / "pyproject.toml").touch()
    (root / "benchmarks" / "slurm" / "run.sbatch").touch()
    for stem in [*GPU_ORDER, *MOLVEC_ORDER]:
        (root / "benchmarks/configs" / f"{stem}.yaml").write_text(f"tool: {stem}\nrun_dir: benchmarks/tools/structure_readers/x\n")
    for stem in sum(TYPE2_ORDER.values(), []):
        (root / "benchmarks/configs" / f"{stem}.yaml").write_text(f"tool: {stem}\nrun_dir: benchmarks/tools/complete_systems/x\n")
    commit_all(root)
    return root


def test_complete_systems_submits_only_type2_configs(fake, tmp_path):
    root = type_repo(tmp_path)
    result = submit(fake, "complete-systems", cwd=root)
    assert result.returncode == 0, result.stderr
    expected = [
        expected_call(partition, limit, gres, configs, name, extra)
        for partition, workers, limit, gres, name, extra, configs in TYPE2_WORKERS
        for _ in range(workers)
    ]
    assert calls(fake) == expected
    assert all("--gres=gpu:h200:1" in call for call in calls(fake))
    assert {call[2] for call in calls(fake)} == {"--partition=gpu", "--partition=gpu-short", "--partition=gpu-interactive"}
    assert not any("sharing" in " ".join(call) for call in calls(fake))
    biominer = [call for call in calls(fake) if "--job-name=molscout2-biominer" in call]
    assert len(biominer) == 1 and "--partition=gpu" in biominer[0] and "--dependency=singleton" in biominer[0]
    resubmit = next(a for a in biominer[0] if a.startswith("--export=ALL,MOLSCOUT_RESUBMIT="))
    assert "--job-name=molscout2-biominer" in resubmit and "--dependency=singleton" in resubmit  # USR1 keeps both
    assert "gpu: submitted 1000" in result.stdout


def test_complete_systems_tops_up_by_job_name(fake, tmp_path):
    root = type_repo(tmp_path)
    (fake / "queued-molscout2-gpu-1gpu").write_text("11\n12\n13\n")
    (fake / "queued-molscout-gpu").write_text("21\n22\n23\n24\n")  # Type 1 workers do not count
    result = submit(fake, "complete-systems", cwd=root)
    assert result.returncode == 0, result.stderr
    names = [next(a for a in call if a.startswith("--job-name=")) for call in calls(fake)]
    assert names.count("--job-name=molscout2-gpu-1gpu") == 0
    assert names.count("--job-name=molscout2-gpu-short-1gpu") == 2
    assert names.count("--job-name=molscout2-biominer") == 1
    assert "gpu: 3 already queued; submitted none" in result.stdout


def test_structure_readers_never_submits_type2_configs(fake, tmp_path):
    root = type_repo(tmp_path)
    result = submit(fake, "structure-readers", cwd=root)
    assert result.returncode == 0, result.stderr
    paths = {arg for call in calls(fake) for arg in call if arg.startswith("benchmarks/configs/")}
    assert paths == {f"benchmarks/configs/{stem}.yaml" for stem in GPU_ORDER + MOLVEC_ORDER}


def test_a_group_with_no_configs_submits_no_workers(fake, tmp_path):
    root = type_repo(tmp_path)
    for config in (root / "benchmarks" / "configs").glob("biominer__*.yaml"):
        config.unlink()
    commit_all_again(root)
    result = submit(fake, "complete-systems", cwd=root)
    assert result.returncode == 0, result.stderr
    assert "no biominer configs: skipping its workers" in result.stdout
    assert calls(fake) == [
        expected_call(partition, limit, gres, configs, name, extra)
        for partition, workers, limit, gres, name, extra, configs in TYPE2_WORKERS[1:]
        for _ in range(workers)
    ]
    assert not any("molscout2-biominer" in " ".join(call) for call in calls(fake))


@pytest.mark.parametrize("kind", ["structure-readers", "complete-systems"])
@pytest.mark.parametrize("path", ["benchmarks/configs/new.yaml", "benchmarks/tools/complete_systems/x/run.py", "src/a.py", "uv.lock"])
def test_a_dirty_tree_refuses_to_submit(fake, tmp_path, kind, path):
    root = type_repo(tmp_path)
    (root / "uv.lock").write_text("x")
    commit_all_again(root)
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(target.read_text() + "changed\n" if target.exists() else "new\n")
    result = submit(fake, kind, cwd=root)
    assert result.returncode == 1
    assert path in result.stderr
    assert not (fake / "sbatch.log").exists()


def test_changes_outside_the_keyed_paths_do_not_block(fake, tmp_path):
    root = type_repo(tmp_path)
    (root / "notes.md").write_text("x")
    assert submit(fake, "complete-systems", cwd=root).returncode == 0
