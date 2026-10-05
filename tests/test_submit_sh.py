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


def submit(fake: Path, cwd: Path = REPO) -> subprocess.CompletedProcess:
    env = {**os.environ, "PATH": f"{fake}{os.pathsep}{os.environ['PATH']}", "FAKE": str(fake)}
    return subprocess.run(["bash", str(SCRIPT)], cwd=cwd, env=env, capture_output=True, text=True)


def expected_call(partition: str, limit: str, gres: str, configs: list[str]) -> list[str]:
    options = f"--partition={partition} --gres={gres} --time={limit} --signal=B:USR1@180 --job-name=molscout-{partition}"
    options += " --output=benchmarks/slurm/logs/%x_%j.out --error=benchmarks/slurm/logs/%x_%j.err"
    paths = [f"benchmarks/configs/{run}.yaml" for run in configs]
    return ["--parsable", f"--export=ALL,MOLSCOUT_RESUBMIT={options}", *options.split(), "benchmarks/slurm/run.sbatch", *paths]


def calls(fake: Path) -> list[list[str]]:
    return [json.loads(line) for line in (fake / "sbatch.log").read_text().splitlines()]


def test_submit_fills_each_partition_to_its_limit_with_the_longest_configs_first(fake):
    result = submit(fake)
    assert result.returncode == 0, result.stderr
    expected = [expected_call(p, limit, gres, configs) for p, workers, limit, gres, configs in WORKERS for _ in range(workers)]
    assert calls(fake) == expected
    assert sorted(GPU_ORDER + MOLVEC_ORDER) == sorted(path.stem for path in (REPO / "benchmarks" / "configs").glob("*.yaml"))
    assert "gpu: submitted 1000 1001 1002 1003" in result.stdout
    assert "short: submitted 1010 1011 1012 1013 1014" in result.stdout
    assert "sharing: sbatch: error: QOSMaxSubmitJobPerUserLimit" in result.stderr  # a refusal does not stop the rest


def test_submit_only_tops_up_partitions_that_already_have_workers(fake):
    (fake / "queued-molscout-gpu").write_text("11\n12\n13\n")
    (fake / "queued-molscout-short").write_text("".join(f"{job}\n" for job in range(20, 25)))
    result = submit(fake)
    assert result.returncode == 0, result.stderr
    partitions = [call[2] for call in calls(fake)]
    assert partitions.count("--partition=gpu") == 1
    assert partitions.count("--partition=short") == 0
    assert "gpu: 3 already queued; submitted 1000" in result.stdout
    assert "short: 5 already queued; submitted none" in result.stdout


def test_submit_must_run_from_the_repository_root(fake, tmp_path):
    result = submit(fake, cwd=tmp_path)
    assert result.returncode == 2
    assert "repository root" in result.stderr
    assert not (fake / "sbatch.log").exists()
