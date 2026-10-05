"""benchmarks/slurm/status.sh shows where each run stands, with fake squeue and git."""

import json
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "slurm" / "status.sh"
HEAD = "a" * 40
FAKE_SQUEUE = """#!/bin/bash
# -j 111 is running on gpu-short; with -u it lists one molscout job and another job.
case "$*" in
    *"-j 111"*) echo gpu-short ;;
    *"-j "*) echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1 ;;
    *) printf '%s\\n' "JOBID NAME PARTITION" "111 molscout-gpu-short gpu-short" "222 other gpu" ;;
esac
"""
FAKE_GIT = """#!/bin/bash
[ "$*" = "rev-parse HEAD" ] && echo "%s"
""" % HEAD


def test_status_shows_each_run_and_the_molscout_jobs(tmp_path):
    root, fake = tmp_path / "repo", tmp_path / "bin"
    results, claims = root / "benchmarks" / "results", root / "benchmarks" / "results" / ".claims"
    images = root / "data" / "img"
    for folder in (root / "benchmarks" / "configs", claims, images, fake):
        folder.mkdir(parents=True)
    for name in ("a.png", "b.png", "c.PNG", "notes.txt", ".hidden.png"):
        (images / name).touch()
    for run in ("done__x", "old__x", "running__x", "partial__x", "failed__x", "pending__x"):
        (root / "benchmarks" / "configs" / f"{run}.yaml").write_text(f"tool: {run}\nimages: data/img\n")
    for run, commit, dirty in (("done__x", HEAD, False), ("old__x", "b" * 40, False)):
        (results / run).mkdir()
        (results / run / "meta.json").write_text(json.dumps({"git": {"commit": commit, "dirty": dirty}}))
    (claims / "running__x").mkdir()
    (claims / "running__x" / "job").write_text("111\n")
    for run, rows in (("running__x", 1), ("partial__x", 2), ("failed__x", 1)):
        (results / ".checkpoints" / run).mkdir(parents=True)
        (results / ".checkpoints" / run / "predictions.csv").write_text("header\n" + "row\n" * rows)
    (claims / "partial__x.failures").write_text(f"{HEAD}\n")
    (claims / "failed__x.failures").write_text(f"{'c' * 40}\n{HEAD}\n{HEAD}\n")
    for name, text in {"squeue": FAKE_SQUEUE, "git": FAKE_GIT}.items():
        (fake / name).write_text(text)
        (fake / name).chmod(0o755)
    env = {**os.environ, "PATH": f"{fake}{os.pathsep}{os.environ['PATH']}"}
    result = subprocess.run(["bash", str(SCRIPT)], cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    table = [line.split(None, 2) for line in result.stdout.splitlines()]
    assert table[:7] == [
        ["RUN", "STATE", "DETAIL"],
        ["done__x", "done", "aaaaaaa"],
        ["failed__x", "failed", "2 failures at HEAD, 1/3 rows"],
        ["old__x", "stale", "bbbbbbb, not HEAD"],
        ["partial__x", "partial", "2/3 rows, 1 failure at HEAD"],
        ["pending__x", "pending"],
        ["running__x", "running", "job 111 (gpu-short), 1/3 rows"],
    ]
    assert result.stdout.splitlines()[-2:] == ["JOBID NAME PARTITION", "111 molscout-gpu-short gpu-short"]
