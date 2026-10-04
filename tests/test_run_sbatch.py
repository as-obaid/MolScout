"""benchmarks/slurm/run.sbatch picks its config from the array index; MOLSCOUT=echo shows the command."""

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "slurm" / "run.sbatch"


def run(*configs: str, task: str | None = None, tmp_path: Path):
    env = {k: v for k, v in os.environ.items() if k != "SLURM_ARRAY_TASK_ID"}
    env.update(MOLSCOUT="echo", SLURM_SUBMIT_DIR=str(tmp_path), MOLSCOUT_STORE=str(tmp_path / "store"))
    if task is not None:
        env["SLURM_ARRAY_TASK_ID"] = task
    return subprocess.run(["bash", str(SCRIPT), *configs], env=env, capture_output=True, text=True)


def test_single_config(tmp_path):
    result = run("a.yaml", tmp_path=tmp_path)
    assert result.returncode == 0
    assert "bench a.yaml" in result.stdout


def test_array_task_picks_its_config(tmp_path):
    result = run("a.yaml", "b.yaml", "c.yaml", task="1", tmp_path=tmp_path)
    assert result.returncode == 0
    assert "bench b.yaml" in result.stdout
    assert "a.yaml" not in result.stdout.split("bench")[-1]


def test_two_configs_without_array_exit_2(tmp_path):
    result = run("a.yaml", "b.yaml", tmp_path=tmp_path)
    assert result.returncode == 2
    assert "array" in result.stderr


def test_no_config_exit_2(tmp_path):
    result = run(tmp_path=tmp_path)
    assert result.returncode == 2
    assert "usage" in result.stderr


def test_out_of_range_task_exit_2(tmp_path):
    result = run("a.yaml", "b.yaml", task="2", tmp_path=tmp_path)
    assert result.returncode == 2
    assert "out of range" in result.stderr


def test_sets_offline_environment(tmp_path):
    script = SCRIPT.read_text()
    for name in ("HF_HOME", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "MOLSCOUT_STORE"):
        assert name in script
