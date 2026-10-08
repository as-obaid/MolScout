"""benchmarks/slurm/run.sbatch runs one config, picks one by array index, or works through several as a worker.

MOLSCOUT=echo shows the command; the worker tests put fake molscout, squeue, sbatch and git first on PATH.
One worker test uses real git and the real `molscout is-current` instead.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "slurm" / "run.sbatch"


def run(*configs: str, task: str | None = None, tmp_path: Path):
    env = {k: v for k, v in os.environ.items() if k not in ("SLURM_ARRAY_TASK_ID", "MOLSCOUT_RESUBMIT", "CUDA_VISIBLE_DEVICES")}
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


def test_files_are_private_by_default():
    lines = SCRIPT.read_text().splitlines()
    assert "umask 077" in [line.split("#")[0].strip() for line in lines[:40]]


HEAD = "a" * 40
CODE = "c" * 64
FAKE_BENCH = """#!/bin/bash
# molscout is-current CONFIG: with MOLSCOUT_PYTHON set, the real one. Otherwise <run>.verdict holds
# "current WHY", "blocked WHY" or "not-current WHY" (none: no results), the code fingerprint is $FAKE/code
# or $FAKE_CODE, and with is-current.broken it fails without output.
# molscout bench CONFIG: logs the call; <run>.exit holds its status, and with <run>.hang it runs until TERM.
# A run that exits 0 leaves results: meta.json at HEAD with real git, else the verdict in <run>.after
# (default: current). <run>.litter names a file it leaves behind; <run>.code becomes $FAKE/code, as if a
# commit to the run's code landed while it ran.
run=$(basename "$2" .yaml)
if [ "$1" = is-current ]; then
    if [ -n "${MOLSCOUT_PYTHON:-}" ]; then exec "$MOLSCOUT_PYTHON" -m molscout "$@"; fi
    if [ -f "$FAKE/is-current.broken" ]; then echo "Traceback: broken" >&2; exit 2; fi
    verdict=$(cat "$FAKE/$run.verdict" 2>/dev/null || echo "not-current no results")
    echo "$run ${verdict%% *} $(cat "$FAKE/code" 2>/dev/null || echo "$FAKE_CODE") ${verdict#* }"
    [ "${verdict%% *}" = current ]
    exit
fi
echo "bench $run" >> "$FAKE/calls"
if [ -f "$FAKE/$run.hang" ]; then
    trap 'echo "TERM $run" >> "$FAKE/calls"; kill $sleeper; exit 143' TERM
    sleep 60 &
    sleeper=$!
    touch "$FAKE/$run.started"
    wait $sleeper
fi
status=$(cat "$FAKE/$run.exit" 2>/dev/null || echo 0)
if [ "$status" = 0 ]; then
    if [ -n "${MOLSCOUT_PYTHON:-}" ]; then
        mkdir -p "benchmarks/results/$run"
        echo "{\\"git\\": {\\"commit\\": \\"$(git rev-parse HEAD)\\", \\"dirty\\": false}}" > "benchmarks/results/$run/meta.json"
    elif [ -f "$FAKE/$run.after" ]; then
        cp "$FAKE/$run.after" "$FAKE/$run.verdict"
    else
        echo "current made at ${FAKE_HEAD:0:7}" > "$FAKE/$run.verdict"
    fi
    if [ -f "$FAKE/$run.litter" ]; then touch "$(cat "$FAKE/$run.litter")"; fi
    if [ -f "$FAKE/$run.code" ]; then cp "$FAKE/$run.code" "$FAKE/code"; fi
fi
exit "$status"
"""
FAKE_SQUEUE = """#!/bin/bash
# squeue -h -j ID -o %T: RUNNING for the jobs listed in $FAKE/alive, else the error for a job that has ended.
while [ $# -gt 0 ]; do [ "$1" = -j ] && job=$2; shift; done
if grep -qx "$job" "$FAKE/alive" 2>/dev/null; then echo RUNNING; exit 0; fi
echo "slurm_load_jobs error: Invalid job id specified" >&2
exit 1
"""
FAKE_SBATCH = """#!/usr/bin/env python3
import json, os, pathlib, sys
fake = pathlib.Path(os.environ["FAKE"])
with (fake / "sbatch").open("a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
(fake / "sbatch.env").write_text(os.environ.get("MOLSCOUT_RESUBMIT", ""))
if (fake / "sbatch.refuse").exists():
    sys.exit("sbatch: error: QOSMaxSubmitJobPerUserLimit")
print("Submitted batch job 900")
"""
FAKE_GIT = """#!/bin/bash
[ "$*" = "rev-parse HEAD" ] && echo "$FAKE_HEAD"
"""
RESUBMIT = "--partition=gpu-short --gres=gpu:h200:1 --time=02:00:00 --signal=B:USR1@180 --job-name=molscout-gpu-short"


class Worker:
    """A submit directory with fake molscout, squeue, sbatch and git, and run.sbatch as job 500 in worker mode."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "repo"
        self.results = self.root / "benchmarks" / "results"
        self.claims = self.results / ".claims"
        self.fake = tmp_path / "fake"
        self.bin = tmp_path / "bin"
        for folder in (self.results, self.fake, self.bin):
            folder.mkdir(parents=True)
        for name, text in {"molscout": FAKE_BENCH, "squeue": FAKE_SQUEUE, "sbatch": FAKE_SBATCH, "git": FAKE_GIT}.items():
            (self.bin / name).write_text(text)
            (self.bin / name).chmod(0o755)
        self.env = {
            **{k: v for k, v in os.environ.items() if not k.startswith("SLURM_") and k != "CUDA_VISIBLE_DEVICES"},
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "HOME": str(tmp_path),
            "MOLSCOUT": str(self.bin / "molscout"),
            "MOLSCOUT_STORE": str(tmp_path / "store"),
            "MOLSCOUT_RESUBMIT": RESUBMIT,
            "SLURM_SUBMIT_DIR": str(self.root),
            "SLURM_JOB_ID": "500",
            "FAKE": str(self.fake),
            "FAKE_HEAD": HEAD,
            "FAKE_CODE": CODE,
        }

    def configs(self, runs: str) -> list[str]:
        return [f"benchmarks/configs/{run}.yaml" for run in runs.split()]

    def run(self, runs: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(SCRIPT), *self.configs(runs)], env=self.env, capture_output=True, text=True)

    def start(self, runs: str) -> subprocess.Popen:
        command = ["bash", str(SCRIPT), *self.configs(runs)]
        return subprocess.Popen(command, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def calls(self) -> list[str]:
        path = self.fake / "calls"
        return path.read_text().splitlines() if path.exists() else []

    def sbatch(self) -> list[list[str]]:
        path = self.fake / "sbatch"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def verdict(self, run: str, text: str, *, after: bool = False) -> None:
        """What the fake `molscout is-current` says of the run ("current WHY", "blocked WHY" or "not-current
        WHY"), now or, with `after`, once a run of it has exited 0."""
        (self.fake / f"{run}.{'after' if after else 'verdict'}").write_text(f"{text}\n")

    def claim(self, run: str, job: str) -> None:
        (self.claims / run).mkdir(parents=True)
        (self.claims / run / "job").write_text(f"{job}\n")


@pytest.fixture
def worker(tmp_path):
    return Worker(tmp_path)


def test_worker_skips_current_runs_and_runs_the_rest(worker):
    worker.verdict("a", "current made at bbbbbbb")
    worker.verdict("b", "not-current code changed since bbbbbbb: src/x.py")
    worker.verdict("c", "not-current uncommitted changes: src/x.py")
    result = worker.run("a b c d")
    assert result.returncode == 0, result.stderr
    assert worker.calls() == ["bench b", "bench c", "bench d"]
    assert "skip a: done at code ccccccc (made at bbbbbbb)" in result.stdout
    assert "claimed b (code changed since bbbbbbb: src/x.py)" in result.stdout
    assert "claimed c (uncommitted changes: src/x.py)" in result.stdout
    assert "claimed d (no results)" in result.stdout and "finished d" in result.stdout
    assert list(worker.claims.iterdir()) == []
    assert worker.sbatch() == []  # a worker that finishes does not resubmit


def test_worker_skips_live_claims_and_takes_over_dead_ones(worker):
    worker.claim("a", "111")
    worker.claim("b", "222")
    worker.claim("c", "333")
    worker.verdict("c", "current made at aaaaaaa")
    (worker.fake / "alive").write_text("111\n")
    result = worker.run("a b c")
    assert result.returncode == 0, result.stderr
    assert worker.calls() == ["bench b"]
    assert "skip a: claimed by job 111" in result.stdout
    assert "claimed b from job 222, which has ended" in result.stdout
    assert "skip c: done at code ccccccc" in result.stdout  # checked under the claim
    assert (worker.claims / "a" / "job").read_text() == "111\n"
    assert not (worker.claims / "b").exists() and not (worker.claims / "c").exists()


def test_worker_counts_failures_against_the_runs_code_and_gives_up_after_two(worker):
    (worker.fake / "a.exit").write_text("3")
    for attempt in (1, 2):
        result = worker.run("a b")
        assert result.returncode == 0, result.stderr
        assert f"failed a with status 3 (failure {attempt} of 2)" in result.stdout
        assert not (worker.claims / "a").exists()
    assert (worker.claims / "a.failures").read_text() == f"{CODE}\n{CODE}\n"
    worker.env["FAKE_HEAD"] = "d" * 40  # a commit that leaves the run's code alone keeps the count
    result = worker.run("a b")
    assert "skip a: gave up after 2 failures at code ccccccc; remove benchmarks/results/.claims/a.failures to retry" in result.stdout
    assert "skip b: done at code ccccccc" in result.stdout
    assert worker.calls() == ["bench a", "bench b", "bench a"]  # b is current once it has finished
    worker.env["FAKE_CODE"] = "e" * 64  # a commit to the run's code gets new attempts
    assert "failed a with status 3 (failure 1 of 2)" in worker.run("a").stdout
    assert (worker.claims / "a.failures").read_text() == f"{CODE}\n{CODE}\n{'e' * 64}\n"


def test_worker_skips_a_blocked_run_without_running_it(worker):
    worker.verdict("a", "blocked uncommitted changes: benchmarks/tools/a/out.png")
    result = worker.run("a b")
    assert result.returncode == 0, result.stderr
    assert worker.calls() == ["bench b"]
    assert (
        "skip a: no result of it could be current until this is fixed: uncommitted changes: benchmarks/tools/a/out.png"
        in result.stdout
    )
    assert not (worker.claims / "a.failures").exists() and not (worker.claims / "a").exists()


def test_a_run_that_finishes_but_is_blocked_counts_as_a_failure_and_is_not_rerun(worker):
    """A tool that leaves a file in its own folder: its results can never be current, so it must not loop."""
    worker.verdict("a", "blocked uncommitted changes: benchmarks/tools/a/out.png", after=True)
    result = worker.run("a")
    assert result.returncode == 0, result.stderr
    assert (
        "finished a, but it is not current (uncommitted changes: benchmarks/tools/a/out.png) (failure 1 of 2)"
        in result.stdout
    )
    assert (worker.claims / "a.failures").read_text() == f"{CODE}\n"
    assert "skip a: no result of it could be current" in worker.run("a").stdout
    assert worker.calls() == ["bench a"]


def test_runs_that_finish_but_stay_not_current_stop_after_two(worker):
    worker.verdict("a", "not-current meta.json is unreadable", after=True)
    for attempt in (1, 2):
        assert f"finished a, but it is not current (meta.json is unreadable) (failure {attempt} of 2)" in worker.run("a").stdout
    assert "skip a: gave up after 2 failures at code ccccccc" in worker.run("a").stdout
    assert worker.calls() == ["bench a", "bench a"]


def test_a_run_whose_code_changed_while_it_ran_is_not_a_failure(worker):
    worker.verdict("a", "not-current code changed since aaaaaaa: src/x.py", after=True)
    (worker.fake / "a.code").write_text("f" * 64 + "\n")
    result = worker.run("a")
    assert "finished a, but its code changed while it ran (code changed since aaaaaaa: src/x.py)" in result.stdout
    assert not (worker.claims / "a.failures").exists()


def test_a_failing_is_current_check_means_not_done(worker):
    (worker.fake / "is-current.broken").touch()
    (worker.fake / "a.exit").write_text("3")
    result = worker.run("a")
    assert worker.calls() == ["bench a"]
    assert "claimed a (molscout is-current failed with status 2)" in result.stdout
    assert (worker.claims / "a.failures").read_text() == "-\n"


@pytest.mark.parametrize(
    ("signum", "resubmits"), [(signal.SIGUSR1, True), (signal.SIGTERM, False)], ids=["time-limit", "scancel"]
)
def test_a_stopped_worker_stops_its_run_releases_the_claim_and_resubmits_only_at_the_time_limit(
    worker, signum, resubmits
):
    (worker.fake / "a.hang").touch()
    job = worker.start("a b")
    try:
        deadline = time.monotonic() + 30
        while not (worker.fake / "a.started").exists():
            assert job.poll() is None, job.stdout.read()
            assert time.monotonic() < deadline, "the run never started"
            time.sleep(0.05)
        assert (worker.claims / "a" / "job").read_text() == "500\n"
        job.send_signal(signum)
        output, _ = job.communicate(timeout=30)
    finally:
        if job.poll() is None:
            job.kill()
            job.wait()
    assert job.returncode == 0, output
    assert worker.calls() == ["bench a", "TERM a"]
    assert not (worker.claims / "a").exists()
    assert not (worker.claims / "a.failures").exists()
    assert "stopped a; its checkpoint is kept" in output
    expected = [[*RESUBMIT.split(), "benchmarks/slurm/run.sbatch", *worker.configs("a b")]] if resubmits else []
    assert worker.sbatch() == expected
    assert ("resubmitted: Submitted batch job 900" in output) is resubmits


def test_a_refused_resubmission_is_logged(worker):
    (worker.fake / "a.hang").touch()
    (worker.fake / "sbatch.refuse").touch()
    job = worker.start("a")
    deadline = time.monotonic() + 30
    while not (worker.fake / "a.started").exists():
        assert job.poll() is None and time.monotonic() < deadline
        time.sleep(0.05)
    job.send_signal(signal.SIGUSR1)
    output, _ = job.communicate(timeout=30)
    assert job.returncode == 0, output
    assert "resubmission refused: sbatch: error: QOSMaxSubmitJobPerUserLimit" in output


def test_one_config_is_a_worker_only_when_submitted_to_resubmit(worker):
    worker.verdict("a", "current made at aaaaaaa")
    assert "skip a: done" in worker.run("a").stdout
    del worker.env["MOLSCOUT_RESUBMIT"]
    assert worker.run("a").returncode == 0
    assert worker.calls() == ["bench a"]


GIT = ("git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false")
CONFIG = """tool: {tool}
name: Tool
version: '1.0'
dataset: uspto
images: data/raw/uspto/images
references: data/raw/uspto/refs
run_dir: benchmarks/tools/structure_readers/{tool}
python: ${{MOLSCOUT_STORE}}/envs/{tool}/bin/python
args: []
checkpoints: []
"""
REPO_FILES = {
    "pyproject.toml": "[project]\nname = 'molscout'\n",
    "src/molscout/x.py": "v1\n",
    "benchmarks/tools/crop_runner.py": "v1\n",
    "benchmarks/tools/paper_runner.py": "v1\n",
    "benchmarks/slurm/run.sbatch": "v1\n",
    "benchmarks/tools/structure_readers/t1/run.py": "v1\n",
    "benchmarks/tools/structure_readers/t2/run.py": "v1\n",
    **{f"benchmarks/configs/{tool}__uspto.yaml": CONFIG.format(tool=tool) for tool in ("t1", "t2")},
}


def test_the_worker_redoes_only_the_runs_whose_code_a_commit_changed(worker):
    """Real git and the real `molscout is-current` (only `molscout bench` is fake): both runs finished at the
    first commit; a commit adding another tool keeps them done, and one to t2's folder makes t2 run again."""
    (worker.bin / "git").unlink()
    worker.env["MOLSCOUT_PYTHON"] = sys.executable

    def commit(files: dict[str, str]) -> str:
        for path, text in files.items():
            (worker.root / path).parent.mkdir(parents=True, exist_ok=True)
            (worker.root / path).write_text(text)
        paths = list(files)
        subprocess.run([*GIT, "-C", str(worker.root), "add", "--", *paths], check=True)
        subprocess.run([*GIT, "-C", str(worker.root), "commit", "-q", "-m", "change"], check=True)
        head = subprocess.run(["git", "-C", str(worker.root), "rev-parse", "HEAD"], capture_output=True, text=True)
        return head.stdout.strip()

    subprocess.run([*GIT, "init", "-q", str(worker.root)], check=True)
    made = commit(REPO_FILES)
    for run in ("t1__uspto", "t2__uspto"):
        (worker.results / run).mkdir()
        (worker.results / run / "meta.json").write_text(json.dumps({"git": {"commit": made, "dirty": False}}))

    commit({"benchmarks/tools/complete_systems/biominer/x.py": "v1\n"})
    result = worker.run("t1__uspto t2__uspto")
    assert result.returncode == 0, result.stderr
    assert worker.calls() == []
    for run in ("t1__uspto", "t2__uspto"):
        assert f"skip {run}: done at code " in result.stdout and f"(made at {made[:7]})" in result.stdout

    own = "benchmarks/tools/structure_readers/t2/run.py"
    commit({own: "v2\n"})
    result = worker.run("t1__uspto t2__uspto")
    assert result.returncode == 0, result.stderr
    assert worker.calls() == ["bench t2__uspto"]
    assert "skip t1__uspto: done at code " in result.stdout
    assert f"claimed t2__uspto (code changed since {made[:7]}: {own})" in result.stdout
    assert "finished t2__uspto\n" in result.stdout  # its new meta.json is from HEAD

    # A run that leaves a file in its tool folder fails once, then waits for the file to go.
    litter = "benchmarks/tools/structure_readers/t1/out.png"
    commit({"benchmarks/tools/structure_readers/t1/run.py": "v2\n"})
    (worker.fake / "t1__uspto.litter").write_text(litter)
    result = worker.run("t1__uspto t2__uspto")
    assert f"finished t1__uspto, but it is not current (uncommitted changes: {litter}) (failure 1 of 2)" in result.stdout
    (worker.fake / "t1__uspto.litter").unlink()
    result = worker.run("t1__uspto")
    assert f"skip t1__uspto: no result of it could be current until this is fixed: uncommitted changes: {litter}" in result.stdout
    (worker.root / litter).unlink()
    assert "skip t1__uspto: done at code " in worker.run("t1__uspto").stdout
    assert worker.calls() == ["bench t2__uspto", "bench t1__uspto"]


GPU_HEALTHY = "#!/bin/bash\necho 0\n"
GPU_BROKEN = "#!/bin/bash\necho '[GPU requires reset]'\n"


def fake_nvidia_smi(worker, text: str) -> None:
    (worker.bin / "nvidia-smi").write_text(text)
    (worker.bin / "nvidia-smi").chmod(0o755)


def gpu_job(worker, *, smi: str | None = GPU_HEALTHY, probe: str = "true", devices: str = "0") -> None:
    if smi is not None:
        fake_nvidia_smi(worker, smi)
    worker.env.update(CUDA_VISIBLE_DEVICES=devices, MOLSCOUT_CUDA_PROBE=probe)


def host() -> str:
    return subprocess.run(["hostname", "-s"], capture_output=True, text=True, check=True).stdout.strip()


def test_bad_gpu_resubmits_excluding_the_node_without_claiming(worker):
    gpu_job(worker, smi=GPU_BROKEN)
    worker.env["MOLSCOUT_RESUBMIT"] = "--partition=gpu --exclude=d4072"
    result = worker.run("a b")
    assert result.returncode == 0, result.stderr
    assert worker.calls() == []
    assert not worker.claims.exists()
    options = f"--partition=gpu --exclude=d4072,{host()}"
    assert worker.sbatch() == [[*options.split(), "benchmarks/slurm/run.sbatch", *worker.configs("a b")]]
    assert (worker.fake / "sbatch.env").read_text() == options
    assert f"bad GPU on {host()}: " in result.stdout
    assert "[GPU requires reset]" in result.stdout


def test_bad_gpu_adds_an_exclude_when_there_is_none(worker):
    gpu_job(worker, smi=GPU_BROKEN)
    assert worker.run("a b").returncode == 0
    options = f"{RESUBMIT} --exclude={host()}"
    assert worker.sbatch() == [[*options.split(), "benchmarks/slurm/run.sbatch", *worker.configs("a b")]]


def test_cuda_probe_failure_is_a_bad_gpu(worker):
    gpu_job(worker, probe="false")
    result = worker.run("a b")
    assert result.returncode == 0, result.stderr
    assert worker.calls() == []
    assert f"bad GPU on {host()}: " in result.stdout
    assert len(worker.sbatch()) == 1


def test_the_probe_is_told_how_many_gpus_to_expect(worker):
    gpu_job(worker, smi="#!/bin/bash\necho 0\necho 5\n", devices="0,1")
    probe = worker.bin / "probe"
    probe.write_text('#!/bin/bash\necho "probe $*" >> "$FAKE/probe"\n')
    probe.chmod(0o755)
    worker.env["MOLSCOUT_CUDA_PROBE"] = str(probe)
    assert worker.run("a b").returncode == 0
    assert (worker.fake / "probe").read_text() == "probe 2\n"
    assert worker.calls() == ["bench a", "bench b"]


def test_a_gpu_missing_from_nvidia_smi_is_a_bad_gpu(worker):
    gpu_job(worker, devices="0,1")  # the fake lists one GPU
    result = worker.run("a b")
    assert worker.calls() == [] and len(worker.sbatch()) == 1
    assert "bad GPU" in result.stdout


def test_healthy_gpu_runs_the_worker(worker):
    gpu_job(worker)
    result = worker.run("a b")
    assert result.returncode == 0, result.stderr
    assert worker.calls() == ["bench a", "bench b"]
    assert "bad GPU" not in result.stdout
    assert worker.sbatch() == []


def test_cpu_job_skips_the_gpu_check(worker):
    result = worker.run("a b")  # no CUDA_VISIBLE_DEVICES, no nvidia-smi
    assert result.returncode == 0, result.stderr
    assert worker.calls() == ["bench a", "bench b"]


def test_single_run_on_bad_gpu_exits_3(worker):
    gpu_job(worker, smi=GPU_BROKEN)
    del worker.env["MOLSCOUT_RESUBMIT"]
    result = worker.run("a")
    assert result.returncode == 3
    assert f"bad GPU on {host()}" in result.stderr
    assert worker.calls() == [] and worker.sbatch() == []


def test_array_run_on_bad_gpu_exits_3(worker):
    gpu_job(worker, smi=GPU_BROKEN)
    del worker.env["MOLSCOUT_RESUBMIT"]
    worker.env["SLURM_ARRAY_TASK_ID"] = "1"
    result = worker.run("a b")
    assert result.returncode == 3
    assert worker.calls() == []


def test_gives_up_after_three_excluded_nodes(worker):
    gpu_job(worker, smi=GPU_BROKEN)
    worker.env["MOLSCOUT_RESUBMIT"] = "--partition=gpu --exclude=d1,d2,d3"
    result = worker.run("a b")
    assert result.returncode == 1
    assert "giving up" in result.stdout
    assert worker.calls() == [] and worker.sbatch() == []


def test_a_refused_bad_gpu_resubmission_fails_the_worker(worker):
    gpu_job(worker, smi=GPU_BROKEN)
    (worker.fake / "sbatch.refuse").touch()
    result = worker.run("a b")
    assert result.returncode == 1
    assert "resubmit failed: sbatch: error: QOSMaxSubmitJobPerUserLimit" in result.stdout
    assert worker.calls() == []
