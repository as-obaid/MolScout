"""benchmarks/slurm/status.sh shows where each run stands, with fake squeue, git and `molscout is-current`."""

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
CODE = "c" * 64
FAKE_MOLSCOUT = """#!/bin/bash
# molscout is-current CONFIG...: one line per config from $FAKE/<run>.verdict ("current WHY" or
# "not-current WHY"; none: no results) with the code fingerprint %s; exit 1 unless all are current.
[ "$1" = is-current ] || exit 2
[ -n "${MOLSCOUT_STORE:-}" ] || { echo "configs use ${MOLSCOUT_STORE}, which is not set" >&2; exit 2; }
shift
status=0
for config in "$@"; do
    run=$(basename "$config" .yaml)
    verdict=$(cat "$FAKE/$run.verdict" 2>/dev/null || echo "not-current no results")
    echo "$run ${verdict%%%% *} %s ${verdict#* }"
    [ "${verdict%%%% *}" = current ] || status=1
done
exit $status
""" % (CODE, CODE)


def fake_tools(fake: Path) -> dict[str, str]:
    """squeue, git and molscout in `fake`, and the environment that puts them first on PATH."""
    for name, text in {"squeue": FAKE_SQUEUE, "git": FAKE_GIT, "molscout": FAKE_MOLSCOUT}.items():
        (fake / name).write_text(text)
        (fake / name).chmod(0o755)
    path = f"{fake}{os.pathsep}{os.environ['PATH']}"
    inherited = {name: value for name, value in os.environ.items() if name != "MOLSCOUT_STORE"}  # status.sh sets it
    return {**inherited, "PATH": path, "MOLSCOUT": str(fake / "molscout"), "FAKE": str(fake)}


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
    (fake / "done__x.verdict").write_text("current made at bbbbbbb\n")
    (fake / "old__x.verdict").write_text("not-current code changed since bbbbbbb: src/x.py\n")
    (claims / "running__x").mkdir()
    (claims / "running__x" / "job").write_text("111\n")
    for run, rows in (("running__x", 1), ("partial__x", 2), ("failed__x", 1)):
        (results / ".checkpoints" / run).mkdir(parents=True)
        (results / ".checkpoints" / run / "predictions.csv").write_text("header\n" + "row\n" * rows)
    (claims / "partial__x.failures").write_text(f"{CODE}\n")
    (claims / "failed__x.failures").write_text(f"{'e' * 64}\n{CODE}\n{CODE}\n")  # only failures with this code count
    env = fake_tools(fake)
    result = subprocess.run(["bash", str(SCRIPT)], cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    table = [line.split(None, 2) for line in result.stdout.splitlines()]
    assert table[:7] == [
        ["RUN", "STATE", "DETAIL"],
        ["done__x", "done", "made at bbbbbbb"],
        ["failed__x", "failed", "2 failures at code ccccccc, 1/3 rows"],
        ["old__x", "stale", "code changed since bbbbbbb: src/x.py"],
        ["partial__x", "partial", "2/3 rows, 1 failure at code ccccccc"],
        ["pending__x", "pending"],
        ["running__x", "running", "job 111 (gpu-short), 1/3 rows"],
    ]
    assert result.stdout.splitlines()[-2:] == ["JOBID NAME PARTITION", "111 molscout-gpu-short gpu-short"]


def paper_repo(tmp_path):
    root, fake = tmp_path / "repo", tmp_path / "bin"
    configs = root / "benchmarks" / "configs"
    for folder in (configs, fake):
        folder.mkdir(parents=True)
    (configs / "t1__x.yaml").write_text("tool: t1\nrun_dir: benchmarks/tools/structure_readers/t1\nimages: nowhere\n")
    (configs / "sys__biovista.yaml").write_text("tool: sys\nrun_dir: benchmarks/tools/complete_systems/sys\n")
    (configs / "sys__internal.yaml").write_text("tool: sys\nrun_dir: benchmarks/tools/complete_systems/sys\n")
    checkpoint = root / "benchmarks" / "results" / ".checkpoints" / "sys__biovista"
    checkpoint.mkdir(parents=True)
    (checkpoint / "predictions.csv").write_text("header\nrow\nrow\nrow\n")
    (checkpoint / "predictions.papers.jsonl").write_text('{"item_id": "p1"}\n{"item_id": "p2"}\n')
    return root, fake_tools(fake)


def status(root, env, *args):
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=root, env=env, capture_output=True, text=True)


def test_a_paper_run_shows_its_finished_papers(tmp_path):
    root, env = paper_repo(tmp_path)
    result = status(root, env)
    assert result.returncode == 0, result.stderr
    table = [line.split(None, 2) for line in result.stdout.splitlines()]
    assert ["sys__biovista", "partial", "2 papers"] in table
    assert ["sys__internal", "pending"] in table
    assert any(row[0] == "t1__x" for row in table)


def test_status_can_show_one_kind_of_run(tmp_path):
    root, env = paper_repo(tmp_path)
    systems = status(root, env, "complete-systems").stdout
    assert "sys__biovista" in systems and "t1__x" not in systems
    readers = status(root, env, "structure-readers").stdout
    assert "t1__x" in readers and "sys__biovista" not in readers


def test_status_rejects_an_unknown_kind(tmp_path):
    root, env = paper_repo(tmp_path)
    assert status(root, env, "everything").returncode == 2


def test_status_fails_when_molscout_is_current_cannot_run(tmp_path):
    root, env = paper_repo(tmp_path)
    (tmp_path / "bin" / "molscout").write_text("#!/bin/bash\necho 'Traceback: broken' >&2\nexit 2\n")
    result = status(root, env)
    assert result.returncode == 1
    assert "molscout is-current failed" in result.stderr


def test_status_needs_a_molscout_command(tmp_path):
    root, env = paper_repo(tmp_path)
    env["MOLSCOUT"] = str(tmp_path / "missing" / "molscout")
    result = status(root, env)
    assert result.returncode == 1
    assert "set MOLSCOUT" in result.stderr
