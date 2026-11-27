"""meta.json keeps the machine out: store paths as $MOLSCOUT_STORE, repository paths relative, the home folder as ~,
the account name as $USER, and no host names or SLURM job IDs. scripts/scrub_meta.py applies it to committed files.

The cluster layout below is made up: account `jdoe`, store /scratch/jdoe/molscout-store, repository /scratch/jdoe/molscout.
"""

import copy
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from molscout.bench.scrub import LOCK_FIELDS, Places, lock_digest, scrub_meta, scrub_text

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "scrub_meta.py"
PLACES = Places(repo="/scratch/jdoe/molscout", store="/scratch/jdoe/molscout-store", home="/home/jdoe", user="jdoe")
STORE, ROOT = PLACES.store, PLACES.repo
ROOTS = ["--repo", ROOT, "--store", STORE, "--home", "/home/jdoe", "--user", "jdoe"]
# A real account's folder; /home/conda/feedstock_root is where conda-forge builds packages, recorded by pip freeze
ABSOLUTE_PRIVATE = re.compile(r"(?<![\w.~$-])/(?:scratch|home|Users)/(?!\$USER\b|conda/feedstock_root/)")


@pytest.mark.parametrize(
    ("text", "scrubbed"),
    [
        (f"{STORE}/models/molscribe/swin.pth", "$MOLSCOUT_STORE/models/molscribe/swin.pth"),
        (STORE, "$MOLSCOUT_STORE"),
        (f"{ROOT}/data/raw/uspto/USPTO", "data/raw/uspto/USPTO"),
        (ROOT, "."),
        (f"--images={ROOT}/data/raw", "--images=./data/raw"),
        ("/home/jdoe/.local/bin/uv", "~/.local/bin/uv"),
        ("/scratch/jdoe/hf_cache", "/scratch/$USER/hf_cache"),
        (f"biominer @ file://{STORE}/src/BioMiner", "biominer @ file://$MOLSCOUT_STORE/src/BioMiner"),
        (f"PATH={STORE}/envs/x/bin:/home/jdoe/bin", "PATH=$MOLSCOUT_STORE/envs/x/bin:~/bin"),
        ("/tmp/tmp.kLscAbm61H/BioMiner", "/tmp/tmp.kLscAbm61H/BioMiner"),
        ("/usr/bin/python3", "/usr/bin/python3"),
        ("/opt/x/scratch/jdoe/y", "/opt/x/scratch/jdoe/y"),  # the root inside another path is not the root
        ("data/jdoe/x", "data/jdoe/x"),  # relative paths keep a folder named like the account
        ("gpu-interactive", "gpu-interactive"),
    ],
)
def test_scrub_text(text, scrubbed):
    assert scrub_text(text, PLACES) == scrubbed


def test_a_root_is_matched_only_as_a_whole_folder():
    repo_only = Places(repo="/scratch/jdoe/molscout")
    assert scrub_text("/scratch/jdoe/molscout-store/envs/x", repo_only) == "/scratch/jdoe/molscout-store/envs/x"
    assert scrub_text("/scratch/jdoe/molscout/src", repo_only) == "src"


def test_the_longest_root_wins_when_one_holds_another():
    nested = Places(repo="/home/jdoe/molscout", store="/home/jdoe/molscout/store", home="/home/jdoe")
    assert scrub_text("/home/jdoe/molscout/store/envs", nested) == "$MOLSCOUT_STORE/envs"
    assert scrub_text("/home/jdoe/molscout/src", nested) == "src"
    assert scrub_text("/home/jdoe/notes", nested) == "~/notes"


@pytest.mark.parametrize("root", ["", "/", "relative/path", None])
def test_roots_that_are_empty_or_not_absolute_are_ignored(root):
    places = Places(repo=root, store=root, home=root)
    assert scrub_text("/scratch/jdoe/x", places) == "/scratch/jdoe/x"


def test_a_trailing_slash_on_a_root_is_ignored():
    assert scrub_text(f"{STORE}/envs", Places(store=f"{STORE}/")) == "$MOLSCOUT_STORE/envs"


def test_places_here_reads_the_store_home_and_account_from_the_environment(tmp_path):
    environ = {"MOLSCOUT_STORE": "/s/store", "HOME": "/home/jdoe", "USER": "jdoe"}
    assert Places.here(tmp_path, environ) == Places(repo=str(tmp_path), store="/s/store", home="/home/jdoe", user="jdoe")
    assert Places.here(tmp_path, {"LOGNAME": "jdoe", "MOLSCOUT_STORE": ""}) == Places(repo=str(tmp_path), user="jdoe")


def cluster_meta() -> dict:
    """A meta.json as a cluster run wrote it before the scrub, cut down to the fields that hold paths or machines."""
    lock = {
        "python_version": "3.10.21",
        "packages": ["numpy==1.26.4", f"biominer @ file://{STORE}/src/BioMiner"],
        "conda_meta": None,
        "lock_commands": [{"command": [f"{STORE}/envs/biominer/bin/python", "-V"], "output": f"{STORE}/envs/biominer\n"}],
    }
    return {
        "run": "molscribe__uspto",
        "tool": {
            "tool": "molscribe",
            "name": "MolScribe",
            "version": "1.1.1",
            "checkpoints": [{"path": f"{STORE}/models/molscribe/swin.pth", "sha256": "3" * 64}],
        },
        "dataset": "uspto",
        "items": 5719,
        "tool_errors": 0,
        "inputs": {"images": f"{ROOT}/data/raw/uspto/USPTO", "images_sha256": "a" * 64},
        "git": {"commit": "1" * 40, "dirty": False, "dirty_paths": [], "code_fingerprint": "f" * 64},
        "sources": [{"path": f"{STORE}/src/BioMiner", "commit": "2" * 40, "dirty": False, "dirty_paths": [], "untracked": 0}],
        "environment": {
            "python": f"{STORE}/envs/molscribe/bin/python",
            "freeze_command": ["/home/jdoe/.local/bin/uv", "pip", "freeze", "--python", f"{STORE}/envs/molscribe/bin/python"],
            **lock,
            "sha256": hashlib.sha256(json.dumps(lock, sort_keys=True).encode("utf-8")).hexdigest(),
            "variables": {"HF_HOME": "/scratch/jdoe/hf_cache", "MOLSCOUT_STORE": STORE, "SLURM_JOB_PARTITION": "gpu"},
        },
        "hardware": {
            "host": "gpu-node-1",
            "platform": "Linux-5.14",
            "cluster": "explorer",
            "cpu_model": "INTEL(R) XEON(R) GOLD 6548Y+",
            "cpus_available": 8,
            "slurm_mem_per_node_mb": 65536,
            "slurm_mem_per_cpu_mb": None,
            "gpus": [{"name": "NVIDIA H200", "memory": "143771 MiB", "driver": "570.86.15"}],
        },
        "slurm": {"job": "5550001", "array_job": None, "array_task": None, "partition": "gpu-interactive", "nodes": "gpu-node-1"},
        "timing": {"started_utc": "2026-10-05T06:05:14.074Z", "wall_seconds": 2059.715},
        "resources": {"tool_peak_rss_mib": 2219.9, "tool_cpu_seconds": 2206.3, "gpu": None},
        "segments": [
            {
                "slurm": {"job": "5550002", "array_job": "5550000", "array_task": "3", "partition": "sharing"},
                "host": "gpu-node-2",
                "gpus": ["NVIDIA H200 NVL"],
                "rows": 0,
                "tool_seconds": 37.635,
                "resources": {"tool_peak_rss_mib": 2106.5, "tool_cpu_seconds": 12.783, "gpu": None},
            }
        ],
        "command": [
            f"{STORE}/envs/molscribe/bin/python",
            "run.py",
            "--images",
            f"{ROOT}/data/raw/uspto/USPTO",
            "--output",
            f"{ROOT}/benchmarks/results/.staging-molscribe__uspto-2643228/predictions.csv",
            "--checkpoint",
            f"{STORE}/models/molscribe/swin.pth",
        ],
    }


def test_scrub_meta_writes_paths_from_their_roots():
    meta = scrub_meta(cluster_meta(), PLACES)
    assert meta["tool"]["checkpoints"] == [{"path": "$MOLSCOUT_STORE/models/molscribe/swin.pth", "sha256": "3" * 64}]
    assert meta["inputs"] == {"images": "data/raw/uspto/USPTO", "images_sha256": "a" * 64}
    assert meta["sources"][0]["path"] == "$MOLSCOUT_STORE/src/BioMiner"
    environment = meta["environment"]
    assert environment["python"] == "$MOLSCOUT_STORE/envs/molscribe/bin/python"
    assert environment["freeze_command"][0] == "~/.local/bin/uv"
    assert environment["packages"][1] == "biominer @ file://$MOLSCOUT_STORE/src/BioMiner"
    assert environment["variables"] == {
        "HF_HOME": "/scratch/$USER/hf_cache",
        "MOLSCOUT_STORE": "$MOLSCOUT_STORE",
        "SLURM_JOB_PARTITION": "gpu",
    }
    assert meta["command"][3:6] == [
        "data/raw/uspto/USPTO",
        "--output",
        "benchmarks/results/.staging-molscribe__uspto-2643228/predictions.csv",
    ]
    text = json.dumps(meta)
    assert "jdoe" not in text
    assert not ABSOLUTE_PRIVATE.search(text)


def test_scrub_meta_drops_host_names_and_job_ids_and_keeps_the_partition():
    meta = scrub_meta(cluster_meta(), PLACES)
    assert "host" not in meta["hardware"]
    assert meta["slurm"] == {"partition": "gpu-interactive"}
    [segment] = meta["segments"]
    assert "host" not in segment
    assert segment["slurm"] == {"partition": "sharing"}
    assert "gpu-node" not in json.dumps(meta)
    assert "555000" not in json.dumps(meta)


def test_scrub_meta_keeps_what_scoring_tables_reports_and_is_current_read():
    before, after = cluster_meta(), scrub_meta(cluster_meta(), PLACES)
    for key in ("run", "dataset", "items", "tool_errors", "git", "timing", "resources"):
        assert after[key] == before[key]
    assert {k: v for k, v in after["tool"].items() if k != "checkpoints"} == {
        k: v for k, v in before["tool"].items() if k != "checkpoints"
    }
    assert after["hardware"] == {k: v for k, v in before["hardware"].items() if k != "host"}
    assert [c["sha256"] for c in after["tool"]["checkpoints"]] == [c["sha256"] for c in before["tool"]["checkpoints"]]
    assert {k: v for k, v in after["sources"][0].items() if k != "path"} == {
        k: v for k, v in before["sources"][0].items() if k != "path"
    }
    assert [s["gpus"] for s in after["segments"]] == [s["gpus"] for s in before["segments"]]


def test_scrub_meta_does_not_change_its_input_and_is_idempotent():
    original = cluster_meta()
    kept = copy.deepcopy(original)
    once = scrub_meta(original, PLACES)
    assert original == kept
    assert scrub_meta(once, PLACES) == once
    assert scrub_meta(once, Places()) == once  # a scrubbed file needs no roots to stay as it is


def test_the_environment_sha256_is_retaken_when_the_scrub_changes_the_lock():
    meta = scrub_meta(cluster_meta(), PLACES)
    environment = meta["environment"]
    assert environment["sha256"] == lock_digest(environment)
    assert environment["sha256"] != cluster_meta()["environment"]["sha256"]


def test_the_environment_sha256_is_kept_when_the_lock_holds_no_paths():
    meta = cluster_meta()
    meta["environment"] = {**meta["environment"], "packages": ["numpy==1.26.4"], "lock_commands": [], "sha256": "recorded"}
    assert scrub_meta(meta, PLACES)["environment"]["sha256"] == "recorded"


def test_lock_digest_is_the_sha256_of_the_lock_fields():
    lock = {"python_version": "3.11", "packages": ["a==1"], "conda_meta": None, "lock_commands": [], "python": "/x"}
    expected = hashlib.sha256(json.dumps({k: lock[k] for k in LOCK_FIELDS}, sort_keys=True).encode()).hexdigest()
    assert lock_digest(lock) == expected


def test_committed_meta_json_files_are_scrubbed():
    """Every committed meta.json names no account, host, job or absolute cluster path."""
    files = sorted((REPO / "benchmarks" / "results").glob("*/meta.json"))
    assert files
    dirty = []
    for path in files:
        meta = json.loads(path.read_text(encoding="utf-8"))
        if scrub_meta(meta, Places()) != meta or ABSOLUTE_PRIVATE.search(json.dumps(meta)):
            dirty.append(path.parent.name)
    assert dirty == [], f"run scripts/scrub_meta.py on: {', '.join(dirty)}"


def write(path: Path, meta: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path


def script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False)


def test_script_scrubs_files_in_place_and_leaves_clean_ones_alone(tmp_path):
    raw = write(tmp_path / "a__uspto" / "meta.json", cluster_meta())
    clean_meta = scrub_meta(cluster_meta(), PLACES)
    clean = write(tmp_path / "b__uspto" / "meta.json", clean_meta)
    clean_bytes = clean.read_bytes()
    done = script(*ROOTS, str(raw), str(clean))
    assert done.returncode == 0, done.stderr
    assert json.loads(raw.read_text()) == clean_meta
    assert raw.read_text() == json.dumps(clean_meta, indent=2, allow_nan=False) + "\n"
    assert clean.read_bytes() == clean_bytes
    assert done.stdout.splitlines() == [f"scrubbed {raw}", "1 of 2 files scrubbed"]


def test_script_check_lists_files_to_scrub_without_writing(tmp_path):
    raw = write(tmp_path / "a__uspto" / "meta.json", cluster_meta())
    before = raw.read_bytes()
    done = script("--check", *ROOTS, str(raw))
    assert done.returncode == 1
    assert done.stdout.splitlines() == [f"needs a scrub: {raw}"]
    assert raw.read_bytes() == before
    assert script(*ROOTS, str(raw)).returncode == 0
    assert script("--check", *ROOTS, str(raw)).returncode == 0


def test_script_reports_unreadable_files(tmp_path):
    broken = tmp_path / "meta.json"
    broken.write_text("{not json")
    done = script(*ROOTS, str(broken))
    assert done.returncode == 1
    assert f"scrub_meta: error: {broken}" in done.stderr
