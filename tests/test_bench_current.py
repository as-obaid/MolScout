"""`molscout is-current`: a finished run stays current across commits that leave its code alone.

A run's code is the harness (GIT_PATHS), its tool folder and its config. The repository here is a small
git repository with that layout; results/<run>/meta.json is written by hand, as an older Type 1 run left it
(commit and dirty flag, no code fingerprint).
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from molscout.bench.current import GIT_PATHS, check_current, code_fingerprint, run_paths
from molscout.bench.config import load_config
from molscout.cli import main

GIT = ("git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false")
RUN = "mytool__uspto"
CONFIG = """\
tool: {tool}
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
OWN_TOOL = "benchmarks/tools/structure_readers/mytool/run.py"
OTHER_TOOL = "benchmarks/tools/structure_readers/other/run.py"
NEW_SYSTEM = "benchmarks/tools/complete_systems/biominer/x.py"
OWN_CONFIG = f"benchmarks/configs/{RUN}.yaml"


class Repo:
    """A repository with the harness paths, two tools and their configs, all committed once."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.results = root / "benchmarks" / "results"
        files = {
            "pyproject.toml": "[project]\nname = 'molscout'\n",
            "src/molscout/harness.py": "v1\n",
            "benchmarks/tools/crop_runner.py": "v1\n",
            "benchmarks/tools/paper_runner.py": "v1\n",
            "benchmarks/slurm/run.sbatch": "v1\n",
            OWN_TOOL: "v1\n",
            OTHER_TOOL: "v1\n",
            OWN_CONFIG: CONFIG.format(tool="mytool"),
            "benchmarks/configs/other__uspto.yaml": CONFIG.format(tool="other"),
            "README.md": "v1\n",
        }
        for path, text in files.items():
            self.write(path, text)
        git("init", "-q", str(root))
        self.commit(*files)

    @property
    def config(self) -> Path:
        return self.root / OWN_CONFIG

    def write(self, path: str, text: str) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def edit(self, path: str) -> None:
        """Change a file without breaking it: a config gets a comment, anything else new text."""
        target = self.root / path
        old = target.read_text() if target.exists() else ""
        self.write(path, old + "# edited\n")

    def commit(self, *paths: str) -> str:
        git("-C", str(self.root), "add", "--", *paths)
        git("-C", str(self.root), "commit", "-q", "-m", "change")
        return self.head()

    def head(self) -> str:
        return git("-C", str(self.root), "rev-parse", "HEAD").strip()

    def finished(self, commit: str | None = None, dirty: object = False) -> str:
        """results/<run>/meta.json as an older run wrote it: its commit and dirty flag, no code fingerprint."""
        commit = commit or self.head()
        folder = self.results / RUN
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "meta.json").write_text(json.dumps({"run": RUN, "git": {"commit": commit, "dirty": dirty}}))
        return commit

    def check(self, config: Path | None = None):
        return check_current(config or self.config, repo_root=self.root, results_root=self.results)


def git(*args: str) -> str:
    return subprocess.run([*GIT, *args], capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("MOLSCOUT_STORE", str(tmp_path / "store"))
    return Repo(tmp_path / "repo")


def test_the_runs_code_is_the_harness_its_tool_folder_and_its_config(repo):
    config = load_config(repo.config, repo.root)
    assert run_paths(config, repo.root) == [
        repo.root / "benchmarks/tools/structure_readers/mytool",
        repo.root / OWN_CONFIG,
        *(repo.root / path for path in GIT_PATHS),
    ]
    assert "src" in GIT_PATHS and "benchmarks/slurm/run.sbatch" in GIT_PATHS


def test_the_fingerprint_hashes_the_sorted_blob_ids_and_paths_of_the_runs_code(repo):
    config = load_config(repo.config, repo.root)
    paths = run_paths(config, repo.root)
    listed = git("-C", str(repo.root), "ls-tree", "-r", "HEAD", "--", *(str(p.relative_to(repo.root)) for p in paths))
    entries = sorted(tuple(reversed(line.split(" ", 2)[2].split("\t"))) for line in listed.splitlines())  # (path, blob)
    expected = hashlib.sha256("".join(f"{blob} {path}\n" for path, blob in entries).encode()).hexdigest()
    assert len(entries) == 7  # README.md and the other tool and config are not the run's code
    assert code_fingerprint(repo.root, repo.head(), paths) == expected
    assert code_fingerprint(repo.root, None, paths) is None
    assert code_fingerprint(repo.root, "f" * 40, paths) is None


def test_a_finished_run_is_current_at_its_own_commit(repo):
    made = repo.finished()
    verdict = repo.check()
    assert verdict.current, verdict.why
    assert verdict.run == RUN
    assert verdict.why == f"made at {made[:7]}"
    assert re.fullmatch("[0-9a-f]{64}", verdict.fingerprint)


@pytest.mark.parametrize("path", [NEW_SYSTEM, OTHER_TOOL, "benchmarks/configs/other__uspto.yaml", "README.md"])
def test_a_commit_outside_the_runs_code_keeps_it_current(repo, path):
    made = repo.finished()
    before = repo.check().fingerprint
    repo.edit(path)
    assert repo.commit(path) != made
    verdict = repo.check()
    assert verdict.current, verdict.why
    assert verdict.why == f"made at {made[:7]}"
    assert verdict.fingerprint == before


@pytest.mark.parametrize(
    "path",
    [
        "src/molscout/harness.py",
        "src/molscout/new.py",
        "pyproject.toml",
        "benchmarks/tools/crop_runner.py",
        "benchmarks/tools/paper_runner.py",
        "benchmarks/slurm/run.sbatch",
        OWN_TOOL,
        "benchmarks/tools/structure_readers/mytool/new.py",
        OWN_CONFIG,
    ],
)
def test_a_commit_to_the_runs_code_makes_it_not_current(repo, path):
    made = repo.finished()
    before = repo.check().fingerprint
    repo.edit(path)
    repo.commit(path)
    verdict = repo.check()
    assert not verdict.current
    assert verdict.why == f"code changed since {made[:7]}: {path}"
    assert verdict.fingerprint not in (None, before)


def test_a_file_removed_from_the_runs_code_makes_it_not_current(repo):
    made = repo.finished()
    git("-C", str(repo.root), "rm", "-q", OWN_TOOL)
    git("-C", str(repo.root), "commit", "-q", "-m", "remove")
    assert repo.check().why == f"code changed since {made[:7]}: {OWN_TOOL}"


def test_the_changed_paths_shown_are_cut_short(repo):
    made = repo.finished()
    paths = [f"src/molscout/m{i}.py" for i in range(5)]
    for path in paths:
        repo.edit(path)
    repo.commit(*paths)
    assert repo.check().why == f"code changed since {made[:7]}: {', '.join(paths[:3])} and 2 more"


@pytest.mark.parametrize(
    ("path", "current"),
    [
        ("src/molscout/harness.py", False),
        (OWN_TOOL, False),
        ("benchmarks/tools/structure_readers/mytool/untracked.py", False),
        (OWN_CONFIG, False),
        (NEW_SYSTEM, True),
        (OTHER_TOOL, True),
        ("README.md", True),
    ],
)
def test_only_uncommitted_changes_to_the_runs_code_make_it_not_current(repo, path, current):
    repo.finished()
    repo.edit(path)
    verdict = repo.check()
    assert verdict.current is current, verdict.why
    if not current:
        assert verdict.why == f"uncommitted changes: {path}"


def test_results_from_a_commit_missing_from_the_repository_are_not_current(repo):
    repo.finished(commit="f" * 40)
    verdict = repo.check()
    assert not verdict.current
    assert verdict.why == "made at fffffff, which is not in this repository"
    assert verdict.fingerprint is not None  # failures are still counted against the code at HEAD


@pytest.mark.parametrize(
    ("meta", "why"),
    [
        (None, "no results"),
        ("{not json", "meta.json is unreadable"),
        ("[]", "meta.json records no commit"),
        ('{"git": null}', "meta.json records no commit"),
        ('{"git": {"dirty": false}}', "meta.json records no commit"),
        ('{"git": {"commit": "HEAD", "dirty": false}}', "meta.json records no commit"),
        ('{"git": {"commit": "--output=/tmp/x", "dirty": false}}', "meta.json records no commit"),
    ],
)
def test_missing_or_unreadable_results_are_not_current(repo, meta, why):
    if meta is not None:
        (repo.results / RUN).mkdir(parents=True)
        (repo.results / RUN / "meta.json").write_text(meta)
    verdict = repo.check()
    assert not verdict.current
    assert verdict.why == why


@pytest.mark.parametrize(("dirty", "why"), [(True, "made with uncommitted changes"), (None, "meta.json does not say")])
def test_results_made_with_uncommitted_changes_are_not_current(repo, dirty, why):
    repo.finished(dirty=dirty)
    verdict = repo.check()
    assert not verdict.current
    assert verdict.why.startswith(why)


def test_a_config_that_cannot_be_read_is_not_current(repo, monkeypatch):
    repo.finished()
    monkeypatch.delenv("MOLSCOUT_STORE")
    verdict = repo.check()
    assert not verdict.current
    assert verdict.run == RUN and verdict.fingerprint is None
    assert verdict.why.startswith("cannot read the config: ")
    assert "\n" not in verdict.why


def test_nothing_is_current_outside_a_git_repository(repo):
    repo.finished()
    git_dir = repo.root / ".git"
    git_dir.rename(repo.root / "not-git")
    verdict = repo.check()
    assert not verdict.current
    assert verdict.fingerprint is None
    assert verdict.why == "not in a git repository"


def test_cli_prints_one_line_per_config_and_exits_0_only_when_all_are_current(repo, monkeypatch, capsys):
    made = repo.finished()
    monkeypatch.chdir(repo.root)
    fingerprint = repo.check().fingerprint
    assert main(["is-current", OWN_CONFIG]) == 0
    assert capsys.readouterr().out == f"{RUN} current {fingerprint} made at {made[:7]}\n"
    assert main(["is-current", OWN_CONFIG, "benchmarks/configs/other__uspto.yaml"]) == 1
    first, second = capsys.readouterr().out.splitlines()
    assert first.startswith(f"{RUN} current ")
    assert re.fullmatch(r"other__uspto not-current [0-9a-f]{64} no results", second)


def test_cli_takes_the_repository_and_results_roots(repo, tmp_path, capsys):
    repo.finished()
    elsewhere = tmp_path / "results"
    args = ["is-current", str(repo.config), "--repo-root", str(repo.root)]
    assert main(args) == 0
    assert main([*args, "--results-root", str(elsewhere)]) == 1
    assert capsys.readouterr().out.splitlines()[-1].endswith(" no results")


def test_python_m_molscout_is_current_runs_in_its_own_process(repo):
    """run.sbatch calls `molscout is-current` as a command, so it must work from a fresh interpreter."""
    repo.finished()
    command = [sys.executable, "-m", "molscout", "is-current", OWN_CONFIG]
    result = subprocess.run(command, cwd=repo.root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith(f"{RUN} current ")
