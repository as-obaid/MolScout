"""Whether a run's results are current: made from the code the run has at HEAD, with none of it uncommitted.

A run's code is every file in the repository that can change its results: the harness (GIT_PATHS), the
run's tool folder and its config (run_paths). Its code fingerprint at a commit is the sha256 of the blob
IDs and paths that `git ls-tree -r <commit>` lists under those paths, so it can be computed for any
commit, including the one an older meta.json records. A finished run therefore stays current across
commits that leave its code alone, such as one that adds another tool. benchmarks/slurm/run.sbatch asks
`molscout is-current` before it runs a config, and paper checkpoints are keyed on the fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from molscout.bench.config import RunConfig, load_config
from molscout.bench.meta import git_state

# The harness's own code: a change to any of these can change every run's results
GIT_PATHS = (
    "src",
    "pyproject.toml",
    "benchmarks/tools/crop_runner.py",
    "benchmarks/tools/paper_runner.py",
    "benchmarks/slurm/run.sbatch",
)
COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")  # a SHA-1 or SHA-256 object name
GIT_TIMEOUT_SECONDS = 60
PATHS_SHOWN = 3


def run_paths(config: RunConfig, repo_root: Path) -> list[Path]:
    """The files that can change a run's results: its tool folder, its config and the harness (GIT_PATHS)."""
    return [config.run_dir, config.path, *(repo_root / path for path in GIT_PATHS)]


def code_fingerprint(repo_root: Path, commit: str | None, paths: Sequence[Path]) -> str | None:
    """The sha256 of the run's code at `commit` (see code_tree); None without a commit git can list."""
    tree = None if commit is None else code_tree(repo_root, commit, paths)
    return None if tree is None else _fingerprint(tree)


def code_tree(repo_root: Path, commit: str, paths: Sequence[Path]) -> dict[str, str] | None:
    """{path: blob ID} for every file under `paths` at `commit`, from `git ls-tree -r`.

    Paths outside the repository are left out, as git_state leaves them out. None when `commit` is not
    an object name or git cannot list it (it is not in this repository, or there is no git).
    """
    git = shutil.which("git")
    if git is None or not COMMIT.fullmatch(commit):
        return None
    inside = [relative for relative in (_relative(path, repo_root) for path in paths) if relative is not None]
    if not inside:
        return {}
    command = [git, "-C", str(repo_root), "ls-tree", "-r", "-z", commit, "--", *inside]
    try:
        result = subprocess.run(command, capture_output=True, timeout=GIT_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    tree = {}
    for record in result.stdout.decode("utf-8", errors="surrogateescape").split("\0"):
        if record:
            info, path = record.split("\t", 1)  # "<mode> <type> <object>\t<path>"
            tree[path] = info.split(" ")[2]
    return tree


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether a run is current, why, and its code fingerprint at HEAD (None when it cannot be computed)."""

    run: str
    current: bool
    fingerprint: str | None
    why: str

    def line(self) -> str:
        """`<run> current|not-current <fingerprint or -> <why>`, the line `molscout is-current` prints.

        run.sbatch and status.sh read it with `read -r run verdict code why`, so it is always one line.
        """
        verdict = "current" if self.current else "not-current"
        return f"{self.run} {verdict} {self.fingerprint or '-'} {_one_line(self.why)}"


def check_current(config_path: str | Path, *, repo_root: str | Path, results_root: str | Path) -> Verdict:
    """Whether results/<run>/ is current: its meta.json records a commit made with no uncommitted changes, the run's
    code has the same fingerprint at that commit as at HEAD, and none of it has uncommitted changes now."""
    root, results = Path(repo_root).absolute(), Path(results_root).absolute()
    run = Path(config_path).stem
    try:
        config = load_config(config_path, root)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return Verdict(run, False, None, f"cannot read the config: {_one_line(str(exc))}")
    run = config.run_name
    paths = run_paths(config, root)
    git = git_state(root, paths)
    head = git["commit"]
    if not isinstance(head, str):
        return Verdict(run, False, None, "not in a git repository")
    tree = code_tree(root, head, paths)
    if tree is None:
        return Verdict(run, False, None, f"git cannot list HEAD {head[:7]}")
    fingerprint = _fingerprint(tree)

    def stale(why: str) -> Verdict:
        return Verdict(run, False, fingerprint, why)

    meta = _read_meta(results / run / "meta.json")
    if isinstance(meta, str):
        return stale(meta)
    made = meta.get("commit")
    if not isinstance(made, str) or not COMMIT.fullmatch(made):
        return stale("meta.json records no commit")
    if meta.get("dirty") is True:
        return stale("made with uncommitted changes")
    if meta.get("dirty") is not False:
        return stale("meta.json does not say whether its code was committed")
    then = code_tree(root, made, paths)
    if then is None:
        return stale(f"made at {made[:7]}, which is not in this repository")
    changed = sorted(path for path in then.keys() | tree.keys() if then.get(path) != tree.get(path))
    if changed:
        return stale(f"code changed since {made[:7]}: {_listed(changed)}")
    if git["dirty"] is not False:
        dirty = git["dirty_paths"]
        return stale(f"uncommitted changes: {_listed(dirty)}" if dirty else "git status failed")
    return Verdict(run, True, fingerprint, f"made at {made[:7]}")


def _read_meta(path: Path) -> Mapping[str, object] | str:
    """meta.json's `git` record, or why there is none."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "no results"
    except (OSError, UnicodeDecodeError):
        return "meta.json is unreadable"
    try:
        meta = json.loads(text)
    except ValueError:
        return "meta.json is unreadable"
    git = meta.get("git") if isinstance(meta, dict) else None
    return git if isinstance(git, dict) else "meta.json records no commit"


def _fingerprint(tree: Mapping[str, str]) -> str:
    lines = "".join(f"{blob} {path}\n" for path, blob in sorted(tree.items()))
    return hashlib.sha256(lines.encode("utf-8", errors="surrogateescape")).hexdigest()


def _relative(path: Path, root: Path) -> str | None:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _listed(paths: Sequence[str]) -> str:
    shown = ", ".join(paths[:PATHS_SHOWN])
    return shown if len(paths) <= PATHS_SHOWN else f"{shown} and {len(paths) - PATHS_SHOWN} more"


def _one_line(text: str) -> str:
    return " ".join(text.split())
