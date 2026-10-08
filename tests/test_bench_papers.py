"""The benchmark harness runs a fake tool on paper datasets (whole PDFs), then checks, scores and records the run.

BioVista papers are the three benchmark papers of tests/fixtures/biovista; each PDF holds the ASCII bytes of its
PDB ID, which is what the fixture manifest's sha256 column pins. 6_ffff has unreadable labels, so it is dropped and
gets no PDF. The Internal ground truth is made here.
"""

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from molscout.bench import BenchError
from molscout.bench.harness import run_benchmark
from molscout.runs import read_paper_seconds, score_run

FIXTURES = Path(__file__).parent / "fixtures"
TOOLS = Path(__file__).parent.parent / "benchmarks" / "tools"
FILES = ["config.yaml", "errors.json", "meta.json", "predictions.csv", "scores.json", "timing.json"]
BIOVISTA_PDFS = {"1_aaaa": b"aaaa", "2_bbbb": b"bbbb", "3_cccc": b"cccc"}
BIOVISTA_ANSWERS = {
    "1_aaaa": [["CCO", 1, [0, 0, 10, 10]], ["c1ccccc1", None, None], ["CCCCl", 2, None], ["CCC", None, None]],
    "2_bbbb": [["CCN", 3, [1, 2, 3, 4]]],
}
INTERNAL_TRUTH = "paperID,name of molecule,canonical_SMILES\np1,ethanol,CCO\np1,benzene,c1ccccc1\np2,ethylamine,CCN\n"
INTERNAL_SPLIT = "paperID,split,molecules\np1,dev,2\np2,test,1\n"
INTERNAL_ANSWERS = {"p1": [["CCO", 1, None]], "p2": [["CCN", 1, None], ["CCC", 2, None]]}


class Workspace:
    """A repository root holding the fake paper tool, the real paper loop, BioVista and Internal data."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.results = root / "results"
        self.answers = root / "answers.json"
        self.markers = root / "markers"
        self.markers.mkdir(parents=True)
        self.lock_commands: list[list[str]] = []
        (root / "pyproject.toml").write_text('[project]\nname = "molscout"\n')
        (root / "benchmarks" / "tools").mkdir(parents=True)
        for name in ("paper_runner.py", "crop_runner.py"):
            shutil.copy(TOOLS / name, root / "benchmarks" / "tools" / name)
        shutil.copytree(
            FIXTURES / "fake_paper_tool", root / "tools" / "fake", ignore=shutil.ignore_patterns("__pycache__")
        )
        self.biovista = root / "data" / "raw" / "biovista"
        shutil.copytree(FIXTURES / "biovista", self.biovista)
        self.biovista_pdfs = self.biovista / "pdfs"
        self.biovista_pdfs.mkdir()
        for paper, content in BIOVISTA_PDFS.items():
            (self.biovista_pdfs / f"{paper.split('_')[1]}.pdf").write_bytes(content)
        self.internal = root / "data" / "internal"
        self.internal.mkdir(parents=True)
        (self.internal / "truth.csv").write_text(INTERNAL_TRUTH)
        (self.internal / "split.csv").write_text(INTERNAL_SPLIT)
        for paper in ("p1", "p2"):
            (self.internal / f"{paper}.pdf").write_bytes(b"%PDF-1.4 " + paper.encode())
        self.set_answers(BIOVISTA_ANSWERS)

    def set_answers(self, answers: dict) -> None:
        self.answers.write_text(json.dumps(answers))

    def config(self, dataset: str, *args: str, sources: tuple[Path, ...] = ()) -> Path:
        paths = (
            {"pdfs": "data/raw/biovista/pdfs", "papers": "data/raw/biovista/manifest.csv", "references": "data/raw/biovista"}
            if dataset == "biovista"
            else {"pdfs": "data/internal", "papers": "data/internal/split.csv", "references": "data/internal/truth.csv"}
        )
        data = {
            "tool": "fake",
            "name": "Fake",
            "version": "1.0 (test)",
            "dataset": dataset,
            **paths,
            "run_dir": "tools/fake",
            "python": sys.executable,
            "args": list(args),
            "checkpoints": [],
        }
        if self.lock_commands:
            data["lock_commands"] = self.lock_commands
        if sources:
            data["sources"] = [str(source) for source in sources]
        path = self.root / "configs" / f"fake__{dataset}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text("# test config\n" + yaml.safe_dump(data, sort_keys=False))
        return path

    def run(self, dataset: str, *args: str, sources: tuple[Path, ...] = ()) -> Path:
        config = self.config(dataset, *args, sources=sources)
        return run_benchmark(config, repo_root=self.root, results_root=self.results)

    def checkpoint(self, dataset: str) -> Path:
        return self.results / ".checkpoints" / f"fake__{dataset}"


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "repo")


GIT = ("git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false")


def rows_without_seconds(path: Path) -> list[dict[str, str]]:
    """The rows of a predictions.csv by column name, minus `seconds`, which is a real wall time."""
    with path.open(newline="") as handle:
        return [{k: v for k, v in row.items() if k != "seconds"} for row in csv.DictReader(handle)]


def make_clone(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "model.py").write_text("v1\n")
    subprocess.run([*GIT, "init", "-q", str(path)], check=True)
    subprocess.run([*GIT, "-C", str(path), "add", "model.py"], check=True)
    subprocess.run([*GIT, "-C", str(path), "commit", "-q", "-m", "one"], check=True)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_biovista_run_writes_six_files_and_scores_like_molscout_score(ws, monkeypatch):
    folder = ws.run("biovista", "--answers", str(ws.answers))
    assert folder == ws.results / "fake__biovista"
    assert sorted(p.name for p in folder.iterdir()) == FILES
    assert not ws.checkpoint("biovista").exists()
    timing = load(folder / "timing.json")
    assert timing["papers"] == 3 and sorted(timing["seconds"]) == list(BIOVISTA_PDFS)
    assert load(folder / "errors.json") == {"papers": 3, "failed": 0, "errors": {}}
    meta = load(folder / "meta.json")
    assert meta["items"] == 3
    assert meta["tool_errors"] == 0
    assert meta["timing"]["item_seconds_total"] == sum(timing["seconds"].values())
    pdfs = {paper: hashlib.sha256(content).hexdigest() for paper, content in BIOVISTA_PDFS.items()}
    lines = "".join(sorted(f"{digest}  {paper}\n" for paper, digest in pdfs.items()))
    assert meta["inputs"] == {
        "pdfs": str(ws.biovista_pdfs),
        "pdfs_sha256": hashlib.sha256(lines.encode()).hexdigest(),
        "papers": str(ws.biovista / "manifest.csv"),
        "papers_sha256": hashlib.sha256((ws.biovista / "manifest.csv").read_bytes()).hexdigest(),
    }
    assert meta["command"][1:9] == [
        "run.py",
        "--papers",
        meta["command"][3],
        "--dataset",
        "biovista",
        "--tool",
        "Fake 1.0 (test)",
        "--output",
    ]
    assert meta["command"][3].endswith("/papers.csv")
    # scores.json is what `molscout score` writes from the repository root, with the run's timing
    monkeypatch.chdir(ws.root)
    expected = score_run(
        Path("results/fake__biovista/predictions.csv"),
        references=Path("data/raw/biovista"),
        papers=Path("data/raw/biovista/manifest.csv"),
        paper_seconds=read_paper_seconds(folder / "timing.json"),
    )
    assert load(folder / "scores.json") == expected
    assert expected["scores"]["papers"]["1_aaaa"]["tp"] == 3  # CCO, benzene and CCCCl found; CCC is not a label


def test_paper_without_molecules_is_scored_and_timed(ws):
    folder = ws.run("biovista", "--answers", str(ws.answers))
    scores = load(folder / "scores.json")["scores"]
    assert scores["papers"]["3_cccc"]["tp"] == 0
    assert scores["papers"]["3_cccc"]["fn"] == 1  # its one label
    assert scores["groups"]["all"]["papers_without_output"] == 1
    assert scores["seconds_per_item"]["items"] == 3


def test_a_dropped_biovista_paper_needs_no_pdf_and_is_not_run(ws):
    assert not (ws.biovista_pdfs / "ffff.pdf").exists()
    folder = ws.run("biovista", "--answers", str(ws.answers))
    assert sorted(load(folder / "timing.json")["seconds"]) == list(BIOVISTA_PDFS)
    report = load(folder / "scores.json")
    assert sorted(report["scores"]["papers"]) == list(BIOVISTA_PDFS)
    assert report["inputs"]["references"]["dropped_papers"] == ["6_ffff"]


def test_a_dropped_biovista_papers_pdf_is_not_checked(ws):
    (ws.biovista_pdfs / "ffff.pdf").write_bytes(b"not the frozen copy")
    folder = ws.run("biovista", "--answers", str(ws.answers))
    assert load(folder / "meta.json")["items"] == 3


def test_biovista_labels_that_differ_from_the_manifest_stop_before_tool_runs(ws):
    path = ws.biovista / "bioactivity_extraction" / "labels" / "6_ffff_structure.csv"
    path.write_text(path.read_text() + "CCBr,10,NA,NA\n")
    record = ws.root / "record.json"
    with pytest.raises(BenchError, match=r"fake__biovista: paper 6_ffff: 3 label rows but the manifest froze 2"):
        ws.run("biovista", "--record", str(record))
    assert not record.exists()


def test_biovista_pdf_sha256_mismatch_stops_before_tool_runs(ws):
    (ws.biovista_pdfs / "bbbb.pdf").write_bytes(b"changed")
    record = ws.root / "record.json"
    with pytest.raises(BenchError, match=r"2_bbbb PDF .*bbbb\.pdf has sha256 .*never re-fetch") as error:
        ws.run("biovista", "--record", str(record))
    assert "but data/raw/biovista/manifest.csv pins 81cc5b17" in str(error.value)
    assert not record.exists()


def test_missing_pdf_stops_before_tool_runs(ws):
    (ws.internal / "p2.pdf").unlink()
    record = ws.root / "record.json"
    with pytest.raises(BenchError, match=r"p2.*p2\.pdf"):
        ws.run("internal", "--record", str(record))
    assert not record.exists()


def test_missing_timing_file_fails_the_run(ws):
    with pytest.raises(BenchError, match="predictions.timing.json"):
        ws.run("biovista", "--answers", str(ws.answers), "--drop-timing")
    assert not (ws.results / "fake__biovista").exists()


def test_timing_file_must_cover_the_runs_papers(ws):
    from molscout.bench import papers

    path = ws.root / "predictions.timing.json"
    path.write_text(json.dumps({"papers": 1, "seconds": {"1_aaaa": 0.5}}))
    with pytest.raises(BenchError, match="2_bbbb"):
        papers.read_timing(path, "fake__biovista", ["1_aaaa", "2_bbbb"])
    assert papers.read_timing(path, "fake__biovista", ["1_aaaa"]) == {"1_aaaa": 0.5}


def test_crashed_paper_is_counted_in_tool_errors(ws):
    folder = ws.run("biovista", "--answers", str(ws.answers), "--crash", "2_bbbb")
    assert load(folder / "errors.json") == {"papers": 3, "failed": 1, "errors": {"2_bbbb": "ValueError: fake crash"}}
    assert load(folder / "meta.json")["tool_errors"] == 1
    assert load(folder / "scores.json")["scores"]["papers"]["2_bbbb"]["tp"] == 0


def test_internal_run_reads_ground_truth_and_split_from_the_config(ws, monkeypatch):
    ws.set_answers(INTERNAL_ANSWERS)
    folder = ws.run("internal", "--answers", str(ws.answers))
    assert sorted(p.name for p in folder.iterdir()) == FILES
    report = load(folder / "scores.json")
    assert report["scores"]["papers"]["p1"]["tp"] == 1
    assert report["scores"]["papers"]["p2"]["fp"] == 1
    assert sorted(report["scores"]["groups"]) == ["all", "dev", "test"]
    meta = load(folder / "meta.json")
    assert meta["items"] == 2
    assert meta["inputs"]["papers"] == str(ws.internal / "split.csv")
    papers = [row.split(",")[1] for row in (folder / "predictions.csv").read_text().splitlines()[1:]]
    assert set(papers) == {"p1", "p2"}
    monkeypatch.chdir(ws.root)
    expected = score_run(
        Path("results/fake__internal/predictions.csv"),
        ground_truth=Path("data/internal/truth.csv"),
        split=Path("data/internal/split.csv"),
        paper_seconds=read_paper_seconds(folder / "timing.json"),
    )
    assert report == expected


def test_resumed_paper_run_matches_an_uninterrupted_one(ws):
    predicted = ws.root / "predicted.txt"
    args = ("--answers", str(ws.answers), "--crash", "2_bbbb", "--stop-once", str(ws.markers / "stopped"))
    args += ("--predicted", str(predicted))
    with pytest.raises(BenchError, match="exited with status 3"):
        ws.run("biovista", *args)
    assert not (ws.results / "fake__biovista").exists()
    log = (ws.checkpoint("biovista") / "predictions.papers.jsonl").read_text().splitlines()
    assert sum('"attempted"' not in line for line in log) == 1  # one paper is done; attempted lines do not count
    folder = ws.run("biovista", *args)
    assert predicted.read_text().split() == ["1_aaaa", "2_bbbb", "3_cccc"]  # the second run did the other two
    assert not ws.checkpoint("biovista").exists()

    assert len(load(folder / "meta.json")["segments"]) == 2
    resumed = {
        "rows": rows_without_seconds(folder / "predictions.csv"),
        "errors": load(folder / "errors.json"),
        "scores": {k: v for k, v in load(folder / "scores.json")["scores"].items() if k != "seconds_per_item"},
        "papers_timed": sorted(load(folder / "timing.json")["seconds"]),
    }
    clean = ws.run("biovista", *args)  # the marker exists, so this run is never stopped
    assert rows_without_seconds(clean / "predictions.csv") == resumed["rows"]
    assert load(clean / "errors.json") == resumed["errors"]
    assert {k: v for k, v in load(clean / "scores.json")["scores"].items() if k != "seconds_per_item"} == resumed["scores"]
    assert sorted(load(clean / "timing.json")["seconds"]) == resumed["papers_timed"]
    assert len(load(clean / "meta.json")["segments"]) == 1


@pytest.mark.parametrize("change", ["none", "edit", "commit"])
def test_checkpoint_is_keyed_on_the_source_clones(ws, tmp_path, change):
    clone = tmp_path / "upstream"
    make_clone(clone)
    predicted = ws.root / "predicted.txt"
    args = ("--answers", str(ws.answers), "--stop-once", str(ws.markers / "stopped"), "--predicted", str(predicted))
    with pytest.raises(BenchError, match="exited with status 3"):
        ws.run("biovista", *args, sources=(clone,))
    if change == "edit":
        (clone / "model.py").write_text("v2\n")
    elif change == "commit":
        (clone / "model.py").write_text("v2\n")
        subprocess.run([*GIT, "-C", str(clone), "commit", "-q", "-am", "two"], check=True)
    folder = ws.run("biovista", *args, sources=(clone,))
    again = predicted.read_text().split()
    if change == "none":
        assert again == ["1_aaaa", "2_bbbb", "3_cccc"]  # carried on from the checkpoint
        assert len(load(folder / "meta.json")["segments"]) == 2
    else:
        assert again == ["1_aaaa", "1_aaaa", "2_bbbb", "3_cccc"]  # started over
        assert len(load(folder / "meta.json")["segments"]) == 1


def test_a_subfolder_of_a_clone_is_not_a_source(ws, tmp_path):
    clone = tmp_path / "upstream"
    make_clone(clone)
    (clone / "sub").mkdir()
    record = ws.root / "record.json"
    with pytest.raises(BenchError, match=r"fake__biovista: source .*sub is not the root of a git clone"):
        ws.run("biovista", "--record", str(record), sources=(clone / "sub",))
    assert not record.exists()


def test_internal_results_are_an_allowlist_and_biovista_ones_are_not_ignored():
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    root = Path(__file__).parent.parent

    def ignored(path: str) -> bool:
        return subprocess.run(["git", "-C", str(root), "check-ignore", "-q", path]).returncode == 0

    assert ignored("benchmarks/results/foo__internal/predictions.csv")
    assert ignored("benchmarks/results/foo__internal/stray.txt")
    for name in ("scores.json", "meta.json", "config.yaml", "timing.json", "errors.json"):
        assert not ignored(f"benchmarks/results/foo__internal/{name}")
    for name in ("predictions.csv", "errors.json", "stray.txt", "scores.json"):
        assert not ignored(f"benchmarks/results/foo__biovista/{name}")


def test_internal_errors_json_keeps_only_the_exception_types(ws):
    ws.set_answers(INTERNAL_ANSWERS)
    secret = "CC(C)Nc1ccc(Br)cc1"
    message = ws.root / "message.txt"
    message.write_text(f"bad {secret}: {secret}")
    folder = ws.run("internal", "--answers", str(ws.answers), "--crash", "p2", "--crash-message-file", str(message))
    assert load(folder / "errors.json") == {"papers": 2, "failed": 1, "errors": {"p2": "ValueError"}}
    assert load(folder / "meta.json")["tool_errors"] == 1
    for path in folder.iterdir():
        assert secret not in path.read_text(), path.name


def test_biovista_errors_json_keeps_the_full_message(ws):
    message = ws.root / "message.txt"
    message.write_text("a: b")
    folder = ws.run("biovista", "--answers", str(ws.answers), "--crash", "2_bbbb", "--crash-message-file", str(message))
    assert load(folder / "errors.json")["errors"] == {"2_bbbb": "ValueError: a: b"}


def test_files_beside_the_output_are_removed_with_a_warning(ws, capsys):
    folder = ws.run("biovista", "--answers", str(ws.answers), "--extra-file", "extra.json")
    assert sorted(p.name for p in folder.iterdir()) == FILES
    err = capsys.readouterr().err
    assert err.count("unexpected file") == 1 and "extra.json" in err


def commit_all(root: Path) -> None:
    subprocess.run([*GIT, "init", "-q", str(root)], check=True)
    subprocess.run([*GIT, "-C", str(root), "add", "-A"], check=True)
    subprocess.run([*GIT, "-C", str(root), "commit", "-q", "-m", "one"], check=True)


@pytest.mark.parametrize("change", ["none", "lock", "edit"])
def test_paper_checkpoint_is_keyed_on_the_environment_lock_and_tracked_edits(ws, change):
    lock = ws.root / "lock.txt"
    lock.write_text("v1\n")
    ws.lock_commands = [["cat", str(lock)]]
    commit_all(ws.root)
    predicted = ws.root / "predicted.txt"
    args = ("--answers", str(ws.answers), "--stop-once", str(ws.markers / "stopped"), "--predicted", str(predicted))
    with pytest.raises(BenchError, match="exited with status 3"):
        ws.run("biovista", *args)
    if change == "lock":
        lock.write_text("v2\n")
    elif change == "edit":
        with (ws.root / "benchmarks" / "tools" / "paper_runner.py").open("a") as handle:
            handle.write("# edited\n")
    folder = ws.run("biovista", *args)
    again = predicted.read_text().split()
    if change == "none":
        assert again == ["1_aaaa", "2_bbbb", "3_cccc"]
        assert len(load(folder / "meta.json")["segments"]) == 2
    else:
        assert again == ["1_aaaa", "1_aaaa", "2_bbbb", "3_cccc"]
        assert len(load(folder / "meta.json")["segments"]) == 1


@pytest.mark.parametrize(
    ("change", "resumes"),
    [
        ("benchmarks/tools/complete_systems/biominer/x.py", True),  # another tool's folder
        ("README.md", True),
        ("src/molscout/x.py", False),
        ("benchmarks/tools/paper_runner.py", False),
        ("tools/fake/x.py", False),  # the run's own tool folder
    ],
)
def test_paper_checkpoint_is_kept_across_commits_that_leave_the_runs_code_alone(ws, change, resumes):
    predicted = ws.root / "predicted.txt"
    args = ("--answers", str(ws.answers), "--stop-once", str(ws.markers / "stopped"), "--predicted", str(predicted))
    ws.config("biovista", *args)
    commit_all(ws.root)
    with pytest.raises(BenchError, match="exited with status 3"):
        ws.run("biovista", *args)
    first = load(ws.checkpoint("biovista") / "state.json")["code_fingerprint"]
    assert first is not None and len(first) == 64
    path = ws.root / change
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write("# a later commit\n")
    subprocess.run([*GIT, "-C", str(ws.root), "add", change], check=True)
    subprocess.run([*GIT, "-C", str(ws.root), "commit", "-q", "-m", "two"], check=True)
    folder = ws.run("biovista", *args)
    again = predicted.read_text().split()
    meta = load(folder / "meta.json")
    assert meta["git"]["dirty"] is False
    if resumes:
        assert again == ["1_aaaa", "2_bbbb", "3_cccc"]
        assert len(meta["segments"]) == 2
        assert meta["git"]["code_fingerprint"] == first
    else:
        assert again == ["1_aaaa", "1_aaaa", "2_bbbb", "3_cccc"]
        assert len(meta["segments"]) == 1
        assert meta["git"]["code_fingerprint"] != first
