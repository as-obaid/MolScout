"""Run the pipeline over papers stage by stage, so each model loads once for all papers; resume per paper and stage.

Each paper works in `<out>/papers/<paper_id>/`. A stage that finishes for a paper writes `.done/<stage>.json`
with its key (stages.stage_keys) and its counts; a later run with the same key skips it. Worker stages run as one
process for all papers that need them: `<python> <worker> --manifest <out>/logs/<stage>.manifest.jsonl --log
<out>/logs/<stage>.log [args]`. A worker writes per-paper failures to `<paper folder>/errors.jsonl` and exits
0 (all papers done), 4 (some papers failed), 1 (model or setup failure), 2 (bad manifest) or 3 (the 32B server
was lost). In-process steps (steps.py) may return counts; a worker may write a paper's counts to
`<paper folder>/.counts/<stage>.json` (a JSON object of integers), which the runner removes before the worker runs.

Failures are loud. A worker exit other than 0 or 4, a step that raises, or a worker that writes no output file for
a paper stops that paper. A stage given records for a paper that writes none (unless the stage is in allow_empty,
or in FILTERS), counts a step or worker reports that do not add up (in != out + dropped) or cannot be read, and every
line a stage adds to errors.jsonl (filed under the stage the line names; short tables among them) are recorded
errors; the paper goes on but counts as failed. Lines of a DIAGNOSTICS kind are counted under `diagnostics`. `stages.json` records every stage of every paper: records in and
out, the step's counts, seconds, peak GPU memory (nvidia-smi while a worker runs; null when unknown) and errors.

With `shared`, render, parse, segment, label OCR and the readers run in `<shared>/<upstream key>/<paper_id>/`, which every
run with the same upstream settings reuses (the ladder and joiner variants), and the paper folder links to it.
Every folder a stage gets is named by its paper ID, so a stage may read the ID from the folder name.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import traceback
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import molscout
from molscout.bench.meta import GpuSampler
from molscout.pipeline import export
from molscout.pipeline import stages as st
from molscout.pipeline.config import PipelineConfig
from molscout.pipeline.inputs import Paper
from molscout.pipeline.steps import STEPS, Step

DONE = ".done"
LOCK = ".lock"
COUNTS = ".counts"  # a worker may write its counts for a paper to <paper folder>/.counts/<stage>.json
ERRORS_FILE = "errors.jsonl"
STAGES_FILE = "stages.json"
NOT_LINKED = frozenset({DONE, LOCK, COUNTS, ERRORS_FILE, STAGES_FILE})
DROPPED_VARIABLES = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
# Stages that may give none out: every vote of a paper may fail (ties), the 32B join may then find nothing to label
# (its failed requests are errors of their own), finalize and export filter, and label OCR may find no text in a
# paper's figures.
FILTERS = frozenset({"vote", "join_32b", "finalize", "export", "ocr_labels"})
NONE_OUT = "{} record(s) in, none out"
_NONE_OUT = re.compile(r"\d+ record\(s\) in, none out")
# A worker's exit status: 0 ok; 4 some papers failed (their errors are in errors.jsonl; the others go on); any
# other status stops every paper it had.
WORKER_STATUS = {1: "a model or setup failure", 2: "a bad manifest or bad arguments", 3: "the 32B server was lost"}
# errors.jsonl lines of these kinds are diagnostics, not failures: the worker still wrote the paper's output
# (join_32b: answer entries that were malformed or named a box the image did not show, a truncated answer).
DIAGNOSTICS = frozenset({"malformed", "not_shown", "truncated"})
PARTIAL = 4
STOP_GRACE_SECONDS = 10
ENDPOINT_TIMEOUT_SECONDS = 10
Echo = Callable[[str], None]


class PipelineError(RuntimeError):
    """The run cannot go on at all, such as a shared folder another run holds."""


@dataclass
class StageRecord:
    """One stage of one paper."""

    records_in: int = 0
    records_out: int = 0
    seconds: float = 0.0
    peak_gpu_mb: float | None = None
    errors: list[str] = field(default_factory=list)
    counts: dict[str, int] | None = None
    diagnostics: dict[str, int] = field(default_factory=dict)
    resumed: bool = False


@dataclass
class PaperWork:
    paper: Paper
    folder: Path
    upstream: Path
    keys: dict[str, str]
    stages: dict[str, StageRecord] = field(default_factory=dict)
    stopped: str | None = None

    def folder_for(self, stage: str) -> Path:
        return self.upstream if st.is_upstream(stage) else self.folder


@dataclass(frozen=True)
class RunSummary:
    """What a run did: per paper its status (ok, failed or stopped), seconds, errors, stages and funnel."""

    out: Path
    papers: Mapping[str, Mapping[str, Any]]
    stages: Mapping[str, Mapping[str, Any]]
    funnel: Mapping[str, int]

    @property
    def failed(self) -> list[str]:
        return [paper for paper, record in self.papers.items() if record["status"] != "ok"]

    @property
    def ok(self) -> bool:
        return not self.failed

    def seconds(self) -> dict[str, float]:
        return {paper: record["seconds"] for paper, record in self.papers.items()}


def run(
    papers: Sequence[Paper],
    out: str | Path,
    config: PipelineConfig,
    *,
    steps: Mapping[str, Step] | None = None,
    dataset: str | None = None,
    tool: str | None = None,
    echo: Echo | None = None,
    before_stage: Callable[[str], None] | None = None,
) -> RunSummary:
    """Run every stage on every paper; write stages.json, molecules.csv and, with `dataset`, predictions.csv.

    `steps` replaces in-process steps by name (tests use it). `before_stage(stage)` is called before a stage that
    has papers to run, such as to start the vLLM server only when the 32B joiner needs it. PipelineError when a
    shared folder is held.
    """
    out = Path(out).absolute()
    say = echo or _stderr
    (out / "papers").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    table = {**STEPS, **(steps or {})}
    work = [_prepare(paper, out, config) for paper in papers]
    names = config.stage_names()
    upstream = [stage for stage in names if st.is_upstream(stage)]
    with _locked(work, config, upstream):
        for stage in upstream:
            _run_stage(stage, work, config, table, out, say, before_stage)
    if config.shared is not None:
        for item in work:
            if item.stopped is None:
                _link_upstream(item)
    for stage in names:
        if not st.is_upstream(stage):
            _run_stage(stage, work, config, table, out, say, before_stage)
    summary = _summary(work, config, out)
    _write_stages(out, work, config, summary)
    kept = {item.paper.paper_id: item.folder / config.output("finalize") for item in work if item.stopped is None}
    export.write_molecules_csv(out / "molecules.csv", kept.values())
    if dataset is not None:
        label = tool or f"MolScout {molscout.__version__}"
        rows = export.predictions(kept, config, dataset=dataset, tool=label, seconds=summary.seconds())
        export.write_predictions_csv(out / "predictions.csv", rows)
    failed = summary.failed
    say(f"{len(work)} papers, {len(failed)} failed" + (f": {', '.join(failed)}" if failed else ""))
    return summary


def _prepare(paper: Paper, out: Path, config: PipelineConfig) -> PaperWork:
    folder = out / "papers" / paper.paper_id
    keys = st.stage_keys(config, paper.sha256())
    upstream = folder if config.shared is None else config.shared / st.upstream_key(keys)[:16] / paper.paper_id
    for path in (folder, upstream):
        if path.name != paper.paper_id:
            raise PipelineError(f"{path}: a paper folder must be named by its paper ID {paper.paper_id!r}")
        path.mkdir(parents=True, exist_ok=True)
    return PaperWork(paper, folder, upstream, keys)


def _run_stage(
    stage: str,
    work: list[PaperWork],
    config: PipelineConfig,
    table: Mapping[str, Step],
    out: Path,
    say: Echo,
    before_stage: Callable[[str], None] | None,
) -> None:
    active = [item for item in work if item.stopped is None]
    todo = []
    for item in active:
        marker = _read_marker(item.folder_for(stage), stage)
        if marker is not None and marker.pop("key", None) == item.keys[stage]:
            item.stages[stage] = _resumed(StageRecord(**{**marker, "resumed": True}), stage, config)
        else:
            todo.append(item)
    say(f"{stage}: {len(todo)} paper(s) to run, {len(active) - len(todo)} done before")
    if stage == "vote":
        todo = [item for item in todo if _reads_intact(item, config, say)]
    if not todo:
        return
    if before_stage is not None:
        before_stage(stage)
    before = {item.paper.paper_id: _error_lines(item.folder_for(stage)) for item in todo}
    given = {item.paper.paper_id: st.records_in(stage, item.folder_for(stage), config) for item in todo}
    if stage in config.workers:
        _clear_outputs(stage, todo, config)
        results = _run_worker(stage, todo, config, out, given, say)
    else:
        results = {item.paper.paper_id: _run_step(table[stage], item, stage, config, out) for item in todo}
    for item in todo:
        pid = item.paper.paper_id
        seconds, peak, stop, counts = results[pid]
        folder = item.folder_for(stage)
        problem = None
        if stage in config.workers:
            counts, problem = _worker_counts(folder, stage)
        record = StageRecord(records_in=given[pid], seconds=seconds, peak_gpu_mb=peak, counts=counts)
        if problem is not None:
            record.errors.append(problem)
        for named, message, kind, count in _new_errors(folder, before[pid]):
            target = record if named in (None, stage) or named not in item.stages else item.stages[named]
            if kind in DIAGNOSTICS:
                target.diagnostics[kind] = target.diagnostics.get(kind, 0) + count
            elif target is record:
                record.errors.append(message if named in (None, stage) else f"[{named}] {message}")
            else:
                target.errors.append(message)
        record.errors += _count_problems(counts)
        if stop is None and stage in config.workers and not (folder / config.output(stage)).is_file():
            stop = f"the worker wrote no {config.output(stage)} for this paper"
        if stop is not None:
            record.errors.append(stop)
            item.stopped = stage
        else:
            record.records_out = st.records_out(stage, folder, config)
            empty_ok = stage in config.allow_empty or stage in FILTERS
            if record.records_in and not record.records_out and not empty_ok:
                record.errors.append(NONE_OUT.format(record.records_in))
            _write_marker(folder, stage, item.keys[stage], record)
        item.stages[stage] = record
        for error in record.errors:
            say(f"{pid}: {stage}: {error}")


def _run_step(step: Step, item: PaperWork, stage: str, config: PipelineConfig, out: Path) -> tuple:
    """(seconds, peak GPU MB (None: in-process), the error that stops the paper or None, the step's counts)."""
    clock = perf_counter()
    counts = None
    try:
        returned = step(item.paper, item.folder_for(stage), config)
        counts = None if returned is None else {str(key): value for key, value in dict(returned).items()}
        error = None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        with (out / "logs" / f"{stage}.log").open("a", encoding="utf-8") as log:
            log.write(f"{item.paper.paper_id}:\n{traceback.format_exc()}\n")
    return perf_counter() - clock, None, error, counts


def _worker_counts(folder: Path, stage: str) -> tuple[dict[str, int] | None, str | None]:
    """The counts a worker wrote for a paper (`.counts/<stage>.json`, a JSON object of integers), and the problem
    with that file when it cannot be read; (None, None) when the worker writes none."""
    path = folder / COUNTS / f"{stage}.json"
    if not path.is_file():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"cannot read the worker's counts {path}: {exc}"
    if not isinstance(data, dict) or not all(type(value) is int for value in data.values()):
        return None, f"the worker's counts {path} are not a JSON object of integers"
    return {str(key): value for key, value in data.items()}, None


def _count_problems(counts: Mapping[str, int] | None) -> list[str]:
    """A step's counts must add up: in == out + dropped (+ every dropped_<why>) when it reports in and out."""
    if not counts or "in" not in counts or "out" not in counts:
        return []
    dropped = sum(value for key, value in counts.items() if key == "dropped" or key.startswith("dropped_"))
    if counts["in"] == counts["out"] + dropped:
        return []
    return [f"counts do not add up: in {counts['in']} != out {counts['out']} + dropped {dropped}"]


def _run_worker(
    stage: str, todo: list[PaperWork], config: PipelineConfig, out: Path, given: Mapping[str, int], say: Echo
) -> dict[str, tuple]:
    """Run one worker process for every paper in `todo`; each paper's share of the time, the GPU peak and any stop."""
    spec = config.workers[stage]
    manifest, log = out / "logs" / f"{stage}.manifest.jsonl", out / "logs" / f"{stage}.log"
    lines = [
        json.dumps({"paper_id": i.paper.paper_id, "pdf": str(i.paper.pdf), "paper_dir": str(i.folder_for(stage))})
        for i in todo
    ]
    manifest.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    problem = None
    if not spec.python.is_file():
        problem = f"worker Python not found: {spec.python}"
    elif not spec.worker.is_file():
        problem = f"worker script not found: {spec.worker}"
    elif stage == "join_32b" and config.check_endpoint:
        problem = _endpoint_problem(str(config.vllm_url))
    if problem is not None:
        return {item.paper.paper_id: (0.0, None, problem, None) for item in todo}
    command = [str(spec.python), str(spec.worker), "--manifest", str(manifest), "--log", str(log)]
    command += config.worker_args(stage)
    say(f"{stage}: {' '.join(command)}")
    clock = perf_counter()
    status, peak = _call(command, {**_environment(), **spec.env}, out)
    seconds = perf_counter() - clock
    stop = None
    if status not in (0, PARTIAL):
        why = WORKER_STATUS.get(status, "an unexpected status")
        stop = f"the {stage} worker exited with status {status} ({why}); see {log}"
    elif status == PARTIAL:
        say(f"{stage}: the worker reports that some papers failed; their errors are recorded")
    total = sum(given.values())
    shares = {
        pid: seconds * (count / total if total else 1 / len(todo)) for pid, count in given.items()
    }
    return {item.paper.paper_id: (shares[item.paper.paper_id], peak, stop, None) for item in todo}


def _call(command: list[str], env: Mapping[str, str], cwd: Path) -> tuple[int, float | None]:
    """Run a worker and wait; on any interruption stop it (TERM, then KILL). Its exit status and the GPU peak."""
    with GpuSampler() as gpu:
        process = subprocess.Popen(command, cwd=cwd, env=dict(env))
        try:
            status = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=STOP_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    summary = gpu.summary()
    return status, None if summary is None else float(summary["peak_memory_mib"])


def _environment() -> dict[str, str]:
    inherited = {name: value for name, value in os.environ.items() if name not in DROPPED_VARIABLES}
    return {**inherited, "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1"}


def _endpoint_problem(url: str) -> str | None:
    """None when the vLLM server lists its models, else why not. Proxy variables are ignored: the server is local,
    and a cluster's proxy cannot reach the node's 127.0.0.1."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url.rstrip("/") + "/models", timeout=ENDPOINT_TIMEOUT_SECONDS) as response:
            json.loads(response.read())
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return f"the vLLM endpoint {url} does not answer: {exc}"
    return None


def _clear_outputs(stage: str, todo: list[PaperWork], config: PipelineConfig) -> None:
    """Remove what an earlier attempt at this stage left; in a file several readers share, only this reader's reads."""
    name = config.output(stage)
    shared = [other for other in config.worker_stages() if other != stage and config.output(other) == name]
    for item in todo:
        (item.folder_for(stage) / COUNTS / f"{stage}.json").unlink(missing_ok=True)
        path = item.folder_for(stage) / name
        if not path.is_file():
            continue
        if shared and stage.startswith("read_"):
            reader = stage.removeprefix("read_")
            kept = [record for record in st.read_records(path) if record.get("reader") != reader]
            path.write_text("".join(json.dumps(record) + "\n" for record in kept), encoding="utf-8")
        else:
            path.unlink()


def _reads_intact(item: PaperWork, config: PipelineConfig, say: Echo) -> bool:
    """Each reader's reads are still all there; a worker that overwrote another's stops the paper."""
    for stage in config.reader_stages():
        recorded = item.stages[stage].records_out
        now = st.records_out(stage, item.folder, config)
        if now != recorded:
            message = f"{stage} wrote {recorded} read(s) but {now} are left; a reader overwrote another's reads"
            item.stages.setdefault("vote", StageRecord()).errors.append(message)
            item.stopped = "vote"
            say(f"{item.paper.paper_id}: vote: {message}")
            return False
    return True


def _error_lines(folder: Path) -> Counter[str]:
    path = folder / ERRORS_FILE
    return Counter(path.read_text(encoding="utf-8").splitlines()) if path.is_file() else Counter()


def _new_errors(folder: Path, before: Counter[str]) -> list[tuple[str | None, str, str | None, int]]:
    """(stage named by the line or None, message, kind or None, count) for each errors.jsonl line the stage added;
    count is the line's `count` (answer entries for a diagnostic), else 1. Lines are compared by content, so a
    worker that clears its own old lines and writes new ones is read right."""
    errors = []
    for line in (_error_lines(folder) - before).elements():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            record = line
        if isinstance(record, dict):
            stage, kind, count = record.get("stage"), record.get("kind"), record.get("count")
            message = str(record.get("error") or record.get("message") or json.dumps(record))
            errors.append((
                stage if isinstance(stage, str) else None,
                message,
                kind if isinstance(kind, str) else None,
                count if isinstance(count, int) and not isinstance(count, bool) and count > 0 else 1,
            ))
        elif str(record).strip():
            errors.append((None, str(record), None, 1))
    return errors


def _resumed(record: StageRecord, stage: str, config: PipelineConfig) -> StageRecord:
    """A done stage's record as its marker kept it, without a "none out" error the empty-output rule now allows:
    the rule is the code's, not the run's."""
    if stage not in config.allow_empty and stage not in FILTERS:
        return record
    return replace(record, errors=[error for error in record.errors if not _NONE_OUT.fullmatch(error)])


def _read_marker(folder: Path, stage: str) -> dict[str, Any] | None:
    path = folder / DONE / f"{stage}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    fields = set(StageRecord.__dataclass_fields__) - {"resumed"}
    if not isinstance(data, dict) or set(data) != fields | {"key"}:
        return None
    return data


def _write_marker(folder: Path, stage: str, key: str, record: StageRecord) -> None:
    (folder / DONE).mkdir(exist_ok=True)
    data = {"key": key, **{k: v for k, v in asdict(record).items() if k != "resumed"}}
    path = folder / DONE / f"{stage}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def _locked(work: list[PaperWork], config: PipelineConfig, upstream: list[str]) -> Iterator[None]:
    """Hold each shared upstream folder this run computes in, so two runs never write one folder at once."""
    held: list[Path] = []
    try:
        if config.shared is not None:
            for item in work:
                pending = any(
                    (_read_marker(item.upstream, stage) or {}).get("key") != item.keys[stage] for stage in upstream
                )
                if pending:
                    held.append(_acquire(item.upstream / LOCK))
        yield
    finally:
        for lock in held:
            shutil.rmtree(lock, ignore_errors=True)


def _acquire(lock: Path) -> Path:
    owner = {"host": socket.gethostname(), "pid": os.getpid(), "job": os.environ.get("SLURM_JOB_ID")}
    for _ in range(2):
        try:
            lock.mkdir()
        except FileExistsError:
            held = _lock_owner(lock)
            if held.get("host") == owner["host"] and isinstance(held.get("pid"), int) and not _alive(held["pid"]):
                shutil.rmtree(lock, ignore_errors=True)  # a run on this host that has ended left it
                continue
            raise PipelineError(
                f"{lock} is held by {held or 'a run that has not written its owner yet'}; another run is computing "
                "these stages, or a stopped run left the lock: remove the folder if no run uses it"
            ) from None
        (lock / "owner.json").write_text(json.dumps(owner) + "\n", encoding="utf-8")
        return lock
    raise PipelineError(f"cannot take the lock {lock}")


def _lock_owner(lock: Path) -> dict[str, Any]:
    for _ in range(3):
        try:
            data = json.loads((lock / "owner.json").read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            sleep(0.1)
    return {}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _link_upstream(item: PaperWork) -> None:
    """Link every file and folder of the shared upstream folder into the paper folder."""
    for entry in sorted(item.upstream.iterdir()):
        if entry.name in NOT_LINKED:
            continue
        target = item.folder / entry.name
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        target.symlink_to(entry)


def _summary(work: list[PaperWork], config: PipelineConfig, out: Path) -> RunSummary:
    papers: dict[str, dict[str, Any]] = {}
    for item in work:
        errors = [{"stage": stage, "error": e} for stage, record in item.stages.items() for e in record.errors]
        status = "stopped" if item.stopped else "failed" if errors else "ok"
        papers[item.paper.paper_id] = {
            "status": status,
            "stopped_at": item.stopped,
            "seconds": sum(record.seconds for record in item.stages.values()),
            "errors": errors,
            "stages": {stage: _record_dict(record) for stage, record in item.stages.items()},
            "funnel": st.funnel(item.folder, config),
        }
    totals: dict[str, dict[str, Any]] = {}
    for stage in config.stage_names():
        records = [item.stages[stage] for item in work if stage in item.stages]
        peaks = [r.peak_gpu_mb for r in records if r.peak_gpu_mb is not None]
        diagnostics: Counter[str] = Counter()
        for r in records:
            diagnostics.update(r.diagnostics)
        totals[stage] = {
            "records_in": sum(r.records_in for r in records),
            "records_out": sum(r.records_out for r in records),
            "seconds": sum(r.seconds for r in records),
            "peak_gpu_mb": max(peaks) if peaks else None,
            "errors": sum(len(r.errors) for r in records),
            "diagnostics": dict(sorted(diagnostics.items())),
            "papers": len(records),
            "resumed": sum(r.resumed for r in records),
        }
    funnel: dict[str, int] = {}
    for record in papers.values():
        for name, count in record["funnel"].items():
            funnel[name] = funnel.get(name, 0) + count
    return RunSummary(out, papers, totals, funnel)


def _record_dict(record: StageRecord) -> dict[str, Any]:
    return asdict(record)


def _write_stages(out: Path, work: list[PaperWork], config: PipelineConfig, summary: RunSummary) -> None:
    for item in work:
        _write_json(item.folder / STAGES_FILE, summary.papers[item.paper.paper_id])
    _write_json(
        out / STAGES_FILE,
        {
            "molscout_version": molscout.__version__,
            "config_sha256": config.sha256(),
            "config": config.public(),
            "papers": len(work),
            "failed": summary.failed,
            "stages": summary.stages,
            "funnel": summary.funnel,
            "per_paper": summary.papers,
        },
    )


def _write_json(path: Path, data: object) -> None:
    """Write through a temporary file renamed into place: rewriting in place on /scratch once left an old tail."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
