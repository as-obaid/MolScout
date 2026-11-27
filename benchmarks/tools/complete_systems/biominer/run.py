"""BioMiner, the complete system, on whole PDFs, written as predictions.csv. Runs in the biominer environment.

BioMiner (jiaxianyan/BioMiner) parses a paper with MinerU 1.3.1, finds structure drawings with MolDetV2,
reads them with MolGlyph, and has BioMiner-Instruct (a 32B MLLM served by vLLM) tie each structure to
its identifier and enumerate Markush structures. run.py starts BioMiner's four servers as child
processes in one process group of their own: vLLM from the biominer_vllm environment with the flags of
scripts/run_local_vllm_server_biominer_instruct.bash, and MinerU, MolDetV2 and MolGlyph with serve.py.
It waits until each answers, then runs BioMiner's pipeline on one paper at a time exactly as
example_open_source.py does (BioMiner(config).opensource([pdf]) with
BioMiner/config/default_open_source.yaml, output_dir set per paper), and stops every server on exit,
on an error and on SIGTERM. A watchdog leads the group and kills it if run.py itself is killed. Model
loading is untimed: the servers are up before the first paper.

Bounds: a paper (the warmup's too) that runs past --paper-timeout (1200 s) raises PaperTimeout from a
SIGALRM handler and is recorded as failed. SIGALRM cannot interrupt C code, so a thread gives up on a
paper at twice the limit: it stops the servers, discards the work folder and exits with status 97. The
work folder is removed by renaming it to <name>.trash and a detached rm -rf, so a kill cannot leave it
half-deleted; the next start sweeps what earlier runs left. Ports are checked before staging.

Start-up: run.py first copies both environments (from the archives setup.sh packs) and BioMiner-Instruct's
weights into its work folder, which is on node-local disk in harness runs, and then runs again from
the staged main environment as the same process (os.execv); stage.py says why. Everything else reads
from /scratch as before: the clone, MinerU's, MolDetV2's and MolGlyph's weights, and the PDFs.

The code runs from a copy of the pinned commit (git archive) in a work folder, because BioMiner reads
its prompts and fonts, and writes its OPSIN cache and logs, relative to the working directory (under
BioMiner/commons/1d_r_groups and in it); the clone is never written to. Each paper gets its own output
folder there, removed when the paper is done. MinerU reads magic-pdf.json from MINERU_TOOLS_CONFIG_JSON:
the pinned file with only models-dir and layoutreader-model-dir pointed at the downloaded models.
Proxy variables are removed: runs are offline, and with the cluster's http_proxy and no no_proxy,
BioMiner's requests to its servers on 127.0.0.1 would go to the proxy.

Output: every entry of BioMiner's structure list for the paper (full structures and Markush-enumerated
ones), one Molecule each, in BioMiner's order, duplicates kept. BioMiner keeps only an identifier and a
SMILES per entry, so page and bbox are None, and it gives no confidence. An entry whose SMILES is None
or empty (OCSR or Markush enumeration failed; BioMiner's own CSVs read these as NaN and drop them) is not
a SMILES and is left out; stderr counts them. BioMiner's own print output (which includes recognised
SMILES) goes to logs/pipeline.out in the work folder, not to the job log.
"""

import contextlib
import itertools
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from crop_runner import _exit_143  # noqa: E402
from paper_runner import Molecule, base_parser, read_papers, run_papers  # noqa: E402
from stage import stage  # noqa: E402  (this folder: the script's own, first on sys.path)

HERE = Path(__file__).resolve().parent
CONFIG = "BioMiner/config/default_open_source.yaml"
SERVED_MODEL = "local-biominer-instruct"  # the model name default_open_source.yaml asks the server for
VLLM_PORT = 8613  # BioMiner's clients call these ports on 127.0.0.1; its code fixes them
LITSERVE_PORTS = {"mineru": 8002, "moldet": 8001, "ocsr": 8003}
# Server start-up after staging (stage.py): imports and 62 GiB of weights from node-local disk, then
# compiling and capturing CUDA graphs. The waits cover the slowest start measured, with margin.
VLLM_START_SECONDS = 2400
LITSERVE_START_SECONDS = 900
STOP_GRACE_SECONDS = 6  # the harness kills run.py 10 s after its SIGTERM
PROXY_VARIABLES = ("http_proxy", "https_proxy", "ftp_proxy", "all_proxy",
                   "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "ALL_PROXY")
MLLM_FAILURE = re.compile(r"local-biominer-instruct \d+ try failed")  # BioMiner prints this and goes on with "[]"
PAPER_TIMEOUT_SECONDS = 1200  # --paper-timeout; the slowest paper measured took 137 s (DECIMER.ai's slowest BioVista one 316 s)
HANG_FACTOR = 2  # the watchdog thread gives up on a paper that outlives this many limits: SIGALRM could not break it
HANG_EXIT_CODE = 97  # run.py's exit when it gives up on a hung paper; no signal or shell uses it, so it names itself in the log
WATCH_INTERVAL_SECONDS = 5
STAGED = "BIOMINER_STAGED"  # run.py's note to itself across its re-exec: the staged paths, as JSON

# Leads the servers' process group. It ignores the group's SIGTERM, so it outlives the servers, and
# when run.py is gone (its parent changes, even after SIGKILL) it stops the whole group itself.
WATCHDOG = """
import os, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
parent = int(sys.argv[1])
while os.getppid() == parent:
    time.sleep(1)
os.killpg(0, signal.SIGTERM)
time.sleep(5)
os.killpg(0, signal.SIGKILL)
"""

# The flags of upstream's scripts/run_local_vllm_server_biominer_instruct.bash, but for the model path,
# --tensor-parallel-size and --gpu-memory-utilization (arguments, set in tool.yaml), the media path
# and the host.
VLLM_FLAGS = (
    "--served-model-name", SERVED_MODEL,
    "--trust-remote-code",
    "--max-model-len", "128000",
    "--swap-space", "32",
    "--port", str(VLLM_PORT),
    "--mm-encoder-tp-mode", "data",
    "--async-scheduling",
    "--host", "127.0.0.1",  # upstream listens on every interface
)


class Servers:
    """BioMiner's servers, in one process group led by the watchdog."""

    def __init__(self, logs, env):
        self.logs, self.env = logs, env
        self.processes = {}
        # a new process group in run.py's session: setpgid can only join a group of the same session
        self.leader = subprocess.Popen([sys.executable, "-c", WATCHDOG, str(os.getpid())],
                                       stdin=subprocess.DEVNULL, preexec_fn=lambda: os.setpgid(0, 0))
        self.pgid = self.leader.pid

    def start(self, name, command, cwd):
        """Start one server in the group, its output going to logs/<name>.log."""
        log = (self.logs / f"{name}.log").open("ab")
        pgid = self.pgid
        try:
            process = subprocess.Popen(list(command), cwd=cwd, env=self.env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, preexec_fn=lambda: os.setpgid(0, pgid))
        finally:
            log.close()
        self.processes[name] = process
        print(f"started the {name} server (pid {process.pid}), log {self.logs / f'{name}.log'}", file=sys.stderr)

    def wait_ready(self, name, url, timeout, accept=lambda body: True):
        """Wait until GET url answers 200 with an accepted body; raise if the server exits or times out."""
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        start = time.monotonic()
        while True:
            self.check(name)
            try:
                with opener.open(url, timeout=10) as response:
                    if response.status == 200 and accept(response.read().decode("utf-8", "replace")):
                        seconds = time.monotonic() - start
                        print(f"the {name} server answers after {seconds:.0f} s", file=sys.stderr)
                        return
            except (urllib.error.URLError, OSError, ValueError):
                pass  # not listening yet, or 503 while its workers set up
            if time.monotonic() - start > timeout:
                raise RuntimeError(f"the {name} server did not answer {url} within {timeout} s{self.tail(name)}")
            time.sleep(3)

    def check(self, name=None, tail=True):
        """Raise if a server (or any, when name is None) has exited; tail=False names its log instead of quoting it."""
        for server, process in self.processes.items():
            if (name is None or server == name) and process.poll() is not None:
                where = self.tail(server) if tail else f"; see {self.logs / f'{server}.log'}"
                raise RuntimeError(f"the {server} server exited with status {process.returncode}{where}")

    def tail(self, name, lines=40):
        path = self.logs / f"{name}.log"
        try:
            text = path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        except OSError:
            return ""
        return f"; last lines of {path}:\n" + "\n".join(text)

    def stop(self):
        """SIGTERM to the group, then SIGKILL to whatever is left after STOP_GRACE_SECONDS."""
        if threading.current_thread() is threading.main_thread():  # signal() raises elsewhere: the hang watchdog stops too
            signal.signal(signal.SIGTERM, signal.SIG_IGN)  # a second SIGTERM must not cut this short
        try:
            os.killpg(self.pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + STOP_GRACE_SECONDS
        while time.monotonic() < deadline and any(p.poll() is None for p in self.processes.values()):
            time.sleep(0.2)
        try:
            os.killpg(self.pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for process in [*self.processes.values(), self.leader]:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        print("stopped the BioMiner servers", file=sys.stderr)


def port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        # SO_REUSEADDR: a TIME_WAIT left by the last segment's servers is not a server in the way
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def busy_ports(ports=(VLLM_PORT, *LITSERVE_PORTS.values())):
    return [port for port in ports if not port_free(port)]


def require_free_ports():
    """Exit at once if a port BioMiner's clients call is taken: better now than after staging's minutes."""
    busy = busy_ports()
    if busy:
        raise SystemExit(f"ports {busy} are in use on {socket.gethostname()}; BioMiner's clients call fixed ports")


def discard(path):
    """Remove a folder past a SIGKILL: rename it to <path>.trash (instant) and let a detached rm -rf, in a
    session of its own, delete it. The harness kills run.py 10 s after SIGTERM and removing the work folder
    (staged environments, 62 GiB of weights) takes longer; a rename cannot be cut short. Returns the new path."""
    path = Path(path)
    target = path
    if path.suffix != ".trash":
        target = path.with_name(path.name + ".trash")
        try:
            os.rename(path, target)
        except OSError:  # a stale <path>.trash in the way, or no rename possible: delete in place
            target = path
    subprocess.Popen(["rm", "-rf", str(target)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    return target


def sweep_stale(parent, keep=None, uid=None):
    """Discard the biominer-* work folders and *.trash left in parent by killed or hung runs, that belong to
    this user and are not keep. One BioMiner job runs per node (--dependency=singleton) and this runs before
    the current run makes its folder, so none of them is in use. Returns the folders handed to discard()."""
    uid = os.getuid() if uid is None else uid
    swept = []
    for entry in sorted(Path(parent).glob("biominer-*")):
        try:
            info = entry.lstat()
        except OSError:
            continue
        if entry == keep or not entry.is_dir() or entry.is_symlink() or info.st_uid != uid:
            continue
        swept.append(discard(entry))
    return swept


class PaperTimeout(TimeoutError):
    """A paper ran past --paper-timeout; paper_runner records it as that paper's error."""


@contextlib.contextmanager
def paper_limit(limit, watch=None):
    """SIGALRM after limit seconds raises PaperTimeout in the main thread; always disarmed on the way out.
    It only fires between Python bytecodes: a paper stuck in C code needs HangWatch."""
    def expired(signum, frame):
        raise PaperTimeout(f"paper exceeded the {limit:g} s limit (--paper-timeout)")

    previous = signal.signal(signal.SIGALRM, expired)
    if watch is not None:
        watch.begin()
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, signal.SIG_DFL if previous is None else previous)
        if watch is not None:
            watch.end()


def unresponsive_servers(timeout=10):
    """Names of the servers that do not answer their health URL within timeout seconds (none: all answer)."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    urls = {"vllm": f"http://127.0.0.1:{VLLM_PORT}/v1/models",
            **{name: f"http://127.0.0.1:{port}/health" for name, port in LITSERVE_PORTS.items()}}
    down = []
    for name, url in urls.items():
        try:
            with opener.open(url, timeout=timeout) as response:
                if response.status == 200:
                    continue
        except (urllib.error.URLError, OSError, ValueError):
            pass
        down.append(name)
    return down


def limited(predict, limit, watch=None, probe=unresponsive_servers):
    """predict with a wall-clock limit per call. Wrapping what run_papers is given covers its untimed warmup too.

    A timed-out paper may leave its HTTP call running in a thread or worker (BioMiner's MinerU request has no
    timeout) and so a server busy or wedged for every paper after it. If a server then fails its health probe,
    the run stops (SystemExit, not an error row: paper_runner only catches Exception) instead of failing the
    rest of the papers one limit each; the checkpoint keeps what finished and the worker counts a failure."""
    def bounded(paper):
        try:
            with paper_limit(limit, watch):
                return predict(paper)
        except PaperTimeout:
            down = probe()
            if down:
                raise SystemExit(f"{paper.paper_id}: timed out and the {', '.join(down)} server(s) no longer answer; "
                                 "stopping rather than timing out every later paper")
            raise

    return bounded


class HangWatch:
    """Decides, from a thread, that the paper in progress is stuck where SIGALRM cannot reach (C code):
    it has run HANG_FACTOR times the limit. give_up is injected (abandon() in a run, a recorder in tests)."""

    def __init__(self, limit, give_up, clock=time.monotonic):
        self.limit, self.give_up, self.clock = limit, give_up, clock
        self.started = None

    def begin(self):
        self.started = self.clock()

    def end(self):
        self.started = None

    def hung(self):
        started = self.started  # one read: the main thread may end the paper meanwhile
        return started is not None and self.clock() - started > HANG_FACTOR * self.limit

    def poll(self):
        if self.hung():
            self.give_up()
            return True
        return False

    def run(self, interval=WATCH_INTERVAL_SECONDS, sleep=time.sleep):
        while not self.poll():
            sleep(interval)

    def start(self):
        threading.Thread(target=self.run, name="hang-watch", daemon=True).start()


def abandon(servers, work, keep, exit_code=HANG_EXIT_CODE, exit_fn=os._exit):
    """Give up on a hung paper: stop the servers, discard the work folder (os._exit skips main's finally),
    and exit non-zero so the Slurm worker counts a failure instead of resubmitting into the same hang."""
    print(f"a paper outlived {HANG_FACTOR} x --paper-timeout in code SIGALRM cannot interrupt; exiting {exit_code}",
          file=sys.stderr, flush=True)
    try:
        servers.stop()
        if not keep:
            discard(work)
    finally:
        exit_fn(exit_code)


def prepare_work(work, source, commit, models, molglyph):
    """The work folder: the pinned code, the weights where the servers look, magic-pdf.json and logs."""
    head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    if head.stdout.strip() != commit:
        raise SystemExit(f"{source} is at {head.stdout.strip()}, not the pinned {commit}; run setup.sh")
    archive = subprocess.Popen(["git", "-C", str(source), "archive", commit, "BioMiner", "scripts", "magic-pdf.json"],
                               stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(work)], stdin=archive.stdout, check=True)
    archive.stdout.close()
    if archive.wait() != 0:
        raise SystemExit(f"git archive of {commit} in {source} failed")
    (work / "scripts" / "moldet_v2_yolo11n_960_doc.pt").symlink_to(models / "moldet_v2_yolo11n_960_doc.pt")
    (work / "scripts" / "molglyph_large.pt").symlink_to(molglyph)
    config = json.loads((work / "magic-pdf.json").read_text(encoding="utf-8"))
    config["models-dir"] = str(models / "mineru" / "mineru_old_version_models")
    config["layoutreader-model-dir"] = str(models / "layoutreader")
    (work / "magic-pdf.json").write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
    for folder in ("logs", "papers", "vllm-cache", "ultralytics"):
        (work / folder).mkdir(exist_ok=True)  # stage.py made logs/


def server_environment(work):
    """run.py's environment for the servers and the pipeline: offline, the env's tools first on PATH."""
    env = {name: value for name, value in os.environ.items() if name not in PROXY_VARIABLES}
    env.update(
        no_proxy="127.0.0.1,localhost", NO_PROXY="127.0.0.1,localhost",
        PATH=f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}",  # pdftoppm and java (py2opsin)
        PYTHONPATH=f"{work}{os.pathsep}{work / 'scripts'}",
        MINERU_TOOLS_CONFIG_JSON=str(work / "magic-pdf.json"),  # MinerU reads ~/magic-pdf.json otherwise
        YOLO_CONFIG_DIR=str(work / "ultralytics"),  # ultralytics writes its settings under ~/.config otherwise
        VLLM_CACHE_ROOT=str(work / "vllm-cache"),  # vLLM's compile cache, under ~/.cache otherwise
        TRITON_CACHE_DIR=str(work / "triton-cache"),  # vLLM's compiled kernels, under ~/.triton otherwise
        VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1",
        PYTHONUNBUFFERED="1",  # BioMiner's joblib workers print the failures report() counts
        # a stale .pyc is compiled in memory instead of written again: a new file on /scratch costs up to 0.4 s
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env


def start_servers(servers, work, model, vllm_python, tensor_parallel, memory_fraction):
    """vLLM first (it needs its whole memory fraction free when it starts), then the three LitServe servers."""
    visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    if len(visible) < tensor_parallel:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} has fewer than "
                         f"{tensor_parallel} GPUs; BioMiner runs on {tensor_parallel}")
    if visible != [str(i) for i in range(len(visible))]:
        # MinerU's worker sets CUDA_VISIBLE_DEVICES to its LitServe index, which is only right for 0..N-1.
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={','.join(visible)} is not 0..{len(visible) - 1}; "
                         "MinerU's server would pick the wrong GPUs")
    require_free_ports()  # again: main checked before staging, minutes ago
    servers.start("vllm", [str(vllm_python), "-m", "vllm.entrypoints.openai.api_server",
                           "--model", str(model),
                           "--tensor-parallel-size", str(tensor_parallel),
                           "--gpu-memory-utilization", str(memory_fraction),
                           # upstream allows /data, its own mount; BioMiner sends images inline (base64)
                           "--allowed-local-media-path", str(work),
                           *VLLM_FLAGS], work)
    servers.wait_ready("vllm", f"http://127.0.0.1:{VLLM_PORT}/v1/models", VLLM_START_SECONDS,
                       accept=lambda body: SERVED_MODEL in body)
    for name in LITSERVE_PORTS:
        servers.start(name, [sys.executable, str(HERE / "serve.py"), name, "--gpus", str(len(visible))], work)
    for name, port in LITSERVE_PORTS.items():
        servers.wait_ready(name, f"http://127.0.0.1:{port}/health", LITSERVE_START_SECONDS, accept=lambda b: b == "ok")


def vllm_metrics():
    """vLLM's Prometheus histograms of prompt and generated tokens per request, summed over engines; None if unreadable."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{VLLM_PORT}/metrics", timeout=30) as response:
            text = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:  # monitoring only: never fails a paper
        print(f"vLLM /metrics unreadable ({exc}); no token report", file=sys.stderr)
        return None
    found = {}
    for line in text.splitlines():
        match = re.match(r"vllm:request_(prompt|generation)_tokens_(bucket|sum|count)(\{[^}]*\})? ([0-9.eE+-]+)$", line)
        if match:
            kind, field, labels, value = match.groups()
            le = re.search(r'le="([^"]+)"', labels or "")
            key = (kind, field, le.group(1) if le else None)
            found[key] = found.get(key, 0.0) + float(value)
    return found


def token_report(paper_id, before, after):
    """stderr: this paper's MLLM requests and prompt tokens (counts only), from two vllm_metrics() readings."""
    if before is None or after is None:
        return
    delta = {key: after.get(key, 0.0) - before.get(key, 0.0) for key in after}
    requests = int(delta.get(("prompt", "count", None), 0))
    if not requests:
        print(f"{paper_id}: 0 BioMiner-Instruct requests", file=sys.stderr)
        return
    buckets = sorted((float(le), count) for (kind, field, le), count in delta.items()
                     if kind == "prompt" and field == "bucket" and le not in (None, "+Inf"))
    largest = next((le for le, count in buckets if count >= requests), None)
    prompt, generated = delta.get(("prompt", "sum", None), 0.0), delta.get(("generation", "sum", None), 0.0)
    print(f"{paper_id}: {requests} BioMiner-Instruct requests, prompt tokens {prompt:.0f} "
          f"(mean {prompt / requests:.0f}, largest <= {largest if largest is None else int(largest)} "
          f"by vLLM's histogram), generated tokens {generated:.0f}", file=sys.stderr)


def safe_name(paper_id):
    """A file stem BioMiner can use: it splits names on '.', and passes paths to the shell."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", paper_id) or "paper"


def make_predict(work, servers, keep):
    """predict(paper) running BioMiner's pipeline in this process, from the work folder."""
    os.chdir(work)  # BioMiner reads prompts and fonts and writes its OPSIN cache relative to here
    sys.path.insert(0, str(work))
    from BioMiner import BioMiner
    from BioMiner.commons.utils import get_config_easydict

    calls = itertools.count(1)  # one folder per call: the untimed warmup runs the first paper twice
    pipeline_log = work / "logs" / "pipeline.out"

    def predict(paper):
        servers.check(tail=False)  # a paper's error goes to predictions.errors.json: no log lines in it
        folder = work / "papers" / f"{next(calls):04d}"
        name = safe_name(paper.paper_id)
        pdf = folder / "input" / f"{name}.pdf"
        pdf.parent.mkdir(parents=True)
        pdf.symlink_to(paper.pdf.resolve())
        start = pipeline_log.stat().st_size
        before = vllm_metrics()
        try:
            config = get_config_easydict(CONFIG)
            config.output_dir = str(folder / "output")  # as example_open_source.py's --output_dir
            _, structure_lists, _ = BioMiner(config).opensource([str(pdf)])
            sys.stdout.flush()
            report(paper.paper_id, folder / "output", name, pipeline_log, start)
            token_report(paper.paper_id, before, vllm_metrics())
            servers.check(tail=False)  # a server that died mid-paper fails the paper, not "no molecules"
            entries = structure_lists[0]
        finally:
            if not keep:
                shutil.rmtree(folder, ignore_errors=True)
        molecules = [Molecule(entry["smiles"]) for entry in entries if is_smiles(entry.get("smiles"))]
        print(f"{paper.paper_id}: {len(entries)} structure entries, {len(entries) - len(molecules)} without a "
              f"SMILES left out", file=sys.stderr)
        return molecules

    return predict


def is_smiles(value):
    return isinstance(value, str) and bool(value.strip())


def report(paper_id, output, name, pipeline_log, start):
    """Say on stderr where BioMiner fell back or swallowed an error for this paper (counts only)."""
    if not (output / "mineru" / name / "auto" / f"{name}_middle.json").is_file():
        print(f"{paper_id}: MinerU gave no layout; BioMiner went on with pypdf text and whole pages", file=sys.stderr)
    with pipeline_log.open("rb") as handle:
        handle.seek(start)
        lines = handle.read().decode("utf-8", "replace").splitlines()
    failures = [line for line in lines if MLLM_FAILURE.search(line)]
    unparsed = sum("image structure extraction" in line for line in lines)
    if failures or unparsed:
        print(f"{paper_id}: {len(failures)} BioMiner-Instruct calls failed (BioMiner used [] for them), "
              f"{unparsed} structure answers did not parse; first failure: {failures[0][:300] if failures else '-'}",
              file=sys.stderr)


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--source", type=Path, required=True, help="the BioMiner clone")
    parser.add_argument("--commit", required=True, help="the pinned commit, copied out of the clone with git archive")
    parser.add_argument("--models", type=Path, required=True, help="the folder setup.sh downloads into")
    parser.add_argument("--molglyph", type=Path, required=True, help="molglyph_large.pt")
    parser.add_argument("--vllm-python", type=Path, required=True, help="the biominer_vllm environment's Python")
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True,
                        help="the fraction of each GPU vLLM takes (upstream 0.6)")
    parser.add_argument("--work-dir", type=Path, help="parent of the work folder (default: the temp folder)")
    parser.add_argument("--keep-work", action="store_true", help="keep the work folder: logs and per-paper output")
    parser.add_argument("--paper-timeout", type=float, default=PAPER_TIMEOUT_SECONDS,
                        help="seconds one paper may take, the warmup's included, before it is recorded as failed "
                             f"(default {PAPER_TIMEOUT_SECONDS}); a paper stuck in C code for {HANG_FACTOR}x this "
                             f"ends the run with status {HANG_EXIT_CODE}")
    args = parser.parse_args()
    if args.paper_timeout <= 0:
        parser.error("--paper-timeout must be positive")
    require_free_ports()
    for path in (args.source, args.models / "BioMiner-Instruct", args.models / "mineru", args.models / "layoutreader",
                 args.models / "moldet_v2_yolo11n_960_doc.pt", args.molglyph, args.vllm_python):
        if not path.exists():
            parser.error(f"{path} does not exist; run setup.sh")
    papers = len(read_papers(args.papers))

    signal.signal(signal.SIGTERM, _exit_143)  # exit 143 through finally, which stops the servers
    os.umask(0o077)  # the work folder holds page images and text of private papers
    staged = json.loads(os.environ.pop(STAGED, "null"))
    if staged is None:
        # First start: copy the environments and the model to the work folder (node-local in harness
        # runs; see stage.py), then run again from the staged main environment, as the same process.
        started = time.monotonic()
        if not args.keep_work:  # a kept folder is wanted; sweeping would delete it
            sweep_stale(args.work_dir or tempfile.gettempdir())
        work = Path(tempfile.mkdtemp(prefix="biominer-", dir=args.work_dir)).resolve()
        try:
            staged = stage(work, sys.executable, args.vllm_python, args.models / "BioMiner-Instruct")
        except BaseException:
            discard(work)
            raise
        os.environ[STAGED] = json.dumps({**staged, "work": str(work), "started": started})
        sys.stderr.flush()
        os.execv(staged["python"], [staged["python"], str(Path(__file__).resolve()), *sys.argv[1:]])
    work, started = Path(staged["work"]), staged["started"]
    servers = None
    stdout = os.dup(1)
    try:
        prepare_work(work, args.source.resolve(), args.commit, args.models.resolve(), args.molglyph.resolve())
        env = server_environment(work)
        os.environ.clear()
        os.environ.update(env)
        # BioMiner prints recognised SMILES; keep its stdout out of the job log
        sys.stdout.flush()
        with (work / "logs" / "pipeline.out").open("ab") as log:
            os.dup2(log.fileno(), 1)
        servers = Servers(work / "logs", env)
        start_servers(servers, work, Path(staged["model"]), Path(staged["vllm_python"]), args.tensor_parallel_size,
                      args.gpu_memory_utilization)
        watch = HangWatch(args.paper_timeout, lambda: abandon(servers, work, args.keep_work))
        predict = limited(make_predict(work, servers, args.keep_work), args.paper_timeout, watch)
        watch.start()
        print(f"start-up: {time.monotonic() - started:.0f} s from run.py's start to every server answering and "
              "the pipeline imported", file=sys.stderr)
        errors = run_papers(predict, args)
    finally:
        if servers is not None:
            servers.stop()
        sys.stdout.flush()
        os.dup2(stdout, 1)  # close pipeline.out: on NFS an open file keeps its folder from being removed
        os.chdir(HERE)
        if args.keep_work:
            print(f"work folder kept: {work}", file=sys.stderr)
        else:
            discard(work)  # a rename and a detached rm: the harness's SIGKILL cannot leave it half-deleted
    return 1 if errors == papers else 0


if __name__ == "__main__":
    sys.exit(main())
