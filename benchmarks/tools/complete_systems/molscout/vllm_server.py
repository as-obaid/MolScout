"""Start BioMiner-Instruct on a vLLM server for MolScout's 32B joiner, and stop it again. Standard library only.

The server is the pipeline's workers/serve_32b.sh: it copies the biominer environments and the weights to its
work folder (node-local disk under $TMPDIR in a job), then becomes vLLM with the flags BioMiner's benchmark runs
used. Here it serves on 127.0.0.1 port 8623, since BioMiner's own runs use 8613. It runs in a process group of
its own, so stopping it stops its workers too; its output goes to a log file, never to the job log.
start_seconds bounds the wait for it to list the model.
"""

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVED_MODEL = "local-biominer-instruct"
PORT = 8623  # BioMiner's own run.py serves on 8613; a MolScout job on the same node must not clash with it
START_SECONDS = 2400
POLL_SECONDS = 5
STOP_GRACE_SECONDS = 10
PROXY_VARIABLES = ("http_proxy", "https_proxy", "ftp_proxy", "all_proxy",
                   "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "ALL_PROXY")


def command(script, work, port=PORT, memory_fraction=0.8):
    """serve_32b.sh with its work folder, port and GPU memory fraction."""
    return ["bash", str(script), "--work", str(work), "--port", str(port),
            "--gpu-memory-utilization", str(memory_fraction)]


class VllmServer:
    """A context manager: `with VllmServer(...) as url:` starts the server and waits until it lists the model."""

    def __init__(self, argv, log, *, port=PORT, start_seconds=START_SECONDS, cache=None):
        self.argv = list(argv)
        self.log = Path(log)
        self.port = port
        self.start_seconds = start_seconds
        self.cache = cache
        self.process = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self):
        env = {name: value for name, value in os.environ.items() if name not in PROXY_VARIABLES}
        if self.cache is not None:
            env["VLLM_CACHE_ROOT"] = str(self.cache)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("ab") as handle:
            self.process = subprocess.Popen(self.argv, stdout=handle, stderr=subprocess.STDOUT, env=env,
                                            start_new_session=True)
        try:
            self._wait()
        except BaseException:
            self.stop()
            raise
        return self.url

    def __exit__(self, *exc_info):
        self.stop()

    def _wait(self):
        deadline = time.monotonic() + self.start_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"vLLM exited with status {self.process.returncode} while starting; see {self.log}")
            if self.ready():
                return
            time.sleep(POLL_SECONDS)
        raise RuntimeError(f"vLLM did not list {SERVED_MODEL} within {self.start_seconds} s; see {self.log}")

    def ready(self):
        """Whether the server lists the model. Proxy variables are ignored, as for the server: a cluster's proxy
        cannot reach the node's 127.0.0.1."""
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(self.url + "/models", timeout=POLL_SECONDS) as response:
                body = json.loads(response.read())
        except (OSError, ValueError, urllib.error.URLError):
            return False
        return any(model.get("id") == SERVED_MODEL for model in body.get("data", []))

    def stop(self):
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=STOP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
        except ProcessLookupError:
            pass
