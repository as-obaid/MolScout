"""Copy BioMiner's two environments and BioMiner-Instruct's weights to node-local disk; run.py does this first.

Why (measured on Explorer, task 9b, 2026-10-07). /scratch is NFS, and on a GPU node whose NFS client
was busy (d4077, sharing) every metadata round trip took about 0.7 s: run.py spent 17 min listing the
vLLM environment's folders and never started a server. A Python import opens and stats thousands of
files, so the servers' imports could not finish there. Large reads fared better on the same node: a
1 GiB O_DIRECT read took 54 s, then 7 s ten minutes later.

So setup.sh packs each environment into one archive beside it (<envs>/<name>.tar.zst; biominer_vllm's
holds the uv Python it runs on too), and this unpacks both into the run's work folder with 64 MiB
direct reads: a few thousand large requests instead of ~150,000 small ones. The weights are copied the
same way. STREAMS reads run at once. The work folder is node-local in harness runs ($TMPDIR) and is
removed after the run, staged files included.

The staged files are the same files. The archives are refused when their packages (site-packages'
*.dist-info and conda-meta records) differ from the environment's, which is what the harness locks;
every weight file is checked for size after the copy.
"""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

STREAMS = 6  # large reads from /scratch at once (dd, 64 MiB each)
EXTRACT = 'dd if="$1" bs=64M iflag=direct status=none | zstd -d -q -c | tar -x -C "$2"'
SPACE_FACTOR = 4  # an archive unpacks to about 2.5 times its size; keep room for 4


def archive_of(prefix):
    """setup.sh's packed copy of an environment folder: <envs>/<name>.tar.zst."""
    return prefix.parent / f"{prefix.name}.tar.zst"


def distributions(prefix):
    """The packages installed in an environment: site-packages' *.dist-info names and conda-meta's records."""
    names = set()
    for folder in [*prefix.glob("lib/python3*/site-packages"), prefix / "conda-meta"]:
        if folder.is_dir():
            names.update(entry.name for entry in os.scandir(folder) if entry.name.endswith((".dist-info", ".json")))
    return names


def run_all(commands, log):
    """Run the commands, STREAMS at a time, their output and finish times appended to log; raise on the first
    failure. Returns each command's seconds from the start, in order."""
    pending, running, start, finished = list(enumerate(commands)), [], time.monotonic(), {}
    with open(log, "ab") as output:
        try:
            while pending or running:
                while pending and len(running) < STREAMS:
                    index, command = pending.pop(0)
                    # a process group each, so stopping a pipeline stops its dd, zstd and tar too
                    running.append((index, subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                                            stderr=subprocess.STDOUT, start_new_session=True)))
                for index, process in list(running):
                    if process.poll() is None:
                        continue
                    running.remove((index, process))
                    command = " ".join(commands[index])
                    if process.returncode:
                        raise RuntimeError(f"staging failed (exit {process.returncode}): {command}; see {log}")
                    finished[index] = time.monotonic() - start
                    output.write(f"{finished[index]:.0f} s: {command}\n".encode())
                    output.flush()
                time.sleep(0.2)
        finally:
            for _, process in running:  # an error or SIGTERM: stop the other copies
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    return [finished[index] for index in range(len(commands))]


def relink_venv(venv, envs):
    """Point a staged uv venv at the staged copy of the Python it runs on (pyvenv.cfg's home, bin/ links)."""
    config = venv / "pyvenv.cfg"
    lines = config.read_text(encoding="utf-8").splitlines()
    home = next(line.partition("=")[2].strip() for line in lines if line.partition("=")[0].strip() == "home")
    staged_home = envs / Path(home).parent.name / "bin"
    if not staged_home.is_dir():
        raise RuntimeError(f"{archive_of(venv)} does not hold the Python {home}")
    config.write_text("".join(f"home = {staged_home}\n" if line.partition("=")[0].strip() == "home" else f"{line}\n"
                              for line in lines), encoding="utf-8")
    for link in (venv / "bin").iterdir():
        target = os.readlink(link) if link.is_symlink() else ""
        if target.startswith(home + "/"):
            link.unlink()
            link.symlink_to(staged_home / target[len(home) + 1:])


def stage(work, main_python, vllm_python, model):
    """Unpack both environments and copy the model folder under work.

    Returns the staged main Python, vLLM Python and model folder, as strings.
    """
    start = time.monotonic()
    pythons = {"main": Path(main_python), "vllm": Path(vllm_python)}
    prefixes = {name: python.parent.parent for name, python in pythons.items()}  # <env>/bin/python
    archives = [archive_of(prefix) for prefix in prefixes.values()]
    for archive in archives:
        if not archive.is_file():
            raise SystemExit(f"{archive} does not exist; run setup.sh, which packs the environments")
    weights = sorted(model.iterdir())
    if any(not path.is_file() for path in weights):
        raise SystemExit(f"{model} holds folders; staging copies files only")
    size = sum(path.stat().st_size for path in weights)
    need = size + SPACE_FACTOR * sum(archive.stat().st_size for archive in archives)
    free = shutil.disk_usage(work).free
    if free < need:
        raise SystemExit(f"{work} has {free / 2**30:.0f} GiB free; staging BioMiner needs about {need / 2**30:.0f} GiB")
    envs, staged_model, logs = work / "envs", work / "models" / model.name, work / "logs"
    envs.mkdir()
    staged_model.mkdir(parents=True)
    logs.mkdir(exist_ok=True)
    commands = [["bash", "-o", "pipefail", "-c", EXTRACT, "extract", str(archive), str(envs)] for archive in archives]
    # largest first, so the long copies overlap the rest; direct writes keep the job's page cache free
    for path in sorted(weights, key=lambda p: -p.stat().st_size):
        commands.append(["dd", f"if={path}", f"of={staged_model / path.name}", "bs=64M", "iflag=direct",
                         "oflag=direct", "status=none"])
    seconds = run_all(commands, logs / "stage.log")
    for path in weights:
        if (staged_model / path.name).stat().st_size != path.stat().st_size:
            raise RuntimeError(f"staged copy of {path} has the wrong size")
    staged = {name: envs / prefix.name for name, prefix in prefixes.items()}
    if (staged["vllm"] / "pyvenv.cfg").is_file():
        relink_venv(staged["vllm"], envs)
    for name, prefix in prefixes.items():
        if distributions(staged[name]) != distributions(prefix):
            raise SystemExit(f"{archive_of(prefix)} holds other packages than {prefix}; rerun setup.sh to pack it again")
    print(f"staged both environments ({sum(a.stat().st_size for a in archives) / 2**30:.1f} GiB packed; unpacked "
          f"after {max(seconds[:len(archives)]):.0f} s) and {model.name} ({size / 2**30:.1f} GiB; copied after "
          f"{max(seconds[len(archives):]):.0f} s) to {work} in {time.monotonic() - start:.0f} s", file=sys.stderr)
    return {"python": str(staged["main"] / "bin" / pythons["main"].name),
            "vllm_python": str(staged["vllm"] / "bin" / pythons["vllm"].name),
            "model": str(staged_model)}
