#!/usr/bin/env bash
# Serve BioMiner-Instruct with vLLM on one GPU for the 32B joiner (join_32b.py), on 127.0.0.1:8613.
#
# The flags are the ones that served it in MolScout's BioMiner benchmark runs (run.py and tool.yaml in
# benchmarks/tools/complete_systems/biominer/): upstream's scripts/run_local_vllm_server_biominer_instruct.bash
# but for --tensor-parallel-size 1 (upstream 4 A800s; here one H200), --gpu-memory-utilization 0.8
# (upstream 0.6 leaves one H200 less KV cache than one 128000-token request needs, so vLLM would not
# start) and --host 127.0.0.1 (upstream listens on every interface). --allowed-local-media-path is left
# out: join_32b.py sends its images inline.
#
# Unless --no-stage, it first copies the biominer and biominer_vllm environments (the archives setup.sh
# packs) and the weights to --work with the benchmark's stage.py, because on a busy node imports from
# /scratch (NFS) can outlast the job. It writes <work>/staged.env (BIOMINER_PYTHON, which runs join_32b.py,
# VLLM_PYTHON, BIOMINER_INSTRUCT) and then becomes the vLLM server (exec), so its PID is the server's.
# The work folder is the job's own by default ($TMPDIR/molscout-32b-<job id>). A staged copy is reused only
# while every file staged.env names is still in the work folder; otherwise what an interrupted job left
# (envs/, models/) is removed and staging starts again, so a resubmitted job can reuse its folder. With
# --no-stage the store's files are served and any staged copy is ignored. The log names the vLLM version
# and the model's snapshot: a sha256 over each weight file's name and size and the contents of the files
# under 64 MiB (configs, tokenizer, index), so it changes with the model without reading the weights.
# The server runs offline: proxy variables removed, no_proxy for 127.0.0.1, Hugging Face offline, no
# usage statistics, compile caches in the work folder.
#
# Usage: serve_32b.sh [--work DIR] [--port N] [--gpu-memory-utilization F] [--no-stage] [--dry-run]
# MOLSCOUT_STORE must name the folder with envs/biominer, envs/biominer_vllm and models/biominer/.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../../.." && pwd)
work=${TMPDIR:-/tmp}/molscout-32b-${SLURM_JOB_ID:-$$}
port=8613
memory=0.8
stage=1
dry=0
while (($#)); do
  case $1 in
    --work) work=$2; shift 2 ;;
    --port) port=$2; shift 2 ;;
    --gpu-memory-utilization) memory=$2; shift 2 ;;
    --no-stage) stage=0; shift ;;
    --dry-run) dry=1; shift ;;
    -h | --help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print } /^set -euo/ { exit }' "$0"; exit 0 ;;
    *) echo "serve_32b.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
if [[ -z $work || $work == / ]]; then
  echo "serve_32b.sh: --work must name a folder" >&2
  exit 2
fi
store=${MOLSCOUT_STORE:?set MOLSCOUT_STORE to the folder with envs/ and models/}
BIOMINER_PYTHON=$store/envs/biominer/bin/python
VLLM_PYTHON=$store/envs/biominer_vllm/bin/python
BIOMINER_INSTRUCT=$store/models/biominer/BioMiner-Instruct
for path in "$BIOMINER_PYTHON" "$VLLM_PYTHON" "$BIOMINER_INSTRUCT"; do
  if [[ ! -e $path ]]; then
    echo "serve_32b.sh: $path does not exist; run benchmarks/tools/complete_systems/biominer/setup.sh" >&2
    exit 2
  fi
done

# True when staged.env names a staged copy that is still complete in the work folder.
staged() {
  [[ -f $work/staged.env ]] || return 1
  (
    # shellcheck source=/dev/null
    source "$work/staged.env"
    for path in "${BIOMINER_PYTHON:-}" "${VLLM_PYTHON:-}" "${BIOMINER_INSTRUCT:-}"; do
      [[ $path == "$work"/* && -e $path ]] || exit 1
    done
  )
}

# The model folder's snapshot: name and size of every file, the contents of those under 64 MiB, hashed.
snapshot() {
  local file size
  for file in "$1"/*; do
    [[ -f $file ]] || continue
    size=$(wc -c <"$file" | tr -d ' ')
    printf '%s %s\n' "${file##*/}" "$size"
    if ((size < 67108864)); then cat "$file"; fi
  done | { sha256sum 2>/dev/null || shasum -a 256; } | cut -c1-16
}

if ((dry)); then
  if ((stage)); then echo "would stage both environments and BioMiner-Instruct to $work"; fi
else
  mkdir -p "$work"
  if ((!stage)); then
    printf 'BIOMINER_PYTHON=%q\nVLLM_PYTHON=%q\nBIOMINER_INSTRUCT=%q\n' \
      "$BIOMINER_PYTHON" "$VLLM_PYTHON" "$BIOMINER_INSTRUCT" >"$work/staged.env"
  elif ! staged; then
    rm -rf "$work/envs" "$work/models" "$work/staged.env"  # what an interrupted staging left
    "$BIOMINER_PYTHON" - "$repo/benchmarks/tools/complete_systems/biominer" "$work" "$BIOMINER_PYTHON" \
      "$VLLM_PYTHON" "$BIOMINER_INSTRUCT" >"$work/staged.env.partial" <<'PY'
import shlex
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from stage import stage

paths = stage(Path(sys.argv[2]), sys.argv[3], sys.argv[4], Path(sys.argv[5]))
for name, key in (("BIOMINER_PYTHON", "python"), ("VLLM_PYTHON", "vllm_python"), ("BIOMINER_INSTRUCT", "model")):
    print(f"{name}={shlex.quote(paths[key])}")
PY
    mv "$work/staged.env.partial" "$work/staged.env"
  fi
  # shellcheck source=/dev/null
  source "$work/staged.env"
fi

environment=(
  -u http_proxy -u https_proxy -u ftp_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u FTP_PROXY -u ALL_PROXY
  "no_proxy=127.0.0.1,localhost" "NO_PROXY=127.0.0.1,localhost"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONDONTWRITEBYTECODE=1
  "VLLM_CACHE_ROOT=$work/vllm-cache" "TRITON_CACHE_DIR=$work/triton-cache"
  "PATH=$(dirname "$BIOMINER_PYTHON"):$PATH"
)
server=(
  "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server
  --model "$BIOMINER_INSTRUCT"
  --served-model-name local-biominer-instruct
  --trust-remote-code
  --tensor-parallel-size 1
  --gpu-memory-utilization "$memory"
  --max-model-len 128000
  --swap-space 32
  --port "$port"
  --mm-encoder-tp-mode data
  --async-scheduling
  --host 127.0.0.1
)
if ((dry)); then
  printf '%q ' env "${environment[@]}" "${server[@]}"
  echo
  exit 0
fi
version=$("$VLLM_PYTHON" -c 'import importlib.metadata as m; print(m.version("vllm"))')
echo "serve_32b.sh: vLLM $version; model $BIOMINER_INSTRUCT, snapshot $(snapshot "$BIOMINER_INSTRUCT")" >&2
cd "$work"
echo "serve_32b.sh: starting vLLM (BioMiner-Instruct) on 127.0.0.1:$port from $work" >&2
exec env "${environment[@]}" "${server[@]}"
