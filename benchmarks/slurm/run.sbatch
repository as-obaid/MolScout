#!/bin/bash
# Run benchmark configs through the harness on one GPU node. Runs are offline (HF_HUB_OFFLINE);
# weights come from /scratch. Submit from the repository root:
#   sbatch benchmarks/slurm/run.sbatch benchmarks/configs/molscribe__uspto.yaml
#   sbatch --array=0-4 benchmarks/slurm/run.sbatch benchmarks/configs/molscribe__*.yaml
#   sbatch --partition=short --gres=none benchmarks/slurm/run.sbatch benchmarks/configs/molvec__uspto.yaml
# Several configs without --array, or any configs with MOLSCOUT_RESUBMIT set (benchmarks/slurm/submit.sh
# sets it), make a worker. It runs, in order, each config not yet done at HEAD that no live job has
# claimed, and gives up on a config after 2 failures at HEAD. USR1 (--signal=B:USR1@180) means the time
# limit is near: the run stops, keeping its checkpoint, and the worker resubmits itself with
# sbatch $MOLSCOUT_RESUBMIT. TERM (scancel) stops it the same way without resubmitting.
# A GPU job first checks its GPUs (gpu_ok). On a bad GPU a worker resubmits itself excluding the node
# (up to 3 nodes) and exits 0 without claiming anything; a single or array run exits 3.
#SBATCH --job-name=molscout-bench
#SBATCH --account=ostadabbas
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=benchmarks/slurm/logs/%x_%A_%a.out
#SBATCH --error=benchmarks/slurm/logs/%x_%A_%a.err
set -euo pipefail
umask 077  # Internal results, checkpoints and logs hold SMILES from private papers
configs=("$@")
if [ "${#configs[@]}" -eq 0 ]; then
    echo "usage: sbatch [--array=0-N] benchmarks/slurm/run.sbatch CONFIG [CONFIG ...]" >&2
    exit 2
fi
results=benchmarks/results
claims=$results/.claims

log() { echo "worker: $*"; }

is_done() {  # whether results/<run>/meta.json is from HEAD with no uncommitted changes
    [ -f "$results/$1/meta.json" ] && python3 -c '
import json, sys
git = json.load(open(sys.argv[1])).get("git") or {}
sys.exit(git.get("commit") != sys.argv[2] or git.get("dirty") is not False)' "$results/$1/meta.json" "$head"
}

alive() {  # whether job $1 is queued or running; when squeue cannot tell, it is taken to be
    local state
    if ! state=$(squeue -h -j "$1" -o %T 2>&1); then
        [[ $state != *"Invalid job id"* ]]
        return
    fi
    case $state in
        "" | COMPLETED | CANCELLED* | FAILED | TIMEOUT | NODE_FAIL | PREEMPTED | OUT_OF_MEMORY | BOOT_FAIL | DEADLINE)
            return 1 ;;
    esac
}

cuda_probe_py='
import ctypes, sys
cuda = ctypes.CDLL("libcuda.so.1")
if cuda.cuInit(0) != 0:
    sys.exit("cuInit failed")
count = ctypes.c_int()
if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
    sys.exit("cuDeviceGetCount failed")
if count.value != int(sys.argv[1]):
    sys.exit("CUDA sees %d device(s), expected %s" % (count.value, sys.argv[1]))'

gpu_ok() {  # whether a GPU job'"'"'s GPUs work; CPU jobs (no CUDA_VISIBLE_DEVICES) pass. Sets gpu_why.
    gpu_why=
    [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || return 0
    local expected utilization line limit=()
    expected=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .) || expected=0
    if command -v timeout > /dev/null; then limit=(timeout 60); fi
    if ! utilization=$("${limit[@]}" nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>&1); then
        gpu_why="nvidia-smi failed: $(echo "$utilization" | head -c 200 | tr '\n' ' ')"
        return 1
    fi
    local seen=0
    while IFS= read -r line; do
        [ -n "${line//[[:space:]]/}" ] || continue
        if [[ ! ${line//[[:space:]]/} =~ ^[0-9]+$ ]]; then gpu_why="nvidia-smi reports $line"; return 1; fi
        seen=$((seen + 1))
    done <<< "$utilization"
    if [ "$seen" -ne "$expected" ]; then
        gpu_why="nvidia-smi lists $seen of $expected GPU(s)"
        return 1
    fi
    local out
    if [ -n "${MOLSCOUT_CUDA_PROBE:-}" ]; then
        # shellcheck disable=SC2086  # the probe may be a command with arguments
        out=$($MOLSCOUT_CUDA_PROBE "$expected" 2>&1) || { gpu_why="CUDA probe failed: ${out:-no output}"; return 1; }
    else
        out=$(python3 -c "$cuda_probe_py" "$expected" 2>&1) || { gpu_why="CUDA probe failed: ${out##*$'\n'}"; return 1; }
    fi
}

bad_gpu_resubmit() {  # this node'"'"'s GPUs are bad: resubmit the worker excluding it, or give up after 3 nodes
    local node options word excluded="" rest=() out
    node=$(hostname -s)
    for word in ${MOLSCOUT_RESUBMIT:-}; do
        case $word in
            --exclude=*) excluded=${word#--exclude=} ;;
            *) rest+=("$word") ;;
        esac
    done
    if [ -n "$excluded" ] && [ "$(echo "$excluded" | tr ',' '\n' | grep -c .)" -ge 3 ]; then
        log "giving up: 3 nodes are already excluded ($excluded)"
        exit 1
    fi
    options="${rest[*]:-} --exclude=${excluded:+$excluded,}$node"
    options=${options# }
    # shellcheck disable=SC2086  # options holds several words; --export=ALL would split the node list at commas
    if out=$(MOLSCOUT_RESUBMIT=$options sbatch $options benchmarks/slurm/run.sbatch "${configs[@]}" 2>&1); then
        log "resubmitted without $node: $out"
    else
        log "resubmit failed: $out"
        return 1
    fi
}

claim() {  # claim run $1 for this job; a claim whose job has ended is taken over. Sets owner.
    local dir=$claims/$1
    owner=
    if mkdir "$dir" 2>/dev/null; then
        echo "$SLURM_JOB_ID" > "$dir/job"
        return 0
    fi
    mkdir "$dir/takeover" 2>/dev/null || return 1  # another worker is taking it over
    owner=$(cat "$dir/job" 2>/dev/null || true)
    if [ -z "$owner" ] || { [ "$owner" != "$SLURM_JOB_ID" ] && alive "$owner"; }; then
        rmdir "$dir/takeover"  # no job yet means another worker has only just made the claim
        return 1
    fi
    echo "$SLURM_JOB_ID" > "$dir/job"
    rmdir "$dir/takeover"
}

release() {  # drop this job's claim on run $1
    if [ "$(cat "$claims/$1/job" 2>/dev/null)" = "$SLURM_JOB_ID" ]; then rm -rf "${claims:?}/$1"; fi
}

stop() {  # trap: $1 is USR1 (the time limit is near) or TERM (scancel); the run in progress gets TERM
    stopped=$1
    if [ -n "$bench" ]; then kill -TERM "$bench" 2>/dev/null || true; fi
}

worker() {
    local head config run failures status
    if ! gpu_ok; then
        log "bad GPU on $(hostname -s): $gpu_why"
        if [ -n "${MOLSCOUT_RESUBMIT:-}" ]; then bad_gpu_resubmit || exit 1; fi
        exit 0
    fi
    head=$(git rev-parse HEAD 2>/dev/null || true)
    stopped='' bench=''
    trap 'stop USR1' USR1
    trap 'stop TERM' TERM
    mkdir -p "$claims"
    log "job ${SLURM_JOB_ID:-} on $(hostname), ${#configs[@]} config(s), HEAD ${head:0:7}"
    for config in "${configs[@]}"; do
        [ -z "$stopped" ] || break
        run=$(basename "$config" .yaml)
        if ! claim "$run"; then  # the checks below run under the claim, so no other worker can finish it meanwhile
            log "skip $run: claimed by job ${owner:-(being claimed)}"
            continue
        fi
        failures=$(grep -cxF "$head" "$claims/$run.failures" 2>/dev/null) || failures=0
        if is_done "$run"; then
            log "skip $run: done at ${head:0:7}"
        elif [ "$failures" -ge 2 ]; then
            log "skip $run: gave up after $failures failures at ${head:0:7}"
        else
            log "claimed $run${owner:+ from job $owner, which has ended}"
            "$molscout" bench "$config" &
            bench=$!
            if [ -n "$stopped" ]; then kill -TERM "$bench" 2>/dev/null || true; fi  # a signal before bench was set
            status=0
            wait "$bench" || status=$?
            if [ -n "$stopped" ] && [ "$status" -gt 128 ]; then  # the trap cut the wait short
                status=0
                wait "$bench" || status=$?
            fi
            bench=
            if [ "$status" -eq 0 ]; then
                log "finished $run"
            elif [ -n "$stopped" ]; then
                log "stopped $run; its checkpoint is kept"
            else
                echo "$head" >> "$claims/$run.failures"
                log "failed $run with status $status (failure $((failures + 1)) of 2)"
            fi
        fi
        release "$run"
    done
    if [ "$stopped" = USR1 ]; then
        local out
        # shellcheck disable=SC2086  # MOLSCOUT_RESUBMIT holds several options
        if out=$(sbatch ${MOLSCOUT_RESUBMIT:-} benchmarks/slurm/run.sbatch "${configs[@]}" 2>&1); then
            log "resubmitted: $out"
        else
            log "resubmission refused: $out"
        fi
    fi
}

cd "${SLURM_SUBMIT_DIR:-.}"
export MOLSCOUT_STORE="${MOLSCOUT_STORE:-/scratch/$USER/molscout-store}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PATH="$HOME/.local/bin:$PATH"
molscout=${MOLSCOUT:-$MOLSCOUT_STORE/envs/molscout/bin/molscout}
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    if [ "$SLURM_ARRAY_TASK_ID" -ge "${#configs[@]}" ]; then
        echo "array task $SLURM_ARRAY_TASK_ID is out of range for ${#configs[@]} config(s)" >&2
        exit 2
    fi
    config=${configs[$SLURM_ARRAY_TASK_ID]}
elif [ "${#configs[@]}" -gt 1 ] || [ -n "${MOLSCOUT_RESUBMIT:-}" ]; then
    worker
    exit 0
else
    config=${configs[0]}
fi
if ! gpu_ok; then
    echo "bad GPU on $(hostname -s): $gpu_why" >&2
    exit 3
fi
echo "bench $config on $(hostname) at $(date -Is)"
"$molscout" bench "$config"
echo "done at $(date -Is)"
