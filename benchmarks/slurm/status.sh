#!/bin/bash
# Where each benchmark run stands, then the molscout jobs in the queue. Run from the repository root:
#   bash benchmarks/slurm/status.sh [structure-readers|complete-systems]   (default: both)
# done: results current (molscout is-current: made from the code the run has at HEAD, none of it
# uncommitted). running: claimed by a live job. blocked: its code has uncommitted changes (or lies outside
# the repository), so workers skip it. failed: given up after 2 failures with the run's code at HEAD
# (remove its .claims/<run>.failures to retry). partial: a checkpoint is left. stale: results a worker will
# redo. MOLSCOUT names the molscout command (default: the one run.sbatch uses, in $MOLSCOUT_STORE).
# Checkpoint rows are counted by line; a complete system (paper config) shows its finished papers, counted
# from .checkpoints/<run>/predictions.papers.jsonl (lines marking an attempt are not counted).
set -euo pipefail
kind=${1:-both}
case $kind in
    both | structure-readers | complete-systems) ;;
    *)
        echo "usage: bash benchmarks/slurm/status.sh [structure-readers|complete-systems]" >&2
        exit 2 ;;
esac
results=benchmarks/results
claims=$results/.claims
export MOLSCOUT_STORE="${MOLSCOUT_STORE:-/scratch/$USER/molscout-store}"  # as in run.sbatch: configs use it
molscout=${MOLSCOUT:-$MOLSCOUT_STORE/envs/molscout/bin/molscout}
if ! command -v "$molscout" > /dev/null; then
    echo "status.sh: $molscout not found; set MOLSCOUT (or MOLSCOUT_STORE) to find molscout" >&2
    exit 1
fi

images() {  # the images in the folder a config names
    local folder
    folder=$(sed -n 's/^images: *//p' "$1")
    [ -d "$folder" ] || { echo "?"; return; }
    find "$folder" -maxdepth 1 -type f ! -name '.*' \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \
        -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.gif' -o -iname '*.bmp' \) | wc -l | tr -d ' '
}

add() { detail="${detail:+$detail, }$1"; }

is_papers() { grep -q '^run_dir:.*complete_systems' "$1"; }  # whether config $1 is a complete system's

selected=()
for config in benchmarks/configs/*.yaml; do
    if is_papers "$config"; then papers=1; else papers=""; fi
    if [ "$kind" = structure-readers ] && [ -n "$papers" ]; then continue; fi
    if [ "$kind" = complete-systems ] && [ -z "$papers" ]; then continue; fi
    selected+=("$config")
done
# One line per config, in order: RUN current|not-current CODE WHY; CODE, the run's code fingerprint at
# HEAD, keys the failures as in run.sbatch. Exit status 1 only means some run is not current.
verdicts=""
if [ "${#selected[@]}" -gt 0 ]; then
    verdicts=$("$molscout" is-current "${selected[@]}") || [ $? -eq 1 ] || {
        echo "status.sh: $molscout is-current failed" >&2
        exit 1
    }
fi

printf '%-30s %-8s %s\n' RUN STATE DETAIL
states=""
while read -r run verdict code why <&3; do
    [ -n "$run" ] || continue
    config=benchmarks/configs/$run.yaml
    if is_papers "$config"; then papers=1; else papers=""; fi
    meta=$results/$run/meta.json
    checkpoint=$results/.checkpoints/$run/predictions.csv
    partition="" detail=""
    job=$(cat "$claims/$run/job" 2>/dev/null) || job=""
    if [ -n "$job" ]; then partition=$(squeue -h -j "$job" -t RUNNING,COMPLETING -o %P 2>/dev/null) || partition=""; fi
    failures=$(grep -cxF -e "$code" "$claims/$run.failures" 2>/dev/null) || failures=0
    if [ "$verdict" = current ]; then
        state="done"
        add "$why"
    elif [ -n "$partition" ]; then
        state=running
        add "job $job ($partition)"
    elif [ "$verdict" = blocked ]; then
        state=blocked
        add "$why"
    elif [ "$failures" -ge 2 ]; then
        state=failed
        add "$failures failures at code ${code:0:7} (remove $claims/$run.failures to retry)"
    elif [ -f "$checkpoint" ]; then
        state=partial
    elif [ -f "$meta" ]; then
        state=stale
        add "$why"
    else
        state=pending
    fi
    if [ -n "$papers" ]; then
        if [ -f "${checkpoint%.csv}.papers.jsonl" ]; then add "$(grep -v '"attempted"' "${checkpoint%.csv}.papers.jsonl" | grep -c . || true) papers"; fi
    elif [ -f "$checkpoint" ]; then
        rows=$(($(wc -l < "$checkpoint") - 1))
        add "$((rows > 0 ? rows : 0))/$(images "$config") rows"
    fi
    if [ "$failures" -gt 0 ] && [ "$state" != failed ]; then
        add "$failures failure$([ "$failures" -eq 1 ] || echo s) at code ${code:0:7}"
    fi
    printf '%-30s %-8s %s\n' "$run" "$state" "$detail"
    states="$states$state"$'\n'
done 3<<< "$verdicts"
printf '%s' "$states" | sort | uniq -c | awk '{ printf "%s%s %s", (NR > 1 ? ", " : ""), $1, $2 } END { print "" }'
echo
squeue -u "$(id -un)" -o '%.10i %.24j %.16P %.9T %.10M %.10L %R' | awk 'NR == 1 || $2 ~ /^molscout/'
