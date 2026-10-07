#!/bin/bash
# Where each benchmark run stands, then the molscout jobs in the queue. Run from the repository root:
#   bash benchmarks/slurm/status.sh [structure-readers|complete-systems]   (default: both)
# done: results at HEAD with no uncommitted changes. running: claimed by a live job. failed: given up
# after 2 failures at HEAD. partial: a checkpoint is left. stale: results a worker will redo.
# Checkpoint rows are counted by line; a complete system (paper config) shows its finished papers, counted
# from .checkpoints/<run>/predictions.papers.jsonl.
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
head=$(git rev-parse HEAD 2>/dev/null || true)

images() {  # the images in the folder a config names
    local folder
    folder=$(sed -n 's/^images: *//p' "$1")
    [ -d "$folder" ] || { echo "?"; return; }
    find "$folder" -maxdepth 1 -type f ! -name '.*' \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \
        -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.gif' -o -iname '*.bmp' \) | wc -l | tr -d ' '
}

add() { detail="${detail:+$detail, }$1"; }

printf '%-30s %-8s %s\n' RUN STATE DETAIL
states=""
for config in benchmarks/configs/*.yaml; do
    run=$(basename "$config" .yaml)
    if grep -q '^run_dir:.*complete_systems' "$config"; then papers=1; else papers=""; fi
    if [ "$kind" = structure-readers ] && [ -n "$papers" ]; then continue; fi
    if [ "$kind" = complete-systems ] && [ -z "$papers" ]; then continue; fi
    meta=$results/$run/meta.json
    checkpoint=$results/.checkpoints/$run/predictions.csv
    commit="" partition="" detail=""
    if [ -f "$meta" ]; then
        commit=$(python3 -c '
import json, sys
git = json.load(open(sys.argv[1])).get("git") or {}
print(git.get("commit") or "unknown", "dirty" if git.get("dirty") is not False else "clean")' "$meta") || commit=""
    fi
    job=$(cat "$claims/$run/job" 2>/dev/null) || job=""
    if [ -n "$job" ]; then partition=$(squeue -h -j "$job" -t RUNNING,COMPLETING -o %P 2>/dev/null) || partition=""; fi
    failures=$(grep -cxF "$head" "$claims/$run.failures" 2>/dev/null) || failures=0
    if [ "$commit" = "$head clean" ]; then
        state="done"
        add "${head:0:7}"
    elif [ -n "$partition" ]; then
        state=running
        add "job $job ($partition)"
    elif [ "$failures" -ge 2 ]; then
        state=failed
        add "$failures failures at HEAD"
    elif [ -f "$checkpoint" ]; then
        state=partial
    elif [ -n "$commit" ]; then
        state=stale
        if [ "${commit% *}" = "$head" ]; then add "${head:0:7}, uncommitted changes"; else add "${commit:0:7}, not HEAD"; fi
    else
        state=pending
    fi
    if [ -n "$papers" ]; then
        if [ -f "${checkpoint%.csv}.papers.jsonl" ]; then add "$(grep -c . "${checkpoint%.csv}.papers.jsonl" || true) papers"; fi
    elif [ -f "$checkpoint" ]; then
        rows=$(($(wc -l < "$checkpoint") - 1))
        add "$((rows > 0 ? rows : 0))/$(images "$config") rows"
    fi
    if [ "$failures" -gt 0 ] && [ "$state" != failed ]; then
        add "$failures failure$([ "$failures" -eq 1 ] || echo s) at HEAD"
    fi
    printf '%-30s %-8s %s\n' "$run" "$state" "$detail"
    states="$states$state"$'\n'
done
printf '%s' "$states" | sort | uniq -c | awk '{ printf "%s%s %s", (NR > 1 ? ", " : ""), $1, $2 } END { print "" }'
echo
squeue -u "$(id -un)" -o '%.10i %.24j %.16P %.9T %.10M %.10L %R' | awk 'NR == 1 || $2 ~ /^molscout/'
