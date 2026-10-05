#!/bin/bash
# Fill every H200 partition with benchmark workers up to the per-user running limits, and the CPU
# queue with MolVec workers. Each worker gets every config of its kind, longest first, and runs those
# not done and not claimed (see run.sbatch); running this again only tops up partitions with fewer
# workers than their limit. Run from the repository root:
#   bash benchmarks/slurm/submit.sh
set -euo pipefail
if [ ! -f pyproject.toml ] || [ ! -f benchmarks/slurm/run.sbatch ]; then
    echo "run benchmarks/slurm/submit.sh from the repository root" >&2
    exit 2
fi

# Estimated seconds = crops × seconds per crop, measured on an H200 (MolVec on a CPU).
estimates=$(awk 'BEGIN {
    crops["uspto"] = 5719; crops["uob"] = 5740; crops["molrecbench_wild"] = 5024; crops["clef"] = 992; crops["jpo"] = 450
    rate["decimer"] = 0.51; rate["molglyph"] = 0.39; rate["molscribe"] = 0.37; rate["molnextr"] = 0.34
    rate["ocsrglyph"] = 0.14; rate["molvec"] = 0.25
    for (i = 1; i < ARGC; i++) {
        name = ARGV[i]; sub(/.*\//, "", name); sub(/\.yaml$/, "", name); split(name, part, "__")
        if (!(part[1] in rate) || !(part[2] in crops)) { print "no estimate for " ARGV[i] > "/dev/stderr"; exit 1 }
        printf "%.0f %s\n", crops[part[2]] * rate[part[1]], ARGV[i]
    }
}' benchmarks/configs/*.yaml | sort -k1,1nr -k2,2)
gpu=() cpu=()
while read -r _ config; do
    case $config in
        */molvec__*) cpu+=("$config") ;;
        *) gpu+=("$config") ;;
    esac
done <<< "$estimates"

submit() {  # PARTITION WORKERS TIME GRES CONFIG...: top the partition up to WORKERS molscout workers
    local partition=$1 workers=$2 options queued i id ids="" note=""
    options="--partition=$1 --gres=$4 --time=$3 --signal=B:USR1@180 --job-name=molscout-$1"
    options="$options --output=benchmarks/slurm/logs/%x_%j.out --error=benchmarks/slurm/logs/%x_%j.err"
    shift 4
    queued=$(($(squeue -h -u "$(id -un)" -n "molscout-$partition" -o %i | wc -l)))
    for ((i = queued; i < workers; i++)); do
        # shellcheck disable=SC2086  # options holds several words
        if id=$(sbatch --parsable --export="ALL,MOLSCOUT_RESUBMIT=$options" $options benchmarks/slurm/run.sbatch "$@" 2>&1)
        then
            ids="$ids $id"
        else
            echo "$partition: $id" >&2
        fi
    done
    [ "$queued" -eq 0 ] || note="$queued already queued; "
    echo "$partition: ${note}submitted${ids:- none}"
}

submit gpu 4 08:00:00 gpu:h200:1 "${gpu[@]}"
submit gpu-short 2 02:00:00 gpu:h200:1 "${gpu[@]}"
submit gpu-interactive 2 02:00:00 gpu:h200:1 "${gpu[@]}"
submit sharing 2 01:00:00 gpu:h200:1 "${gpu[@]}"
submit short 5 24:00:00 none "${cpu[@]}"
