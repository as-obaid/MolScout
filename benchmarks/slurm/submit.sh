#!/bin/bash
# Fill the H200 partitions with benchmark workers up to the per-user running limits. Each worker gets
# every config of its kind, longest first, and runs those not done and not claimed (see run.sbatch);
# running this again only tops up partitions with fewer workers than their limit. Run from the
# repository root, with the kind of configs to submit (no argument is an error, so Type 1 runs are
# never resubmitted by accident):
#   bash benchmarks/slurm/submit.sh structure-readers   # Type 1: the 30 configs, MolVec on the CPU queue
#   bash benchmarks/slurm/submit.sh complete-systems    # Type 2: the 6 configs, recognised by run_dir
set -euo pipefail
if [ ! -f pyproject.toml ] || [ ! -f benchmarks/slurm/run.sbatch ]; then
    echo "run benchmarks/slurm/submit.sh from the repository root" >&2
    exit 2
fi
kind=${1:-}
case $kind in
    structure-readers | complete-systems) ;;
    *)
        echo "usage: bash benchmarks/slurm/submit.sh structure-readers|complete-systems" >&2
        exit 2 ;;
esac

# A run is current only when its code (tool folder, config, paper manifest, harness) has no uncommitted
# changes (molscout is-current), and a worker skips a run whose code has some, so refuse to submit until
# configs, tools, manifests and sources are committed.
keyed="benchmarks/configs benchmarks/tools benchmarks/slurm/run.sbatch src data/manifests pyproject.toml uv.lock"
# shellcheck disable=SC2086  # keyed holds several paths
dirty=$(git status --porcelain -uall -- $keyed) ||
    { echo "git status failed: submit from a git checkout" >&2; exit 1; }
if [ -n "$dirty" ]; then
    echo "uncommitted changes in ${keyed// /, }; commit them first" >&2
    echo "$dirty" >&2
    exit 1
fi

# Type 2 estimates: papers per dataset × seconds per paper, plus start-up seconds for each segment the
# run needs. DECIMER.ai and OpenChemIE were timed on an H200: 57.7 and 6.9 s per paper, and about 210
# and 140 s of start-up (model load and a warm-up paper), rounded up below; BioMiner's are not yet timed.
# A segment is at most the time limit less the 180 s USR1 warning: TYPE2_SEGMENT_SECONDS for the 2 h
# partitions that DECIMER.ai and OpenChemIE also use, TYPE2_BIOMINER_SEGMENT_SECONDS for BioMiner's one
# 8 h worker.
TYPE2_PAPERS="biovista=145 internal=6"
TYPE2_SECONDS_PER_PAPER="biominer=120 decimer_ai=60 openchemie=30"
TYPE2_STARTUP_SECONDS="biominer=600 decimer_ai=240 openchemie=180"
TYPE2_SEGMENT_SECONDS=7020
TYPE2_BIOMINER_SEGMENT_SECONDS=28620

type2=() type1=()
for config in benchmarks/configs/*.yaml; do
    if grep -q '^run_dir:.*complete_systems' "$config"; then type2+=("$config"); else type1+=("$config"); fi
done

# Estimated seconds = items × seconds per item, measured on an H200 (MolVec on a CPU); longest first.
estimate() {  # ITEMS RATES STARTUPS SEGMENT CONFIG...: "seconds config" lines; the first three are
    # "name=number ..." lists. Each started segment of SEGMENT seconds (0: one run) costs its STARTUP seconds.
    local items=$1 rates=$2 startups=$3 segment=$4
    shift 4
    awk -v items="$items" -v rates="$rates" -v startups="$startups" -v segment="$segment" 'BEGIN {
        n = split(items, a, " "); for (i = 1; i <= n; i++) { split(a[i], kv, "="); count[kv[1]] = kv[2] }
        n = split(rates, a, " "); for (i = 1; i <= n; i++) { split(a[i], kv, "="); rate[kv[1]] = kv[2] }
        n = split(startups, a, " "); for (i = 1; i <= n; i++) { split(a[i], kv, "="); startup[kv[1]] = kv[2] }
        for (i = 1; i < ARGC; i++) {
            name = ARGV[i]; sub(/.*\//, "", name); sub(/\.yaml$/, "", name); split(name, part, "__")
            if (!(part[1] in rate) || !(part[2] in count)) { print "no estimate for " ARGV[i] > "/dev/stderr"; exit 1 }
            seconds = count[part[2]] * rate[part[1]]
            if (segment > 0) {
                segments = int((seconds + segment - 1) / segment); if (segments < 1) segments = 1
                seconds += segments * startup[part[1]]
            }
            printf "%.0f %s\n", seconds, ARGV[i]
        }
    }' "$@" | sort -k1,1nr -k2,2
}

submit() {  # PARTITION WORKERS TIME GRES NAME EXTRA CONFIG...: top the partition up to WORKERS workers named NAME
    local partition=$1 workers=$2 options queued i id ids="" note="" name=$5 extra=$6
    options="--partition=$1 --gres=$4 --time=$3 --signal=B:USR1@180 --job-name=$name"
    options="$options${extra:+ $extra} --output=benchmarks/slurm/logs/%x_%j.out --error=benchmarks/slurm/logs/%x_%j.err"
    shift 6
    queued=$(($(squeue -h -u "$(id -un)" -n "$name" -o %i | wc -l)))
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

if [ "$kind" = structure-readers ]; then
    # Estimated seconds = crops × seconds per crop.
    estimates=$(estimate "uspto=5719 uob=5740 molrecbench_wild=5024 clef=992 jpo=450" \
        "decimer=0.51 molglyph=0.39 molscribe=0.37 molnextr=0.34 ocsrglyph=0.14 molvec=0.25" "" 0 "${type1[@]}")
    gpu=() cpu=()
    while read -r _ config; do
        case $config in
            */molvec__*) cpu+=("$config") ;;
            *) gpu+=("$config") ;;
        esac
    done <<< "$estimates"
    if [ "${#gpu[@]}" -eq 0 ]; then
        echo "no GPU configs: skipping their workers"
    else
        submit gpu 4 08:00:00 gpu:h200:1 molscout-gpu "" "${gpu[@]}"
        submit gpu-short 2 02:00:00 gpu:h200:1 molscout-gpu-short "" "${gpu[@]}"
        submit gpu-interactive 2 02:00:00 gpu:h200:1 molscout-gpu-interactive "" "${gpu[@]}"
        submit sharing 2 01:00:00 gpu:h200:1 molscout-sharing "" "${gpu[@]}"
    fi
    if [ "${#cpu[@]}" -eq 0 ]; then
        echo "no molvec configs: skipping its workers"
    else
        submit short 5 24:00:00 none molscout-short "" "${cpu[@]}"
    fi
else
    # Internal configs first (short canaries), then BioVista; longest first within each.
    ordered() {  # SEGMENT CONFIG...
        local segment=$1 all
        shift
        all=$(estimate "$TYPE2_PAPERS" "$TYPE2_SECONDS_PER_PAPER" "$TYPE2_STARTUP_SECONDS" "$segment" "$@")
        { grep '__internal\.yaml$' <<< "$all" || true; grep -v '__internal\.yaml$' <<< "$all" || true; } | cut -d' ' -f2
    }
    biominer=() others=()
    for config in "${type2[@]}"; do
        case $config in
            */biominer__*) biominer+=("$config") ;;
            *) others+=("$config") ;;
        esac
    done
    # Every Type 2 job takes one H200. BioMiner is one worker on gpu (its servers use fixed ports, so
    # --dependency=singleton keeps two BioMiner jobs, resubmissions included, from sharing a node).
    # Fallback, not used: 2 H200 on gpu-short.
    if [ "${#biominer[@]}" -eq 0 ]; then
        echo "no biominer configs: skipping its workers"
    else
        sorted=()
        while read -r config; do sorted+=("$config"); done < <(ordered "$TYPE2_BIOMINER_SEGMENT_SECONDS" "${biominer[@]}")
        biominer=("${sorted[@]}")
        submit gpu 1 08:00:00 gpu:h200:1 molscout2-biominer "--cpus-per-task=16 --mem=192G --dependency=singleton" "${biominer[@]}"
    fi
    if [ "${#others[@]}" -eq 0 ]; then
        echo "no decimer_ai or openchemie configs: skipping their workers"
    else
        sorted=()
        while read -r config; do sorted+=("$config"); done < <(ordered "$TYPE2_SEGMENT_SECONDS" "${others[@]}")
        others=("${sorted[@]}")
        submit gpu 3 08:00:00 gpu:h200:1 molscout2-gpu-1gpu "" "${others[@]}"
        submit gpu-short 2 02:00:00 gpu:h200:1 molscout2-gpu-short-1gpu "" "${others[@]}"
        submit gpu-interactive 2 02:00:00 gpu:h200:1 molscout2-gpu-interactive-1gpu "" "${others[@]}"
    fi
fi
