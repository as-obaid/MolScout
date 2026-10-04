#!/bin/bash
# Run benchmark configs through the harness on one GPU node. Runs are offline (HF_HUB_OFFLINE);
# weights come from /scratch. Submit from the repository root:
#   sbatch benchmarks/slurm/run.sbatch benchmarks/configs/molscribe__uspto.yaml
#   sbatch --array=0-4 benchmarks/slurm/run.sbatch benchmarks/configs/molscribe__*.yaml
#   sbatch --partition=short --gres=none benchmarks/slurm/run.sbatch benchmarks/configs/molvec__uspto.yaml
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
configs=("$@")
if [ "${#configs[@]}" -eq 0 ]; then
    echo "usage: sbatch [--array=0-N] benchmarks/slurm/run.sbatch CONFIG [CONFIG ...]" >&2
    exit 2
fi
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    if [ "$SLURM_ARRAY_TASK_ID" -ge "${#configs[@]}" ]; then
        echo "array task $SLURM_ARRAY_TASK_ID is out of range for ${#configs[@]} config(s)" >&2
        exit 2
    fi
    config=${configs[$SLURM_ARRAY_TASK_ID]}
elif [ "${#configs[@]}" -eq 1 ]; then
    config=${configs[0]}
else
    echo "${#configs[@]} configs given without an array; submit with --array=0-$((${#configs[@]} - 1))" >&2
    exit 2
fi
cd "${SLURM_SUBMIT_DIR:-.}"
export MOLSCOUT_STORE="${MOLSCOUT_STORE:-/scratch/$USER/molscout-store}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PATH="$HOME/.local/bin:$PATH"
echo "bench $config on $(hostname) at $(date -Is)"
"${MOLSCOUT:-$MOLSCOUT_STORE/envs/molscout/bin/molscout}" bench "$config"
echo "done at $(date -Is)"
