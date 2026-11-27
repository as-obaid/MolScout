#!/bin/bash
# Build one tool's environment and download its weights on a CPU node. Compute nodes reach
# PyPI, GitHub and Hugging Face through the cluster proxy, and create files on /scratch far
# faster than login nodes do. Submit from the repository root:
#   sbatch benchmarks/slurm/setup.sbatch benchmarks/tools/structure_readers/molscribe/setup.sh
# On a cluster that charges jobs to an account, add --account=<account> (or set SBATCH_ACCOUNT).
#SBATCH --job-name=molscout-setup
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=benchmarks/slurm/logs/%x_%j.out
set -euo pipefail
script=${1:?usage: sbatch benchmarks/slurm/setup.sbatch <tool>/setup.sh}
echo "setup $script on $(hostname) at $(date -Is)"
bash "$script"
echo "done at $(date -Is)"
