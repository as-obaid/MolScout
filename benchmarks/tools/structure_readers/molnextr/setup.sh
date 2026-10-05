#!/bin/bash
# Build the MolNexTR environment and download its checkpoint. Run as a CPU job, from the
# repository root on Explorer:
#   sbatch benchmarks/slurm/setup.sbatch benchmarks/tools/structure_readers/molnextr/setup.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
store=${MOLSCOUT_STORE:-/scratch/$USER/molscout-store}
env=$store/envs/molnextr
models=$store/models/molnextr
uv=${UV:-$HOME/.local/bin/uv}
export UV_PYTHON_INSTALL_DIR=$store/python UV_CACHE_DIR=$store/uv-cache UV_LINK_MODE=copy
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

[[ -x $env/bin/python ]] || "$uv" venv -q -p 3.10 "$env"
"$uv" pip install -q -p "$env/bin/python" -r "$here/requirements.txt"
# MolNexTR's own requirements pin opencv-python next to the headless build; install it alone.
"$uv" pip install -q -p "$env/bin/python" --no-deps \
    "MolNexTR @ git+https://github.com/CYF2000127/MolNexTR.git@6f6502b4ed9733dba8b1ee45d2da474576683194"

# The weights sit in a Hugging Face dataset repo. Its 2024 molnextr_best.pth was the MolScribe
# checkpoint byte for byte; the file was replaced on 2026-01-10, so the revision is pinned.
mkdir -p "$models"
"$env/bin/python" - "$models" <<'PY'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download("CYF200127/MolNexTR", "molnextr_best.pth", repo_type="dataset",
                      revision="9ac2da6fa553e1d053af3d8d534b8a6c74b79749", local_dir=sys.argv[1]))
PY
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$env/bin/python" -c "import torch, MolNexTR; from MolNexTR.model import molnextr; \
print('MolNexTR', MolNexTR.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
sha256sum "$models/molnextr_best.pth"
