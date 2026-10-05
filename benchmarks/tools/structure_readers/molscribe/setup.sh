#!/bin/bash
# Build the MolScribe environment and download its checkpoint. Run as a CPU job, from the
# repository root on Explorer:
#   sbatch benchmarks/slurm/setup.sbatch benchmarks/tools/structure_readers/molscribe/setup.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
store=${MOLSCOUT_STORE:-/scratch/$USER/molscout-store}
env=$store/envs/molscribe
models=$store/models/molscribe
uv=${UV:-$HOME/.local/bin/uv}
export UV_PYTHON_INSTALL_DIR=$store/python UV_CACHE_DIR=$store/uv-cache UV_LINK_MODE=copy
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

[[ -x $env/bin/python ]] || "$uv" venv -q -p 3.10 "$env"
"$uv" pip install -q -p "$env/bin/python" -r "$here/requirements.txt"
# MolScribe's own requirements pin opencv-python next to the headless build; install it alone.
"$uv" pip install -q -p "$env/bin/python" --no-deps \
    "MolScribe @ git+https://github.com/thomas0809/MolScribe.git@7296a30413eb55436702011efdff78131f66d162"

mkdir -p "$models"
"$env/bin/python" - "$models" <<'PY'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download("yujieq/MolScribe", "swin_base_char_aux_1m680k.pth",
                      revision="a0189776b7415b82795c7ee81eed311bf5c8724b", local_dir=sys.argv[1]))
PY
"$env/bin/python" -c "import torch, molscribe; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
sha256sum "$models/swin_base_char_aux_1m680k.pth"
