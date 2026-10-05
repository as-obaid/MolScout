#!/bin/bash
# Build the OCSRGlyph environment and download its checkpoint. Run as a CPU job, from the
# repository root on Explorer:
#   sbatch benchmarks/slurm/setup.sbatch benchmarks/tools/structure_readers/ocsrglyph/setup.sh
# The environment file is upstream's uv.lock at the pinned glyph commit (Python 3.12,
# torch 2.11.0+cu128, timm 1.0.28, rdkit 2026.3.3). `uv sync --frozen` installs it unchanged
# into $MOLSCOUT_STORE/src/glyph/.venv, the python that tool.yaml names.
set -euo pipefail
store=${MOLSCOUT_STORE:-/scratch/$USER/molscout-store}
src=$store/src/glyph
commit=0bf782f863d26b041ace157668928ef07c38b972
models=$store/models/ocsrglyph
# Hugging Face LFS sha256 of model.pth at revision da0d049.
model_sha256=6f966e0ea5bf2c5cf1ccd046c1523cb166acc76c556ad0745964f536735fb6be
uv=${UV:-$HOME/.local/bin/uv}
export UV_PYTHON_INSTALL_DIR=$store/python UV_CACHE_DIR=$store/uv-cache UV_LINK_MODE=copy
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

[[ -d $src/.git ]] || git clone -q https://github.com/EdisonScientific/glyph.git "$src"
if [[ $(git -C "$src" rev-parse HEAD) != "$commit" ]]; then
    git -C "$src" fetch -q origin "$commit"
    git -C "$src" checkout -q --detach "$commit"
fi
[[ $(git -C "$src" rev-parse HEAD) == "$commit" ]] || { echo "glyph is not at $commit" >&2; exit 1; }
[[ -z $(git -C "$src" status --porcelain --untracked-files=no) ]] \
    || { echo "glyph clone has local changes" >&2; exit 1; }
(cd "$src" && "$uv" sync -q --frozen --extra ocsr --python 3.12)
py=$src/.venv/bin/python

mkdir -p "$models"
"$py" - "$models" <<'PY'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download("EdisonScientific/OCSRGlyph", "model.pth",
                      revision="da0d049fa56effd3a07ecb15c715efdd78d9e8a0", local_dir=sys.argv[1]))
PY
echo "$model_sha256  $models/model.pth" | sha256sum -c -

# Nothing else is fetched on first use: the vocabulary ships in the package, the Swin backbone
# is built with pretrained=False and the recipe comes from the checkpoint. Check offline on CPU.
HF_HUB_OFFLINE=1 "$py" - "$models/model.pth" "$src/examples/imatinib.png" <<'PY'
import sys
import importlib.metadata as m
import torch
from glyph.ocsr.predict import OCSRPredictor
print("glyph", m.version("glyph"), "torch", torch.__version__, "cuda", torch.version.cuda,
      "timm", m.version("timm"), "rdkit", m.version("rdkit"), "python", sys.version.split()[0])
smiles = OCSRPredictor(sys.argv[1], device="cpu").predict(sys.argv[2])
expected = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
print("imatinib.png ->", smiles, "(matches README)" if smiles == expected else "(README expects %s)" % expected)
sys.exit(smiles != expected)
PY
sha256sum "$models/model.pth"
