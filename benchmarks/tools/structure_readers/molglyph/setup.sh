#!/bin/bash
# Build the MolGlyph environment and download its checkpoint. Run as a CPU job, from the
# repository root on Explorer:
#   sbatch benchmarks/slurm/setup.sbatch benchmarks/tools/structure_readers/molglyph/setup.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
store=${MOLSCOUT_STORE:-/scratch/$USER/molscout-store}
env=$store/envs/molglyph
models=$store/models/molglyph
src=$store/src/BioMiner
commit=17c6161b40e34e2c377b2871f760baab3fdeea06
uv=${UV:-$HOME/.local/bin/uv}
export UV_PYTHON_INSTALL_DIR=$store/python UV_CACHE_DIR=$store/uv-cache UV_LINK_MODE=copy
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}
# The MolGlyph repository is gated, and HF_HOME no longer holds the token.
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}

# The code is BioMiner/MolScribe (package molscribe) plus BioMiner/commons/molglyph_utils.py,
# which run.py loads from this clone; tool.yaml pins that file and abbrevs.csv by sha256.
[[ -d $src/.git ]] || git clone -q https://github.com/jiaxianyan/BioMiner.git "$src"
[[ $(git -C "$src" rev-parse HEAD) == "$commit" ]] || git -C "$src" checkout -q "$commit"
git -C "$src" diff --quiet HEAD -- BioMiner/commons || { echo "local changes in $src" >&2; exit 1; }

[[ -x $env/bin/python ]] || "$uv" venv -q -p 3.10 "$env"
"$uv" pip install -q -p "$env/bin/python" -r "$here/requirements.txt"
# Install the package from the pinned commit, built outside the clone. Its setup.py pins
# opencv-python, timm 0.4.12 and albumentations 1.1.0, so it goes in alone.
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git -C "$src" archive "$commit" BioMiner/MolScribe | tar -x -C "$tmp"
"$uv" pip install -q -p "$env/bin/python" --no-deps --reinstall-package MolScribe "$tmp/BioMiner/MolScribe"

mkdir -p "$models"
"$env/bin/python" - "$models" <<'PY'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download("jiaxianustc/MolGlyph", "molglyph_large.pt",
                      revision="18db8e0e09dbde08dfc82604ccb815c47d36bff5", local_dir=sys.argv[1]))
PY

# MolGlyph loads weights with strict=False and ignores mismatched keys; check that all match,
# offline, as runs are.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
    "$env/bin/python" - "$models/molglyph_large.pt" <<'PY'
import sys
import torch, timm, albumentations, cv2, rdkit
from molscribe import MolGlyph
print("torch", torch.__version__, "cuda", torch.version.cuda, "timm", timm.__version__,
      "albumentations", albumentations.__version__, "opencv", cv2.__version__, "rdkit", rdkit.__version__)
states = torch.load(sys.argv[1], map_location="cpu")
print("args", {k: states["args"].get(k) for k in ("encoder", "decoder", "formats", "input_size", "vocab_file")})
model = MolGlyph(sys.argv[1])
for part in ("encoder", "decoder"):
    have = set(getattr(model, part).state_dict())
    want = {k.replace("module.", "") for k in states[part]}
    print(part, len(have), "tensors, missing", sorted(have - want)[:5], "unexpected", sorted(want - have)[:5])
    assert have == want, f"{part} keys do not match the checkpoint"
PY
sha256sum "$models/molglyph_large.pt" "$src/BioMiner/commons/molglyph_utils.py" "$src/BioMiner/commons/abbrevs.csv"
