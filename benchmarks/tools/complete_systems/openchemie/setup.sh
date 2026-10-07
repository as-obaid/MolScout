#!/bin/bash
# Build the OpenChemIE environment and download its weights. Run as a CPU job, from the
# repository root on Explorer:
#   sbatch --time=04:00:00 --mem=64G benchmarks/slurm/setup.sbatch benchmarks/tools/complete_systems/openchemie/setup.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
store=${MOLSCOUT_STORE:-/scratch/$USER/molscout-store}
env=$store/envs/openchemie
poppler=$store/envs/openchemie-poppler
models=$store/models/openchemie
src=$store/src/OpenChemIE
commit=d9b50bb4fb094538eb071d6610f8e746a4fba32b
uv=${UV:-$HOME/.local/bin/uv}
export UV_PYTHON_INSTALL_DIR=$store/python UV_CACHE_DIR=$store/uv-cache UV_LINK_MODE=copy
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export CONDA_PKGS_DIRS=$store/conda-pkgs

[[ -d $src/.git ]] || git clone -q https://github.com/CrystalEye42/OpenChemIE.git "$src"
[[ -z $(git -C "$src" status --porcelain) ]] || { echo "local changes in $src" >&2; exit 1; }
[[ $(git -C "$src" rev-parse HEAD) == "$commit" ]] || git -C "$src" checkout -q "$commit"

# pdf2image needs the pdftoppm binary and the pdftotext package needs libpoppler-cpp; Explorer has
# neither, so poppler comes from conda-forge into its own prefix (run.py puts its bin on PATH).
if [[ ! -x $poppler/bin/pdftoppm ]]; then
    module load miniconda3/25.9.1
    conda create -y -q -p "$poppler" -c conda-forge --override-channels poppler=24.12.0 pkg-config
fi
"$poppler/bin/pdftoppm" -v 2>&1 | head -1

[[ -x $env/bin/python ]] || "$uv" venv -q -p 3.10 "$env"
"$uv" pip install -q -p "$env/bin/python" setuptools wheel
# pdftotext compiles against poppler; the rpath lets it find libpoppler-cpp without LD_LIBRARY_PATH.
# uv's Python is built with clang, which Explorer lacks, so g++ is named explicitly.
PKG_CONFIG_PATH=$poppler/lib/pkgconfig PATH=$poppler/bin:$PATH \
    CC=gcc CXX=g++ CFLAGS="-I$poppler/include" CXXFLAGS="-I$poppler/include" LDFLAGS="-L$poppler/lib -Wl,-rpath,$poppler/lib" \
    "$uv" pip install -q -p "$env/bin/python" --no-build-isolation pdftotext==2.2.2
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
grep -v '^pdftotext' "$here/requirements.txt" > "$tmp/requirements.txt"
"$uv" pip install -q -p "$env/bin/python" -r "$tmp/requirements.txt"
# OpenChemIE and its git dependencies, each at a pinned commit and without their own (clashing)
# requirements. OpenChemIE is built from an archive outside the clone.
git -C "$src" archive "$commit" | tar -x -C "$tmp" --one-top-level=OpenChemIE
"$uv" pip install -q -p "$env/bin/python" --no-deps --no-build-isolation \
    "$tmp/OpenChemIE" \
    "RxnScribe @ git+https://github.com/Ozymandias314/MolDetect.git@e97f1971bfca3ddbbff6688e88caf6e9e9c41a0b" \
    "MolScribe @ git+https://github.com/CrystalEye42/MolScribe.git@250f683f52f5050eb624870ccfd04bccbcaa27e1" \
    "ChemIENER @ git+https://github.com/Ozymandias314/ChemIENER.git@a169dad9f2729de00c89367e45b5daec39a8cc33" \
    "chemrxnextractor @ git+https://github.com/CrystalEye42/ChemRxnExtractor.git@86be85e966e8f54fc14dddbc48fdf0bfd3925d3b"

# Weights. OpenChemIE fetches them on first use from three places, so each is fetched here, into
# $models, and later runs are offline:
#   HF_HOME           Hugging Face: MolScribe swin_base_char_aux_1m.pth (loaded by OpenChemIE and,
#                     again, by MolDetect) and MolDetectCkpt best_hf.ckpt, at pinned revisions
#   FVCORE_CACHE      layoutparser's PubLayNet EfficientDet-D1. Its Dropbox link now serves a
#                     "File Deleted" page, so the file comes from layoutparser's own Hugging Face
#                     repo (same file name) and is put where the Dropbox handler looks for it
#   TORCH_HOME        effdet also loads the COCO tf_efficientdet_d1 weights from GitHub before the
#                     PubLayNet checkpoint replaces them
#   EASYOCR_MODULE_PATH  EasyOCR's English detector and recogniser, which MolDetect loads when built
mkdir -p "$models"
export HF_HOME=$models/hf FVCORE_CACHE=$models/iopath_cache TORCH_HOME=$models/torch EASYOCR_MODULE_PATH=$models/easyocr
export PATH=$poppler/bin:$PATH
"$env/bin/python" - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
import os
for repo, name, rev in (("yujieq/MolScribe", "swin_base_char_aux_1m.pth", "a0189776b7415b82795c7ee81eed311bf5c8724b"),
                        ("Ozymandias314/MolDetectCkpt", "best_hf.ckpt", "7cf4ba5ffdae2aec35c8339693f1dafd70e0613c")):
    print(hf_hub_download(repo, name, revision=rev))
    # OpenChemIE asks for these files without a revision, which offline means refs/main: point it at the pin
    refs = Path(os.environ["HF_HOME"]) / "hub" / ("models--" + repo.replace("/", "--")) / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(rev)
# layoutparser's cache path for the Dropbox link of tf_efficientdet_d1 (iopath keeps the query string)
dropbox = Path(os.environ["FVCORE_CACHE"]) / "s" / "gxy11xkkiwnpgog" / "publaynet-tf_efficientdet_d1.pth.tar?dl=1"
dropbox.parent.mkdir(parents=True, exist_ok=True)
fetched = hf_hub_download("layoutparser/efficientdet", "PubLayNet/tf_efficientdet_d1/publaynet-tf_efficientdet_d1.pth.tar",
                          revision="98c70a9ac8e021a7b52132828f5ca5765d2d8b0a")
print(fetched)
import shutil
shutil.copyfile(fetched, dropbox)
PY
# The first build downloads the rest: EasyOCR's models and the ResNet-50 backbone MolDetect
# initialises (torch hub), both overwritten or unused after loading but needed to build the model.
# pdftotext goes before torch: it loads the newer libstdc++ that libpoppler-cpp needs (torch would load Explorer's older one first).
"$env/bin/python" - <<'PY'
import pdftotext  # noqa: F401
import torch
from openchemie import OpenChemIE
model = OpenChemIE(device=torch.device("cpu"))
model.pdfparser, model.moldet, model.molscribe
PY

# Offline import check: build all three models the default call uses, as run.py does, from the stored weights.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$env/bin/python" - <<'PY'
import pdftotext  # noqa: F401
import torch
from openchemie import OpenChemIE
model = OpenChemIE(device=torch.device("cpu"))
model.pdfparser, model.moldet, model.molscribe
print("torch", torch.__version__, "cuda", torch.version.cuda, "built OpenChemIE's pdfparser, moldet and molscribe offline")
PY
(cd "$models" && find . \( -name '*.pth' -o -name '*.ckpt' -o -name '*.tar*' \) ! -name '*.lock' -print0 | sort -z | xargs -0 -n1 sh -c 'echo "$(sha256sum "$0" | cut -d" " -f1)  $0"')
