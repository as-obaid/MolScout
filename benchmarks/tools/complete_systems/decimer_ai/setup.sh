#!/bin/bash
# Build the DECIMER.ai environment and download its weights. Run as a CPU job, from the
# repository root on Explorer (the build is long: TensorFlow, CUDA libraries, three git installs):
#   sbatch --time=04:00:00 --mem=64G benchmarks/slurm/setup.sbatch benchmarks/tools/complete_systems/decimer_ai/setup.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
store=${MOLSCOUT_STORE:-/scratch/$USER/molscout-store}
env=$store/envs/decimer_ai
models=$store/models/decimer_ai
src=$store/src/DECIMER.ai
commit=1fc858a35eb89de636c1002f1ca2d12afe04cdd8
# The commits docker/Dockerfile installs with --no-deps (Segmentation's v1.3.0 tag is 2e0b78f).
transformer=3db69546ed706af2be474e774de888bd3a067c6e
segmentation=2e0b78fc6d117a0efa816c280082b3462eed0865
classifier=146b00be2fe6f8fa6670a4255969d4747502b7f2
uv=${UV:-$HOME/.local/bin/uv}
export UV_PYTHON_INSTALL_DIR=$store/python UV_CACHE_DIR=$store/uv-cache UV_LINK_MODE=copy
# DECIMER fetches the Image Transformer on first import into pystow.join("DECIMER-V2").
export PYSTOW_HOME=$models

# The app's code (its Python scripts hold the pipeline run.py reproduces) from the pinned commit.
[[ -d $src/.git ]] || git clone -q https://github.com/OBrink/DECIMER.ai.git "$src"
[[ -z $(git -C "$src" status --porcelain) ]] || { echo "local changes in $src" >&2; exit 1; }
[[ $(git -C "$src" rev-parse HEAD) == "$commit" ]] || git -C "$src" checkout -q "$commit"

# Python 3.9 as in the app's image; uv fetches it into the store.
[[ -x $env/bin/python ]] || "$uv" venv -q -p 3.9 "$env"
"$uv" pip install -q -p "$env/bin/python" -r "$here/requirements.txt"
# The Dockerfile installs keras 2.3.0 after TensorFlow 2.7.0 (whose metadata asks for keras 2.7), so
# it goes in on its own, without dependency resolution.
"$uv" pip install -q -p "$env/bin/python" --no-deps keras==2.3.0
for spec in "DECIMER-Image_Transformer@$transformer" "DECIMER-Image-Segmentation@$segmentation"; do
    "$uv" pip install -q -p "$env/bin/python" --no-deps "git+https://github.com/Kohulan/${spec}"
done
"$uv" pip install -q -p "$env/bin/python" --no-deps "git+https://github.com/Iagea/DECIMER-Image-Classifier@$classifier"

# pdf2image needs poppler's pdftoppm, which Explorer lacks; the app's image has Debian's poppler-utils.
if [[ ! -x $env/poppler/bin/pdftoppm ]]; then
    source /usr/share/Modules/init/bash
    module load miniconda3/25.9.1
    conda create -y -q -p "$env/poppler" --override-channels -c conda-forge poppler
fi
"$env/poppler/bin/pdftoppm" -v 2>&1 | head -1

# DECIMER Segmentation downloads mask_rcnn_molecule.h5 into its package folder on first import
# (Zenodo 10142866, the URL in v1.3.0); fetch it here, keep a copy under models, and put it where
# the package looks, so that runs download nothing.
mkdir -p "$models"
h5=$models/mask_rcnn_molecule.h5
[[ -s $h5 ]] || curl -fsSL -o "$h5" "https://zenodo.org/records/10142866/files/mask_rcnn_molecule.h5?download=1"
site=$("$env/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "d42382a22585f98d8d7c8b480d2e0fcab58a9fa503133199fac839cb0f5fd4f9  $h5" | sha256sum -c -
cp -f "$h5" "$site/decimer_segmentation/mask_rcnn_molecule.h5"

# Importing DECIMER downloads and loads the Image Transformer (Zenodo 8300489, models.zip).
"$env/bin/python" -c "import DECIMER"

# Self-check, offline like a benchmark job and without the proxy so any download would fail: the
# Transformer reads upstream's caffeine test image and the Classifier calls it a structure.
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL -o "$tmp/caffeine.png" \
    "https://raw.githubusercontent.com/Kohulan/DECIMER-Image_Transformer/$transformer/Tests/caffeine.png"
echo "31df300ab9cae7a4761a278a952f22881d058ec6c4a39baba537d370e528ac8b  $tmp/caffeine.png" | sha256sum -c -
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY HF_HUB_OFFLINE=1 \
    "$env/bin/python" - "$tmp/caffeine.png" "$site" <<'PY'
import hashlib, sys
from pathlib import Path
import numpy, tensorflow as tf
from DECIMER import predict_SMILES
from DECIMER.decimer import default_path
from decimer_image_classifier import DecimerImageClassifier
import decimer_segmentation  # loads the Mask R-CNN weights

print("tensorflow", tf.__version__, "numpy", numpy.__version__,
      "built for CUDA", tf.sysconfig.get_build_info().get("cuda_version"))
smiles = predict_SMILES(sys.argv[1])
if smiles != "CN1C=NC2=C1C(=O)N(C)C(=O)N2C":
    sys.exit(f"self-check failed: caffeine read as {smiles!r}")
print("self-check ok: caffeine read as", smiles)
classifier = DecimerImageClassifier()
score = classifier.get_classifier_score(sys.argv[1])
if not classifier.is_chemical_structure(sys.argv[1]):
    sys.exit(f"self-check failed: caffeine scored {score} by the classifier")
print("classifier ok: caffeine score", score)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

# A folder's sha256: sorted "<sha256>  <relative path>" lines of every file, hidden ones included.
def folder_sha256(folder):
    files = sorted((f.relative_to(folder).as_posix(), f) for f in folder.rglob("*") if f.is_file())
    lines = "".join(f"{sha256_file(f)}  {rel}\n" for rel, f in files)
    return hashlib.sha256(lines.encode()).hexdigest(), len(files)

for folder in sorted(Path(default_path).iterdir()):
    if folder.is_dir():
        digest, count = folder_sha256(folder)
        print(digest, folder, f"({count} files)")
    else:
        print("file", folder)
classifier_dir = Path(sys.modules["decimer_image_classifier"].__file__).parent / "model"
digest, count = folder_sha256(classifier_dir)
print(digest, classifier_dir, f"({count} files)")
PY
sha256sum "$h5"
for f in convert_pdf_to_images decimer_segmentation_server decimer_classifier_server decimer_predictor_server; do
    sha256sum "$src/app/Python/$f.py"
done
"$uv" pip freeze -p "$env/bin/python" | grep -i -E "^(tensorflow|keras|numpy|protobuf|decimer|nvidia|opencv|pdf2image|pillow|h5py)"
