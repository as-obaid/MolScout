#!/bin/bash
# Build the DECIMER environment and download its models. Run as a CPU job, from the
# repository root on Explorer:
#   sbatch benchmarks/slurm/setup.sbatch benchmarks/tools/structure_readers/decimer/setup.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
store=${MOLSCOUT_STORE:-/scratch/$USER/molscout-store}
env=$store/envs/decimer
uv=${UV:-$HOME/.local/bin/uv}
export UV_PYTHON_INSTALL_DIR=$store/python UV_CACHE_DIR=$store/uv-cache UV_LINK_MODE=copy
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}
# DECIMER downloads its models on first import into pystow.join("DECIMER-V2").
export PYSTOW_HOME=$store/models/decimer

[[ -x $env/bin/python ]] || "$uv" venv -q -p 3.12 "$env"
"$uv" pip install -q -p "$env/bin/python" -r "$here/requirements.txt"
# The PyPI 2.8.0 wheel's DECIMER/ is identical to d927ed1; --no-deps keeps opencv-python out.
"$uv" pip install -q -p "$env/bin/python" --no-deps decimer==2.8.0

# Importing DECIMER downloads and loads both models (Zenodo 8300489 and 10781330).
mkdir -p "$PYSTOW_HOME"
"$env/bin/python" -c "import DECIMER"

# Self-check: upstream's test image (from the clone, else fetched at the pinned commit) must read
# as caffeine, offline like a benchmark job and without the proxy, so any download would fail.
caffeine=$store/src/DECIMER-Image_Transformer/tests/caffeine.png
if [[ ! -f $caffeine ]]; then
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    caffeine=$tmp/caffeine.png
    curl -fsSL -o "$caffeine" \
        https://raw.githubusercontent.com/Kohulan/DECIMER-Image_Transformer/d927ed1228d8db66cc84678818ff9466cb850f8c/tests/caffeine.png
fi
echo "31df300ab9cae7a4761a278a952f22881d058ec6c4a39baba537d370e528ac8b  $caffeine" | sha256sum -c -
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY HF_HUB_OFFLINE=1 \
    "$env/bin/python" - "$caffeine" <<'PY'
import hashlib, sys
from pathlib import Path
import keras, numpy, tensorflow as tf
from DECIMER import predict_SMILES
from DECIMER.decimer import default_path

print("tensorflow", tf.__version__, "keras", keras.__version__, "numpy", numpy.__version__,
      "built for CUDA", tf.sysconfig.get_build_info().get("cuda_version"))
smiles = predict_SMILES(sys.argv[1])
if smiles != "CN1C=NC2=C1C(=O)N(C)C(=O)N2C":
    sys.exit(f"self-check failed: caffeine read as {smiles!r}")
print("self-check ok: caffeine read as", smiles)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

# A folder's sha256: sorted "<sha256>  <relative path>" lines of every file, hidden ones included.
for folder in sorted(Path(default_path).iterdir()):
    if folder.is_dir():
        files = sorted((f.relative_to(folder).as_posix(), f) for f in folder.rglob("*") if f.is_file())
        lines = "".join(f"{sha256_file(f)}  {rel}\n" for rel, f in files)
        print(hashlib.sha256(lines.encode()).hexdigest(), folder, f"({len(files)} files)")
    else:
        print("file", folder)
PY
