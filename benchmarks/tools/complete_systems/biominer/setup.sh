#!/bin/bash
# Build BioMiner's two environments, pack a copy of each, and download its weights. Run as a CPU job,
# from the repository root on Explorer (two environments and 70 GB of weights need more than
# setup.sbatch's defaults):
#   sbatch --time=08:00:00 --mem=64G benchmarks/slurm/setup.sbatch benchmarks/tools/complete_systems/biominer/setup.sh
# The environments are BioMiner's two (main_environment.yml and vllm_environment.yml): "biominer" runs
# the pipeline and the MinerU, MolDetV2 and MolGlyph servers, "biominer_vllm" serves BioMiner-Instruct.
# Exits 1 if a weight repository could not be fetched (BioMiner-Instruct is gated: the Hugging Face
# account of the token must have requested access); everything else still runs.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
store=${MOLSCOUT_STORE:-/scratch/$USER/molscout-store}
env=$store/envs/biominer
vllm_env=$store/envs/biominer_vllm
models=$store/models/biominer
molglyph=$store/models/molglyph/molglyph_large.pt
src=$store/src/BioMiner
commit=17c6161b40e34e2c377b2871f760baab3fdeea06
uv=${UV:-$HOME/.local/bin/uv}
export UV_PYTHON_INSTALL_DIR=$store/python UV_CACHE_DIR=$store/uv-cache UV_LINK_MODE=copy
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}
# Only compileall writes bytecode, after the mtimes are refreshed (see below); Python runs here don't.
export PYTHONDONTWRITEBYTECODE=1
# BioMiner-Instruct is gated, and HF_HOME does not hold the token.
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}

# The clone at the pinned commit; run.py copies the code out of it with git archive at every run.
[[ -d $src/.git ]] || git clone -q https://github.com/jiaxianyan/BioMiner.git "$src"
[[ $(git -C "$src" rev-parse HEAD) == "$commit" ]] || git -C "$src" checkout -q "$commit"
# GIT_OPTIONAL_LOCKS=0: git status does not rewrite the clone's index while it looks
[[ -z $(GIT_OPTIONAL_LOCKS=0 git -C "$src" status --porcelain) ]] || { echo "local changes in $src" >&2; exit 1; }

# Main environment: a conda prefix with only Python and the two programs BioMiner calls (pdftoppm from
# poppler, java for OPSIN), then the pip packages with uv. Python and openjdk are main_environment.yml's
# versions. Its poppler 24.09.0 is only on Anaconda's own channel, whose terms of service conda would
# need accepted; conda-forge's nearest release, 24.08.0, renders the pages instead.
# /scratch purges files whose mtime is about 60 days old, and conda's files keep their package build
# dates: on 2026-10-07 the purge removed most of this prefix's standard library, a day after it was built.
# A prefix whose Python cannot start is rebuilt, from a package cache of its own (the shared one is
# purged too), and the end of this script gives every file in both environments a current mtime.
if [[ -e $env ]] && ! "$env/bin/python" -c "import os, encodings" >/dev/null 2>&1; then
    echo "$env is broken (its Python cannot start); rebuilding it"
    rm -rf "$env"
fi
if [[ ! -x $env/bin/python ]]; then
    set +u  # the module scripts read unset variables
    type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
    module load miniconda3/25.9.1
    set -u
    pkgs=$(mktemp -d "$store/conda-pkgs-biominer.XXXXXX")
    CONDA_PKGS_DIRS=$pkgs conda create -q -y -p "$env" --override-channels -c conda-forge \
        python=3.10.14 poppler=24.08.0 openjdk=21.0.6
    rm -rf "$pkgs"
fi
# ProDy has no Linux wheel for Python 3.10 and its setup.py imports numpy, so it builds against the
# environment's own pinned numpy and setuptools, installed first, instead of in an isolated build.
"$uv" pip install -q -p "$env/bin/python" numpy==1.26.4 setuptools==75.8.0
"$uv" pip install -q -p "$env/bin/python" --no-build-isolation-package prody -r "$here/requirements.txt"
# opencv-python and opencv-python-headless both write cv2/; the headless build goes in last, so cv2 is one build.
"$uv" pip install -q -p "$env/bin/python" --reinstall --no-deps opencv-python-headless==4.11.0.86

# vLLM environment, on the yml's Python (uv fetches 3.10.19 into $store/python).
[[ -x $vllm_env/bin/python ]] || "$uv" venv -q -p 3.10.19 "$vllm_env"
"$uv" pip install -q -p "$vllm_env/bin/python" numpy==2.2.6 setuptools==80.9.0
"$uv" pip install -q -p "$vllm_env/bin/python" --no-build-isolation-package prody -r "$here/requirements-vllm.txt"

# Current mtimes on every file and link, so the /scratch purge leaves the environments for ~60 days
# (rerun this script before then; it rebuilds a purged prefix). This covers the uv Python that
# biominer_vllm runs on. Parallel, because each touch is a round trip to the file server.
vllm_base=$("$vllm_env/bin/python" -c 'import sys; print(sys.base_prefix)')
find "$env" "$vllm_env" "$vllm_base" \( -type f -o -type l \) -print0 | xargs -0 -r -P 16 -n 1000 touch -h -c

# Then compile all bytecode, standard libraries included. The order matters: a .pyc records its
# source's mtime, so the touch above makes every older .pyc stale, and Python then recompiles the
# module and writes its .pyc again at every import. Creating a file on /scratch from a compute node
# took 0.4 s (2026-10-07), so stale .pyc files (14,818 of 19,702 in biominer_vllm, 17,854 of 22,522
# in biominer, left by this script compiling before touching) made vLLM's first start take over
# 30 min. run.py also sets PYTHONDONTWRITEBYTECODE, so a stale file costs a compile, never a write.
# A few files in packages are not Python 3.10 source (templates, Python 2 examples); they stay
# uncompiled and compileall's errors are not shown.
for python in "$env/bin/python" "$vllm_env/bin/python"; do
    # the standard library and site-packages, a folder inside another listed once
    mapfile -t trees < <("$python" -c 'import sysconfig
paths = sorted({sysconfig.get_paths()[key] for key in ("stdlib", "purelib", "platlib")})
print(*[p for p in paths if not any(p.startswith(q + "/") for q in paths)], sep="\n")')
    "$python" -m compileall -q -j 32 "${trees[@]}" >/dev/null 2>&1 ||
        echo "compileall: some files under ${trees[*]} are not Python 3.10 source"
    "$python" "$here/bytecode.py" "${trees[@]}"
done

# A packed copy of each environment beside it (<envs>/<name>.tar.zst), which run.py unpacks to
# node-local disk at every start (stage.py says why). biominer_vllm's holds the uv Python it runs on.
# Written under a temporary name and renamed, so a run never reads half an archive. run.py refuses an
# archive whose packages differ from its environment's: rerun this script after changing either.
pack() {  # ARCHIVE TAR-ARGUMENTS...
    local archive=$1
    shift
    tar -cf - "$@" | zstd -q -f -T16 -3 -o "$archive.partial"
    mv -f "$archive.partial" "$archive"
    echo "packed $(du -h "$archive" | cut -f1) into $archive"
}
pack "$store/envs/biominer.tar.zst" -C "$store/envs" biominer
pack "$store/envs/biominer_vllm.tar.zst" -C "$store/envs" biominer_vllm \
    -C "$(dirname "$vllm_base")" "$(basename "$vllm_base")"

# Weights, each at a pinned revision and checked file by file (the vLLM environment's huggingface_hub
# has hf_xet, which the 67 GB of BioMiner-Instruct need to download in reasonable time). MolGlyph's
# checkpoint is the one the molglyph tool downloaded.
fetched=0
"$vllm_env/bin/python" "$here/fetch_weights.py" "$models" || fetched=$?
echo "15a0fab1aeb75d35e1aeaf8a49da4d39c15131892e64fd5c415a66bbddbc3d51  $molglyph" | sha256sum -c -

# Offline checks, without the proxy, so any download would fail. The pipeline imports from a copy of
# the pinned code, as at run time (importing BioMiner writes its OPSIN cache under BioMiner/commons).
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git -C "$src" archive "$commit" BioMiner scripts | tar -x -C "$tmp"
offline=(env -u http_proxy -u https_proxy -u ftp_proxy -u HTTP_PROXY -u HTTPS_PROXY
         HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 PATH="$env/bin:$PATH")
(cd "$tmp" && "${offline[@]}" PYTHONPATH="$tmp:$tmp/scripts" YOLO_CONFIG_DIR="$tmp/ultralytics" \
    "$env/bin/python" - "$molglyph" <<'PY'
import importlib.metadata as md
import subprocess
import sys

import cv2, litserve, magic_pdf, rdkit, torch, transformers, ultralytics
from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton  # noqa: F401 (MinerU's models)
from magic_pdf.tools.cli import do_parse  # noqa: F401 (what MinerU's server runs)
from py2opsin import py2opsin

print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda,
      "transformers", transformers.__version__, "magic-pdf", md.version("magic-pdf"),
      "ultralytics", ultralytics.__version__, "litserve", md.version("litserve"), "opencv", cv2.__version__,
      "rdkit", rdkit.__version__, "timm", md.version("timm"), "albumentations", md.version("albumentations"))
print("cv2 GUI backends (none for the headless build):",
      [line.strip() for line in cv2.getBuildInformation().splitlines() if line.strip().startswith(("QT", "GTK"))])
print(subprocess.run(["pdftoppm", "-v"], capture_output=True, text=True).stderr.splitlines()[0])
print(subprocess.run(["java", "-version"], capture_output=True, text=True).stderr.splitlines()[0])
from rdkit import Chem
smiles = py2opsin("ethanol")  # OPSIN writes its own (non-canonical) SMILES
assert smiles and Chem.CanonSmiles(smiles) == "CCO", f"OPSIN read ethanol as {smiles!r}"
print("OPSIN ok: ethanol ->", smiles)

from BioMiner import BioMiner  # noqa: F401 (the whole pipeline)
import mineru_server, moldet_server, ocsr_server  # noqa: F401,E401 (the three servers' modules)

# MolGlyph loads its weights with strict=False and ignores mismatched keys; check that all match.
from BioMiner.MolScribe.molscribe import MolGlyph
states = torch.load(sys.argv[1], map_location="cpu")
model = MolGlyph(sys.argv[1], "cpu")
for part in ("encoder", "decoder"):
    have = set(getattr(model, part).state_dict())
    want = {k.replace("module.", "") for k in states[part]}
    print("MolGlyph", part, len(have), "tensors, missing", sorted(have - want)[:5], "unexpected", sorted(want - have)[:5])
    assert have == want, f"MolGlyph {part} keys do not match the checkpoint"
print("main environment ok")
PY
)
"${offline[@]}" "$vllm_env/bin/python" - "$models/BioMiner-Instruct" <<'PY'
import sys
from pathlib import Path

import torch, transformers, vllm

print("python", sys.version.split()[0], "vllm", vllm.__version__, "torch", torch.__version__,
      "cuda", torch.version.cuda, "transformers", transformers.__version__)
if Path(sys.argv[1], "config.json").is_file():
    from transformers import AutoConfig, AutoProcessor
    config = AutoConfig.from_pretrained(sys.argv[1])
    print("BioMiner-Instruct:", config.model_type, config.architectures, "dtype", getattr(config, "torch_dtype", None))
    AutoProcessor.from_pretrained(sys.argv[1])
    print("BioMiner-Instruct processor loads offline")
else:
    print("BioMiner-Instruct not downloaded; skipped its offline load check")
print("vllm environment ok")
PY
exit "$fetched"
