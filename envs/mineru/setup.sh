#!/bin/bash
# Build the MinerU 4.2.3 environment for the parse stage and download its models into the store. Run as a
# CPU job, from the repository root on Explorer:
#   sbatch benchmarks/slurm/setup.sbatch envs/mineru/setup.sh
# Then run the worker with this environment and MINERU_HOME=$MOLSCOUT_STORE/models/mineru (where the
# models are), e.g. on a GPU node:
#   MINERU_HOME=$MOLSCOUT_STORE/models/mineru $MOLSCOUT_STORE/envs/mineru/bin/python \
#       src/molscout/pipeline/workers/mineru_parse.py --manifest work.jsonl --log parse.log \
#       --tier standard --backend vllm --small-backend torch
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
store=${MOLSCOUT_STORE:-/scratch/$USER/molscout-store}
env=$store/envs/mineru
home=$store/models/mineru  # MINERU_HOME: MinerU keeps its models under $home/models/<repo>
uv=${UV:-$HOME/.local/bin/uv}
export UV_PYTHON_INSTALL_DIR=$store/python UV_CACHE_DIR=$store/uv-cache UV_LINK_MODE=copy
export MINERU_HOME=$home MINERU_TELEMETRY_DISABLED=1
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

[[ -x $env/bin/python ]] || "$uv" venv -q -p 3.12.11 "$env"
# every package pinned and checked against the sha256 recorded in requirements.txt
"$uv" pip install -q -p "$env/bin/python" --require-hashes -r "$here/requirements.txt"
"$env/bin/mineru-kit" version

# Models: the standard tier on GPU nodes (PyTorch small models and the 1.2B VLM in safetensors for vLLM)
# and the ONNX small models for the basic tier on CPU nodes. MinerU fetches each repository at its current
# commit and checks file sizes only; the commits and sha256 below make a changed upstream repository stop
# this script instead of changing results.
mkdir -p "$home"
"$env/bin/mineru-kit" models download --tier standard --small-backend torch --vlm-engine vllm --source huggingface
"$env/bin/mineru-kit" models download --tier basic --small-backend onnx --source huggingface
"$env/bin/mineru-kit" models verify --tier standard --small-backend torch --vlm-engine vllm
"$env/bin/mineru-kit" models verify --tier basic --small-backend onnx
"$env/bin/python" - "$home/models" <<'PY'
import hashlib
import sys
from pathlib import Path

# repository: (commit, {file: sha256}) for every LFS file MinerU uses (Hugging Face API, 2026-11-25)
PINS = {
    "MinerU-4_models_torch": ("5b0dc6a7012ecc77ab8db30fe9b899d61fbdc86e", {
        "Layout/PP-DocLayoutV2/model.safetensors": "e60f3725aeedc88fd319416ef166bda79171a41516a301c27cab9132dc2739d2",
        "MFR/pp_formulanet_plus_m/PP-FormulaNet_plus-M.pth": "bd9d40fd9c7759805e93cb54f14f5e91448a80b14aadc2bfb25c4c948b2974e8",
        "OCR/paddleocr/ch_PP-OCRv6_small_rec_infer.safetensors": "f65a332afe5aa663f0b9d5706f4ae8457b5b4058a842d5c1eb22df505c27d642",
        "OCR/paddleocr/ch_PP-OCRv6_tiny_det_infer.safetensors": "cae3c88d2a9902fd0293e6b17990428f54bfa7ec98f800a4368e95423a754d16",
        "OCR/paddleocr/seal_PP-OCRv4_det_infer.pth": "a7777ca66448ab90948ce5a3257e4c959d6eacf0489fbadd5133dbe8f89662ae",
        "Table/PP-LCNet_x1_0_table_cls.onnx": "c84bf1d79c1c74d534b5b12adb14dd12151c42f7ae3e4be4f1042b830f80b949",
        "Table/slanet-plus.onnx": "f74953de6868a1da7117407ef3a51ff877a97bad72e9368a857c8c274e1f5d69",
        "Table/unet.onnx": "0ea48d3a17e35ef5c2e498a5e799566073234d39b1079ca21d9f4fafe73c6d20",
    }),
    "MinerU-4_models_onnx": ("de70e917d417d8bdee7b985f9431d52b35efcd23", {
        "Layout/PP-DocLayoutV2/inference.onnx": "cd540dc296ff3115fe78efa65b68501e6a8dc74b198acc9834a725ffaa095aac",
        "MFR/pp_formulanet_plus_m/PP-FormulaNet_plus-M.onnx": "20a32595c2b30282dcd069e6295e9fad96115697beb66acb9b681b814006d193",
        "OCR/paddleocr/ch_PP-OCRv6_small_rec_infer.onnx": "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634",
        "OCR/paddleocr/ch_PP-OCRv6_tiny_det_infer.onnx": "193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8",
        "OCR/paddleocr/seal_PP-OCRv4_det_infer.onnx": "3e190df84542c2756975febe8f73603b587a23c7f7badfdd3fc8a986540d5bc0",
        "Table/PP-LCNet_x1_0_table_cls.onnx": "c84bf1d79c1c74d534b5b12adb14dd12151c42f7ae3e4be4f1042b830f80b949",
        "Table/slanet-plus.onnx": "f74953de6868a1da7117407ef3a51ff877a97bad72e9368a857c8c274e1f5d69",
        "Table/unet.onnx": "0ea48d3a17e35ef5c2e498a5e799566073234d39b1079ca21d9f4fafe73c6d20",
    }),
    "MinerU2.5-Pro-2605-1.2B": ("08aaea840498d49ce16247b6263196cf02814885", {
        "model.safetensors": "abf8681ca63b8dec7b67de257af47b821f179442f72998d0696ae2ed9232a5f0",
        "tokenizer.json": "dceac5fc54a795ee7570d17902b47bd05412dc2afa62bdf325c3f97fcb5b87fe",
    }),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


failures = []
for name, (commit, files) in PINS.items():
    root = Path(sys.argv[1], name)
    # huggingface_hub records the commit of every downloaded file in the first line of its .metadata
    found = {p.read_text().splitlines()[0] for p in (root / ".cache/huggingface/download").rglob("*.metadata")}
    if found != {commit}:
        failures.append(f"{name}: downloaded from commit(s) {sorted(found)}, pinned {commit}")
    for rel, want in files.items():
        path = root / rel
        have = sha256(path) if path.is_file() else "missing"
        if have != want:
            failures.append(f"{name}/{rel}: sha256 {have}, pinned {want}")
    print(f"{name} @ {commit}: {len(files)} files checked")
if failures:
    sys.exit("model check failed:\n  " + "\n  ".join(failures))
print("models match their pins")
PY

# /scratch purges files whose mtime is about 60 days old: give every file a current mtime (rerun this
# script before then), then compile bytecode so that imports do not write .pyc files on /scratch.
python_base=$("$env/bin/python" -c 'import sys; print(sys.base_prefix)')
find "$env" "$home" "$python_base" \( -type f -o -type l \) -print0 | xargs -0 -r -P 16 -n 1000 touch -h -c
"$env/bin/python" -m compileall -q -j 8 "$("$env/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    >/dev/null 2>&1 || echo "compileall: some files in site-packages are not Python 3.12 source"

# Offline self-check without the proxy, so any download fails: the worker parses a one-page PDF with a
# heading, a paragraph and a table at the basic tier on CPU (the standard tier needs a GPU).
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
"$env/bin/python" - "$tmp/check.pdf" <<'PY'
import sys
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

page = canvas.Canvas(sys.argv[1], pagesize=A4)
page.setFont("Helvetica-Bold", 16)
page.drawString(72, 770, "Table 1. Inhibition of PHD2 by compounds 8-10")
page.setFont("Helvetica", 11)
page.drawString(72, 740, "Compounds 8, 9 and 10 were assayed against PHD2; IC50 values are in the table.")
rows = [("Compound", "R", "IC50 (uM)"), ("8", "Cl", "0.26"), ("9", "OMe", "0.21"), ("10", "CF3", "0.40")]
for i, row in enumerate(rows):
    y = 700 - 22 * i
    for j, cell in enumerate(row):
        page.drawString(80 + 140 * j, y, cell)
    page.line(72, y - 6, 500, y - 6)
page.showPage()
page.save()
PY
printf '{"paper_id": "check", "pdf": "%s", "paper_dir": "%s"}\n' "$tmp/check.pdf" "$tmp/check" > "$tmp/work.jsonl"
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY HF_HUB_OFFLINE=1 \
    "$env/bin/python" "$repo/src/molscout/pipeline/workers/mineru_parse.py" \
    --manifest "$tmp/work.jsonl" --log "$tmp/parse.log" --tier basic --small-backend onnx
cat "$tmp/parse.log"
[[ -s $tmp/check/layout.jsonl && ! -e $tmp/check/errors.jsonl ]] || { echo "self-check failed" >&2; exit 1; }
echo "self-check ok: $(wc -l < "$tmp/check/layout.jsonl") layout blocks"
"$env/bin/python" -c 'import mineru.version, onnxruntime, torch
print("mineru", mineru.version.__version__, "torch", torch.__version__, "cuda", torch.version.cuda,
      "onnxruntime", onnxruntime.__version__)'
# vLLM runs only on a GPU node (the first standard-tier smoke run tests it); here it only has to import.
"$env/bin/python" -c 'import vllm; print("vllm", vllm.__version__)' ||
    echo "vllm does not import on this CPU node; check it in the first GPU run"
