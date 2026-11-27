#!/bin/bash
# Install MolScout into the molscout environment and check what its benchmark runs need. Run as a CPU job, from the
# repository root on Explorer:
#   sbatch benchmarks/slurm/setup.sbatch benchmarks/tools/complete_systems/molscout/setup.sh
# The stage environments come from their own setup scripts (envs/mineru/setup.sh; the decimer_ai, molglyph,
# molscribe, molnextr and biominer tool folders); this checks that each is there. scripts/fetch_opsin.py puts the
# pinned OPSIN jar (opsin-cli-<version>-jar-with-dependencies.jar, sha256-checked) in $MOLSCOUT_STORE/models/opsin,
# the folder the pipeline configs name; it keeps a jar already there with the pinned sha256.
set -euo pipefail
store=${MOLSCOUT_STORE:-/scratch/$USER/molscout-store}
env=$store/envs/molscout
uv=${UV:-$HOME/.local/bin/uv}
export UV_PYTHON_INSTALL_DIR=$store/python UV_CACHE_DIR=$store/uv-cache UV_LINK_MODE=copy

[[ -x $env/bin/python ]] || "$uv" venv -q -p 3.12 "$env"
# This repository, editable, so runs use the checked-out code (pyproject.toml pins RDKit and pypdfium2).
"$uv" pip install -q -p "$env/bin/python" -e .
"$env/bin/python" -c "import molscout.pipeline.runner, pypdfium2; print('molscout', molscout.__version__)"

opsin=$store/models/opsin
"$env/bin/python" scripts/fetch_opsin.py --dest "$opsin"
jars=("$opsin"/opsin-cli-*-jar-with-dependencies.jar)
[[ -f ${jars[0]} ]] || { echo "no OPSIN jar in $opsin after scripts/fetch_opsin.py" >&2; exit 1; }
"$store/models/molvec/jdk/bin/java" -version 2>&1 | head -1

missing=0
for name in mineru decimer_ai molglyph molscribe molnextr biominer biominer_vllm; do
    if [[ -x $store/envs/$name/bin/python ]]; then
        echo "ok: envs/$name"
    else
        echo "missing: envs/$name (run its setup.sh)" >&2
        missing=1
    fi
done
exit $missing
