#!/bin/bash
# Build MolVec 0.9.8: a pinned Temurin JDK 17, the molvec jars from Maven Central, the compiled
# MolvecBatch class and a Python 3.12 venv with RDKit. Run as a CPU job, from the repository
# root on Explorer:
#   sbatch benchmarks/slurm/setup.sbatch benchmarks/tools/structure_readers/molvec/setup.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
store=${MOLSCOUT_STORE:-/scratch/$USER/molscout-store}
env=$store/envs/molvec
models=$store/models/molvec
build=$models/build
uv=${UV:-$HOME/.local/bin/uv}
export UV_PYTHON_INSTALL_DIR=$store/python UV_CACHE_DIR=$store/uv-cache UV_LINK_MODE=copy
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

jdk_url=https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.20.1%2B1/OpenJDK17U-jdk_x64_linux_hotspot_17.0.20.1_1.tar.gz
jdk_sha256=3808d1d15e3ec6bd5b84057fb5d84c33d8a1536a258146bcea2e603fc726e08e
maven_url=https://archive.apache.org/dist/maven/maven-3/3.9.16/binaries/apache-maven-3.9.16-bin.tar.gz
maven_sha256=80ffca22aed9e8b9713a232f3394fd81d7f20322df75efdb2b047dbd3e3a23bb

# fetch URL SHA256 DIR: download a tarball, check its sha256 and unpack it into DIR, once.
fetch() {
    [[ -d $3 ]] && return
    local tarball=$build/$(basename "$3").tar.gz
    curl -fsSL --retry 3 -o "$tarball" "$1"
    echo "$2  $tarball" | sha256sum -c -
    rm -rf "$3.part" && mkdir -p "$3.part"
    tar -xzf "$tarball" -C "$3.part" --strip-components=1
    mv "$3.part" "$3" && rm -f "$tarball"
}

mkdir -p "$build"
[[ -x $env/bin/python ]] || "$uv" venv -q -p 3.12 "$env"
"$uv" pip install -q -p "$env/bin/python" -r "$here/requirements.txt"

fetch "$jdk_url" "$jdk_sha256" "$models/jdk"
fetch "$maven_url" "$maven_sha256" "$build/maven"
export JAVA_HOME=$models/jdk

# Maven ignores http(s)_proxy, which is how Explorer's compute nodes reach the internet.
proxy=${https_proxy:-${http_proxy:-}}
proxy=${proxy#*://}
proxy=${proxy%/}
{
    echo "<settings><proxies>"
    if [[ -n $proxy ]]; then
        for protocol in https http; do
            echo "<proxy><id>$protocol</id><active>true</active><protocol>$protocol</protocol>"
            echo "<host>${proxy%:*}</host><port>${proxy##*:}</port></proxy>"
        done
    fi
    echo "</proxies></settings>"
} > "$build/settings.xml"

# Copy the jars into a fresh lib/; the pom is copied so Maven writes nothing in the repository.
cp "$here/pom.xml" "$build/pom.xml"
rm -rf "$models/lib.part"
"$build/maven/bin/mvn" -B -q -C -s "$build/settings.xml" -Dmaven.repo.local="$build/m2" \
    -f "$build/pom.xml" dependency:copy-dependencies \
    -DincludeScope=runtime -DoutputDirectory="$models/lib.part"
rm -rf "$models/lib" && mv "$models/lib.part" "$models/lib"

rm -rf "$models/classes" && mkdir -p "$models/classes"
"$JAVA_HOME/bin/javac" --release 17 -encoding UTF-8 -d "$models/classes" \
    -cp "$models/lib/*" "$here/MolvecBatch.java"

"$JAVA_HOME/bin/java" -version
"$env/bin/python" -c "import rdkit; print('rdkit', rdkit.__version__)"
# Upstream tag 0.9.8 in the clone, for the version string in tool.yaml.
git -C "$store/src/molvec" rev-parse "0.9.8^{commit}" 2>/dev/null || echo "molvec clone: tag 0.9.8 not found"
(cd "$models/lib" && sha256sum *.jar)
"$env/bin/python" - "$models/lib" "$models/classes" <<'PY'
import hashlib
import sys
from pathlib import Path

def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for folder in map(Path, sys.argv[1:]):
    files = sorted((p.relative_to(folder).as_posix(), p) for p in folder.rglob("*") if p.is_file())
    lines = "".join(f"{sha256_file(p)}  {rel}\n" for rel, p in files)
    print(hashlib.sha256(lines.encode("utf-8")).hexdigest(), f"{folder}/")
PY
