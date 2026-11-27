"""MolScout's setup.sh, the device setting, the OPSIN folder and the worker code hash."""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from pipeline_helpers import fake_config, pipeline_yaml

from molscout.pipeline import stages as st
from molscout.pipeline.config import load_pipeline_config

REPO = Path(__file__).resolve().parents[1]
SETUP = REPO / "benchmarks" / "tools" / "complete_systems" / "molscout" / "setup.sh"
# The real scripts/fetch_opsin.py, with only its download replaced: its arguments are the real ones.
FETCH_STUB = f'''import importlib.util, sys
spec = importlib.util.spec_from_file_location("fetch_opsin", {str(REPO / "scripts" / "fetch_opsin.py")!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
def download(url, path, sha256):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"jar")
module.download = download
sys.exit(module.main())
'''


def executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


def test_setup_fetches_the_versioned_opsin_jar_into_the_folder_the_configs_name(tmp_path, monkeypatch):
    repo, store = tmp_path / "repo", tmp_path / "store"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "fetch_opsin.py").write_text(FETCH_STUB)
    python = f'#!/bin/bash\nexec "{sys.executable}" "$@"\n'
    for name in ("molscout", "mineru", "decimer_ai", "molglyph", "molscribe", "molnextr", "biominer", "biominer_vllm"):
        executable(store / "envs" / name / "bin" / "python", python)
    executable(store / "models" / "molvec" / "jdk" / "bin" / "java", "#!/bin/bash\necho 'openjdk 17' >&2\n")
    executable(tmp_path / "uv", "#!/bin/bash\nexit 0\n")
    env = {**os.environ, "MOLSCOUT_STORE": str(store), "UV": str(tmp_path / "uv")}
    result = subprocess.run(["bash", str(SETUP)], cwd=repo, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    jars = list((store / "models" / "opsin").glob("opsin-cli-*-jar-with-dependencies.jar"))
    assert len(jars) == 1
    monkeypatch.setenv("MOLSCOUT_STORE", str(store))
    for config in (REPO / "src/molscout/pipeline/pipeline.yaml", SETUP.parent / "pipeline.yaml"):
        assert load_pipeline_config(config).opsin_jar_file() == jars[0]


def test_the_opsin_folder_gives_its_newest_jar(tmp_path):
    folder = tmp_path / "opsin"
    folder.mkdir()
    for version in ("2.9.0", "2.10.0", "2.8.1"):
        (folder / f"opsin-cli-{version}-jar-with-dependencies.jar").write_text("jar")
    (folder / "notes.txt").write_text("x")
    config = replace(fake_config(tmp_path), opsin_jar=folder)
    assert config.opsin_jar_file() == folder / "opsin-cli-2.10.0-jar-with-dependencies.jar"
    (tmp_path / "empty").mkdir()
    assert replace(config, opsin_jar=tmp_path / "empty").opsin_jar_file() is None
    jar = folder / "opsin-cli-2.9.0-jar-with-dependencies.jar"
    assert replace(config, opsin_jar=jar).opsin_jar_file() == jar  # a file is used as it is


def test_device_chooses_the_worker_arguments(tmp_path, monkeypatch):
    monkeypatch.setenv("STORE", "/store")
    text = f"""workers:
  parse:
    python: {sys.executable}
    worker: mineru_parse.py
    env: {{MINERU_HOME: "${{STORE}}/models/mineru"}}
    device_args: {{gpu: [--tier, standard, --backend, vllm], cpu: [--tier, basic]}}
  segment: {{python: {sys.executable}, worker: segment.py, args: [--device, "{{device}}"]}}
  read_molglyph: {{python: {sys.executable}, worker: r.py, args: [--checkpoint, "${{STORE}}/m.pt", --device, "{{torch_device}}"]}}
  read_molscribe: {{python: {sys.executable}, worker: r.py}}
  read_molnextr: {{python: {sys.executable}, worker: r.py}}
opsin_jar: /o.jar
"""
    path = tmp_path / "p.yaml"
    path.write_text(text)
    gpu, cpu = load_pipeline_config(path), load_pipeline_config(path, device="cpu")
    assert gpu.device == "gpu" and gpu.worker_args("parse") == ["--tier", "standard", "--backend", "vllm"]
    assert cpu.worker_args("parse") == ["--tier", "basic"] and cpu.worker_args("segment") == ["--device", "cpu"]
    assert gpu.worker_args("read_molglyph") == ["--checkpoint", "/store/m.pt", "--device", "cuda"]
    assert gpu.workers["parse"].env == {"MINERU_HOME": "/store/models/mineru"}
    assert st.stage_keys(gpu, "0" * 64)["parse"] != st.stage_keys(cpu, "0" * 64)["parse"]
    assert st.stage_keys(gpu, "0" * 64)["render"] == st.stage_keys(cpu, "0" * 64)["render"]
    with pytest.raises(ValueError, match="device"):
        load_pipeline_config(path, device="tpu")
    path.write_text(text.replace("cpu: [--tier, basic]", "tpu: [--tier, basic]"))
    with pytest.raises(ValueError, match="device_args"):
        load_pipeline_config(path)


def test_a_change_to_a_shared_worker_helper_reruns_the_stage(tmp_path):
    workers = tmp_path / "workers"
    workers.mkdir()
    (workers / "segment.py").write_text("print('segment')\n")
    (workers / "_common.py").write_text("VERSION = 1\n")
    path = pipeline_yaml(tmp_path)
    path.write_text(path.read_text().replace(str(Path(__file__).parent / "fixtures" / "pipeline" / "fake_worker.py"),
                                             str(workers / "segment.py")))
    config = load_pipeline_config(path)
    before = st.stage_keys(config, "0" * 64)["segment"]
    (workers / "_common.py").write_text("VERSION = 2\n")
    assert st.stage_keys(config, "0" * 64)["segment"] != before
