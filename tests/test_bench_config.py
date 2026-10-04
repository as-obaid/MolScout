"""Loading a benchmark config: keys, file name, ${VAR} expansion and paths from the repo root."""

from pathlib import Path

import pytest
import yaml

from molscout.bench.config import Checkpoint, load_config

SHA = "a" * 64
BASE = {
    "tool": "fake",
    "name": "Fake",
    "version": "1.0 (test)",
    "dataset": "uspto",
    "images": "data/images",
    "references": "data/refs",
    "run_dir": "tools/fake",
    "python": "${STORE}/bin/python",
    "args": ["--checkpoint", "${STORE}/model.pt", "--device", "cpu"],
    "checkpoints": [{"path": "${STORE}/model.pt", "sha256": SHA}],
}


def write_config(directory: Path, data: dict, name: str | None = None) -> Path:
    path = directory / (name or f"{data['tool']}__{data['dataset']}.yaml")
    path.write_text("# a test config\n" + yaml.safe_dump(data, sort_keys=False))
    return path


@pytest.fixture
def store(tmp_path, monkeypatch):
    store = tmp_path / "store"
    monkeypatch.setenv("STORE", str(store))
    return store


def test_loads_expands_variables_and_resolves_paths(tmp_path, store):
    path = write_config(tmp_path, BASE)
    config = load_config(path, tmp_path)
    assert config.path == path
    assert config.text == path.read_text()
    assert (config.tool, config.name, config.version, config.dataset) == ("fake", "Fake", "1.0 (test)", "uspto")
    assert config.images == tmp_path / "data" / "images"
    assert config.references == tmp_path / "data" / "refs"
    assert config.run_dir == tmp_path / "tools" / "fake"
    assert config.python == store / "bin" / "python"
    assert config.args == ("--checkpoint", f"{store}/model.pt", "--device", "cpu")
    assert config.checkpoints == (Checkpoint(store / "model.pt", SHA),)
    assert dict(config.env) == {}
    assert config.lock_commands == ()
    assert config.run_name == "fake__uspto"
    assert config.tool_label == "Fake 1.0 (test)"


def test_absolute_paths_are_kept(tmp_path, store):
    images = tmp_path / "elsewhere" / "images"
    config = load_config(write_config(tmp_path, {**BASE, "images": str(images)}), tmp_path / "repo")
    assert config.images == images
    assert config.references == tmp_path / "repo" / "data" / "refs"


def test_env_and_lock_commands_are_optional_and_expanded(tmp_path, store):
    data = {**BASE, "env": {"PYSTOW_HOME": "${STORE}/models", "THREADS": 4}, "lock_commands": [["java", "-version"]]}
    config = load_config(write_config(tmp_path, data), tmp_path)
    assert dict(config.env) == {"PYSTOW_HOME": f"{store}/models", "THREADS": "4"}
    assert config.lock_commands == (("java", "-version"),)


def test_lock_commands_expand_variables_like_the_molvec_tool_yaml(tmp_path, store):
    data = {**BASE, "lock_commands": [["${STORE}/models/molvec/jdk/bin/java", "-version"]]}
    config = load_config(write_config(tmp_path, data), tmp_path)
    assert config.lock_commands == ((f"{store}/models/molvec/jdk/bin/java", "-version"),)


def test_unset_variable_in_lock_commands_is_named(tmp_path, store, monkeypatch):
    monkeypatch.delenv("JDK_HOME_FOR_TEST", raising=False)
    data = {**BASE, "lock_commands": [["${JDK_HOME_FOR_TEST}/bin/java", "-version"]]}
    with pytest.raises(ValueError, match=r"lock_commands\[0\] uses \$\{JDK_HOME_FOR_TEST\}, which is not set"):
        load_config(write_config(tmp_path, data), tmp_path)


def test_numeric_args_become_strings(tmp_path, store):
    config = load_config(write_config(tmp_path, {**BASE, "args": ["--batch-size", 16]}), tmp_path)
    assert config.args == ("--batch-size", "16")


def test_unset_variable_is_named(tmp_path, monkeypatch):
    monkeypatch.delenv("STORE", raising=False)
    with pytest.raises(ValueError, match=r"\$\{STORE\}.*not set"):
        load_config(write_config(tmp_path, BASE), tmp_path)


def test_empty_variable_counts_as_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("STORE", "")
    with pytest.raises(ValueError, match="STORE"):
        load_config(write_config(tmp_path, BASE), tmp_path)


def test_file_name_must_match_tool_and_dataset(tmp_path, store):
    path = write_config(tmp_path, BASE, name="fake__uob.yaml")
    with pytest.raises(ValueError, match="fake__uspto.yaml"):
        load_config(path, tmp_path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"dataset": "biovista"}, "paper dataset"),
        ({"dataset": "nope"}, "unknown dataset 'nope'"),
        ({"checkpoints": [{"path": "model.pt", "sha256": "abc"}]}, "sha256 must be 64 lowercase hex"),
        ({"checkpoints": [{"path": "model.pt", "sha256": SHA, "size": 1}]}, r"unknown key\(s\): size"),
        ({"checkpoints": "model.pt"}, "checkpoints must be a list"),
        ({"extra": 1}, r"unknown key\(s\): extra"),
        ({"args": "--device cpu"}, "args must be a list"),
        ({"args": ["--flag", True]}, r"args\[1\]"),
        ({"tool": "Fake Tool"}, "tool must match"),
        ({"version": 2.0}, "version must be a string"),
        ({"env": ["A=1"]}, "env must be a mapping"),
        ({"lock_commands": ["java -version"]}, r"lock_commands\[0\] must be a non-empty list"),
    ],
)
def test_rejects_bad_configs(tmp_path, store, change, message):
    with pytest.raises(ValueError, match=message):
        load_config(write_config(tmp_path, {**BASE, **change}), tmp_path)


def test_missing_key_is_named(tmp_path, store):
    data = {key: value for key, value in BASE.items() if key != "checkpoints"}
    with pytest.raises(ValueError, match=r"missing key\(s\): checkpoints"):
        load_config(write_config(tmp_path, data), tmp_path)


def test_config_must_be_a_mapping(tmp_path):
    path = tmp_path / "fake__uspto.yaml"
    path.write_text("- tool\n- fake\n")
    with pytest.raises(ValueError, match="expected a YAML mapping"):
        load_config(path, tmp_path)


def test_invalid_yaml_is_a_value_error_naming_the_file(tmp_path):
    path = tmp_path / "fake__uspto.yaml"
    path.write_text("tool: [fake\n")
    with pytest.raises(ValueError, match="fake__uspto.yaml"):
        load_config(path, tmp_path)
