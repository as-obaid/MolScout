"""Complete-system variants and subsets in tool.yaml, and the harness running a subset of a paper dataset."""

from pathlib import Path

import pytest
import yaml
from test_bench_configs import SHA, write_system

from molscout.bench import BenchError
from molscout.bench.config import load_config
from molscout.bench.configs import make_configs
from molscout.bench.current import run_paths
from molscout.bench.papers import paper_pdfs, subset_papers
from molscout.data.biovista_dev import DevPaper, write_dev_manifest

REPO = Path(__file__).resolve().parents[1]
SYSTEM = {
    "tool": "scout",
    "name": "Scout",
    "version": "0.1.0",
    "python": "${STORE}/envs/scout/bin/python",
    "args": ["--work", "${STORE}/work/{run}", "--shared", "${STORE}/shared/{dataset}"],
    "checkpoints": [{"path": "${STORE}/models/m.pt", "sha256": SHA}],
    "subset": "dev",
    "variants": {"rules": ["--joiner", "rules"], "l0_drawings": ["--fixes-only", "none"]},
}


def test_variants_give_one_tool_each_with_their_args_and_the_subset(tmp_path):
    written = make_configs(write_system(tmp_path, SYSTEM), tmp_path / "configs", repo_root=tmp_path)
    assert [p.name for p in written] == [
        "scout_rules__biovista.yaml", "scout_rules__internal.yaml",
        "scout_l0_drawings__biovista.yaml", "scout_l0_drawings__internal.yaml",
    ]
    data = yaml.safe_load(written[0].read_text())
    assert (data["tool"], data["name"], data["subset"]) == ("scout_rules", "Scout (rules)", "dev")
    assert data["args"] == [
        "--work", "${STORE}/work/scout_rules__biovista", "--shared", "${STORE}/shared/biovista", "--joiner", "rules",
    ]
    assert data["run_dir"] == "benchmarks/tools/complete_systems/scout"


def test_the_default_variant_keeps_the_tools_name(tmp_path):
    system = {**SYSTEM, "variants": {"default": ["--joiner", "rules"], "rules": []}}
    written = make_configs(write_system(tmp_path, system), tmp_path / "configs", repo_root=tmp_path)
    assert [p.name for p in written][:2] == ["scout__biovista.yaml", "scout__internal.yaml"]
    data = yaml.safe_load(written[0].read_text())
    assert (data["tool"], data["name"]) == ("scout", "Scout")
    assert data["args"][2:] == ["--shared", "${STORE}/shared/biovista", "--joiner", "rules"]
    assert data["args"][1] == "${STORE}/work/scout__biovista"


def test_the_committed_molscout_configs_match_tool_yaml(tmp_path):
    tool = REPO / "benchmarks" / "tools" / "complete_systems" / "molscout" / "tool.yaml"
    for path in make_configs(tool, tmp_path, repo_root=REPO):
        assert path.read_text() == (REPO / "benchmarks" / "configs" / path.name).read_text()
    assert (REPO / "benchmarks" / "configs" / "molscout__biovista.yaml").is_file()


def test_without_variants_or_subset_the_configs_are_unchanged(tmp_path):
    plain = {key: value for key, value in SYSTEM.items() if key not in ("variants", "subset")}
    written = make_configs(write_system(tmp_path, plain), tmp_path / "configs", repo_root=tmp_path)
    assert [p.name for p in written] == ["scout__biovista.yaml", "scout__internal.yaml"]
    assert "subset" not in yaml.safe_load(written[0].read_text())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"variants": {"Bad-Name": []}}, "variant"),
        ({"variants": {"rules": "--joiner"}}, "variants.rules"),
        ({"variants": {}}, "variants"),
        ({"subset": "test"}, "subset"),
    ],
)
def test_bad_variants_and_subsets_are_refused(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        make_configs(write_system(tmp_path, {**SYSTEM, **change}), tmp_path / "configs", repo_root=tmp_path)


def test_a_structure_reader_cannot_have_variants(tmp_path):
    from test_bench_configs import TOOL, write_tool

    with pytest.raises(ValueError, match="complete system"):
        make_configs(write_tool(tmp_path, {**TOOL, "subset": "dev"}), tmp_path / "configs", repo_root=tmp_path)


def test_a_crop_config_cannot_have_a_subset(tmp_path):
    config = tmp_path / "reader__uspto.yaml"
    data = {
        "tool": "reader", "name": "R", "version": "1", "dataset": "uspto", "images": "i", "references": "r",
        "run_dir": "t", "python": "/p", "args": [], "checkpoints": [], "subset": "dev",
    }
    config.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="subset"):
        load_config(config, tmp_path)


# The harness on a subset


@pytest.fixture
def ws(tmp_path):
    from test_bench_papers import Workspace

    workspace = Workspace(tmp_path / "repo")
    dev = (DevPaper("1_aaaa", "aaaa", "drawn", 3, 2, 1), DevPaper("3_cccc", "cccc", "table", 1, 0, 1))
    write_dev_manifest(workspace.biovista / "biovista_dev.csv", dev)
    return workspace


def subset_config(ws, dataset):
    path = ws.config(dataset)
    data = yaml.safe_load(path.read_text())
    path.write_text(yaml.safe_dump({**data, "subset": "dev"}, sort_keys=False))
    return load_config(path, ws.root)


def test_a_dev_subset_runs_only_the_dev_papers(ws):
    assert list(paper_pdfs(subset_config(ws, "biovista"))) == ["1_aaaa", "3_cccc"]
    assert list(paper_pdfs(subset_config(ws, "internal"))) == ["p1"]
    assert subset_papers(load_config(ws.config("biovista"), ws.root)) is None


def test_a_dev_subset_needs_the_dev_manifest(ws):
    (ws.biovista / "biovista_dev.csv").unlink()
    with pytest.raises(BenchError, match="biovista_dev.csv"):
        paper_pdfs(subset_config(ws, "biovista"))


def test_the_dev_manifest_is_part_of_a_biovista_runs_code(ws):
    paths = run_paths(load_config(ws.config("biovista"), ws.root), ws.root)
    assert ws.biovista / "biovista_dev.csv" in paths


def test_the_harness_scores_a_subset_run_on_its_papers(ws):
    import json

    path = subset_config(ws, "biovista").path
    from molscout.bench.harness import run_benchmark

    folder = run_benchmark(path, repo_root=ws.root, results_root=ws.results)
    scores = json.loads((folder / "scores.json").read_text())
    assert list(scores["scores"]["papers"]) == ["1_aaaa", "3_cccc"]
    assert set(scores["scores"]["groups"]) == {"all", "without_submitted", "dev"}
    assert scores["inputs"]["subset"] == ["1_aaaa", "3_cccc"]
    assert set(json.loads((folder / "timing.json").read_text())["seconds"]) == {"1_aaaa", "3_cccc"}
