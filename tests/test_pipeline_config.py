"""PipelineConfig: the YAML schema, overrides, validation and the stage plan."""

import json
from pathlib import Path

import pytest

from molscout.pipeline import config as pc
from molscout.pipeline import stages as st

WORKERS = """
workers:
  parse: {python: /envs/mineru/bin/python, worker: mineru_parse.py}
  segment: {python: /envs/decimer_ai/bin/python, worker: segment.py, args: [--threshold, "{thresholds.structure_score}"]}
  read_molglyph: {python: /envs/molglyph/bin/python, worker: read_molglyph.py}
  read_molscribe: {python: /envs/molscribe/bin/python, worker: read_molscribe.py}
  read_molnextr: {python: /envs/molnextr/bin/python, worker: read_molnextr.py}
  join_32b: {python: /envs/biominer/bin/python, worker: /abs/join_32b.py, args: [--endpoint, "{vllm_url}"], output: m32.jsonl}
"""


def write(tmp_path, text, name="pipeline.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def load(tmp_path, extra="", **overrides):
    return pc.load_pipeline_config(write(tmp_path, WORKERS + extra), **overrides)


def test_defaults(tmp_path):
    config = load(tmp_path, "opsin_jar: opsin.jar\n")
    assert config.joiner == "rules" and config.export == "accepted"
    assert config.export_statuses == frozenset({"accepted"})
    assert config.readers == ("molglyph", "molscribe", "molnextr") and config.primary_reader == "molglyph"
    assert all(config.fixes[name] for name in pc.FIX_NAMES)
    assert config.opsin_jar == tmp_path / "opsin.jar"  # relative to the config file
    assert config.dpi == 300


def test_relative_worker_paths_resolve_to_the_packaged_workers(tmp_path):
    config = load(tmp_path, "opsin_jar: /o.jar\n")
    assert config.workers["parse"].worker == pc.WORKERS_DIR / "mineru_parse.py"
    assert "join_32b" not in config.workers  # the rules joiner does not use it
    llm = config.with_overrides(joiner="32b", vllm_url="http://h:1/v1")
    assert llm.workers["join_32b"].worker == Path("/abs/join_32b.py")


def test_stage_plan_follows_the_joiner(tmp_path):
    rules = load(tmp_path, "opsin_jar: /o.jar\n")
    reads = ["read_molglyph", "read_molscribe", "read_molnextr"]
    assert rules.stage_names() == ["render", "parse", "segment", *reads, "vote", "link", "finalize", "export"]
    llm = rules.with_overrides(joiner="32b", vllm_url="http://h:1/v1")
    assert llm.stage_names() == ["render", "parse", "segment", *reads, "vote", "join_32b", "finalize", "export"]
    both = rules.with_overrides(joiner="both", vllm_url="http://h:1/v1")
    assert both.stage_names()[-5:] == ["link", "join_32b", "merge", "finalize", "export"]


def test_java_and_vllm_url_expand_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("MOLSCOUT_STORE", "/store")
    monkeypatch.setenv("VLLM_HOST", "node7")
    config = load(
        tmp_path,
        "opsin_jar: /o.jar\njava: ${MOLSCOUT_STORE}/jdk/bin/java\nvllm_url: http://${VLLM_HOST}:8623/v1\n",
    )
    assert config.java == "/store/jdk/bin/java"
    assert config.vllm_url == "http://node7:8623/v1"


def test_32b_needs_a_vllm_url(tmp_path):
    with pytest.raises(ValueError, match="vllm_url"):
        load(tmp_path, "opsin_jar: /o.jar\njoiner: 32b\n")
    with pytest.raises(ValueError, match="vllm_url"):
        load(tmp_path, "opsin_jar: /o.jar\n").with_overrides(joiner="both")


def test_rules_needs_the_opsin_jar(tmp_path):
    with pytest.raises(ValueError, match="opsin_jar"):
        load(tmp_path)
    assert load(tmp_path, "joiner: 32b\nvllm_url: http://h:1/v1\n").opsin_jar is None


def test_overrides_switch_fixes_off_and_export(tmp_path):
    config = load(tmp_path, "opsin_jar: /o.jar\n", fixes_off=("vote", "stereo"), export="accepted+review")
    assert not config.fixes.vote and not config.fixes["stereo"] and "vote" not in config.fixes
    assert "tables" in config.fixes and config.fixes.get("tables") is True
    assert config.export_statuses == frozenset({"accepted", "review"})
    assert config.fixes.enabled() == (
        "labels", "relevance", "tables", "core_check", "junk", "names", "formulas",
        "attachments", "slot_labels", "legends", "partial_stereo",
    )


def test_fixes_only_keeps_the_named_fixes(tmp_path):
    config = load(tmp_path, "opsin_jar: /o.jar\n", fixes_only=("labels", "relevance"))
    assert config.fixes.enabled() == ("labels", "relevance")
    assert load(tmp_path, "opsin_jar: /o.jar\n", fixes_only=()).fixes.enabled() == ()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("opsin_jar: /o.jar\njoiner: llm\n", "joiner"),
        ("opsin_jar: /o.jar\nexport: everything\n", "export"),
        ("opsin_jar: /o.jar\nfixes: {vote: maybe}\n", "fixes.vote"),
        ("opsin_jar: /o.jar\nfixes: {speed: true}\n", "speed"),
        ("opsin_jar: /o.jar\nthresholds: {structure_score: high}\n", "thresholds.structure_score"),
        ("opsin_jar: /o.jar\ncolour: blue\n", "colour"),
        ("opsin_jar: /o.jar\nprimary_reader: decimer\n", "primary_reader"),
        ("opsin_jar: /o.jar\nallow_empty: [teleport]\n", "teleport"),
        ("opsin_jar: /o.jar\ndpi: 0\n", "dpi"),
    ],
)
def test_bad_values_name_the_key(tmp_path, extra, message):
    with pytest.raises(ValueError, match=message):
        load(tmp_path, extra)


def test_missing_worker_for_a_used_stage(tmp_path):
    text = WORKERS.replace("  read_molnextr: {python: /envs/molnextr/bin/python, worker: read_molnextr.py}\n", "")
    with pytest.raises(ValueError, match="read_molnextr"):
        pc.load_pipeline_config(write(tmp_path, text + "opsin_jar: /o.jar\n"))
    config = pc.load_pipeline_config(write(tmp_path, text + "opsin_jar: /o.jar\nreaders: [molglyph, molscribe]\n"))
    assert config.readers == ("molglyph", "molscribe")


def test_environment_variables_expand_only_for_used_workers(tmp_path, monkeypatch):
    monkeypatch.setenv("STORE", "/store")
    monkeypatch.delenv("UNSET_FOR_TEST", raising=False)
    text = WORKERS.replace("/envs/mineru/bin/python", "\"${STORE}/envs/mineru/bin/python\"").replace(
        "/envs/biominer/bin/python", "\"${UNSET_FOR_TEST}/b\""
    )
    config = pc.load_pipeline_config(write(tmp_path, text + "opsin_jar: /o.jar\n"))
    assert config.workers["parse"].python == Path("/store/envs/mineru/bin/python")
    with pytest.raises(ValueError, match="UNSET_FOR_TEST"):
        config.with_overrides(joiner="32b", vllm_url="http://h:1/v1")


def test_worker_args_fill_placeholders(tmp_path):
    config = load(tmp_path, "joiner: 32b\nvllm_url: http://h:1/v1\nthresholds: {structure_score: 0.25}\n")
    assert config.worker_args("join_32b") == ["--endpoint", "http://h:1/v1"]
    assert config.worker_args("segment") == ["--threshold", "0.25"]
    assert config.output("join_32b") == "m32.jsonl" and config.output("link") == "molecules_rules.jsonl"


def test_an_unknown_placeholder_is_an_error(tmp_path):
    text = WORKERS.replace('"{vllm_url}"', '"{nope}"')
    config = pc.load_pipeline_config(write(tmp_path, text + "joiner: 32b\nvllm_url: http://h:1/v1\n"))
    with pytest.raises(ValueError, match="nope"):
        config.worker_args("join_32b")


def test_sha256_changes_with_settings_but_not_with_the_vllm_url(tmp_path):
    a = load(tmp_path, "joiner: 32b\nvllm_url: http://h:1/v1\n")
    assert a.sha256() == a.with_overrides(vllm_url="http://other:2/v1").sha256()
    assert a.sha256() != a.with_overrides(fixes_off=("junk",)).sha256()


def test_the_packaged_default_config_loads(monkeypatch):
    monkeypatch.setenv("MOLSCOUT_STORE", "/store")
    config = pc.load_pipeline_config(pc.DEFAULT_CONFIG)
    assert config.joiner == "rules"
    assert set(config.workers) >= {"parse", "segment", "read_molglyph", "read_molscribe", "read_molnextr"}
    assert all(spec.worker.parent == pc.WORKERS_DIR for spec in config.workers.values())


# --- the cells stage -------------------------------------------------------------------------------

CELLS_WORKER = '  cells: {python: /envs/decimer_ai/bin/python, worker: cells.py, args: [--device, "{device}"]}\n'


def test_the_cells_stage_runs_after_segment_when_its_worker_is_configured(tmp_path):
    config = pc.load_pipeline_config(write(tmp_path, WORKERS + CELLS_WORKER + "opsin_jar: /o.jar\n"))
    reads = ["read_molglyph", "read_molscribe", "read_molnextr"]
    assert config.cells and config.stage_names()[:4] == ["render", "parse", "segment", "cells"]
    assert config.stage_names()[4:7] == reads
    assert "cells" in config.worker_stages() and config.workers["cells"].worker == pc.WORKERS_DIR / "cells.py"
    assert config.output("cells") == "cells.jsonl" and config.worker_args("cells") == ["--device", "gpu"]
    assert not load(tmp_path, "opsin_jar: /o.jar\n").cells


def test_the_readers_depend_on_the_cells_stage(tmp_path):
    config = pc.load_pipeline_config(write(tmp_path, WORKERS + CELLS_WORKER + "opsin_jar: /o.jar\n"))
    assert st.dependencies("cells", config) == ("render", "parse", "segment")
    assert st.dependencies("read_molglyph", config) == ("segment", "cells")
    assert st.is_upstream("cells")
    plain = load(tmp_path, "opsin_jar: /o.jar\n")
    assert st.dependencies("read_molglyph", plain) == ("segment",)
    keys = st.stage_keys(config, "0" * 64)
    assert keys["segment"] == st.stage_keys(plain, "0" * 64)["segment"]
    assert keys["read_molglyph"] != st.stage_keys(plain, "0" * 64)["read_molglyph"]


def test_cells_counts_the_table_images_in_and_the_cell_drawings_out(tmp_path):
    config = pc.load_pipeline_config(write(tmp_path, WORKERS + CELLS_WORKER + "opsin_jar: /o.jar\n"))
    folder = tmp_path / "paper"
    folder.mkdir()
    layout = [{"type": "table", "html": "<table><tr><td><img src='a.jpg'/></td><td><img src='b.jpg'/></td></tr></table>"},
              {"type": "image", "html": None}, {"type": "table", "html": "<table><tr><td>1</td></tr></table>"}]
    (folder / "layout.jsonl").write_text("".join(json.dumps(b) + "\n" for b in layout))
    (folder / "cells.jsonl").write_text("".join(json.dumps({"region_id": f"c{i}"}) + "\n" for i in range(3)))
    regions = [{"region_id": "p1-r1"}, {"region_id": "c0", "kind": "cell"}]
    (folder / "regions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in regions))
    assert st.records_in("cells", folder, config) == 2
    assert st.records_out("cells", folder, config) == 3
    assert st.funnel(folder, config)["cells"] == 1


def test_label_ocr_and_cells_both_run_after_segment_and_ocr_skips_cells(tmp_path):
    ocr = '  ocr_labels: {python: /envs/mineru/bin/python, worker: ocr_labels.py}\n'
    config = pc.load_pipeline_config(write(tmp_path, WORKERS + ocr + CELLS_WORKER + "opsin_jar: /o.jar\n"))
    assert config.stage_names()[3:5] == ["ocr_labels", "cells"] and config.optional_stages() == ["ocr_labels", "cells"]
    assert st.dependencies("link", config) == ("vote", "parse", "ocr_labels", "cells")
    folder = tmp_path / "paper"
    folder.mkdir()
    (folder / "layout.jsonl").write_text("")
    regions = [{"region_id": "p1-r1", "page": 1, "bbox": [0, 0, 10, 10], "is_structure": True},
               {"region_id": "p1-t2-r1-c1", "page": 1, "bbox": [0, 0, 10, 10], "is_structure": True, "kind": "cell"}]
    (folder / "regions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in regions))
    assert st.figure_areas(folder, config) == 1


def test_the_packaged_configs_run_the_cells_stage(monkeypatch):
    monkeypatch.setenv("MOLSCOUT_STORE", "/store")
    benchmark = Path(__file__).parents[1] / "benchmarks" / "tools" / "complete_systems" / "molscout" / "pipeline.yaml"
    for path in (pc.DEFAULT_CONFIG, benchmark):
        config = pc.load_pipeline_config(path)
        assert config.cells and config.workers["cells"].python == Path("/store/envs/decimer_ai/bin/python")
