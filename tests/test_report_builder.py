"""The W&B report builder in scripts/report/: run-set filters, figure panels and the complete-system table.

The builder needs the report-builder extra; without it these tests are skipped. Specs are built offline: the one
GraphQL call wandb-workspaces makes per run set (the project's internal ID) is stubbed.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("wandb_workspaces")
pytest.importorskip("markdown")

import wandb_workspaces.reports.v2 as wr  # noqa: E402
import wandb_workspaces.reports.v2.interface as interface  # noqa: E402

from molscout.report import paper_figures  # noqa: E402
from molscout.report.wandb_publish import FIGURE_PREFIX, panel_rows  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "scripts" / "report"
SYSTEMS = ("biominer", "decimer_ai", "openchemie")


@pytest.fixture(scope="module")
def builder():
    sys.path.insert(0, str(BUILDER))
    try:
        yield importlib.import_module("build_report"), importlib.import_module("spec_tools")
    finally:
        sys.path.remove(str(BUILDER))


@pytest.fixture
def build(builder, monkeypatch):
    """stage -> the spec a save would upload, built without network."""
    module, tools = builder
    monkeypatch.setattr(interface, "execute_graphql", lambda api, query, variables: {"project": {"internalId": "UHJvamVjdA=="}})
    monkeypatch.setattr(interface, "_get_api", lambda: None)

    def spec(stage):
        module.set_stage(stage)
        try:
            report = wr.Report(project=module.PROJECT, entity=module.ENTITY, title=module.TITLE, width="fixed", blocks=module.blocks())
            built = json.loads(report._to_model().spec.model_dump_json(by_alias=True, exclude_none=True))
        finally:
            module.set_stage("full")
        tools.fix_table_types(built)
        return built

    return spec


def grids(spec, tools):
    for block in tools.panel_grids(spec):
        yield block["metadata"]["panelBankSectionConfig"]["panels"], block["metadata"]["runSets"]


@pytest.mark.parametrize("stage", ["type1", "full"])
def test_every_grid_with_panels_reads_one_named_summary_run(builder, build, stage):
    module, tools = builder
    spec = build(stage)
    assert tools.runset_problems(spec, module.SUMMARY_NAMES) == []
    for panels, runsets in grids(spec, tools):
        if not panels:
            continue
        assert len(runsets) == 1
        filters = tools.runset_filters(runsets[0])
        names = [value for key, _, value in filters if key == "displayName"]
        assert ("jobType", "=", "analysis") in filters
        keys = [key for panel in panels for key in panel["config"].get("mediaKeys", [])]
        if any(key.startswith(module.SYSTEMS_PREFIX) for key in keys):
            assert names == [module.SYSTEMS_SUMMARY]
        else:
            assert names == [module.READERS_SUMMARY]


def test_the_structure_reader_grids_only_gain_the_run_name_filter(builder, build):
    """Narrowing the old grids adds one filter and changes nothing else in them."""
    _, tools = builder
    spec = build("type1")
    for panels, runsets in grids(spec, tools):
        if panels:
            assert tools.runset_filters(runsets[0]) == [("jobType", "=", "analysis"), ("displayName", "=", "summary")]


def test_a_job_type_filter_alone_is_a_problem(builder):
    _, tools = builder
    loose = {"filters": {"filterFormat": "filterV2", "filters": [{"key": {"section": "run", "name": "jobType"}, "op": "=", "value": "analysis", "disabled": False}]}}
    named = json.loads(json.dumps(loose))
    named["filters"]["filters"].append({"key": {"section": "run", "name": "displayName"}, "op": "=", "value": "summary", "disabled": False})
    panel = {"viewType": "Markdown Panel", "config": {}}
    spec = {
        "blocks": [
            {"type": "panel-grid", "metadata": {"panelBankSectionConfig": {"panels": [panel]}, "runSets": [loose]}},
            {"type": "panel-grid", "metadata": {"panelBankSectionConfig": {"panels": [panel]}, "runSets": [named]}},
            {"type": "panel-grid", "metadata": {"panelBankSectionConfig": {"panels": []}, "runSets": [loose]}},
        ]
    }
    problems = tools.runset_problems(spec, {"summary"})
    assert len(problems) == 2
    assert problems[0].startswith("block 0") and "run-name" in problems[0]
    assert problems[1].startswith("block 2") and "group" in problems[1]
    assert tools.runset_problems(spec, {"other"})[1].startswith("block 1")


def test_runs_tables_filter_each_benchmark_by_group(builder, build):
    _, tools = builder
    groups = []
    for panels, runsets in grids(build("full"), tools):
        if not panels:
            filters = tools.runset_filters(runsets[0])
            assert ("jobType", "=", "eval") in filters
            groups += [value for key, _, value in filters if key == "group"]
    assert groups == ["structure-readers", "complete-systems"]


def test_figure_panels_use_the_published_prefixes_and_their_figures_heights(builder, build):
    module, tools = builder
    wandb_papers = pytest.importorskip("molscout.report.wandb_papers")
    assert module.READERS_PREFIX == FIGURE_PREFIX
    assert module.SYSTEMS_PREFIX == wandb_papers.PAPER_FIGURE_PREFIX
    heights = {key: panel["layout"]["h"] for panels, _ in grids(build("full"), tools) for panel in panels for key in panel["config"].get("mediaKeys", [])}
    figures = _system_figures()
    assert {key for key in heights if key.startswith(module.SYSTEMS_PREFIX)} == {module.SYSTEMS_PREFIX + name for name in figures}
    for name, figure in figures.items():
        assert heights[module.SYSTEMS_PREFIX + name] == panel_rows(figure), name
    assert len([key for key in heights if key.startswith(module.READERS_PREFIX)]) == 7


def test_the_full_stage_replaces_the_placeholder_with_the_results(builder, build):
    text = json.dumps(build("full"))
    assert "not yet run" not in text and "will follow" not in text
    assert "352_6kqi" in text and "Complete-system runs" in text
    assert "not yet run" in json.dumps(build("type1"))


def test_complete_system_table_has_a_block_per_paper_set_with_intervals(builder):
    module, _ = builder
    table = module.tables.systems_md()
    rows = [line for line in table.splitlines() if line.startswith("| ")]
    assert [row.split(" | ")[0].split("<br>")[0] for row in rows[1:]] == [
        "| **BioVista**", "| BioMiner", "| DECIMER.ai", "| OpenChemIE",
        "| **Internal**", "| BioMiner", "| DECIMER.ai", "| OpenChemIE",
    ]
    for row in rows[2:5] + rows[6:]:
        cells = row.strip("| ").split(" | ")
        assert len(cells) == 9
        assert all("<br>" in cell for cell in cells[1:4])
        assert all("<br>" not in cell for cell in cells[4:])
    assert "145 papers, 2,304 molecules" in rows[1] and "6 papers, 220 molecules" in rows[5]
    versions = module.tables.versions_md()
    assert all(name in versions for name in ("BioMiner", "DECIMER.ai", "OpenChemIE", "MolScribe"))


def _system_figures():
    """The complete-system figures drawn from made-up numbers; only their heights matter here."""
    metrics = {}
    for tool in SYSTEMS:
        for dataset, prefixes in (("biovista", ("micro", "drawn", "without_submitted")), ("internal", ("micro", "dev", "test"))):
            row = {"speed/s_per_paper_mean": 10.0}
            for prefix in prefixes:
                for measure in ("precision", "recall", "f1"):
                    row[f"{prefix}/{measure}"] = 0.5
                    row[f"{prefix}/{measure}_ci_low"], row[f"{prefix}/{measure}_ci_high"] = 0.4, 0.6
            metrics[(tool, dataset)] = row
    sizes = {(d, v): 3 for d, views in paper_figures.VIEWS.items() for v, _, _ in views}
    columns = ["system", "dataset", "paper", "molecules", "tp", "recall"]
    papers = (columns, [[paper_figures.SYSTEM_NAMES[tool], "biovista", "1_aaaa", 2, 1, 0.5] for tool in SYSTEMS])
    return {
        "biovista_views": paper_figures.view_bars(metrics, "biovista", sizes),
        "recall_by_paper": paper_figures.recall_by_paper(papers),
        "internal_views": paper_figures.view_bars(metrics, "internal", sizes),
        "f1_vs_time": paper_figures.f1_vs_time(metrics),
    }
