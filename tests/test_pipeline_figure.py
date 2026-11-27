"""scripts/pipeline_figure.py: two SVGs that parse, name every stage of docs/pipeline.md and carry nothing else."""

from __future__ import annotations

import importlib.util
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NS = "{http://www.w3.org/2000/svg}"
STAGES = ("Parse", "Find drawings", "Read and vote", "Link and build", "Check and export")
DRAWN = {"svg", "rect", "g", "path", "text"}  # every element the figure needs


def _load_script():
    spec = importlib.util.spec_from_file_location("pipeline_figure", REPO / "scripts" / "pipeline_figure.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


figure = _load_script()


@pytest.fixture(scope="module")
def written(tmp_path_factory) -> dict[str, Path]:
    paths = figure.write_figures(tmp_path_factory.mktemp("figures"))
    return {path.stem.removeprefix("pipeline-"): path for path in paths}


def test_writes_a_light_and_a_dark_svg(written):
    assert sorted(path.name for path in written.values()) == ["pipeline-dark.svg", "pipeline-light.svg"]


@pytest.mark.parametrize(("theme", "background"), [("light", "#ffffff"), ("dark", "#0d1117")])
def test_each_svg_parses_on_its_background(written, theme, background):
    root = ET.parse(written[theme]).getroot()
    assert root.tag == f"{NS}svg"
    assert root.find(f"{NS}rect").get("fill") == background
    assert {element.tag.removeprefix(NS) for element in root.iter()} <= DRAWN


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_every_stage_is_named_once_from_pdf_to_molecules_csv(written, theme):
    texts = [element.text for element in ET.parse(written[theme]).getroot().iter(f"{NS}text")]
    for stage in STAGES:
        assert texts.count(stage) == 1, stage
    assert texts[0] == "PDF"
    assert "molecules.csv" in texts


def test_stage_names_are_the_docs():
    assert figure.STAGES == STAGES
    doc = (REPO / "docs" / "pipeline.md").read_text(encoding="utf-8")
    for stage in STAGES:
        assert re.search(rf"^\| {re.escape(stage)} \|", doc, re.MULTILINE), f"{stage} is not a row of the Stages table"


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_no_generator_metadata(written, theme):
    text = written[theme].read_text(encoding="utf-8").lower()
    for word in ("generator", "generated", "inkscape", "<metadata", "<!--", "<desc", "<title"):
        assert word not in text, word


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_committed_figure_is_the_scripts(theme):
    committed = REPO / "docs" / "figures" / f"pipeline-{theme}.svg"
    assert committed.read_text(encoding="utf-8") == figure.svg(theme), "run scripts/pipeline_figure.py"


def test_wide_enough_to_stay_legible_at_github_width():
    """About 880 px of content width: a figure at most 960 units wide keeps its 12-unit notes above 11 px."""
    assert figure.layout().width <= 960
