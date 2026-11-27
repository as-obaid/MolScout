"""The W&B entity and project come from --entity and --project, else WANDB_ENTITY and WANDB_PROJECT.

No script names an entity of its own; the project defaults to MolScout. scripts/report/target.py serves
build_report.py (which needs the report-builder extra, so it is tested in test_report_builder.py);
scripts/wandb_upload.py reads the same through argparse.
"""

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def target():
    return load(REPO / "scripts" / "report" / "target.py", "report_target")


@pytest.fixture
def upload(monkeypatch):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    return load(REPO / "scripts" / "wandb_upload.py", "wandb_upload_target")


@pytest.mark.parametrize(
    ("argv", "environ", "expected"),
    [
        (["--entity", "lab", "--project", "p"], {"WANDB_ENTITY": "env", "WANDB_PROJECT": "q"}, ("lab", "p")),
        (["--entity=lab", "--project=p"], {}, ("lab", "p")),
        (["--dry-run"], {"WANDB_ENTITY": "env", "WANDB_PROJECT": "q"}, ("env", "q")),
        (["--stage", "type1"], {}, (None, "MolScout")),
        ([], {"WANDB_ENTITY": "", "WANDB_PROJECT": ""}, (None, "MolScout")),
    ],
)
def test_report_target_reads_flags_then_the_environment(target, argv, environ, expected):
    assert target.wandb_target(argv, environ) == expected


@pytest.mark.parametrize("argv", [["--entity"], ["--project"], ["--entity", "--dry-run"], ["--entity="]])
def test_report_target_flag_without_a_value_is_an_error(target, argv):
    with pytest.raises(SystemExit, match="needs a value"):
        target.wandb_target(argv, {})


def test_upload_takes_the_entity_and_project_from_the_environment(upload, monkeypatch):
    monkeypatch.setenv("WANDB_ENTITY", "env-entity")
    monkeypatch.setenv("WANDB_PROJECT", "env-project")
    args = upload.build_parser().parse_args([])
    assert (args.entity, args.project) == ("env-entity", "env-project")
    args = upload.build_parser().parse_args(["--entity", "e", "--project", "p"])
    assert (args.entity, args.project) == ("e", "p")


def test_upload_needs_an_entity_and_defaults_the_project(upload, capsys):
    with pytest.raises(SystemExit):
        upload.build_parser().parse_args([])
    assert "--entity" in capsys.readouterr().err
    assert upload.build_parser().parse_args(["--entity", "e"]).project == "MolScout"


@pytest.mark.parametrize(
    "path", ["scripts/wandb_upload.py", "scripts/report/build_report.py", "scripts/report/target.py", "docs/reproduce.md"]
)
def test_no_script_or_doc_names_an_entity_of_its_own(path):
    text = (REPO / path).read_text(encoding="utf-8")
    assert not re.search(r"wandb\.ai/[A-Za-z0-9_-]", text)  # https://wandb.ai/{entity}/... or <entity> only
    assert not re.search(r"(?i)\bENTITY\s*(,\s*\w+)?\s*=\s*[\"']", text)  # no ENTITY = "<name>"


def test_build_report_reads_its_target_through_target_py():
    """build_report.py itself needs the report-builder extra; test_report_builder.py imports it."""
    text = (REPO / "scripts" / "report" / "build_report.py").read_text(encoding="utf-8")
    assert "wandb_target(sys.argv[1:], os.environ)" in text
