"""`molscout run`: the pipeline from the command line, end to end with fake workers and steps."""

import csv
import json

import pytest
from pipeline_helpers import FAKE_STEPS, make_papers, pipeline_yaml

from molscout.cli import main
from molscout.pipeline import runner
from molscout.predictions import read_predictions


@pytest.fixture(autouse=True)
def fake_steps(monkeypatch):
    monkeypatch.setattr(runner, "STEPS", {**runner.STEPS, **FAKE_STEPS})


def molscout_run(tmp_path, source, *args):
    return main(["run", str(source), "-o", str(tmp_path / "out"), "--config", str(pipeline_yaml(tmp_path)), *args])


def test_run_a_folder_of_pdfs(tmp_path, capsys):
    make_papers(tmp_path)
    assert molscout_run(tmp_path, tmp_path / "pdfs") == 0
    out = tmp_path / "out"
    with (out / "molecules.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 6
    assert not (out / "predictions.csv").exists()  # no dataset, so no scorer format
    captured = capsys.readouterr()
    assert "2 papers, 0 failed" in captured.err and captured.out == f"wrote {out / 'molecules.csv'}\n"


def test_run_one_pdf_with_a_dataset_writes_predictions(tmp_path):
    [paper, _] = make_papers(tmp_path)
    assert molscout_run(tmp_path, paper.pdf, "--dataset", "biovista", "--export", "accepted+review") == 0
    predictions = read_predictions(tmp_path / "out" / "predictions.csv")
    assert {p.item_id for p in predictions} == {"p1"} and len(predictions) == 3
    assert predictions[0].tool.startswith("MolScout ")


def test_run_a_papers_csv_with_overrides(tmp_path):
    make_papers(tmp_path)
    listing = tmp_path / "papers.csv"
    listing.write_text("paper_id,pdf\nalpha,pdfs/p1.pdf\n")
    assert molscout_run(tmp_path, listing, "--fixes-off", "vote,junk") == 0
    stages = json.loads((tmp_path / "out" / "stages.json").read_text())
    assert list(stages["per_paper"]) == ["alpha"]
    assert stages["config"]["fixes"]["vote"] is False and stages["funnel"]["voted_agree_1"] == 2


def test_a_failed_paper_fails_the_run_unless_keep_going(tmp_path, capsys):
    make_papers(tmp_path)
    config = pipeline_yaml(tmp_path, worker_args={"segment": ["--fail", "p2"]})
    args = ["run", str(tmp_path / "pdfs"), "-o", str(tmp_path / "out"), "--config", str(config)]
    assert main(args) == 1
    assert "1 paper(s) failed: p2" in capsys.readouterr().err
    assert main([*args, "--keep-going"]) == 0


def test_the_32b_joiner_needs_an_endpoint(tmp_path, capsys):
    make_papers(tmp_path)
    assert molscout_run(tmp_path, tmp_path / "pdfs", "--joiner", "32b") == 1
    assert "vllm_url" in capsys.readouterr().err


def test_bad_inputs_are_reported(tmp_path, capsys):
    assert molscout_run(tmp_path, tmp_path / "nothing.pdf") == 1
    assert "molscout run: error:" in capsys.readouterr().err
    make_papers(tmp_path)
    assert molscout_run(tmp_path, tmp_path / "pdfs", "--fixes-off", "speed") == 1
    assert "speed" in capsys.readouterr().err
