import hashlib
import json

import pytest

from molscout.scoring.report import build_report, write_scores
from molscout.scoring.stats import BOOTSTRAP_SEED


def test_report_records_how_scores_were_made(tmp_path):
    predictions = tmp_path / "predictions.csv"
    predictions.write_bytes(b"x\n")
    report = build_report({"kind": "crop"}, dataset="uspto", tool="T 1", predictions_path=predictions)
    assert (report["dataset"], report["tool"]) == ("uspto", "T 1")
    assert report["predictions"]["sha256"] == hashlib.sha256(b"x\n").hexdigest()
    assert report["scoring"]["rdkit_version"] == "2026.03.2"
    assert report["scoring"]["ci95_macro"] == {
        "method": "percentile bootstrap over papers",
        "resamples": 10_000,
        "seed": BOOTSTRAP_SEED,
    }
    assert report["scores"] == {"kind": "crop"}


def test_write_scores_round_trips_json(tmp_path):
    out = tmp_path / "run" / "scores.json"
    write_scores(out, {"a": [0.5, 1]})
    assert json.loads(out.read_text()) == {"a": [0.5, 1]}


def test_write_scores_refuses_nan(tmp_path):
    with pytest.raises(ValueError):
        write_scores(tmp_path / "scores.json", {"a": float("nan")})
