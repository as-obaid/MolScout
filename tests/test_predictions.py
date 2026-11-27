import re

import pytest

from molscout.predictions import (
    COLUMNS,
    Prediction,
    PredictionsFormatError,
    read_predictions,
    write_predictions,
)

HEADER = ",".join(COLUMNS)


def write(tmp_path, text):
    path = tmp_path / "predictions.csv"
    path.write_bytes(text.encode("utf-8"))
    return path


def test_reads_crop_rows(tmp_path):
    path = write(tmp_path, HEADER + "\nuspto,US1-C001,CCO,,,0.9,MolScribe 1.1.1,0.25\n")
    assert read_predictions(path) == (
        Prediction("uspto", "US1-C001", "CCO", None, None, 0.9, "MolScribe 1.1.1", 0.25),
    )


def test_reads_paper_rows_with_page_and_bbox(tmp_path):
    path = write(tmp_path, HEADER + '\ninternal,1,c1ccccc1,3,"72.0,100.5,200,300",,BioMiner abc,41.5\n')
    (prediction,) = read_predictions(path)
    assert prediction.page == 3
    assert prediction.bbox == (72.0, 100.5, 200.0, 300.0)
    assert prediction.confidence is None


def test_empty_smiles_is_kept_as_no_output(tmp_path):
    path = write(tmp_path, HEADER + "\nuspto,c1,,,,,DECIMER 2.7,1.0\n")
    assert read_predictions(path)[0].smiles == ""


def test_accepts_bom_and_crlf(tmp_path):
    path = write(tmp_path, "﻿" + HEADER + "\r\nuspto,c1,CCO,,,,T 1,0.1\r\n")
    assert len(read_predictions(path)) == 1


def test_columns_may_come_in_any_order(tmp_path):
    columns = list(reversed(COLUMNS))
    row = {"dataset": "uspto", "item_id": "c1", "smiles": "CCO", "page": "", "bbox": "",
           "confidence": "", "tool": "T 1", "seconds": "0.1"}
    text = ",".join(columns) + "\n" + ",".join(row[c] for c in columns) + "\n"
    assert read_predictions(write(tmp_path, text))[0].smiles == "CCO"


def test_header_only_file_has_no_predictions(tmp_path):
    assert read_predictions(write(tmp_path, HEADER + "\n")) == ()


def test_strips_whitespace_around_item_id(tmp_path):
    path = write(tmp_path, HEADER + "\ninternal, 1 ,CCO,,,,T 1,0.1\n")
    assert read_predictions(path)[0].item_id == "1"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "file is empty"),
        ("dataset,item_id,smiles\n", "missing column(s): page, bbox, confidence, tool, seconds"),
        (HEADER + ",extra\n", "unknown column(s): extra"),
    ],
)
def test_rejects_bad_headers(tmp_path, text, message):
    with pytest.raises(PredictionsFormatError, match=re.escape(message)):
        read_predictions(write(tmp_path, text))


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("USPTO,c1,CCO,,,,T 1,0.1", "unknown dataset 'USPTO'"),
        ("uspto,,CCO,,,,T 1,0.1", "item_id is empty"),
        ("uspto,c1,CCO,,,,,0.1", "tool is empty"),
        ("uspto,c1,CCO,,,,T 1,", "seconds is empty"),
        ("uspto,c1,CCO,,,,T 1,-1", "seconds must be >= 0"),
        ("uspto,c1,CCO,,,,T 1,nan", "not a finite number"),
        ("uspto,c1,CCO,,,high,T 1,0.1", "confidence"),
        ("uspto,c1,CCO,2,,,T 1,0.1", "page must be empty for a crop dataset"),
        ('uspto,c1,CCO,,"0,0,1,1",,T 1,0.1', "bbox must be empty for a crop dataset"),
        ("internal,1,CCO,0,,,T 1,0.1", "pages are 1-based"),
        ("internal,1,CCO,1.0,,,T 1,0.1", "page"),
        ('internal,1,CCO,1,"0,0,1",,T 1,0.1', "expected x0,y0,x1,y1"),
        ('internal,1,CCO,1,"5,0,1,1",,T 1,0.1', "x0 <= x1"),
        ("uspto,c1,CCO,,,,T 1", "expected 8 fields"),
        ("uspto,c1,CCO,,,,T 1,0.1,surplus", "extra field"),
    ],
)
def test_rejects_bad_rows_with_line_number(tmp_path, row, message):
    with pytest.raises(PredictionsFormatError) as info:
        read_predictions(write(tmp_path, HEADER + "\n" + row + "\n"))
    assert any(e.startswith("line 2:") and message in e for e in info.value.errors), info.value.errors


def test_reports_every_bad_row_at_once(tmp_path):
    text = HEADER + "\nuspto,,CCO,,,,T 1,0.1\nuspto,c2,CCO,,,,,0.1\n"
    with pytest.raises(PredictionsFormatError) as info:
        read_predictions(write(tmp_path, text))
    assert [e.split(":")[0] for e in info.value.errors] == ["line 2", "line 3"]


def test_rejects_mixed_datasets_and_tools(tmp_path):
    text = HEADER + "\nuspto,c1,CCO,,,,T 1,0.1\nuob,c2,CCO,,,,T 2,0.1\n"
    with pytest.raises(PredictionsFormatError) as info:
        read_predictions(write(tmp_path, text))
    joined = "\n".join(info.value.errors)
    assert "one dataset" in joined
    assert "one tool" in joined


def test_rejects_two_rows_for_one_crop(tmp_path):
    text = HEADER + "\nuspto,c1,CCO,,,,T 1,0.1\nuspto,c1,CCN,,,,T 1,0.1\n"
    with pytest.raises(PredictionsFormatError, match="more than one row.*c1"):
        read_predictions(write(tmp_path, text))


def test_allows_many_rows_per_paper(tmp_path):
    text = HEADER + "\ninternal,1,CCO,1,,,T 1,9\ninternal,1,CCN,2,,,T 1,9\n"
    assert len(read_predictions(write(tmp_path, text))) == 2


def test_write_then_read_round_trips(tmp_path):
    rows = (
        Prediction("internal", "1", "C[C@H](N)C(=O)O", 2, (72.0, 10.25, 300.5, 400.0), 0.875, "MolScout 0.1.0", 12.5),
        Prediction("internal", "1", "", None, None, None, "MolScout 0.1.0", 12.5),
    )
    path = tmp_path / "out.csv"
    write_predictions(path, rows)
    assert read_predictions(path) == rows
