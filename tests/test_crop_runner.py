"""The shared crop loop that every tool's run.py calls."""

import sys
from pathlib import Path

import pytest

from molscout.predictions import read_predictions

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "tools"))
import crop_runner  # noqa: E402


def make_images(folder: Path, names: list[str]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"x")
    return folder


def parse(images: Path, output: Path, dataset: str = "uspto", tool: str = "fake 1.0"):
    parser = crop_runner.base_parser("test")
    return parser.parse_args(
        ["--images", str(images), "--dataset", dataset, "--tool", tool, "--output", str(output)]
    )


def test_list_images_sorted_and_filtered(tmp_path):
    folder = make_images(tmp_path / "img", ["b.png", "a.JPG", "c.tiff", "d.txt", ".hidden.png", "e.bmp"])
    (folder / ".cache").mkdir()
    (folder / ".cache" / "z.png").write_bytes(b"x")
    (folder / "sub.png").mkdir()
    names = [p.name for p in crop_runner.list_images(folder)]
    assert names == ["a.JPG", "b.png", "c.tiff", "e.bmp"]


def test_writes_predictions_the_reader_accepts(tmp_path):
    images = make_images(tmp_path / "img", ["a.png", "b.png", "c.png"])
    output = tmp_path / "predictions.csv"
    answers = {"a": ("CCO", 0.9), "b": ("c1ccccc1", None), "c": ("N", 1)}
    errors = crop_runner.run_crops(lambda image: answers[image.stem], parse(images, output))
    assert errors == 0
    rows = read_predictions(output)
    assert [r.item_id for r in rows] == ["a", "b", "c"]
    assert [r.smiles for r in rows] == ["CCO", "c1ccccc1", "N"]
    assert [r.confidence for r in rows] == [0.9, None, 1.0]
    assert all(r.dataset == "uspto" and r.tool == "fake 1.0" for r in rows)
    assert all(r.page is None and r.bbox is None and r.seconds >= 0 for r in rows)
    assert not Path(f"{output}.part").exists()


def test_failed_image_gives_empty_row_and_run_continues(tmp_path, capsys):
    images = make_images(tmp_path / "img", ["a.png", "bad.png", "c.png"])
    output = tmp_path / "predictions.csv"

    def predict(image):
        if image.stem == "bad":
            raise ValueError("empty drawing")
        return "C", None

    errors = crop_runner.run_crops(predict, parse(images, output))
    assert errors == 1
    assert [(r.item_id, r.smiles) for r in read_predictions(output)] == [("a", "C"), ("bad", ""), ("c", "C")]
    assert "bad.png: ValueError: empty drawing" in capsys.readouterr().err


def test_warmup_runs_first_image_untimed_and_extra(tmp_path):
    images = make_images(tmp_path / "img", ["a.png", "b.png"])
    seen = []
    crop_runner.run_crops(lambda image: (seen.append(image.stem), ("C", None))[1], parse(images, tmp_path / "p.csv"))
    assert seen == ["a", "a", "b"]
    seen.clear()
    crop_runner.run_crops(
        lambda image: (seen.append(image.stem), ("C", None))[1], parse(images, tmp_path / "q.csv"), warmup=False
    )
    assert seen == ["a", "b"]


def test_failing_warmup_does_not_stop_the_run(tmp_path):
    images = make_images(tmp_path / "img", ["a.png", "b.png"])
    calls = {"a": 0}

    def predict(image):
        if image.stem == "a":
            calls["a"] += 1
            if calls["a"] == 1:
                raise RuntimeError("cold start")
        return "C", None

    assert crop_runner.run_crops(predict, parse(images, tmp_path / "p.csv")) == 0
    assert [r.smiles for r in read_predictions(tmp_path / "p.csv")] == ["C", "C"]


def test_smiles_with_whitespace_and_newline_round_trip(tmp_path):
    images = make_images(tmp_path / "img", ["a.png", "b.png"])
    answers = {"a": "CC O", "b": "C\nC"}
    output = tmp_path / "predictions.csv"
    crop_runner.run_crops(lambda image: (answers[image.stem], None), parse(images, output), warmup=False)
    assert [r.smiles for r in read_predictions(output)] == ["CC O", "C\nC"]


def test_non_finite_confidence_is_written_empty(tmp_path):
    images = make_images(tmp_path / "img", ["a.png"])
    output = tmp_path / "predictions.csv"
    crop_runner.run_crops(lambda image: ("C", float("nan")), parse(images, output), warmup=False)
    assert read_predictions(output)[0].confidence is None


def test_interrupt_leaves_no_predictions_and_no_part_file(tmp_path):
    images = make_images(tmp_path / "img", ["a.png", "b.png"])
    output = tmp_path / "predictions.csv"

    def predict(image):
        if image.stem == "b":
            raise KeyboardInterrupt
        return "C", None

    with pytest.raises(KeyboardInterrupt):
        crop_runner.run_crops(predict, parse(images, output), warmup=False)
    assert not output.exists()
    assert not Path(f"{output}.part").exists()


def test_progress_every_500_images_and_final_count(tmp_path, capsys):
    images = make_images(tmp_path / "img", [f"{i:04d}.png" for i in range(501)])
    crop_runner.run_crops(lambda image: ("C", None), parse(images, tmp_path / "p.csv"), warmup=False)
    err = capsys.readouterr().err
    assert "500/501" in err
    assert "501 images" in err
