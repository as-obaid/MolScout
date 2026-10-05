"""The shared crop loop that every tool's run.py calls."""

import json
import os
import signal
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


def parse(images: Path, output: Path, dataset: str = "uspto", tool: str = "fake 1.0", resume: Path | None = None):
    parser = crop_runner.base_parser("test")
    argv = ["--images", str(images), "--dataset", dataset, "--tool", tool, "--output", str(output)]
    return parser.parse_args(argv + (["--resume", str(resume)] if resume else []))


def sidecar(output: Path) -> dict:
    """The errors file run_crops writes beside predictions.csv."""
    return json.loads(output.with_name(f"{output.stem}.errors.json").read_text(encoding="utf-8"))


def leftovers(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.name.endswith(".part"))


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
    assert sidecar(output) == {"images": 3, "failed": 0, "errors": {}}
    assert (tmp_path / "predictions.errors.json").is_file()
    assert leftovers(tmp_path) == []


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
    assert sidecar(output) == {"images": 3, "failed": 1, "errors": {"bad": "ValueError: empty drawing"}}


def test_25_failures_in_a_row_stop_the_run_and_write_nothing(tmp_path):
    names = [f"{i:02d}.png" for i in range(40)]
    images = make_images(tmp_path / "img", names)
    output = tmp_path / "out" / "predictions.csv"
    output.parent.mkdir()
    seen = []

    def predict(image):
        seen.append(image.stem)
        if image.stem == "00":
            return "C", None
        raise RuntimeError(f"CUDA error on {image.stem}")

    assert crop_runner.MAX_CONSECUTIVE_FAILURES == 25
    with pytest.raises(RuntimeError, match=r"25 images failed in a row.*25\.png: RuntimeError: CUDA error on 25"):
        crop_runner.run_crops(predict, parse(images, output), warmup=False)
    assert seen == [f"{i:02d}" for i in range(26)]
    assert list(output.parent.iterdir()) == []


def test_failures_that_are_not_in_a_row_do_not_stop_the_run(tmp_path):
    images = make_images(tmp_path / "img", [f"{i:02d}.png" for i in range(49)])
    output = tmp_path / "predictions.csv"

    def predict(image):
        if image.stem == "24":
            return "C", None
        raise ValueError("odd drawing")

    assert crop_runner.run_crops(predict, parse(images, output), warmup=False) == 48
    assert sidecar(output)["failed"] == 48
    assert [r.smiles for r in read_predictions(output)].count("C") == 1


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


def test_failing_warmup_is_reported_and_does_not_stop_the_run(tmp_path, capsys):
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
    assert "warmup on a.png failed: RuntimeError: cold start" in capsys.readouterr().err
    assert sidecar(tmp_path / "p.csv") == {"images": 2, "failed": 0, "errors": {}}


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
    assert sorted(p.name for p in tmp_path.iterdir()) == ["img"]


def test_progress_every_500_images_and_final_count(tmp_path, capsys):
    images = make_images(tmp_path / "img", [f"{i:04d}.png" for i in range(501)])
    crop_runner.run_crops(lambda image: ("C", None), parse(images, tmp_path / "p.csv"), warmup=False)
    err = capsys.readouterr().err
    assert "500/501" in err
    assert "501 images" in err


# With --resume PATH every finished row goes to PATH at once, so a stopped run carries on where it left off.

NAMES = [f"{stem}.png" for stem in "abcdefgh"]
ANSWERS = {"a": "CCO", "b": "", "c": "C\nC", "d": "N", "e": "O", "f": "", "g": "CCC", "h": "Cl"}
HEADER = "dataset,item_id,smiles,page,bbox,confidence,tool,seconds\r\n"


class Stop(BaseException):
    """Stands in for a kill: not an Exception, so run_crops does not catch it."""


def answer(image):
    """b and f raise; the rest get ANSWERS."""
    if image.stem in ("b", "f"):
        raise ValueError(f"odd drawing {image.stem}")
    return ANSWERS[image.stem], 0.5


def stopping_at(stem, seen=None):
    def predict(image):
        if seen is not None:
            seen.append(image.stem)
        if image.stem == stem:
            raise Stop
        return answer(image)

    return predict


def errors_log(checkpoint: Path) -> list[dict]:
    path = checkpoint.with_name(f"{checkpoint.stem}.errors.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def rows_without_seconds(path: Path) -> list[list[str]]:
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return [row[:-1] for row in csv.reader(handle)]


def folders(tmp_path: Path) -> tuple[Path, Path, Path]:
    images = make_images(tmp_path / "img", NAMES)
    output = tmp_path / "out" / "predictions.csv"
    output.parent.mkdir()
    return images, output, tmp_path / "checkpoint" / "predictions.csv"


def test_resume_keeps_the_rows_done_and_predicts_only_the_rest(tmp_path):
    images, output, checkpoint = folders(tmp_path)
    checkpoint.parent.mkdir()
    with pytest.raises(Stop):
        crop_runner.run_crops(stopping_at("d"), parse(images, output, resume=checkpoint))
    assert [r.item_id for r in read_predictions(checkpoint)] == ["a", "b", "c"]
    assert list(output.parent.iterdir()) == []
    seen = []
    assert crop_runner.run_crops(stopping_at(None, seen), parse(images, output, resume=checkpoint)) == 2
    assert seen == ["d", "d", "e", "f", "g", "h"]  # the untimed warm-up is on the first image left
    assert [r.item_id for r in read_predictions(output)] == list("abcdefgh")
    assert [r.item_id for r in read_predictions(checkpoint)] == list("abcdefgh")  # never deleted


def test_a_resumed_run_writes_what_an_uninterrupted_run_writes_apart_from_seconds(tmp_path):
    images, output, checkpoint = folders(tmp_path)
    checkpoint.parent.mkdir()
    whole = tmp_path / "predictions.csv"
    assert crop_runner.run_crops(answer, parse(images, whole)) == 2
    for stem in ("c", "g"):
        with pytest.raises(Stop):
            crop_runner.run_crops(stopping_at(stem), parse(images, output, resume=checkpoint))
    assert crop_runner.run_crops(answer, parse(images, output, resume=checkpoint)) == 2
    assert rows_without_seconds(output) == rows_without_seconds(whole)
    assert errors_path_bytes(output) == errors_path_bytes(whole)


def errors_path_bytes(output: Path) -> bytes:
    return crop_runner.errors_path(output).read_bytes()


def test_errors_from_every_segment_are_merged_in_image_order(tmp_path):
    images, output, checkpoint = folders(tmp_path)
    checkpoint.parent.mkdir()
    with pytest.raises(Stop):
        crop_runner.run_crops(stopping_at("e"), parse(images, output, resume=checkpoint), warmup=False)
    assert errors_log(checkpoint) == [{"item_id": "b", "error": "ValueError: odd drawing b"}]
    assert crop_runner.run_crops(answer, parse(images, output, resume=checkpoint), warmup=False) == 2
    assert sidecar(output) == {
        "images": 8,
        "failed": 2,
        "errors": {"b": "ValueError: odd drawing b", "f": "ValueError: odd drawing f"},
    }
    assert [entry["item_id"] for entry in errors_log(checkpoint)] == ["b", "f"]


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        (HEADER + "uspto,a,OLD,,,,fake 1.0,0.5\r\nuspto,b,CC", ["a"]),
        (HEADER + 'uspto,a,OLD,,,,fake 1.0,0.5\r\nuspto,b,"C\nC', ["a"]),
        (HEADER + "uspto,a,OLD,,,,fake 1.0,0.5\r", []),
        ("dataset,item_", []),
    ],
    ids=["mid-row", "inside-quotes", "before-newline", "header"],
)
def test_a_last_row_cut_short_by_a_kill_is_dropped(tmp_path, text, kept):
    images = make_images(tmp_path / "img", ["a.png", "b.png", "c.png"])
    output, checkpoint = tmp_path / "predictions.csv", tmp_path / "checkpoint.csv"
    checkpoint.write_text(text, encoding="utf-8", newline="")
    seen = []
    args = parse(images, output, resume=checkpoint)
    crop_runner.run_crops(lambda image: (seen.append(image.stem), ("N", None))[1], args, warmup=False)
    assert seen == [stem for stem in "abc" if stem not in kept]
    expected = [(stem, "OLD" if stem in kept else "N") for stem in "abc"]
    assert [(r.item_id, r.smiles) for r in read_predictions(output)] == expected
    assert [(r.item_id, r.smiles) for r in read_predictions(checkpoint)] == expected  # the cut row is gone


def test_rows_for_crops_without_an_image_are_refused(tmp_path):
    images = make_images(tmp_path / "img", ["a.png"])
    output, checkpoint = tmp_path / "out" / "predictions.csv", tmp_path / "checkpoint.csv"
    output.parent.mkdir()
    text = (HEADER + "uspto,zz,C,,,,fake 1.0,0.5\r\n").encode()
    checkpoint.write_bytes(text)
    with pytest.raises(ValueError, match=r"checkpoint\.csv.*zz"):
        crop_runner.run_crops(lambda image: ("C", None), parse(images, output, resume=checkpoint))
    assert list(output.parent.iterdir()) == []
    assert checkpoint.read_bytes() == text


def test_failures_that_stop_the_run_are_left_out_of_the_checkpoint(tmp_path):
    images = make_images(tmp_path / "img", [f"{i:02d}.png" for i in range(30)])
    output, checkpoint = tmp_path / "predictions.csv", tmp_path / "checkpoint.csv"

    def broken(image):
        if image.stem == "00":
            return "C", None
        raise RuntimeError("CUDA error")

    with pytest.raises(RuntimeError, match="25 images failed in a row"):
        crop_runner.run_crops(broken, parse(images, output, resume=checkpoint), warmup=False)
    assert [r.item_id for r in read_predictions(checkpoint)] == ["00"]
    assert errors_log(checkpoint) == []
    seen = []
    args = parse(images, output, resume=checkpoint)
    assert crop_runner.run_crops(lambda image: (seen.append(image.stem), ("N", None))[1], args, warmup=False) == 0
    assert seen == [f"{i:02d}" for i in range(1, 30)]


def test_sigterm_exits_143_keeps_the_checkpoint_and_removes_only_the_part_file(tmp_path):
    images, output, checkpoint = folders(tmp_path)
    checkpoint.parent.mkdir()

    def predict(image):
        if image.stem == "d":
            os.kill(os.getpid(), signal.SIGTERM)
        return answer(image)

    def not_installed(signum, frame):
        raise AssertionError("run_crops did not install its SIGTERM handler")

    previous = signal.signal(signal.SIGTERM, not_installed)
    try:
        with pytest.raises(SystemExit) as stopped:
            crop_runner.run_crops(predict, parse(images, output, resume=checkpoint), warmup=False)
        assert signal.getsignal(signal.SIGTERM) is not_installed
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert stopped.value.code == 143
    assert list(output.parent.iterdir()) == []
    assert [r.item_id for r in read_predictions(checkpoint)] == ["a", "b", "c"]


def test_without_resume_only_the_two_files_are_written_and_sigterm_is_left_alone(tmp_path):
    images, output, _ = folders(tmp_path)
    args = parse(images, output)
    assert args.resume is None
    handlers = []
    crop_runner.run_crops(lambda image: (handlers.append(signal.getsignal(signal.SIGTERM)), ("C", None))[1], args)
    assert set(handlers) == {signal.getsignal(signal.SIGTERM)}
    assert sorted(p.name for p in output.parent.iterdir()) == ["predictions.csv", "predictions.errors.json"]


def test_resume_leaves_the_tools_own_checkpoint_option_alone(tmp_path):
    parser = crop_runner.base_parser("test")
    parser.add_argument("--checkpoint", type=Path, required=True)  # model weights, in molscribe's run.py and others
    argv = ["--images", "img", "--dataset", "uspto", "--tool", "t", "--output", "p.csv"]
    args = parser.parse_args([*argv, "--checkpoint", "model.pth", "--resume", "c.csv"])
    assert (args.checkpoint, args.resume) == (Path("model.pth"), Path("c.csv"))
