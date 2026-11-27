import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("figures", REPO / "scripts" / "figures.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


figures = _load_script()

# paper -> (tp, fp, fn); small made-up splits in the shape of the real ones
INTERNAL_DEV = {"1": (10, 1, 10), "16": (6, 0, 9), "19": (4, 1, 5)}
INTERNAL_TEST = {"2": (5, 2, 10), "4": (3, 1, 6), "6": (2, 2, 4)}
BIOVISTA_DEV = {f"dev_{i}": (3, 1, 2) for i in range(20)}
BIOVISTA_HELDOUT = {f"paper_{i}": (4, 2, 3) for i in range(125)}


def _metric(top: float, bottom: float) -> dict:
    value = top / bottom if bottom else 0.0
    return {"value": value, "ci95": [max(0.0, value - 0.1), min(1.0, value + 0.1)]}


def _paper_run(results: Path, run: str, groups: dict[str, dict[str, tuple[int, int, int]]]) -> None:
    """A paper run's scores.json: each group's papers with their counts, and micro scores from the totals."""
    papers, out = {}, {}
    for name, counts in groups.items():
        for paper, (tp, fp, fn) in counts.items():
            papers[paper] = {"molecules": tp + fn, "tp": tp, "fp": fp, "fn": fn}
        tp, fp, fn = (sum(paper[i] for paper in counts.values()) for i in range(3))
        out[name] = {
            "papers": list(counts), "molecules": tp + fn, "counts": {"tp": tp, "fp": fp, "fn": fn},
            "micro": {"precision": _metric(tp, tp + fp), "recall": _metric(tp, tp + fn),
                      "f1": _metric(2 * tp, 2 * tp + fp + fn)},
        }
    (results / run).mkdir(parents=True)
    report = {"scores": {"kind": "paper", "papers": papers, "groups": out}}
    (results / run / "scores.json").write_text(json.dumps(report))


def _crop_runs(results: Path) -> None:
    for i, tool in enumerate(figures.READERS):
        for j, dataset in enumerate(figures.CROP_DATASETS):
            (results / f"{tool}__{dataset}").mkdir(parents=True)
            accuracy = {"value": 0.5 + (i + j) / 100, "ci95": [0.4, 0.7]}
            (results / f"{tool}__{dataset}" / "scores.json").write_text(json.dumps({"scores": {"accuracy": accuracy}}))


def _rows(panel):
    return dict(panel.rows)


def _png(path: Path) -> tuple[tuple[int, int], set[bytes]]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    names, i = set(), 8
    while i < len(data):
        length, name = struct.unpack(">I4s", data[i : i + 8])
        names.add(name)
        i += 12 + length
    return struct.unpack(">II", data[16:24]), names


def test_crop_accuracy_reads_every_reader_on_every_dataset(tmp_path):
    _crop_runs(tmp_path)
    table = figures.crop_accuracy(tmp_path)
    assert [name for name, _ in table] == ["MolScribe", "MolNexTR", "DECIMER", "MolVec", "MolGlyph", "OCSRGlyph"]
    rows = dict(table)
    assert rows["MolScribe"]["uspto"] == pytest.approx(50.0)
    assert rows["OCSRGlyph"]["molrecbench_wild"] == pytest.approx(59.0)
    assert all(len(values) == 5 for values in rows.values())


def test_crop_accuracy_names_a_missing_run(tmp_path):
    _crop_runs(tmp_path)
    (tmp_path / "molnextr__uspto" / "scores.json").unlink()
    with pytest.raises(figures.FigureError, match="molnextr__uspto"):
        figures.crop_accuracy(tmp_path)


def test_systems_read_the_internal_dev_and_test_groups(tmp_path):
    _paper_run(tmp_path, "biominer__internal", {"dev": INTERNAL_DEV, "test": INTERNAL_TEST})
    dev, heldout = figures.systems(tmp_path, "dev")[0], figures.systems(tmp_path, "heldout")[0]
    assert (dev.dataset, dev.title) == ("internal", "Internal 1, 16 and 19 · 44 molecules")
    assert heldout.title == "Internal 2, 4 and 6 · 30 molecules"
    scores = _rows(dev)["BioMiner"]
    assert [scores.precision.value, scores.recall.value, scores.f1.value] == pytest.approx(
        [100 * 20 / 22, 100 * 20 / 44, 100 * 40 / 66])
    assert scores.f1.low == pytest.approx(100 * 40 / 66 - 10)
    held = _rows(heldout)["BioMiner"]
    assert held.accuracy.value == pytest.approx(100 * 10 / 35)
    assert held.accuracy.low <= held.accuracy.value <= held.accuracy.high


def test_accuracy_is_tp_over_tp_fp_and_fn_with_a_paper_bootstrap_interval(tmp_path):
    _paper_run(tmp_path, "openchemie__biovista", {"heldout": BIOVISTA_HELDOUT})
    panel = figures.systems(tmp_path, "heldout")[1]
    assert panel.title == "BioVista · 125 papers, 875 molecules"
    accuracy = _rows(panel)["OpenChemIE"].accuracy
    # every paper has the same counts, so every resample gives the same accuracy
    assert (accuracy.value, accuracy.low, accuracy.high) == pytest.approx((100 * 4 / 9,) * 3)


def test_missing_runs_and_groups_are_pending(tmp_path):
    _paper_run(tmp_path, "molscout_both__internal", {"dev": INTERNAL_DEV, "all": INTERNAL_DEV})
    dev_internal, dev_biovista = figures.systems(tmp_path, "dev")
    assert [name for name, _ in dev_internal.rows] == list(figures.SYSTEMS)
    assert dev_internal.pending == ["BioMiner", "DECIMER.ai", "OpenChemIE"]
    assert (dev_biovista.title, dev_biovista.pending) == ("BioVista", list(figures.SYSTEMS))
    heldout_internal = figures.systems(tmp_path, "heldout")[0]
    assert heldout_internal.pending == list(figures.SYSTEMS)  # the dev run's `all` group is never read


def test_molscout_reads_the_both_run_on_dev_and_its_own_run_on_heldout(tmp_path):
    _paper_run(tmp_path, "molscout_both__biovista", {"dev": BIOVISTA_DEV})
    _paper_run(tmp_path, "molscout__biovista", {"dev": {"dev_0": (1, 0, 4)}, "heldout": BIOVISTA_HELDOUT})
    assert _rows(figures.systems(tmp_path, "dev")[1])["MolScout"].accuracy.value == pytest.approx(50.0)
    assert _rows(figures.systems(tmp_path, "heldout")[1])["MolScout"].accuracy.value == pytest.approx(100 * 4 / 9)


def test_a_group_without_micro_scores_is_an_error(tmp_path):
    (tmp_path / "decimer_ai__internal").mkdir()
    report = {"scores": {"groups": {"test": {"papers": ["2"], "molecules": 30}}}}
    (tmp_path / "decimer_ai__internal" / "scores.json").write_text(json.dumps(report))
    with pytest.raises(figures.FigureError, match="decimer_ai__internal"):
        figures.systems(tmp_path, "heldout")


def test_per_paper_counts_that_miss_the_group_total_are_an_error(tmp_path):
    _paper_run(tmp_path, "biominer__internal", {"test": INTERNAL_TEST})
    path = tmp_path / "biominer__internal" / "scores.json"
    report = json.loads(path.read_text())
    report["scores"]["papers"]["2"]["tp"] = 6
    path.write_text(json.dumps(report))
    with pytest.raises(figures.FigureError, match="do not add up"):
        figures.systems(tmp_path, "heldout")


def test_runs_scored_on_different_papers_are_an_error(tmp_path):
    _paper_run(tmp_path, "biominer__internal", {"dev": INTERNAL_DEV})
    _paper_run(tmp_path, "openchemie__internal", {"dev": {"1": (1, 1, 1), "16": (1, 1, 1)}})
    with pytest.raises(figures.FigureError, match="different papers"):
        figures.systems(tmp_path, "dev")


def test_ladder_reads_each_step_and_ends_at_molscout(tmp_path):
    _paper_run(tmp_path, "molscout_l0_drawings__internal", {"dev": {"1": (5, 0, 15), "16": (5, 0, 10), "19": (0, 3, 9)}})
    _paper_run(tmp_path, "molscout_rules__internal", {"dev": INTERNAL_DEV})
    _paper_run(tmp_path, "molscout_both__internal", {"dev": INTERNAL_DEV})
    internal, biovista = figures.ladder(tmp_path)
    assert [name for name, _ in internal.rows] == [*figures.STEPS, "rules", "MolScout"]
    rows = _rows(internal)
    assert rows["l0_drawings"].value == pytest.approx(100 * 20 / 57)
    assert rows["MolScout"].value == pytest.approx(100 * 40 / 66)
    assert rows["l1_labels"] is None
    assert biovista.pending == [*figures.STEPS, "rules", "MolScout"]


def test_main_writes_light_and_dark_png_at_twice_the_size(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    results = tmp_path / "results"
    _crop_runs(results)
    _paper_run(results, "molscout_both__internal", {"dev": INTERNAL_DEV})
    _paper_run(results, "biominer__biovista", {"dev": BIOVISTA_DEV, "heldout": BIOVISTA_HELDOUT})
    out = tmp_path / "out"
    assert figures.main(["--results", str(results), "--out", str(out)]) == 0
    assert sorted(path.name for path in out.iterdir()) == sorted(
        f"{name}-{theme.name}.png" for name in figures.CHARTS for theme in figures.THEMES)
    for name, (width, height) in figures.CHARTS.items():
        for theme in figures.THEMES:
            size, chunks = _png(out / f"{name}-{theme.name}.png")
            assert size == (round(width * figures.DPI), round(height * figures.DPI))
            assert not chunks & figures.DROPPED_CHUNKS
    assert "systems_heldout: Internal: pending MolScout, BioMiner" in capsys.readouterr().err


def test_main_draws_only_the_named_chart(tmp_path):
    pytest.importorskip("matplotlib")
    (tmp_path / "results").mkdir()
    out = tmp_path / "out"
    assert figures.main(["--results", str(tmp_path / "results"), "--out", str(out), "--chart", "ladder"]) == 0
    assert sorted(path.name for path in out.iterdir()) == ["ladder-dark.png", "ladder-light.png"]


def test_main_reports_missing_results(tmp_path, capsys):
    assert figures.main(["--results", str(tmp_path / "none"), "--out", str(tmp_path / "out")]) == 1
    assert "figures: error: no results folder" in capsys.readouterr().err
    (tmp_path / "empty").mkdir()
    assert figures.main(["--results", str(tmp_path / "empty"), "--out", str(tmp_path / "out")]) == 1
    assert "molscribe__uspto" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
