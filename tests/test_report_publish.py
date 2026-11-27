"""Loading result folders, refusing inconsistent ones, the W&B payload, and an offline end-to-end upload.

Result folders are built on the hand-made crop references (c1 to c7, c5 unreadable) and scored
by the real scorer, as the harness would.
"""

import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest
from PIL import Image

from molscout.predictions import Prediction, write_predictions
from molscout.report.analysis import (
    InconsistentResults,
    ResultsError,
    check_consistency,
    check_references,
    load_run,
    load_runs,
)
from molscout.runs import score_run
from molscout.scoring import write_scores

REPO = Path(__file__).resolve().parents[1]
REFERENCES = Path(__file__).parent / "fixtures" / "handmade" / "crops_references"
CROPS = ("c1", "c2", "c3", "c4", "c5", "c6", "c7")
COMMIT = "4cbaee3e2d4abdb94afaac495b0922ecf3c159fc"
# MolScribe: 3 of 6 right (c2 has the wrong enantiomer, c6 crashed, c7 is wrong); MolNexTR: 4 of 6.
MOLSCRIBE = {
    "c1": "OCC",
    "c2": "C[C@@H](N)C(=O)O",
    "c3": "C1=CC=CC=C1",
    "c4": "*c1ccccc1",
    "c5": "C",
    "c6": "",
    "c7": "C1CC",
}
MOLNEXTR = {"c1": "CCO", "c2": "C[C@H](N)C(=O)O", "c3": "c1ccccc1", "c4": "C(", "c5": "", "c6": "CC(=O)O", "c7": "CCO"}
RESOURCES = {
    "tool_peak_rss_mib": 3072.0,
    "tool_cpu_seconds": 41.5,
    "gpu": {
        "name": "NVIDIA H200",
        "peak_memory_mib": 6144.0,
        "mean_utilization_pct": 37.5,
        "samples": 12,
        "interval_seconds": 1.0,
    },
}


@pytest.fixture
def repo(tmp_path):
    """A repository root with the USPTO references and one tiny image per crop under data/raw/uspto."""
    root = tmp_path / "repo"
    shutil.copytree(REFERENCES, root / "data" / "raw" / "uspto" / "USPTO_mol_ref")
    images = root / "data" / "raw" / "uspto" / "USPTO"
    images.mkdir(parents=True)
    for crop in CROPS:
        Image.new("RGB", (8, 6), "white").save(images / f"{crop}.png")
    return root


def make_result(repo, tool, answers, *, commit=COMMIT, dirty=False, crashed=(), resources=RESOURCES):
    """A `<tool>__uspto` result folder as the harness writes it, scored by the real scorer."""
    folder = repo / "benchmarks" / "results" / f"{tool}__uspto"
    folder.mkdir(parents=True)
    label = f"{tool.capitalize()} 1.0 (test)"
    rows = [
        Prediction("uspto", crop, smiles, None, None, None, label, 0.2 + n / 10)
        for n, (crop, smiles) in enumerate(answers.items())
    ]
    write_predictions(folder / "predictions.csv", rows)
    write_scores(
        folder / "scores.json", score_run(folder / "predictions.csv", references=repo / "data/raw/uspto/USPTO_mol_ref")
    )
    meta = {
        "run": f"{tool}__uspto",
        "tool": {
            "tool": tool,
            "name": tool.capitalize(),
            "version": "1.0 (test)",
            "checkpoints": [{"path": f"/store/models/{tool}/model.pth", "sha256": "c" * 64}],
        },
        "dataset": "uspto",
        "items": len(answers),
        "tool_errors": len(crashed),
        "git": {"commit": commit, "dirty": dirty, "dirty_paths": ["src/molscout/x.py"] if dirty else []},
        "environment": {"sha256": "e" * 64},
        "hardware": {
            "cpu_model": "Test CPU",
            "gpus": [{"name": "NVIDIA H200", "memory": "143771 MiB", "driver": "570"}],
        },
        "slurm": {"job": "1", "partition": "gpu-short"},
        **({"resources": resources} if resources is not None else {}),
    }
    (folder / "meta.json").write_text(json.dumps(meta))
    (folder / "config.yaml").write_text(f"tool: {tool}\ndataset: uspto\n")
    errors = {
        "images": len(answers),
        "failed": len(crashed),
        "errors": {crop: "RuntimeError: boom" for crop in crashed},
    }
    (folder / "errors.json").write_text(json.dumps(errors))
    return folder


def two_tools(repo, **molnextr):
    make_result(repo, "molscribe", MOLSCRIBE, crashed=("c6",))
    make_result(repo, "molnextr", MOLNEXTR, **molnextr)
    return repo / "benchmarks" / "results"


def test_load_runs_reads_crop_folders_in_tool_order_and_skips_the_rest(repo):
    results = two_tools(repo)
    for leftover in (".staging-molscribe__uspto-1", "biominer__internal"):
        (results / leftover).mkdir()
        (results / leftover / "scores.json").write_text("{}")
    runs = load_runs(results)
    assert [run.name for run in runs] == ["molscribe__uspto", "molnextr__uspto"]
    assert runs[0].errors == {"c6": "RuntimeError: boom"}
    assert check_consistency(runs) == [] and check_references(runs, repo) == []


def test_load_run_refuses_incomplete_or_mismatched_folders(repo):
    folder = make_result(repo, "molscribe", MOLSCRIBE)
    (folder / "config.yaml").unlink()
    with pytest.raises(ResultsError, match="missing config.yaml"):
        load_run(folder)
    (folder / "config.yaml").write_text("tool: molscribe\n")
    with (folder / "predictions.csv").open("a") as handle:
        handle.write('uspto,c8,C,,,,"Molscribe 1.0 (test)",0.1\n')
    with pytest.raises(ResultsError, match="another predictions.csv"):
        load_run(folder)
    renamed = folder.with_name("molvec__uspto")
    folder.rename(renamed)
    with pytest.raises(ResultsError, match="the folder is molvec on uspto"):
        load_run(renamed)


def test_consistency_refuses_mixed_commits_uncommitted_code_and_different_references(repo):
    results = two_tools(repo, commit="0" * 40, dirty=True)
    scores = results / "molnextr__uspto" / "scores.json"
    report = json.loads(scores.read_text())
    report["inputs"]["references"]["sha256"] = "f" * 64
    scores.write_text(json.dumps(report))
    problems = check_consistency(load_runs(results))
    assert problems == [
        "molnextr__uspto ran with uncommitted changes: src/molscout/x.py",
        f"the runs come from 2 commits: {COMMIT[:12]} (molscribe__uspto); 000000000000 (molnextr__uspto)",
        f"uspto: the tools were scored against 2 reference sets: "
        f"{report_sha(results, 'molscribe')[:12]} (molscribe); ffffffffffff (molnextr)",
    ]


def test_consistency_refuses_runs_that_do_not_record_their_commit(repo):
    make_result(repo, "molscribe", MOLSCRIBE, commit=None)
    meta_path = repo / "benchmarks/results/molscribe__uspto/meta.json"
    meta = json.loads(meta_path.read_text())
    meta["git"]["dirty"] = None
    meta_path.write_text(json.dumps(meta))
    assert check_consistency(load_runs(repo / "benchmarks" / "results")) == [
        "molscribe__uspto does not record whether its code was committed",
        "no git commit recorded for molscribe__uspto",
    ]


def test_references_that_changed_since_scoring_are_reported(repo):
    results = two_tools(repo)
    (repo / "data/raw/uspto/USPTO_mol_ref/c7.mol").write_text((REFERENCES / "c1.MOL").read_text())
    [problem] = check_references(load_runs(results), repo)
    assert problem.startswith("uspto: ") and problem.endswith(
        "than molscribe__uspto, molnextr__uspto was scored against"
    )


@pytest.fixture
def publisher():
    """molscout.report.wandb_publish; skips where the report extra (wandb, plotly) is not installed."""
    pytest.importorskip("wandb")
    pytest.importorskip("plotly")
    from molscout.report import wandb_publish

    return wandb_publish


def test_payload_for_one_folder_names_the_run_and_carries_config_and_summary(repo, publisher):
    two_tools(repo)
    benchmark = publisher.prepare(repo / "benchmarks" / "results", repo)
    run = benchmark.runs[0]
    assert publisher.run_id(run) == f"molscribe-uspto-{run.predictions_sha256[:10]}"
    assert publisher.run_config(run) == {
        "tool": "molscribe",
        "name": "Molscribe",
        "version": "1.0 (test)",
        "dataset": "uspto",
        "git_commit": COMMIT,
        "checkpoints": [{"file": "model.pth", "sha256": "c" * 64}],
        "environment_sha256": "e" * 64,
        "device": "NVIDIA H200",
        "slurm_partition": "gpu-short",
    }
    summary = benchmark.metrics[("molscribe", "uspto")]
    assert summary["accuracy/stereo_aware"] == 0.5 and summary["items/scored"] == 6
    assert (summary["outcome/correct"], summary["outcome/stereo_only"], summary["outcome/crashed"]) == (3, 1, 1)
    assert summary["resources/gpu_peak_memory_gib"] == 6.0 and summary["items/crashed"] == 1
    assert publisher.summary_run_id(benchmark.runs) == publisher.summary_run_id(benchmark.runs[::-1])


def test_tables_hold_every_scored_crop_and_a_seeded_failure_sample(repo, publisher):
    two_tools(repo)
    benchmark = publisher.prepare(repo / "benchmarks" / "results", repo)
    columns, rows = publisher.prediction_rows(benchmark)
    assert columns == [
        "dataset",
        "item_id",
        "reference",
        "molscribe/smiles",
        "molscribe/outcome",
        "molnextr/smiles",
        "molnextr/outcome",
    ]
    assert [row[1] for row in rows] == ["c1", "c2", "c3", "c4", "c6", "c7"]
    assert rows[4][3:] == ["", "crashed", "CC(=O)O", "correct"]
    columns, rows = publisher.failure_rows(benchmark, 3, repo)
    assert columns[:4] == ["dataset", "item_id", "image", "reference"] and len(rows) == 3
    assert all(row[2] == repo / "data/raw/uspto/USPTO" / f"{row[1]}.png" for row in rows)
    columns, rows = publisher.leaderboard_rows(benchmark)
    assert columns[:3] == ["tool", "dataset", "accuracy/stereo_aware"] and [row[0] for row in rows] == [
        "molscribe",
        "molnextr",
    ]


def test_figure_panels_compare_tools_only_when_there_are_two(repo, publisher):
    two_tools(repo)
    both = publisher.figure_panels(publisher.prepare(repo / "benchmarks" / "results", repo), repo)
    assert list(both) == panel_keys(publisher, NO_WILD_FIGURES)
    shutil.rmtree(repo / "benchmarks" / "results" / "molnextr__uspto")
    alone = publisher.figure_panels(publisher.prepare(repo / "benchmarks" / "results", repo), repo)
    assert panel_keys(publisher, ["headroom"])[0] not in alone and len(alone) == len(both) - 1


def test_panel_rows_are_the_nearest_whole_rows_once_the_panel_chrome_is_added(publisher):
    go = pytest.importorskip("plotly.graph_objects")
    heights = (300, 320, 380, 400)  # +150 px of chrome: 9.6, 10.0, 11.3 and 11.7 rows of 47 px
    assert [publisher.panel_rows(go.Figure(layout={"height": height})) for height in heights] == [10, 10, 11, 12]
    assert publisher.panel_rows(go.Figure(layout={"height": 10})) == 4  # never less than the minimum


def test_a_figure_without_a_height_cannot_be_sized_for_a_panel(publisher):
    go = pytest.importorskip("plotly.graph_objects")
    with pytest.raises(ValueError, match="layout.height"):
        publisher.panel_rows(go.Figure())


NO_WILD_FIGURES = ["accuracy_by_dataset", "stereo_gain", "error_mix", "headroom", "accuracy_vs_time"]


def panel_keys(publisher, names):
    """The W&B keys of the named figures, under the one prefix the publisher uses."""
    return [f"{publisher.FIGURE_PREFIX}{name}" for name in names]


def sized_figure(go, height=400):
    """A figure the way figures.py builds one: a container-anchored title and legend, and a fixed height."""
    return go.Figure(
        layout={
            "height": height,
            "margin": {"t": 72, "b": 40},
            "title": {"text": "T", "y": 1 - 8 / height, "yref": "container", "yanchor": "top"},
            "legend": {"y": 1 - 40 / height, "yref": "container", "yanchor": "top"},
        }
    )


def embedded(html, name):
    """The figure the page embeds under `name` ("light" or "dark"), parsed back from the page."""
    match = re.search(rf"^\s*{name}: (.*),$", html, re.MULTILINE)
    assert match, name
    return json.loads(match.group(1).replace("<\\/", "</"))


def test_panel_html_holds_both_figures_each_sized_to_whole_rows(publisher):
    go = pytest.importorskip("plotly.graph_objects")
    light, dark = sized_figure(go, 400), sized_figure(go, 400).update_layout(title_text="dark")
    html = publisher.panel_html(light, dark)
    height = publisher.PANEL_ROW_PX * publisher.panel_rows(light) - publisher.HTML_PANEL_CHROME_PX
    assert embedded(html, "light")["layout"]["height"] == embedded(html, "dark")["layout"]["height"] == height
    assert embedded(html, "dark")["layout"]["title"]["text"] == "dark"
    assert light.layout.height == 400  # the inputs are left as they were


def test_panel_html_keeps_the_title_and_legend_the_same_pixels_from_the_top(publisher):
    go = pytest.importorskip("plotly.graph_objects")
    layout = embedded(publisher.panel_html(sized_figure(go, 400), sized_figure(go, 400)), "light")["layout"]
    for part, pixels in (("title", 8), ("legend", 40)):
        assert (1 - layout[part]["y"]) * layout["height"] == pytest.approx(pixels)


def test_panel_html_is_a_small_page_that_follows_the_viewers_theme(publisher):
    go = pytest.importorskip("plotly.graph_objects")
    from plotly.offline import get_plotlyjs_version

    html = publisher.panel_html(sized_figure(go), sized_figure(go))
    assert html.startswith("<!DOCTYPE html>") and '<meta charset="utf-8">' in html
    assert '<meta name="color-scheme" content="light dark">' in html
    assert f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js" in html
    assert "cdn.plot.ly" in html.split("onerror=")[1].split(">")[0]  # the message names what failed to load
    assert "prefers-color-scheme: dark" in html and "addEventListener('change'" in html
    assert "addEventListener('resize'" in html and "window.innerHeight" in html and "Plotly.react" in html
    assert "modeBarButtonsToRemove: ['toImage']" in html and "displaylogo: false" in html
    assert "background: transparent" in html and "overflow: hidden" in html


def test_panel_html_cannot_be_closed_early_by_a_figure_title(publisher):
    go = pytest.importorskip("plotly.graph_objects")
    figure = sized_figure(go).update_layout(title_text="</script><script>alert(1)</script><!--")
    html = publisher.panel_html(figure, figure)
    assert html.count("</script>") == 2  # the loader and the page's own script
    assert "<!--" not in html.split("<body>")[1]
    assert embedded(html, "light")["layout"]["title"]["text"] == "</script><script>alert(1)</script><!--"


def test_panel_html_embeds_a_title_that_looks_like_a_placeholder_unchanged(publisher):
    go = pytest.importorskip("plotly.graph_objects")
    title = "LOGGED_HEIGHT DARK_FIGURE LIGHT_FIGURE MUTED_DARK MUTED_LIGHT PLOTLY_URL"
    light = sized_figure(go).update_layout(title_text=title)
    dark = sized_figure(go).update_layout(title_text=title)
    html = publisher.panel_html(light, dark)
    assert html.count(title) == 2
    assert (
        embedded(html, "light")["layout"]["title"]["text"] == embedded(html, "dark")["layout"]["title"]["text"] == title
    )


def test_panel_html_refuses_figures_of_different_heights(publisher):
    go = pytest.importorskip("plotly.graph_objects")
    with pytest.raises(ValueError, match="height"):
        publisher.panel_html(sized_figure(go, 400), sized_figure(go, 380))


def test_panel_html_skips_drawing_in_a_frame_with_no_height(publisher):
    go = pytest.importorskip("plotly.graph_objects")
    html = publisher.panel_html(sized_figure(go), sized_figure(go))
    assert "if (!window.Plotly || !window.innerHeight) return;" in html


@pytest.mark.parametrize("labels", range(6, 15))
def test_two_line_heatmap_labels_fit_the_row_pitch_at_the_settled_frame_height(publisher, labels):
    figures = publisher.figures
    shares = {("molscribe", f"Label {n} " + "word " * 12): figures.Share(1, 100) for n in range(labels)}
    figure = figures.hardcase_heatmap(shares)
    frame = publisher.PANEL_ROW_PX * publisher.panel_rows(figure) - publisher.HTML_PANEL_CHROME_PX
    pitch = (frame - figure.layout.margin.t - figure.layout.margin.b) / labels
    assert pitch >= 2 * 12 * 1.2  # two lines of 12 px text at the usual 1.2 line height


def test_figures_are_logged_under_a_prefix_never_used_before(publisher):
    assert publisher.FIGURE_PREFIX == "viz/"
    source = Path(publisher.__file__).read_text()
    used = source.split("Used so far:")[1].split("\n")[0]
    assert "fig/" in used and "viz/" not in used


def test_panel_html_draws_at_no_more_than_the_logged_height(publisher):
    go = pytest.importorskip("plotly.graph_objects")
    html = publisher.panel_html(sized_figure(go), sized_figure(go))
    assert "const height = Math.min(window.innerHeight, loggedHeight);" in html
    assert "const height = window.innerHeight;" not in html
    assert "if (!window.Plotly || !window.innerHeight) return;" in html  # the zero-height guard stays


NODE_HARNESS = """
const vm = require('vm');
const [html, inner] = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const script = html.split('<script>')[1].split('</script>')[0];
const calls = [], listeners = {}, rootListeners = {};
const plot = {style: {}};
const window = {
  innerHeight: inner,
  matchMedia: () => ({matches: false, addEventListener() {}}),
  addEventListener: (name, fn) => { listeners[name] = fn; },
  Plotly: {react: (el, data, layout) => { plot.data = data; calls.push(['react', layout, el.style.height]); },
           Fx: {unhover: (el) => calls.push(['unhover', el === plot])}},
};
const document = {
  getElementById: () => plot,
  documentElement: {addEventListener: (name, fn) => { rootListeners[name] = fn; }},
};
vm.runInNewContext(script, {window, document, Plotly: window.Plotly, setTimeout, clearTimeout});
if (process.argv[1] === 'leave') rootListeners.mouseleave();
console.log(JSON.stringify(calls));
"""


def run_page(html, inner, *args):
    """What the page's script calls on Plotly, run in node against a stub window `inner` px tall."""
    import subprocess

    done = subprocess.run(
        ["node", "-e", NODE_HARNESS, *args],
        input=json.dumps([html, inner]),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(done.stdout)


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run the page's script")


@needs_node
@pytest.mark.parametrize("inner", [3, 120, 367, 414, 442])
def test_the_page_draws_at_the_smaller_of_the_frame_and_the_logged_height(publisher, inner):
    go = pytest.importorskip("plotly.graph_objects")
    html = publisher.panel_html(sized_figure(go), sized_figure(go))  # logged at 12 rows: 414 px
    [(_, layout, container)] = run_page(html, inner)
    assert layout["height"] == min(inner, 414)
    # Plotly's responsive resize refits the figure to its container, so the container is sized to match.
    assert container == f"{min(inner, 414)}px"
    for part, pixels in (("title", 8), ("legend", 40)):
        assert (1 - layout[part]["y"]) * layout["height"] == pytest.approx(pixels)


@needs_node
def test_the_page_draws_nothing_in_a_frame_with_no_height(publisher):
    go = pytest.importorskip("plotly.graph_objects")
    assert run_page(publisher.panel_html(sized_figure(go), sized_figure(go)), 0) == []


@needs_node
def test_the_page_unhovers_when_the_pointer_leaves_the_document(publisher):
    go = pytest.importorskip("plotly.graph_objects")
    calls = run_page(publisher.panel_html(sized_figure(go), sized_figure(go)), 414, "leave")
    assert [call[0] for call in calls] == ["react", "unhover"] and calls[1][1] is True


def test_panel_html_clears_the_hover_when_the_pointer_leaves_the_frame(publisher):
    go = pytest.importorskip("plotly.graph_objects")
    html = publisher.panel_html(sized_figure(go), sized_figure(go))
    assert "document.documentElement.addEventListener('mouseleave'" in html
    assert "Plotly.Fx.unhover(plot)" in html  # exported by the plotly.js the page pins (4.1.1)


def test_publish_logs_one_html_page_per_figure(repo, publisher, monkeypatch):
    from unittest.mock import MagicMock

    import wandb

    benchmark = publisher.prepare(two_tools(repo), repo)
    run = MagicMock()
    monkeypatch.setattr(publisher.wandb, "init", lambda **_: run)
    publisher.publish(benchmark, entity="e", project="p", repo_root=repo, images_per_dataset=0, echo=lambda _: None)
    [logged] = [call.args[0] for call in run.__enter__.return_value.log.call_args_list]
    keys = panel_keys(publisher, NO_WILD_FIGURES)
    assert [key for key in logged if key in keys] == keys
    assert all(isinstance(logged[key], wandb.Html) for key in keys)
    light = publisher.figure_panels(benchmark, repo)
    dark = publisher.figure_panels(benchmark, repo, theme=publisher.figures.DARK)
    for key in keys:
        page = Path(logged[key]._path).read_text()  # what W&B uploads, unchanged: no stylesheet injected
        assert page == publisher.panel_html(light[key], dark[key])
        assert embedded(page, "dark")["layout"]["paper_bgcolor"] == "rgba(0,0,0,0)"
        assert "bdata" not in page  # no typed-array encoding, which not every Plotly.js renders
    assert not any(isinstance(value, wandb.Plotly) for value in logged.values())


def test_figure_panels_follow_the_theme_they_are_given(repo, publisher):
    benchmark = publisher.prepare(two_tools(repo), repo)
    bars = publisher.figure_panels(benchmark, repo, theme=publisher.figures.DARK)[
        panel_keys(publisher, ["error_mix"])[0]
    ]
    assert bars.data[0].marker.line.color == publisher.figures.DARK.surface


def test_upload_refuses_an_incomplete_meta_before_touching_wandb(repo, publisher, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb-dir"))
    results = two_tools(repo)
    meta_path = results / "molnextr__uspto" / "meta.json"
    meta = json.loads(meta_path.read_text())
    del meta["slurm"]
    meta_path.write_text(json.dumps(meta))
    assert upload(results, repo, "--offline", "--allow-inconsistent") == 1
    assert "molnextr__uspto: meta.json lacks 'slurm'" in capsys.readouterr().err
    assert not (tmp_path / "wandb-dir").exists()


def test_upload_refuses_inconsistent_results_before_touching_wandb(repo, publisher, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb-dir"))
    results = two_tools(repo, commit="0" * 40)
    assert upload(results, repo, "--offline") == 1
    assert "2 commits" in capsys.readouterr().err
    assert not (tmp_path / "wandb-dir").exists()


def test_offline_upload_logs_one_run_per_folder_and_one_summary(repo, publisher, tmp_path, monkeypatch):
    wandb_dir = tmp_path / "wandb-dir"
    wandb_dir.mkdir()
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(wandb_dir))
    results = two_tools(repo)
    assert upload(results, repo, "--offline", "--images-per-dataset", "2") == 0
    offline = wandb_dir / "wandb"
    assert len(list(offline.glob("offline-run-*"))) == 3
    [eval_dir] = offline.glob("offline-run-*-molscribe-uspto-*")
    logged = logged_values(eval_dir)
    assert logged["summary"]["accuracy/stereo_aware"] == 0.5 and logged["summary"]["outcome/crashed"] == 1
    assert {"accuracy/stereo_aware_ci_low", "speed/s_per_crop_mean", "resources/peak_rss_gib", "items/excluded"} <= set(
        logged["summary"]
    )
    assert logged["config"]["tool"] == "molscribe" and logged["config"]["device"] == "NVIDIA H200"
    assert logged["history"] == {}  # an eval run has a summary, not panels
    assert logged["run"] == {"tags": ["molscribe", "uspto"], "group": "structure-readers", "job_type": "eval"}
    [analysis_dir] = offline.glob("offline-run-*-summary-*")
    analysis = logged_values(analysis_dir)
    assert analysis["run"] == {"tags": [], "group": "", "job_type": "analysis"}  # nothing carried over from eval runs
    history = analysis["history"]
    plots = {key.removesuffix("/_type") for key, value in history.items() if value == "html-file"}
    assert plots == set(panel_keys(publisher, NO_WILD_FIGURES))  # no MolRecBench-Wild, so no journal-crop figures
    assert {key.removesuffix("/_type") for key, value in history.items() if value == "table-file"} == {
        "leaderboard",
        "predictions",
        "failures",
    }


def logged_values(run_dir):
    """Summary, config and history values from an offline run's transaction log (LevelDB-style records)."""
    from wandb.proto import wandb_internal_pb2

    data = next(run_dir.glob("*.wandb")).read_bytes()
    assert data[:4] == b":W&B"
    values = {"summary": {}, "config": {}, "history": {}}
    position, block, pending = 7, 32768, b""
    while position + 7 <= len(data):
        remaining = block - position % block
        length, kind = int.from_bytes(data[position + 4 : position + 6], "little"), data[position + 6]
        if remaining < 7 or (length == 0 and kind == 0):
            position += remaining
            continue
        pending += data[position + 7 : position + 7 + length]
        position += 7 + length
        if kind in (1, 4):  # a full record, or the last chunk of one
            record = wandb_internal_pb2.Record()
            record.ParseFromString(pending)
            pending = b""
            kind_of = record.WhichOneof("record_type")
            if kind_of == "run":
                run = record.run
                values["run"] = {"tags": sorted(run.tags), "group": run.run_group, "job_type": run.job_type}
                continue
            if kind_of == "summary":
                items = record.summary.update
            elif kind_of == "config":
                items = record.config.update
            elif kind_of == "history":
                items = [item for item in record.history.item if not item.key.startswith("_")]
            else:
                continue
            for item in items:
                values[kind_of][item.key or "/".join(item.nested_key)] = json.loads(item.value_json)
    return values


def report_sha(results, tool):
    return json.loads((results / f"{tool}__uspto" / "scores.json").read_text())["inputs"]["references"]["sha256"]


def upload(results, repo, *args):
    """Run scripts/wandb_upload.py's main in-process."""
    spec = importlib.util.spec_from_file_location("wandb_upload", REPO / "scripts" / "wandb_upload.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = [
        "--results",
        str(results),
        "--repo-root",
        str(repo),
        "--entity",
        "test",
        "--project",
        "molscout-test",
        *args,
    ]
    return module.main(argv)


def test_inconsistent_results_list_every_problem():
    error = InconsistentResults(["a", "b"])
    assert error.problems == ("a", "b") and str(error).endswith("  - a\n  - b")
    assert isinstance(error, ResultsError)
