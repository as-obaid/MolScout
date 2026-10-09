"""Build every block of the MolScout baseline W&B report and save it in place.

    python scripts/report/raw_spec.py BACKUP.json          # the live spec, read with GraphQL view_report
    python scripts/report/diff_live.py BACKUP.json         # what a save would change
    python scripts/report/build_report.py [--dry-run] [--stage type1|full]

Needs the report and report-builder extras (pip install -e ".[report,report-builder]") and W&B credentials.
`--stage type1` builds the structure-reader report alone, with every grid narrowed to its own summary run;
`full` (the default) adds the complete-system section. Report figures are wandb.Html pages logged by
scripts/wandb_upload.py; each panel is as tall as its figure (figure_rows.json, keyed by figure key prefix), so
no panel scrolls. Markdown tables are styled inline (table_rows.json holds their measured heights in grid rows).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import markdown
import wandb_workspaces.reports.v2 as wr
from wandb_workspaces.reports.v2 import gql
from wandb_workspaces.reports.v2.interface import _get_api, execute_graphql

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import report_tables as tables  # noqa: E402
import spec_tools  # noqa: E402

REPORT_ID = "VmlldzoxODA1ODcwOQ=="
URL = (
    "https://wandb.ai/muhammadobaidullah98-northeastern-university/MolScout/reports/"
    "MolScout-baseline-six-structure-readers-on-five-datasets--VmlldzoxODA1ODcwOQ"
)
ENTITY, PROJECT = "muhammadobaidullah98-northeastern-university", "MolScout"
REPO = "https://github.com/as-obaid/MolScout"
DOCS = f"{REPO}/blob/benchmark/docs"
SUMMARY_RUN = f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/summary-5cb0bf0bd0"
# The summary runs the grids read from, by W&B display name (wandb_publish.SUMMARY_NAME, wandb_papers.SUMMARY_NAME).
READERS_SUMMARY = "summary"
SYSTEMS_SUMMARY = "summary-complete-systems"
SUMMARY_NAMES = frozenset({READERS_SUMMARY, SYSTEMS_SUMMARY})

STAGES = ("type1", "full")
STAGE = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else "full"
FULL = STAGE == "full"  # read by the block builders; set_stage changes it


def set_stage(stage: str) -> None:
    global STAGE, FULL
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    STAGE, FULL = stage, stage == "full"


set_stage(STAGE)

TITLE = "Baseline Benchmark: Extracting SMILES from Scientific Papers"


def description() -> str:
    if FULL:
        return (
            "Six structure readers scored on 15,241 molecule crops from five public datasets, and three complete "
            "systems on 151 whole papers."
        )
    return (
        "Six structure readers scored on 15,241 molecule crops from five public datasets; "
        "results for three whole-PDF systems will follow."
    )

# Figure key prefixes: wandb_publish.FIGURE_PREFIX (structure readers) and wandb_papers.PAPER_FIGURE_PREFIX.
READERS_PREFIX, SYSTEMS_PREFIX = "viz/", "sys/"
FIGURE_ROWS = json.loads((HERE / "figure_rows.json").read_text(encoding="utf-8"))
FIGURES = [
    "accuracy_by_dataset",
    "stereo_gain",
    "journal_crops",
    "hardcases",
    "error_mix",
    "headroom",
    "accuracy_vs_time",
    "biovista_views",
    "recall_by_paper",
    "internal_views",
    "f1_vs_time",
]
SYSTEM_FIGURES = frozenset(FIGURES[7:])
TABLE_ROWS = json.loads((HERE / "table_rows.json").read_text(encoding="utf-8"))
TABLES: dict[str, str] = {}  # name -> panel HTML, for measuring (--dump-tables)


def compact_html(md: str, min_height: int = 0) -> str:
    """Markdown as one line of styled HTML for a markdown panel.

    W&B markdown panels keep the renderer's newlines (pre-wrap), which stack up as blank lines above a table,
    and raw HTML tables lose the report's table style, so cells, rules and notes are styled inline. Borders use
    a translucent gray so they read in light and dark mode.
    """
    html = re.sub(r">\s+<", "><", markdown.markdown(md, extensions=["tables"])).strip()
    cell = "padding:5px 10px;border-bottom:1px solid rgba(128,128,128,.25);"
    head = "padding:5px 10px;border-bottom:1px solid rgba(128,128,128,.6);font-weight:600;"
    for tag, style in (("td", cell), ("th", head)):
        html = html.replace(f'<{tag} style="', f'<{tag} style="{style}').replace(f"<{tag}>", f'<{tag} style="{style}text-align:left;">')
    html = html.replace("<table>", '<table style="border-collapse:collapse;width:100%;font-size:15px;line-height:22px;">')
    html = re.sub(r"<h3>(.*?)</h3>", r'<div style="font-size:18px;line-height:24px;font-weight:600;margin:0 0 10px;">\1</div>', html)
    html = re.sub(r"<p>(<strong>.*?</strong>)</p>(?=<table)", r'<div style="font-size:18px;line-height:24px;font-weight:600;margin:0 0 10px;">\1</div>', html)
    html = html.replace("<p>", '<p style="font-family:\'Source Sans 3\',sans-serif;font-size:14px;line-height:20px;margin:10px 0 0;opacity:.75;">')
    # Side padding, and one height for panels that sit side by side so their titles line up.
    return f'<div style="padding:0 14px;min-height:{min_height}px;box-sizing:border-box;">{html}</div>'


def summary_runset(name: str = READERS_SUMMARY) -> wr.Runset:
    """The one summary run a grid reads its panels from, by job type and display name."""
    if not FULL and name == READERS_SUMMARY and "--unnarrowed" in sys.argv:
        return wr.Runset(entity=ENTITY, project=PROJECT, name="Summary run", filters="JobType = 'analysis'")
    title = "Summary run" if name == READERS_SUMMARY else "Complete-system summary run"
    return wr.Runset(entity=ENTITY, project=PROJECT, name=title, filters=f"JobType = 'analysis' and Name = '{name}'")


def grid(*panels, summary: str = READERS_SUMMARY) -> wr.PanelGrid:
    return wr.PanelGrid(runsets=[summary_runset(summary)], hide_run_sets=True, panels=list(panels))


def figure_rows(name: str) -> int:
    prefix = SYSTEMS_PREFIX if name in SYSTEM_FIGURES else READERS_PREFIX
    rows = FIGURE_ROWS[prefix][name]
    assert rows == int(rows), (name, rows)
    return int(rows)


def media(name: str, x: int = 0, y: int = 0, w: int = 24) -> wr.MediaBrowser:
    """One logged figure, titled by its number, exactly as tall as the figure so the panel never scrolls."""
    prefix = SYSTEMS_PREFIX if name in SYSTEM_FIGURES else READERS_PREFIX
    title = f"Figure {FIGURES.index(name) + 1}"
    return wr.MediaBrowser(title=title, media_keys=[prefix + name], layout=wr.Layout(x=x, y=y, w=w, h=figure_rows(name)))


def figure(name: str) -> wr.PanelGrid:
    return grid(media(name), summary=SYSTEMS_SUMMARY if name in SYSTEM_FIGURES else READERS_SUMMARY)


def table_panel(name: str, md: str) -> wr.PanelGrid:
    """A full-width markdown panel holding one table, styled like the Tools and Datasets panels.

    Report-body markdown tables pad cells by 1 px; the panel's inline styles give them room and rules.
    """
    html = compact_html(md)
    TABLES[name] = html
    return grid(wr.MarkdownPanel(markdown=html, layout=wr.Layout(x=0, y=0, w=24, h=TABLE_ROWS[name])))


def bullets(*items: str) -> wr.MarkdownBlock:
    """A list with bold lead-ins; UnorderedList takes no bold text."""
    return wr.MarkdownBlock(text="\n".join(f"- {item}" for item in items))


def opening() -> list:
    intro = (
        "This report sets the baseline for MolScout, a system for extracting the molecules drawn in scientific "
        "papers as SMILES. It measures existing tools of two types: structure readers, which read one cropped "
        "molecule drawing, and complete systems, which take a whole PDF and return every molecule they find. "
    )
    intro += (
        "All runs are complete: 30 structure-reader runs (6 readers on 5 public datasets, 15,241 scored crops) and "
        "6 complete-system runs (3 systems on 145 public BioVista papers and 6 private Internal papers)."
        if FULL
        else "All 30 structure-reader runs are complete (6 readers on 5 public datasets, 15,241 scored crops); "
        "complete-system results will be added when those runs finish."
    )
    return [
        wr.P(text=intro),
        # W&B's header already names the author, so the byline holds the links only.
        wr.P(
            text=[
                wr.Link(text="Code", url=REPO),
                " · ",
                wr.Link(text="Benchmark protocol", url=f"{DOCS}/Benchmarking.md"),
                " · ",
                wr.Link(text="Scoring rules", url=f"{DOCS}/dev.md#scoring-rules"),
            ]
        ),
        grid(
            wr.MarkdownPanel(markdown=compact_html(tables.TOOLS_MD, 436), layout=wr.Layout(x=0, y=0, w=12, h=10)),
            wr.MarkdownPanel(
                markdown=compact_html(tables.DATASETS_MD_FULL if FULL else tables.DATASETS_MD, 436),
                layout=wr.Layout(x=12, y=0, w=12, h=10),
            ),
        ),
    ]


def at_a_glance() -> list:
    callout = (
        "MolScribe is the most accurate structure reader overall, no reader wins every dataset, and every "
        "reader scores 15 to 48 points lower on journal figures than on the four standard sets."
    )
    if FULL:
        callout += (
            " Among complete systems, BioMiner leads on both paper sets, with about twice the F1 of DECIMER.ai and "
            "OpenChemIE on BioVista."
        )
    items = [
        "**Overall.** MolScribe leads with 83.7% stereo-aware exact match pooled over 15,241 crops and 76.0% "
        "averaged over the five datasets; OCSRGlyph is second on both (81.8% and 73.2%).",
        "**By dataset.** OCSRGlyph leads USPTO (94.0%) and CLEF (90.7%), MolScribe UOB (87.3%), MolVec JPO "
        "(66.6%) and MolGlyph MolRecBench-Wild (63.2%).",
        "**Headroom.** At least one reader is right on 91.6% of crops, so the readers fail on different drawings; "
        "a plain plurality vote over the six reaches 85.9%.",
    ]
    if FULL:
        items.append(
            "**Complete systems.** BioMiner reaches an F1 of 60.6% on BioVista and 46.8% on Internal, against 31.4% "
            "and 26.7% for DECIMER.ai and 27.1% and 22.0% for OpenChemIE. OpenChemIE is the fastest, at 8.1 s per "
            "BioVista paper against 65.6 s (DECIMER.ai) and 107.3 s (BioMiner)."
        )
    return [
        wr.H1(text="Results at a glance"),
        wr.CalloutBlock(text=callout),
        bullets(*items),
        table_panel(
            "leaderboard",
            tables.LEADERBOARD_MD
            + "\n\n"
            + "Exact match and valid output in %. Pooled weights each dataset by its scored crops, so USPTO and UOB "
            "carry 11,444 of the 15,241; macro gives the five datasets equal weight. Time per crop is the mean "
            "over all 17,925 crops each reader read, on different GPU and CPU models, so it compares readers "
            "roughly. Bold marks the best value in each column; rows follow the pooled stereo-aware score.",
        ),
    ]


def structure_readers() -> list:
    return [
        wr.HorizontalRule(),
        wr.H1(text="Structure readers (Type 1)"),
        wr.P(
            text="Each reader gets one cropped drawing and returns one SMILES. A crop counts as correct only when the "
            "canonical SMILES of the prediction equals that of the reference, stereochemistry included unless a "
            "score says stereo-stripped."
        ),
        wr.H2(text="Accuracy by dataset"),
        wr.P(
            text="The dataset moves the score more than the reader: averaged over the six readers, scores run from "
            "49.0% (MolRecBench-Wild) to 85.6% (UOB), while the readers' five-dataset averages run from 58.6% to "
            "76.0%. On UOB five of the six readers land between 85.9% and 87.3%; on USPTO they spread from 56.9% "
            "(DECIMER) to 94.0% (OCSRGlyph). JPO is the hardest of the four standard sets, and with 449 crops its "
            "intervals are wide: MolVec's lead (66.6%, 95% CI 62.1–⁠70.8%) overlaps MolScribe (59.5%, "
            "54.9–⁠63.9%)."
        ),
        figure("accuracy_by_dataset"),
        table_panel("aware", tables.AWARE_MD),
        wr.P(
            text="UOB scores are capped by its references. On 386 UOB crops (6.7%) all six readers return the same "
            "molecule, and it differs from the reference only in where a hydrogen sits (a tautomer). Each reader "
            "has 443 to 481 such mismatches on UOB and at most 2 on any other dataset, which points to the "
            "references' tautomer convention; they are scored as wrong here."
        ),
        wr.H2(text="Stereochemistry"),
        wr.P(
            text="Ignoring stereochemistry adds 2.0 to 2.6 points pooled for five readers and 5.6 for MolGlyph, which "
            "gains 8.6 on USPTO and 9.6 on MolRecBench-Wild; scored that way, MolGlyph moves from fifth to third "
            "overall (81.6%). UOB and JPO carry little stereochemistry: no reader gains 2.5 points there. Part of "
            "the MolRecBench-Wild gain is a reference artifact: its references carry no cis/trans, so any E/Z a "
            "reader states counts as a stereo error."
        ),
        figure("stereo_gain"),
        table_panel("stripped", tables.STRIPPED_MD),
        wr.H2(text="Journal figures (MolRecBench-Wild)"),
        wr.P(
            text="MolRecBench-Wild crops come from figures in 818 journal articles and carry the dataset's own "
            "hard-case labels, such as colored backgrounds, blur and special atoms. Rankings from the four "
            "standard sets do not carry over: OCSRGlyph, first on them at 89.1%, falls to fourth at 42.7%, while "
            "MolGlyph rises from fifth (78.4%) to first (63.2%). Within the dataset, accuracy falls from subset A "
            "(few visual difficulties) through B (more) to C (chemical-notation difficulties) for every reader. "
            "MolGlyph's lead comes from subset B, 58.9% against 50.9% for MolScribe; on subset C no reader "
            "passes 31%."
        ),
        figure("journal_crops"),
        wr.P(
            text="Chemical notation costs more than image quality. The best reader reaches 31.2% on crops with special "
            "atoms or ions and 40.1% on crops with charge symbols, against 70.7% on blurry crops and 85.5% on "
            "crops with no hard-case label. DECIMER falls to 7.9% when a crop also holds arrows, boxes or "
            "explanatory text, and MolVec to 7.2% on special atoms and on charge symbols."
        ),
        figure("hardcases"),
        wr.H2(text="Error types"),
        wr.P(
            text="Most wrong answers are valid SMILES for the wrong molecule, so a parse check catches few of them. "
            "Pooled, wrong structures make up 69% to 84% of each reader's errors and 12.0% (MolScribe) to 29.1% "
            "(DECIMER) of all crops, while invalid SMILES stay between 1.3% and 3.6%. They cluster on JPO and "
            "MolRecBench-Wild (up to 10.7% and 12.1% of crops) and on DECIMER for CLEF "
            "(11.4%). MolGlyph's errors lean most to stereochemistry: 23% of them have the right graph and the "
            "wrong stereo."
        ),
        figure("error_mix"),
        wr.H2(text="Agreement and voting"),
        wr.P(
            text="The readers fail on different crops. At least one of the six is right on 91.6% of crops (13,966 of "
            "15,241), against 83.7% for MolScribe alone. A plurality vote over the six, with no tuning, reaches "
            "85.9% pooled: it gains most on JPO (73.3% against MolVec's 66.6%) and loses on CLEF (85.2% against "
            "OCSRGlyph's 90.7%)."
        ),
        figure("headroom"),
        wr.P(
            text="Agreement also works as a confidence signal. When all six readers give the same answer, on 53.1% of "
            "crops, that answer is right 94.7% of the time, and 99.4% outside UOB, where most unanimous misses "
            "are the tautomer mismatches noted above. The most similar pair is MolScribe and MolNexTR, which "
            "agree on 84.6% of crops; the least similar, DECIMER and MolVec, on 61.7%."
        ),
        table_panel("agreement", tables.AGREEMENT_MD),
        wr.H2(text="Speed and resources"),
        wr.P(
            text="OCSRGlyph is the fastest reader at 0.089 s per crop, 2.7 times faster than the next (MolGlyph, "
            "0.242 s), and second in pooled accuracy. DECIMER is the slowest (0.432 s) and the heaviest, peaking "
            "at 9.6 GiB of GPU memory against 1.2 to 1.4 GiB for the other GPU readers."
        ),
        grid(
            media("accuracy_vs_time", x=0, y=0, w=12),
            wr.MarkdownPanel(
                markdown=compact_html(tables.RESOURCES_MD),
                layout=wr.Layout(x=12, y=0, w=12, h=figure_rows("accuracy_vs_time")),
            ),
        ),
        wr.H2(text="Published results"),
        wr.P(
            text="On USPTO, OCSRGlyph scores 94.0% against its reported 93.8%, and MolScribe (92.1%) and MolNexTR "
            "(86.0%) fall inside the 82.1–⁠93.8% range reported across USPTO and JPO. JPO does not reproduce: "
            "this benchmark measures 59.5% (MolScribe) and 51.2% (MolNexTR) there. The published MolRecBench-Wild "
            "scores come from an earlier snapshot of the dataset and do not carry over to the current release."
        ),
        table_panel("published", tables.PUBLISHED_MD),
    ]


def complete_systems_placeholder() -> list:
    # P takes no bold text either; a markdown block keeps the bold status.
    status = (
        "**Status: not yet run.** Results will be added to this report when the runs finish. "
        "BioMiner, DECIMER.ai and OpenChemIE each take a whole PDF and return SMILES for every molecule they find, "
        "run with released code and weights at default settings. They are scored on BioVista (500 public papers, "
        "8,735 molecules) and Internal (6 private papers, 222 molecules)."
    )
    measures = (
        "Precision, recall and F1, pooled over every molecule (micro).",
        "Precision and recall per paper, then averaged (macro).",
        "Stereo-stripped precision and recall.",
        "The share of BioVista PDFs obtained by DOI, and seconds per paper.",
    )
    return [
        wr.HorizontalRule(),
        wr.H1(text="Complete systems (Type 2)"),
        # One markdown block, so the status, the lead-in and its list keep paragraph spacing between them.
        wr.MarkdownBlock(
            text=status
            + "\n\nEach system and dataset will report:\n\n"
            + "\n".join(f"- {item}" for item in measures)
        ),
        wr.P(
            text="Predictions and ground truth are reduced to unique canonical SMILES per paper before counting. The "
            "published reference point is BioMiner's F1 of 52.8% on BioVista."
        ),
    ]


def complete_systems() -> list:
    return [
        wr.HorizontalRule(),
        wr.H1(text="Complete systems (Type 2)"),
        wr.P(
            text="Each system takes a whole PDF and returns SMILES for every molecule it finds, run with released "
            "code and weights. A paper's outputs and labels are each reduced to unique canonical SMILES before "
            "counting. BioVista is scored on the 145 of its 500 papers that have a legally obtainable PDF and only "
            "readable labels; their 2,435 labels hold 2,304 distinct molecules, as some papers label a molecule "
            "twice. Internal is scored on all 6 papers, whose 222 ground-truth rows hold 220 distinct molecules. "
            "BioMiner's published F1 of 52.8% on BioVista is not comparable with these scores: it was measured "
            "with every paper's PDF, so it covers papers and labels this benchmark leaves out."
        ),
        table_panel("systems", tables.systems_md()),
        wr.H2(text="BioVista"),
        wr.P(
            text="BioMiner leads every accuracy column on BioVista. Its F1 of 60.6% (95% CI 54.9–⁠66.3%) is "
            "about twice DECIMER.ai's 31.4% and OpenChemIE's 27.1%, whose intervals stay below 36%. Those two find "
            "about a third of the molecules (recall 34.7% and 36.3%), but most of what they return is wrong "
            "(precision 28.6% and 21.6%). Ignoring stereochemistry adds 6.5 F1 points for BioMiner and under 3 "
            "for the others. Scored on drawn structures only (142 papers, 1,264 molecules), recall rises to 74.0%, "
            "58.4% and 62.3% while precision falls to 54.4%, 27.4% and 20.9%. Leaving out the 35 submitted-version "
            "papers moves no F1 by more than 1.5 points."
        ),
        figure("biovista_views"),
        wr.H2(text="Paper by paper"),
        wr.P(
            text="Recall spreads across the whole range for every system. BioMiner's median paper recall is 66.7%: "
            "it finds every labeled molecule in 42 of the 145 papers and none in 23, 7 of which get no output at "
            "all. DECIMER.ai and OpenChemIE have medians of 31.6% and 37.5% and find nothing in 34 papers each. "
            "BioMiner's recall is at least as high as both others' on 120 papers."
        ),
        figure("recall_by_paper"),
        wr.H2(text="Internal (private)"),
        wr.P(
            text="Internal papers and molecules are private, so this section shows metrics only. On the six papers "
            "BioMiner again leads, with an F1 of 46.8% against 26.7% (DECIMER.ai) and 22.0% (OpenChemIE). Its "
            "precision holds (72.4%) but its recall falls to 34.5%, from 58.7% on BioVista. With six papers the "
            "intervals are wide and overlap: BioMiner's F1 interval runs from 35.6% to 69.4%. Every system scores "
            "lower on the three test papers than on the three development papers (BioMiner 44.9% against 48.3%)."
        ),
        figure("internal_views"),
        wr.H2(text="Speed"),
        wr.P(
            text="OpenChemIE is the fastest by far, at 8.1 s per BioVista paper, against 65.6 s for DECIMER.ai and "
            "107.3 s for BioMiner, which runs a 32-billion-parameter vision-language model. Internal papers hold "
            "more than twice as many molecules on average (37 against 16), and every system takes longer on them "
            "(12.1, 109.2 and 190.8 s). BioMiner peaks at 135.9 GiB of GPU memory, with vLLM and three model "
            "servers sharing the card, against 6.0 GiB for DECIMER.ai and 33.9 GiB for OpenChemIE."
        ),
        figure("f1_vs_time"),
        wr.P(
            text="One run crashed on one paper. OpenChemIE stopped on BioVista paper 352_6kqi when MolScribe was "
            "handed an empty batch, as no molecule crop was found; the paper stays in the score, so its 4 labels "
            "count as misses. BioMiner and DECIMER.ai return nothing for that paper either. Among the outputs, "
            "invalid SMILES make up 20.8% of DECIMER.ai's on BioVista and 15.7% of OpenChemIE's; BioMiner "
            "returns none."
        ),
    ]


def methods() -> list:
    items = [
        "**Match rule.** Predictions and references go through RDKit 2026.3.2 canonical SMILES; a crop is correct "
        "only when the strings are identical. Stereo-stripped scores first remove tetrahedral and double-bond "
        "stereochemistry; isotopes are kept.",
    ]
    if FULL:
        items.append(
            "**Complete-system counts.** Each paper's outputs and labels are reduced to unique canonical SMILES. An "
            "output matching a label of the same paper is a true positive, any other output a false positive "
            "(invalid SMILES included), and each unmatched label a false negative. Micro scores pool the counts over "
            "all papers; macro scores average each paper's precision and recall, a paper with no output scoring 0."
        )
    items += [
        "**No answer is a wrong answer.** Empty, unparsable or crashed output scores as wrong and lowers the "
        "valid-output rate. R‑group labels must match: a drawn R1 is `[1*]`, so a bare `*` is wrong (70 of "
        "the 977 scored CLEF references carry numbered R‑groups).",
        "**Scored crops.** References RDKit cannot read are left out: 15 USPTO, 1 JPO and 15 CLEF crops. "
        "MolRecBench-Wild references are molecular graphs, converted as its official SMILES track does (commit "
        "500da87): 2,392 of its 5,024 crops convert, and 2,371 are scored after dropping 21 whose references keep "
        "counter-ion labels such as OTf as tokens.",
        "**MolRecBench-Wild stereochemistry.** Its references carry no cis/trans (1 of 2,371 has E/Z), so a "
        "stated E/Z bond counts against the stereo-aware score. Compare readers there on the stereo-stripped "
        "score.",
        "**UOB tautomers.** On 386 UOB crops (6.7%) all six readers agree and differ from the reference only by "
        "tautomer. Each reader has 443 to 481 tautomer-only mismatches on UOB and at most 2 on any other dataset. "
        "Scores are reported as measured, so UOB likely understates every reader.",
        "**Reader settings.** MolNexTR pads each crop to a square (its released API stretches non-square crops). "
        "OCSRGlyph runs in fp32, as in its published USPTO evaluation, and keeps its own clean-up. MolGlyph "
        "writes a MolParser caption that BioMiner's code converts to SMILES; captions with ring or circle groups "
        "convert to nothing.",
    ]
    if FULL:
        items.append(
            "**System settings.** All three run at their defaults except where one GPU requires otherwise. BioMiner "
            "serves BioMiner-Instruct with vLLM at tensor-parallel 1 (upstream uses 4 GPUs) and GPU memory "
            "utilization 0.8 instead of 0.6, so its 128,000-token context fits. DECIMER.ai reads every page and "
            "segment, without the web app's cap of 10 pages and 20 segments. OpenChemIE runs its figure-molecule "
            "call alone, on torch 2.5.1 with CUDA 12.1, since its pinned torch has no H200 kernels."
        )
    items.append(
        "**Hardware and timing.** Each GPU run used one NVIDIA H200 or H200 NVL; MolVec used 8 CPU cores (Xeon "
        "E5-2680 v4 or Platinum 8276). Time per crop runs from reading the image to the final SMILES; model "
        "loading and one warm-up image are excluded."
        + (" Every complete-system run used one H200 NVL; s / paper is the mean over the papers." if FULL else "")
    )
    if FULL:
        items.append(
            "**Uncertainty and provenance.** Structure-reader intervals are Wilson 95%. Complete-system intervals "
            "come from a paper bootstrap (10,000 resamples, seed 6630), since molecules cluster by paper; macro "
            "intervals are percentiles of the same resampling. All 30 structure-reader runs ran at commit 139105a; "
            "the 6 complete-system runs ran at 4 commits with the same scorer, references and PDFs. No run had "
            "uncommitted changes, and each run's result folder (scores, environment lock, and predictions except "
            "for Internal) is attached to it as an artifact."
        )
    else:
        items.append(
            "**Uncertainty and provenance.** Intervals are Wilson 95%. All 30 runs ran at commit 139105a with no "
            "uncommitted changes; each run's result folder (predictions, scores, environment lock) is attached to it "
            "as an artifact."
        )
    return [
        wr.HorizontalRule(),
        wr.H1(text="Methods and caveats"),
        bullets(*items),
        table_panel("versions_all", tables.versions_md()) if FULL else table_panel("versions", tables.READER_VERSIONS_MD),
    ]


# The runs tables' columns: the structure-reader one as published, the complete-system one alike.
RUN_COLUMNS = [
    "config:tool.value",
    "config:dataset.value",
    "config:version.value",
    "config:device.value",
    "config:git_commit.value",
    "summary:accuracy/stereo_aware",
    "summary:accuracy/stereo_stripped",
    "summary:valid_output_rate",
    "summary:speed/s_per_crop_mean",
    "summary:items/scored",
]
SYSTEM_RUN_COLUMNS = [
    "config:tool.value",
    "config:dataset.value",
    "config:version.value",
    "config:device.value",
    "config:git_commit.value",
    "summary:micro/precision",
    "summary:micro/recall",
    "summary:micro/f1",
    "summary:speed/s_per_paper_mean",
    "summary:items/papers",
    "summary:items/molecules",
]


def runs_table(name: str, group: str, columns: list[str]) -> wr.PanelGrid:
    runs = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name=name,
        filters=f"JobType = 'eval' and Group = '{group}'",
        pinned_columns=["run:displayName"],
        visible_columns=columns,
        column_order=["run:displayName", *columns],
        lock_columns=True,
    )
    return wr.PanelGrid(runsets=[runs], hide_run_sets=False, panels=[])


def data_and_runs() -> list:
    runs = [
        wr.H2(text="Runs"),
        wr.P(
            text="The table lists the 30 eval runs, one per reader and dataset. Each holds its config (tool, version, "
            "commit, checkpoints, device), its summary metrics and its result folder as a benchmark-run artifact."
        ),
        runs_table("Structure-reader runs", "structure-readers", RUN_COLUMNS),
    ]
    if FULL:
        runs += [
            wr.P(
                text="The 6 complete-system runs, one per system and paper set, hold the same, with paper-level "
                "metrics. Internal runs publish metrics and per-paper counts only: their artifacts leave out the "
                "predictions."
            ),
            runs_table("Complete-system runs", "complete-systems", SYSTEM_RUN_COLUMNS),
        ]
    return [
        wr.HorizontalRule(),
        wr.H1(text="Data and runs"),
        wr.H2(text="Failure examples"),
        wr.P(
            text=[
                "200 crops per dataset where at least one reader is wrong, drawn at random with seed 6630 (1,000 "
                "rows): the crop image, the canonical reference, and each reader's answer and outcome. Every "
                "scored crop, with all six answers, is in the ",
                wr.Link(text="summary run", url=SUMMARY_RUN),
                "'s predictions table.",
            ]
        ),
        grid(wr.WeavePanelSummaryTable(table_name="failures", layout=wr.Layout(x=0, y=0, w=24, h=20))),
        *runs,
        wr.P(
            text=[
                "Setup, run and scoring commands are in ",
                wr.Link(text="docs/dev.md", url=f"{DOCS}/dev.md"),
                ".",
            ]
        ),
    ]


def blocks() -> list:
    middle = complete_systems() if FULL else complete_systems_placeholder()
    return opening() + at_a_glance() + structure_readers() + middle + methods() + data_and_runs()


def build_spec() -> tuple[wr.Report, dict]:
    """The report object (from the live report, for its ID and project) and the spec a save would upload."""
    report = wr.Report.from_url(URL)
    assert report.id == REPORT_ID, report.id
    report.title, report.description, report.width, report.blocks = TITLE, description(), "fixed", blocks()
    spec = json.loads(report._to_model().spec.model_dump_json(by_alias=True, exclude_none=True))
    spec_tools.fix_table_types(spec)
    problems = spec_tools.runset_problems(spec, SUMMARY_NAMES)
    if problems and "--unnarrowed" not in sys.argv:
        raise SystemExit("run-set problems:\n" + "\n".join(problems))
    return report, spec


def save(report: wr.Report, spec: dict) -> None:
    """Upload the spec as the report's saved version (Report.save() would rebuild it without the fixes)."""
    model = report._to_model()
    result = execute_graphql(
        _get_api(),
        gql.upsert_view,
        {
            "id": model.id,
            "name": model.name,
            "entityName": model.project.entity_name,
            "projectName": model.project.name,
            "description": model.description,
            "displayName": model.display_name,
            "type": "runs",
            "spec": json.dumps(spec),
        },
    )
    assert result["upsertView"]["view"]["id"] == model.id, result["upsertView"]["view"]["id"]


def main() -> None:
    new_blocks = blocks()
    print(f"stage {STAGE}: {len(new_blocks)} blocks")
    if "--dump-tables" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--dump-tables") + 1])
        out.write_text(json.dumps(TABLES, ensure_ascii=False), encoding="utf-8")
        print("tables:", ", ".join(TABLES), "->", out)
    if "--dry-run" in sys.argv or "--dump-tables" in sys.argv:
        return
    report, spec = build_spec()
    save(report, spec)
    print("saved", report.id, report.url)


if __name__ == "__main__":
    main()
