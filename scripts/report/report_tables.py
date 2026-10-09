"""The report's tables as markdown, for build_report.py's markdown panels.

The structure-reader tables hold the numbers of the published results; the complete-system table is read from the
runs' scores.json and timing.json, which hold metrics only (no SMILES), so the Internal rows are safe to show.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "benchmarks" / "results"

TOOLS_MD = """### Tools

| Tool | Approach |
|:--|:--|
| **Type 1** | **Structure readers** |
| [MolScribe](https://github.com/thomas0809/MolScribe) | Swin Transformer → graph |
| [MolNexTR](https://github.com/CYF2000127/MolNexTR) | ConvNeXt + ViT → graph |
| [DECIMER](https://github.com/Kohulan/DECIMER-Image_Transformer) | EfficientNet + Transformer |
| [MolVec](https://github.com/ncats/molvec) | Rule-based vectorization |
| [MolGlyph](https://github.com/jiaxianyan/BioMiner/blob/main/BioMiner/MolScribe/molscribe/interface_molglyph.py) | Swin-B + Transformer |
| [OCSRGlyph](https://github.com/EdisonScientific/glyph) | Swin-B + 6-layer decoder |
| **Type 2** | **Complete systems** |
| [BioMiner](https://github.com/jiaxianyan/BioMiner) | MolDetv2 → MolGlyph |
| [DECIMER.ai](https://github.com/OBrink/DECIMER.ai) | Segmentation → DECIMER |
| [OpenChemIE](https://github.com/CrystalEye42/OpenChemIE) | MolDet → MolScribe |"""

DATASETS_MD = """### Datasets

| Dataset | Source | Size |
|:--|:--|--:|
| **Type 1** | **Crops** | |
| [USPTO](https://github.com/Kohulan/OCSR_Review) | US patents | 5,704 |
| [UOB](https://github.com/Kohulan/OCSR_Review) | Maybridge | 5,740 |
| [JPO](https://github.com/Kohulan/OCSR_Review) | JP patents | 449 |
| [CLEF](https://github.com/Kohulan/OCSR_Review) | EP patents | 977 |
| [MolRecBench-Wild](https://huggingface.co/datasets/opendatalab/MolRecBench-Wild) | 818 papers | 2,371 |
| **Type 2** | **Papers** | |
| [BioVista](https://github.com/jiaxianyan/BioMiner#statistics-and-access-of-biovista) | 500 papers | 8,735 |
| Internal | 6 papers | 222 |

Size: scored crops or molecules.

Internal is private; the rest are public."""

# With the complete-system results in, the note says what the Type 2 sizes count; the section says what is scored.
DATASETS_MD_FULL = DATASETS_MD.replace(
    "Size: scored crops or molecules.", "Size: scored crops, or labeled molecules in the whole paper set."
)
assert DATASETS_MD_FULL != DATASETS_MD

LEADERBOARD_MD = """**Overall results**

| Reader | Stereo-aware pooled | Stereo-aware macro | Stereo-stripped pooled | Stereo-stripped macro | Valid output | s/crop | Best on |
|:--|--:|--:|--:|--:|--:|--:|:--|
| MolScribe | **83.7** | **76.0** | **86.3** | **79.4** | **98.3** | 0.246 | UOB |
| OCSRGlyph | 81.8 | 73.2 | 84.1 | 75.9 | 98.2 | **0.089** | USPTO, CLEF |
| MolNexTR | 79.5 | 70.9 | 81.4 | 73.8 | 98.2 | 0.249 | — |
| MolVec | 76.2 | 71.1 | 78.6 | 73.6 | 96.3 | 0.291 | JPO |
| MolGlyph | 76.0 | 70.4 | 81.6 | 76.2 | 98.2 | 0.242 | MolRecBench‑Wild |
| DECIMER | 65.5 | 58.6 | 67.7 | 61.2 | 96.8 | 0.432 | — |"""

AWARE_MD = """**Stereo-aware exact match (%)**

| Reader | USPTO | UOB | JPO | CLEF | MolRecBench-Wild |
|:--|--:|--:|--:|--:|--:|
| MolScribe | 92.1 | **87.3** | 59.5 | 80.8 | 60.1 |
| MolNexTR | 86.0 | 85.9 | 51.2 | 76.9 | 54.5 |
| DECIMER | 56.9 | 86.4 | 39.4 | 72.2 | 37.8 |
| MolVec | 88.3 | 80.2 | **66.6** | 84.7 | 35.6 |
| MolGlyph | 72.5 | 87.0 | 57.5 | 72.1 | **63.2** |
| OCSRGlyph | **94.0** | 86.8 | 51.9 | **90.7** | 42.7 |
| *Scored crops* | *5,704* | *5,740* | *449* | *977* | *2,371* |"""

STRIPPED_MD = """**Stereo-stripped exact match (%)**

| Reader | USPTO | UOB | JPO | CLEF | MolRecBench-Wild |
|:--|--:|--:|--:|--:|--:|
| MolScribe | 94.2 | **88.1** | 60.6 | 86.9 | 67.4 |
| MolNexTR | 87.1 | 86.6 | 52.1 | 83.0 | 60.2 |
| DECIMER | 59.9 | 87.1 | 40.5 | 78.0 | 40.4 |
| MolVec | 91.6 | 80.7 | **68.8** | 86.1 | 40.9 |
| MolGlyph | 81.1 | 87.8 | 59.9 | 79.3 | **72.8** |
| OCSRGlyph | **96.4** | 87.5 | 53.9 | **92.6** | 48.9 |"""

AGREEMENT_MD = """**Agreement among the six readers**

| Readers giving the same answer | Crops | Share of crops | That answer is correct |
|:--|--:|--:|--:|
| All 6 | 8,089 | 53.1% | 94.7% |
| 5 | 3,130 | 20.5% | 95.8% |
| 4 | 1,563 | 10.3% | 88.7% |
| 3 | 890 | 5.8% | 75.4% |
| 2 | 663 | 4.4% | 56.6% |
| No agreed answer | 906 | 5.9% | — |

All five datasets pooled. The agreed answer is the one shared by the largest group of readers; a crop where two answers tie for largest, or where no two readers agree, has none. Empty, invalid and crashed outputs agree with nothing."""

RESOURCES_MD = """**Time and memory, all datasets**

| Reader | Mean | Median | GPU | RAM |
|:--|--:|--:|--:|--:|
| MolScribe | 0.246 | 0.194 | 1.3 | 2.2 |
| MolNexTR | 0.249 | 0.195 | 1.3 | 2.2 |
| DECIMER | 0.432 | 0.347 | 9.6 | 5.0 |
| MolVec | 0.291 | 0.203 | CPU | 2.5 |
| MolGlyph | 0.242 | 0.193 | 1.4 | 2.9 |
| OCSRGlyph | 0.089 | 0.073 | 1.2 | 1.7 |

Seconds per crop; peak memory in GiB."""

PUBLISHED_MD = """**Published and measured exact match (%)**

| Reader | Dataset | Published | This benchmark |
|:--|:--|--:|--:|
| MolScribe | USPTO | 82.1–93.8 ¹ | 92.1 |
| MolNexTR | USPTO | 82.1–93.8 ¹ | 86.0 |
| OCSRGlyph | USPTO | 93.8 ² | 94.0 |
| MolScribe | MolRecBench-Wild | 41.05 ³ | 60.1 |
| MolNexTR | MolRecBench-Wild | 40.90 ³ | 54.5 |
| MolGlyph | BioVista crops | 76.4 | not run |

¹ One range reported across USPTO and JPO, not per dataset.

² All 5,719 USPTO images with stereochemistry required; 5,704 are scored here.

³ The paper's 5,029-crop v1 snapshot. On the 2026-08-19 release, the official SMILES track (commit 500da87, cis/trans ignored) gives this MolScribe run 60.91% and the MolScribe outputs bundled with the benchmark 62.25%. The gap comes from labels such as R1, Ar or X, which the official references keep as tokens and a `*` in the output never matches."""

READER_VERSIONS = """| MolScribe | 1.1.1 (git 7296a30, swin_base_char_aux_1m680k) |
| MolNexTR | 1.0.2 (git 6f6502b, molnextr_best.pth @ 9ac2da6, pad-to-square) |
| DECIMER | 2.8.0 (git d927ed1, Zenodo 8300489) |
| MolVec | 0.9.8 (git b412dd0, Maven Central) |
| MolGlyph | 1.1.1 (BioMiner git 17c6161, molglyph_large) |
| OCSRGlyph | 0.1.0 (git 0bf782f, model.pth @ da0d049) |"""

# The structure-reader report's table, kept as published before the complete systems were added.
READER_VERSIONS_MD = "**Reader versions**\n\n| Reader | Version |\n|:--|:--|\n" + READER_VERSIONS

SYSTEMS = ("biominer", "decimer_ai", "openchemie")
SYSTEM_NAMES = {"biominer": "BioMiner", "decimer_ai": "DECIMER.ai", "openchemie": "OpenChemIE"}
PAPER_SETS = {"biovista": "BioVista", "internal": "Internal"}
SMALL = '<br><span style="font-size:12px;opacity:.7;">{}</span>'  # a second, smaller line in a table cell


def versions_md(results: Path = RESULTS) -> str:
    """Every tool's version: the six readers, then each complete system's, as its runs' scores.json name it."""
    systems = []
    for tool in SYSTEMS:
        names = {_scores(results, tool, dataset)["tool"] for dataset in PAPER_SETS}
        assert len(names) == 1, f"{tool} ran as {len(names)} versions"
        name = names.pop()
        # scores.json names the system first ("BioMiner 1.0.0 (...)"); the table names it in its own column.
        version = name.removeprefix(SYSTEM_NAMES[tool]).strip()
        systems.append(f"| {SYSTEM_NAMES[tool]} | {version} |")
    return (
        "**Versions**\n\n| Tool | Version |\n|:--|:--|\n| **Type 1** | **Structure readers** |\n"
        + READER_VERSIONS
        + "\n| **Type 2** | **Complete systems** |\n"
        + "\n".join(systems)
    )


def systems_md(results: Path = RESULTS) -> str:
    """The complete-system results: a block of rows per paper set, micro scores with their 95% intervals below."""
    lines = [
        "**Complete-system results (%)**",
        "",
        "| System | Precision | Recall | F1 | Macro P | Macro R | Stripped P | Stripped R | s / paper |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for dataset, title in PAPER_SETS.items():
        rows = {tool: _row(_scores(results, tool, dataset)["scores"]) for tool in SYSTEMS}
        first = _scores(results, SYSTEMS[0], dataset)["scores"]["groups"]["all"]
        papers, molecules = len(first["papers"]), first["molecules"]
        lines.append(f"| **{title}**{SMALL.format(f'{papers} papers, {molecules:,} molecules')} | | | | | | | | |")
        best = {
            column: (min if column == "seconds" else max)(row[column][0] for row in rows.values())
            for column in rows[SYSTEMS[0]]
        }
        for tool in SYSTEMS:
            cells = []
            for column, (value, interval) in rows[tool].items():
                text = f"{value:.1f}" if column == "seconds" else f"{100 * value:.1f}"
                if value == best[column]:
                    text = f"**{text}**"
                if interval:
                    low, high = interval
                    text += SMALL.format(f"{100 * low:.1f}–⁠{100 * high:.1f}")
                cells.append(text)
            lines.append(f"| {SYSTEM_NAMES[tool]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Precision, recall and F1 are pooled over every molecule of the paper set (micro), with 95% intervals from a "
        "paper bootstrap below; macro averages each paper's precision and recall. Stripped scores ignore "
        "stereochemistry. s / paper is the mean on one NVIDIA H200 NVL. Bold marks the best value in each column "
        "of a paper set.",
    ]
    return "\n".join(lines)


def _row(scores: dict) -> dict[str, tuple[float, tuple[float, float] | None]]:
    group = scores["groups"]["all"]
    micro, macro, stripped = group["micro"], group["macro"], group["stereo_stripped"]
    return {
        "precision": (micro["precision"]["value"], tuple(micro["precision"]["ci95"])),
        "recall": (micro["recall"]["value"], tuple(micro["recall"]["ci95"])),
        "f1": (micro["f1"]["value"], tuple(micro["f1"]["ci95"])),
        "macro_precision": (macro["precision"]["value"], None),
        "macro_recall": (macro["recall"]["value"], None),
        "stripped_precision": (stripped["precision"]["value"], None),
        "stripped_recall": (stripped["recall"]["value"], None),
        "seconds": (scores["seconds_per_item"]["mean"], None),
    }


def _scores(results: Path, tool: str, dataset: str) -> dict:
    return json.loads((results / f"{tool}__{dataset}" / "scores.json").read_text(encoding="utf-8"))
