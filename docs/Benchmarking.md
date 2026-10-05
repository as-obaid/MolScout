# Baseline Benchmark: SMILES Extraction from Papers

Existing ways of extracting SMILES from papers, benchmarked before any new pipeline is
built. The benchmark covers three types of approach, 39 runs in total, all scored by exact
match of canonical SMILES under RDKit 2026.3.2. Results are added as runs complete; — marks
a result that is not yet available.

| Type | Runs | Input | Datasets |
|:---------------------|:-----|:---------------|:----------------------------------|
| 1. Structure readers | 30 | Cut-out molecule | • [USPTO](https://github.com/Kohulan/OCSR_Review)<br>• [UOB](https://github.com/Kohulan/OCSR_Review)<br>• [JPO](https://github.com/Kohulan/OCSR_Review)<br>• [CLEF](https://github.com/Kohulan/OCSR_Review)<br>• [MolRecBench-Wild](https://huggingface.co/datasets/opendatalab/MolRecBench-Wild) |
| 2. Complete systems | 6 | Whole PDF | • [BioVista](https://github.com/jiaxianyan/BioMiner#statistics-and-access-of-biovista)<br>• Internal |
| 3. AI agent | 3 | Whole PDF | • Internal |

---

## Type 1: Structure readers

Each tool reads a single cropped molecule image and returns SMILES. Released checkpoints are
run at default settings.

### Setup

| Tool | Design | Output | Hardware | Version |
|:-----|:-------|:-------|:---------|:--------|
| [MolScribe](https://github.com/thomas0809/MolScribe) | Swin Transformer | Graph → SMILES | Explorer, NVIDIA H200 / Explorer, NVIDIA H200 NVL | 1.1.1 (git 7296a30, swin_base_char_aux_1m680k) |
| [MolNexTR](https://github.com/CYF2000127/MolNexTR) | ConvNeXt + ViT | Graph → SMILES | Explorer, NVIDIA H200 / Explorer, NVIDIA H200 NVL | 1.0.2 (git 6f6502b, molnextr_best.pth @ 9ac2da6, pad-to-square) |
| [DECIMER](https://github.com/Kohulan/DECIMER-Image_Transformer) | EfficientNet-V2 + Transformer | SMILES | Explorer, NVIDIA H200 NVL / Explorer, NVIDIA H200 | 2.8.0 (git d927ed1, Zenodo 8300489) |
| [MolVec](https://github.com/ncats/molvec) | Rule-based vectorization | Molfile → SMILES | Explorer, CPU, 8 cores | 0.9.8 (git b412dd0, Maven Central) |
| [MolGlyph](https://github.com/jiaxianyan/BioMiner/blob/main/BioMiner/MolScribe/molscribe/interface_molglyph.py) | Swin-B + Transformer on MolScribe's code; BioMiner's reader | SMILES | Explorer, NVIDIA H200 / Explorer, NVIDIA H200 NVL | 1.1.1 (BioMiner git 17c6161, molglyph_large) |
| [OCSRGlyph](https://github.com/EdisonScientific/glyph) | Swin-B + 6-layer Transformer decoder | SMILES | Explorer, NVIDIA H200 NVL / Explorer, NVIDIA H200 | 0.1.0 (git 0bf782f, model.pth @ da0d049) |

All five datasets are public. USPTO, JPO and CLEF are patent crops; UOB mixes patent and
synthetic crops; [MolRecBench-Wild](https://huggingface.co/datasets/opendatalab/MolRecBench-Wild)
has 5,024 real journal crops from 818 papers in its 2026-08-19 release (5,029 from 820 in the
paper).

Metrics:

- **Exact match, stereo-aware:** prediction matches the ground truth, stereochemistry included.
- **Exact match, stereo-stripped:** prediction matches once stereochemistry is removed.
- **Valid output:** share of crops that return a SMILES RDKit can parse.
- **Speed:** seconds per crop on the hardware listed above.

### Results

**Exact match, stereo-aware (%)**

| Tool | USPTO | UOB | JPO | CLEF | MolRecBench-Wild |
|:-----|------:|----:|----:|-----:|-----------------:|
| MolScribe | 92.1 | 87.3 | 59.5 | 80.8 | 60.1 |
| MolNexTR | 86.0 | 85.9 | 51.2 | 76.9 | 54.5 |
| DECIMER | 56.9 | 86.4 | 39.4 | 72.2 | 37.8 |
| MolVec | 88.3 | 80.2 | 66.6 | 84.7 | 35.6 |
| MolGlyph | 72.5 | 87.0 | 57.5 | 72.1 | 63.2 |
| OCSRGlyph | 94.0 | 86.8 | 51.9 | 90.7 | 42.7 |

**Exact match, stereo-stripped (%)**

| Tool | USPTO | UOB | JPO | CLEF | MolRecBench-Wild |
|:-----|------:|----:|----:|-----:|-----------------:|
| MolScribe | 94.2 | 88.1 | 60.6 | 86.9 | 67.4 |
| MolNexTR | 87.1 | 86.6 | 52.1 | 83.0 | 60.2 |
| DECIMER | 59.9 | 87.1 | 40.5 | 78.0 | 40.4 |
| MolVec | 91.6 | 80.7 | 68.8 | 86.1 | 40.9 |
| MolGlyph | 81.1 | 87.8 | 59.9 | 79.3 | 72.8 |
| OCSRGlyph | 96.4 | 87.5 | 53.9 | 92.6 | 48.9 |

**Valid output and speed, all datasets pooled**

| Tool | Valid output (%) | s / crop |
|:-----|-----------------:|---------:|
| MolScribe | 98.3 | 0.246 |
| MolNexTR | 98.2 | 0.249 |
| DECIMER | 96.8 | 0.432 |
| MolVec | 96.3 | 0.291 |
| MolGlyph | 98.2 | 0.242 |
| OCSRGlyph | 98.2 | 0.089 |

**Published accuracy (%), for reference**

| Tool | USPTO | UOB | JPO | CLEF | MolRecBench-Wild |
|:-----|------:|----:|----:|-----:|-----------------:|
| MolScribe | 82.1–93.8\* | — | 82.1–93.8\* | — | 41.05 |
| MolNexTR | 82.1–93.8\* | — | 82.1–93.8\* | — | 40.90 |
| DECIMER | — | — | — | — | — |
| MolVec | — | — | — | — | — |
| MolGlyph | — | — | — | — | — |
| OCSRGlyph | 93.8 | — | — | — | — |

\* Reported as a range across USPTO and JPO for MolScribe and MolNexTR, not per dataset.

MolGlyph's published OCSR score is 76.4% overall and 50.4% on chirality, measured on
BioVista crops rather than on these datasets (Yan et al., 2026). Its weights are gated on
[Hugging Face](https://huggingface.co/jiaxianustc/MolGlyph). OCSRGlyph's 93.8% is on the
5,719-image USPTO set with all stereochemistry required, as reported in its repository.

---

## Type 2: Complete systems

Each system takes a whole PDF and returns SMILES for every molecule it finds. Released code
and weights are run at default settings.

### Setup

| System | Detect | Recognize | Extra | Hardware | Version |
|:-------|:-------|:----------|:------|:---------|:--------|
| [BioMiner](https://github.com/jiaxianyan/BioMiner) | MolDetv2 | MolGlyph | Qwen3-VL-32B agents;<br>coreference | 4× H200 | — |
| [DECIMER.ai](https://github.com/OBrink/DECIMER.ai) | [DECIMER-Segmentation](https://github.com/Kohulan/DECIMER-Image-Segmentation) | [DECIMER](https://github.com/Kohulan/DECIMER-Image_Transformer) | Image classifier | Local / 1 GPU | — |
| [OpenChemIE](https://github.com/CrystalEye42/OpenChemIE) | MolDet | [MolScribe](https://github.com/thomas0809/MolScribe) | Coreference;<br>text and reaction models | Explorer GPU | — |

[BioVista](https://github.com/jiaxianyan/BioMiner#statistics-and-access-of-biovista) is
public: 8,735 structures from 500 papers, with PDFs fetched by DOI. Internal is private:
222 molecules from 6 papers. Existing tools are scored on all six; papers 2, 4 and 6
(96 molecules) are also MolScout's held-out test set, so per-paper scores allow a direct
comparison later.

Metrics:

- **Precision, recall, F1 (micro):** pooled over every molecule in the dataset.
- **Precision, recall (macro):** computed per paper, then averaged.
- **Stereo-stripped precision, recall:** micro scores with stereochemistry removed.
- **PDFs obtained:** papers whose PDF could be fetched by DOI (BioVista only).
- **Speed:** seconds per paper on the hardware listed above.

### Results

Precision, recall and F1 in %.

**BioVista**

| System | Precision | Recall | F1 | Macro P | Macro R | Stripped P | Stripped R | PDFs | s / paper |
|:-------|----------:|-------:|---:|--------:|--------:|-----------:|-----------:|-----:|----------:|
| BioMiner | — | — | — | — | — | — | — | — | — |
| DECIMER.ai | — | — | — | — | — | — | — | — | — |
| OpenChemIE | — | — | — | — | — | — | — | — | — |
| *BioMiner, published* | — | — | *52.8* | — | — | — | — | — | — |

**Internal**

| System | Precision | Recall | F1 | Macro P | Macro R | Stripped P | Stripped R | s / paper |
|:-------|----------:|-------:|---:|--------:|--------:|-----------:|-----------:|----------:|
| BioMiner | — | — | — | — | — | — | — | — |
| DECIMER.ai | — | — | — | — | — | — | — | — |
| OpenChemIE | — | — | — | — | — | — | — | — |

---

## Type 3: AI agent

A coding agent is given the PDFs and one goal prompt and must build its own extraction. It
runs three times with an identical setup, each starting from scratch.

### Setup

| Item | Value |
|:--------|:--------------------------------------------------|
| Agent | [Claude Code](https://claude.com/claude-code) |
| Model | Opus 5.5 |
| Effort | Extra High |
| Dataset | Internal (6 papers, 222 molecules; private) |
| Input | 6 internal PDFs + one fixed goal prompt |
| Hidden | Ground-truth file, kept outside the workspace |
| Session | Fresh per run, empty repository |
| Log | `AGENTIC.md`: steps, errors, human interventions |

Metrics:

- **Precision, recall, F1 (micro) and precision, recall (macro):** as in Type 2.
- **Cost:** input, output and total tokens; wall-clock time; tool calls.
- **Human interventions:** times a person had to step in, as logged in `AGENTIC.md`.

### Results

Mean ± standard deviation over the three runs. Precision, recall and F1 in %.

**Accuracy**

| Agent | Precision | Recall | F1 | Macro P | Macro R |
|:------|----------:|-------:|---:|--------:|--------:|
| Claude Code, Opus 5.5 | — | — | — | — | — |
| *Claude Code, published (Gulluoglu et al., 2026)* | *~21* | *~11* | — | — | — |

**Cost**

| Agent | Input tokens | Output tokens | Total tokens | Wall-clock | Tool calls | Human interventions |
|:------|-------------:|--------------:|-------------:|-----------:|-----------:|--------------------:|
| Claude Code, Opus 5.5 | — | — | — | — | — | — |
