# Baseline Benchmark: SMILES Extraction from Papers

Existing ways of extracting SMILES from papers, benchmarked before any new pipeline is
built. The benchmark covers three types of approach, 29 runs in total, all scored by exact
match of canonical SMILES under RDKit 2026.3.2. Results are added as runs complete; — marks
a result that is not yet available.

| Type | Runs | Input | Datasets |
|:---------------------|:-----|:---------------|:----------------------------------|
| 1. Structure readers | 20 | Cut-out molecule | • [USPTO](https://github.com/Kohulan/OCSR_Review)<br>• [UOB](https://github.com/Kohulan/OCSR_Review)<br>• [JPO](https://github.com/Kohulan/OCSR_Review)<br>• [CLEF](https://github.com/Kohulan/OCSR_Review)<br>• [MolRecBench-Wild](https://huggingface.co/datasets/opendatalab/MolRecBench-Wild) |
| 2. Complete systems | 6 | Whole PDF | • [BioVista](https://github.com/jiaxianyan/BioMiner#statistics-and-access-of-biovista)<br>• Internal |
| 3. AI agent | 3 | Whole PDF | • Internal |

---

## Type 1: Structure readers

Each tool reads a single cropped molecule image and returns SMILES. Released checkpoints are
run at default settings.

### Setup

| Tool | Design | Output | Hardware | Version |
|:-----|:-------|:-------|:---------|:--------|
| [MolScribe](https://github.com/thomas0809/MolScribe) | Swin Transformer | Graph → SMILES | Explorer GPU | — |
| [MolNexTR](https://github.com/CYF2000127/MolNexTR) | ConvNeXt + ViT | Graph → SMILES | Explorer GPU | — |
| [DECIMER](https://github.com/Kohulan/DECIMER-Image_Transformer) | EfficientNet-V2 + Transformer | SMILES | Local | — |
| [MolVec](https://github.com/ncats/molvec) | Rule-based vectorization | Molfile → SMILES | Local CPU | — |

All five datasets are public. USPTO, JPO and CLEF are patent crops; UOB mixes patent and
synthetic crops; [MolRecBench-Wild](https://huggingface.co/datasets/opendatalab/MolRecBench-Wild)
has 5,029 real journal crops from 820 papers.

Metrics:

- **Exact match, stereo-aware:** prediction matches the ground truth, stereochemistry included.
- **Exact match, stereo-stripped:** prediction matches once stereochemistry is removed.
- **Valid output:** share of crops that return a SMILES RDKit can parse.
- **Speed:** seconds per crop on the hardware listed above.

### Results

**Exact match, stereo-aware (%)**

| Tool | USPTO | UOB | JPO | CLEF | MolRecBench-Wild |
|:-----|------:|----:|----:|-----:|-----------------:|
| MolScribe | — | — | — | — | — |
| MolNexTR | — | — | — | — | — |
| DECIMER | — | — | — | — | — |
| MolVec | — | — | — | — | — |

**Exact match, stereo-stripped (%)**

| Tool | USPTO | UOB | JPO | CLEF | MolRecBench-Wild |
|:-----|------:|----:|----:|-----:|-----------------:|
| MolScribe | — | — | — | — | — |
| MolNexTR | — | — | — | — | — |
| DECIMER | — | — | — | — | — |
| MolVec | — | — | — | — | — |

**Valid output and speed, all datasets pooled**

| Tool | Valid output (%) | s / crop |
|:-----|-----------------:|---------:|
| MolScribe | — | — |
| MolNexTR | — | — |
| DECIMER | — | — |
| MolVec | — | — |

**Published accuracy (%), for reference**

| Tool | USPTO | UOB | JPO | CLEF | MolRecBench-Wild |
|:-----|------:|----:|----:|-----:|-----------------:|
| MolScribe | 82.1–93.8\* | — | 82.1–93.8\* | — | 41.05 |
| MolNexTR | 82.1–93.8\* | — | 82.1–93.8\* | — | 40.90 |
| DECIMER | — | — | — | — | — |
| MolVec | — | — | — | — | — |

\* Reported as a range across USPTO and JPO for MolScribe and MolNexTR, not per dataset.

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
222 molecules from 6 papers.

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
