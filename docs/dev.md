# Development Guide

- `benchmarks/` runs existing tools.
- `src/molscout/` is the MolScout pipeline.
- Both write the same predictions format.
- `src/molscout/scoring/` canonicalizes and matches SMILES for every run, then scores crops by
  accuracy and whole PDFs by precision, recall and F1.

## Repository Layout

```
MolScout/
├── pyproject.toml
├── src/molscout/
│   ├── scoring/                # canonical SMILES, exact match; accuracy, precision, recall, F1
│   ├── data/                   # loaders: molfile, SDF, ground-truth CSV, manifests
│   └── pipeline/               # see Architecture.md
├── benchmarks/
│   ├── tools/                  # per tool: environment file + run.py → predictions.csv
│   │   ├── structure_readers/
│   │   │   ├── molscribe/
│   │   │   ├── molnextr/
│   │   │   ├── decimer/
│   │   │   ├── molvec/
│   │   │   ├── molglyph/
│   │   │   └── ocsrglyph/
│   │   ├── complete_systems/
│   │   │   ├── biominer/
│   │   │   ├── decimer_ai/
│   │   │   └── openchemie/
│   │   └── agents/
│   │       └── claude_code/    # goal prompt; AGENTIC.md per run
│   ├── configs/                # one YAML per tool × dataset
│   ├── slurm/                  # sbatch template
│   └── results/                # one folder per run
├── data/
│   ├── manifests/              # DOIs, URLs, checksums, internal_split.csv
│   └── raw/                    # gitignored
├── scripts/                    # fetch_data.py
├── tests/
└── docs/
```

## Benchmark Flow

```
                    Benchmark harness
                           │
             ┌─────────────┴─────────────┐
             │                           │
        MolScout env               External tools
      (Python package)             (isolated envs)
             │                           │
      import molscout                subprocess
             │                           │
             └─────────────┬─────────────┘
                           │
                    predictions.csv
                           │
                     shared scorer
```

Agent runs are interactive sessions outside the harness; their output is converted to
`predictions.csv` and scored by the same scorer.

## Rules

- **One environment per tool** (conda or Apptainer). DECIMER needs TensorFlow, MolScribe needs
  PyTorch and MolVec needs Java.
- **`molscout` never imports external benchmark tools.** Tools run as separate processes.
- **No paper PDFs, dataset images or private data in git.** Public data is rebuilt by
  `fetch_data.py` from the manifests; the internal set stays local.
- **Internal papers 2, 4 and 6 are the held-out test set** (96 molecules), frozen in
  `data/manifests/internal_split.csv` before MolScout development. MolScout is tuned only
  on papers 1, 16 and 19 (126 molecules). Existing tools are scored on all six.
- **Scoring always uses RDKit 2026.3.2.** Canonical SMILES can differ between versions, so
  `molscout score` refuses to run under any other version.
- **Every run saves its config, git commit, environment lock, hardware and timing
  metadata**, so the run can be reproduced.

## Predictions Format

`predictions.csv`, one row per predicted molecule:

| Column | Type | Content |
|:-------------|:--------|:-------------------------------------------|
| `dataset` | string | `uspto`, `uob`, `jpo`, `clef`, `molrecbench_wild`, `biovista`, `internal` |
| `item_id` | string | Crop ID (image file name without extension), or paper ID for whole PDFs |
| `smiles` | string | Raw tool output; empty if the tool returned nothing |
| `page` | integer | PDF page, 1-based; empty for crops |
| `bbox` | string | `x0,y0,x1,y1` in PDF points; empty for crops |
| `confidence` | float | Empty if the tool gives none |
| `tool` | string | Name and version |
| `seconds` | float | Wall-clock time for the item |

One file holds one tool's run on one dataset, with at most one row per crop.
`molscout.predictions` reads and validates the format and reports every error with its line
number.

## Scoring Rules

`molscout score` applies the same rules to every run.

- **Canonical form.** Predictions and ground truth both go through RDKit's default
  `MolToSmiles`, and a match is an identical string. Stereo-stripped scores first remove
  tetrahedral and double-bond stereo with `RemoveStereochemistry`; isotopes are kept.
- **`*` is an atom.** Five internal ground-truth molecules contain `*`, so `*` is
  canonicalized like any other atom, never rejected. Labels are kept: `*`, `[1*]` and `[*:1]`
  are different atoms.
- **Invalid output counts as emitted and wrong.** An empty or unparsable SMILES is a wrong
  answer for its crop and a false positive for its paper, and it counts against the
  valid-output rate. Text after whitespace makes an output invalid (`CCO CCN` is not read as
  `CCO`); CXSMILES extensions such as `*C |$R1$|` are accepted.
- **Valid-output rate counts what a tool emitted:** crops with a parsable SMILES ÷ scored
  crops, and parsable rows ÷ rows for papers.
- **Duplicates collapse within a paper.** Predictions and ground truth are each reduced to
  unique canonical SMILES per paper before counting; unparsable outputs collapse by their
  trimmed text. On Internal, papers 4 and 19 each list one structure under two names, so 220
  structures are scored rather than 222: 125 in dev and 95 in test.
- **Unreadable references are left out.** A crop whose reference file RDKit cannot read is
  dropped from the denominator and listed in `scores.json`: 15 USPTO, 1 JPO and 15 CLEF crops.
- **0 ÷ 0 is 0.** A paper with no output has precision 0; `papers_without_output` counts
  them.
- **95% confidence intervals.** Accuracy, valid-output rate and micro precision, recall and
  F1 use Wilson intervals. F1's interval uses TP out of TP + (FP + FN)/2, since
  F1 = TP ÷ (TP + (FP + FN)/2). Macro scores use a percentile bootstrap over papers with
  10,000 resamples and seed 6630. Two limits apply. Pooled intervals treat molecules as
  independent, though they cluster by paper (paper 19 holds 105 of 222 rows), so they are
  optimistic. With three papers per split, the dev and test bootstrap intervals reduce to the
  range of the per-paper values.
- **`scores.json` pins its inputs:** the predictions sha256; the ground-truth and split sha256
  (Internal) or the reference-set sha256 with the reason each crop was excluded (crops); and
  the RDKit, NumPy, Python and MolScout versions.
- **Internal is reported three ways:** dev (papers 1, 16, 19), test (2, 4, 6) and all six.

## Running Benchmarks

Fetch the public datasets once. Each file is checked against the sha256 in its manifest:

```bash
pip install -e ".[dev]"
python scripts/fetch_data.py            # or: python scripts/fetch_data.py uspto jpo
```

Crop datasets unpack to `data/raw/<dataset>/`, with images in `USPTO/` and references in
`USPTO_mol_ref/` (likewise for UOB, JPO and CLEF). MolRecBench-Wild arrives as Parquet in
`data/raw/molrecbench_wild/data/`.

One run is one tool on one dataset:

```bash
sbatch benchmarks/slurm/run.sbatch benchmarks/configs/molscribe__uspto.yaml
```

Output goes to `benchmarks/results/molscribe__uspto/`: `predictions.csv`, `scores.json`,
`config.yaml` and `meta.json` (commit, environment lock, hardware, timing).

`scores.json` comes from the shared scorer:

```bash
molscout score benchmarks/results/molscribe__uspto/predictions.csv \
    --references data/raw/uspto/USPTO_mol_ref -o benchmarks/results/molscribe__uspto/scores.json
molscout score benchmarks/results/biominer__internal/predictions.csv \
    -o benchmarks/results/biominer__internal/scores.json   # reads data/internal/ and the split
```

## Build Order

| | Step | Done when |
|:-:|:-----------------------------------------------|:-----------------------------------------------|
| ☑ | Scoring, loaders, predictions format, internal split | Unit tests pass |
| ☐ | MolScribe on USPTO, on Explorer | Accuracy within the published 82.1–93.8% range |
| ☐ | MolNexTR, DECIMER, MolVec, MolGlyph, OCSRGlyph | All 30 structure-reader runs scored |
| ☐ | DECIMER.ai, OpenChemIE | Scored on BioVista and Internal |
| ☐ | BioMiner | Scored on BioVista and Internal |
| ☐ | MolScout pipeline | Scored by the harness on BioVista and Internal |

## Target Usage

```bash
pip install -e .
molscout run papers/ -o molecules.csv
```

`molecules.csv` uses the predictions format.
