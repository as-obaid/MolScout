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
│   │   │   └── molvec/
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
│   ├── manifests/              # DOIs, URLs, checksums, held-out paper list
│   └── raw/                    # gitignored
├── scripts/                    # fetch_data.py, select_test_papers.py
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
- **Held-out test papers are chosen before MolScout development** and never used for tuning.
- **Scoring always uses RDKit 2026.3.2.** Canonical SMILES can differ between versions.
- **Every run saves its config, git commit, environment lock, hardware and timing
  metadata**, so the run can be reproduced.

## Predictions Format

`predictions.csv`, one row per predicted molecule:

| Column | Type | Content |
|:-------------|:--------|:-------------------------------------------|
| `dataset` | string | `uspto`, `biovista`, `internal`, … |
| `item_id` | string | Crop ID, or paper ID for whole PDFs |
| `smiles` | string | Raw tool output |
| `page` | integer | PDF page; empty for crops |
| `bbox` | string | `x0,y0,x1,y1` in PDF points; empty for crops |
| `confidence` | float | Empty if the tool gives none |
| `tool` | string | Name and version |
| `seconds` | float | Wall-clock time for the item |

## Running Benchmarks

One run is one tool on one dataset:

```bash
sbatch benchmarks/slurm/run.sbatch benchmarks/configs/molscribe__uspto.yaml
```

Output goes to `benchmarks/results/molscribe__uspto/`: `predictions.csv`, `scores.json`,
`config.yaml` and `meta.json` (commit, environment lock, hardware, timing).

## Build Order

| | Step | Done when |
|:-:|:-----------------------------------------------|:-----------------------------------------------|
| ☐ | Scoring, loaders, predictions format, held-out papers | Unit tests pass |
| ☐ | MolScribe on USPTO, on Explorer | Accuracy within the published 82.1–93.8% range |
| ☐ | MolNexTR, DECIMER, MolVec | All 20 structure-reader runs scored |
| ☐ | DECIMER.ai, OpenChemIE | Scored on BioVista and Internal |
| ☐ | BioMiner | Scored on BioVista and Internal |
| ☐ | MolScout pipeline | Scored by the harness on BioVista and Internal |

## Target Usage

```bash
pip install -e .
molscout run papers/ -o molecules.csv
```

`molecules.csv` uses the predictions format.
