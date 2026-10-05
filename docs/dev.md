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
│   ├── slurm/                  # run.sbatch, submit.sh (workers), status.sh
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
  RDKit's canonical SMILES is not always stable, so each side is read and written again
  until the string stops changing, at most five passes. An aromatic ring with `*` atoms comes
  back Kekulé (MolRecBench-Wild `10.1002_anie.202411707_3_figure_0_mol_3`), and stripping
  stereo can leave an explicit `[H]` that the next pass drops (USPTO
  `US07317016-20080108-C00012`). A tool that already writes RDKit's canonical SMILES is
  scored the same as one that does not.
- **`*` is an atom.** Five internal ground-truth molecules contain `*`, so `*` is
  canonicalized like any other atom, never rejected. Labels are kept: `*`, `[1*]` and `[*:1]`
  are different atoms, so a reader that writes a bare `*` for a drawn R1 is wrong. CLEF
  R-groups canonicalize as `[1*]`, `[2*]` (85 of 992 references); the reader has to keep the
  number.
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
- **MolRecBench-Wild references come from its CARBON graphs**, converted as the official
  SMILES track does (evaluator commit 500da87): graphs with Greek letters or `?` in a label,
  repeat brackets, drawing-specific bonds or attachment points are left out, and drawn
  abbreviations are expanded from the evaluator's table. R-group and variable labels become
  `*` atoms (R1 becomes `[1*]`); a graph with any other label left is left out. 2,371 of
  5,024 crops are scored. Unlike the official track, cis/trans counts in the stereo-aware
  score.
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
`data/raw/molrecbench_wild/data/`, and `scripts/fetch_data.py molrecbench_wild` also exports
each crop's image to `data/raw/molrecbench_wild/images/<crop ID>.png`.

One run is one tool on one dataset:

```bash
sbatch benchmarks/slurm/run.sbatch benchmarks/configs/molscribe__uspto.yaml
```

Output goes to `benchmarks/results/molscribe__uspto/`: `predictions.csv`, `scores.json`,
`config.yaml`, `meta.json` (commit, environment lock, hardware, timing) and `errors.json` (the
crops whose prediction crashed).

`meta.json` also records the tool's peak memory and CPU time, and the GPU's peak memory and
mean utilization, sampled every 5 s (`resources`).

Each row's `seconds` times one `predict` call: reading the image, inference and the tool's own
post-processing. Imports, model loading and one untimed warm-up image are not counted.

A crash on a single image gives that crop an empty SMILES, which scores as wrong. The crash is
listed in `errors.json` and counted in `meta.json` (`tool_errors`). 25 crashes in a row stop the
run.

All 30 runs go through workers that fill every H200 partition, and the CPU queue for MolVec, up to
the per-user limits:

```bash
bash benchmarks/slurm/submit.sh     # from the repository root; running it again only tops up
bash benchmarks/slurm/status.sh     # one line per run, then the molscout jobs in the queue
```

Each worker gets every config of its kind, longest first. It skips configs with results from HEAD
and no uncommitted changes, and configs another live job has claimed (`benchmarks/results/.claims/`);
after two failures at HEAD it gives up on a config. Three minutes before its time limit a worker
stops its run and resubmits itself; `scancel` stops it without resubmitting.

Runs resume. `run.py` appends each finished row to `benchmarks/results/.checkpoints/<run>/`, and the
next run of the same config at the same commit predicts only the images left; a checkpoint from
another config or commit is deleted. `meta.json` records each part as a segment (job, host, GPU,
start and finish, rows done, resources), and its `timing.tool_seconds` and `resources` cover all
of them. The results equal those of one uninterrupted run, apart from each row's `seconds`.

`scores.json` comes from the shared scorer:

```bash
molscout score benchmarks/results/molscribe__uspto/predictions.csv \
    --references data/raw/uspto/USPTO_mol_ref -o benchmarks/results/molscribe__uspto/scores.json
molscout score benchmarks/results/biominer__internal/predictions.csv \
    -o benchmarks/results/biominer__internal/scores.json   # reads data/internal/ and the split
```

MolRecBench-Wild is scored against its Parquet shards, whose folder is the references path:

```bash
molscout score predictions.csv --references data/raw/molrecbench_wild
```

## Publishing to W&B

Install the report extra, copy the result folders from the cluster, then publish them:

```bash
pip install -e ".[report]"
rsync -a <cluster>:<repo>/benchmarks/results/ benchmarks/results/
python scripts/wandb_upload.py --entity ENTITY --project PROJECT
```

Each `<tool>__<dataset>/` folder becomes one `eval` run in the `structure-readers` group. The run
holds its config (tool, version, commit, checkpoints, environment, device), namespaced summary
metrics (`accuracy/`, `speed/`, `resources/`, `items/`, `outcome/`) and the folder as a
`benchmark-run` artifact. One `analysis` run, `summary`, holds the figures and the `leaderboard`,
`predictions` and `failures` tables. Run IDs come from each `predictions.csv` sha256, so
publishing the same results again updates the same runs.

The upload refuses runs from more than one commit, runs with uncommitted code, and runs scored
against different references; `--allow-inconsistent` is for development only. It reads the
references and images under `data/raw/` and needs W&B credentials (`wandb login`).

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
