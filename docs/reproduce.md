# Reproduce

Every committed result can be rebuilt from this repository: fetch the data, build each tool's
environment, run its config, and score the predictions. The datasets and rules are described in
[benchmarks.md](benchmarks.md).

## 1. Environment

Python 3.11 or later and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/as-obaid/MolScout.git
cd MolScout
uv sync --extra dev
uv run pytest -q
```

Scoring needs RDKit 2026.3.2, which `pyproject.toml` pins. `molscout score` refuses any other
version, since canonical SMILES change between RDKit releases.

## 2. Data

```bash
uv run python scripts/fetch_data.py                 # every public dataset
uv run python scripts/fetch_data.py uspto jpo       # selected datasets
uv run python scripts/fetch_data.py biovista        # BioVista labels and open-access PDFs
```

Files go to `data/raw/`, which git ignores, and each is checked against the sha256 in its manifest
in `data/manifests/`.

BioVista ships labels only. The script takes each paper's DOI from its PDB entry and downloads the
first direct open-access PDF offered by PubMed Central, OpenAlex or Europe PMC. Unpaywall is also
asked when `UNPAYWALL_EMAIL` is set. Legal copies found by hand, such as author manuscripts, are
pinned by URL and sha256 in `data/manifests/biovista_recovered.csv`. Landing pages are never
scraped. Every paper's source, version, checksum and failure reason are written to
`data/manifests/biovista_papers.csv`.

The Internal set is private and not distributed. Its committed results hold metrics and counts only.

## 3. Tool environments

Each tool runs in its own environment under `$MOLSCOUT_STORE/envs/<tool>`, built by the tool's
`setup.sh` from pinned requirements. The script also downloads the weights. `molscout` never
imports tool code; tools run as separate processes and write the same `predictions.csv`.

```bash
export MOLSCOUT_STORE=/path/to/store
bash benchmarks/tools/structure_readers/molscribe/setup.sh
bash benchmarks/tools/complete_systems/openchemie/setup.sh
```

Configs in `benchmarks/configs/` are generated from each tool's `tool.yaml`:

```bash
uv run python scripts/make_configs.py
```

## 4. Run and score

One run is one tool on one dataset:

```bash
uv run molscout bench benchmarks/configs/molscribe__uspto.yaml
uv run molscout bench benchmarks/configs/openchemie__biovista.yaml
```

Each run writes `benchmarks/results/<tool>__<dataset>/`:

| File | Content |
|:--|:--|
| `predictions.csv` | one row per crop, or per molecule for papers |
| `scores.json` | every metric with its 95% interval, and the sha256 of each input |
| `config.yaml` | the config that ran |
| `meta.json` | commit, environment lock, hardware, timing, peak memory |
| `errors.json` | items whose prediction crashed |
| `timing.json` | seconds per paper (paper runs only) |

`meta.json` names no machine: paths are written as `$MOLSCOUT_STORE/...`, relative to the repository
root, or under `~`, and host names and SLURM job IDs are left out. `scripts/scrub_meta.py` does the
same to a `meta.json` written before this rule.

Runs resume: finished rows are checkpointed, and a rerun at the same commit predicts only what is
left. To score predictions again:

```bash
uv run molscout score benchmarks/results/molscribe__uspto/predictions.csv \
    --references data/raw/uspto/USPTO_mol_ref -o scores.json

uv run molscout score benchmarks/results/openchemie__biovista/predictions.csv \
    --references data/raw/biovista --papers data/manifests/biovista_papers.csv \
    --paper-seconds benchmarks/results/openchemie__biovista/timing.json -o scores.json
```

Because `biovista_dev.csv` lies beside `biovista_papers.csv`, a BioVista `scores.json` reports the
`dev` and `heldout` groups as well as `all`. An Internal one reports `dev` and `test` from
`data/manifests/internal_split.csv`. `--subset FILE` scores only the papers the file lists.

## 5. On a SLURM cluster

`benchmarks/slurm/` holds the job scripts. The partition names in their `#SBATCH` headers and in
`submit.sh` are site-specific; set them for your cluster. The scripts name no account. If your
cluster charges jobs to one, set `MOLSCOUT_ACCOUNT` and `submit.sh` passes `--account` to every
job and its resubmissions; for a direct `sbatch`, set `SBATCH_ACCOUNT` or pass `--account`.

```bash
export MOLSCOUT_ACCOUNT=<account>     # only if your cluster needs one
export SBATCH_ACCOUNT=<account>       # the same, for direct sbatch calls
sbatch benchmarks/slurm/setup.sbatch benchmarks/tools/structure_readers/molscribe/setup.sh
sbatch benchmarks/slurm/run.sbatch benchmarks/configs/molscribe__uspto.yaml
bash benchmarks/slurm/submit.sh structure-readers    # or: complete-systems, pipeline
bash benchmarks/slurm/status.sh
```

`submit.sh pipeline` runs every MolScout config in one worker: Internal first, then BioVista, with
the scored configuration first on each. `submit.sh pipeline CONFIG ...` runs the named configs in
that order. `submit.sh` refuses to submit from a tree with uncommitted changes, so every result maps
to a commit. Workers stop before their time limit, keep their checkpoint and resubmit themselves.

## 6. MolScout

MolScout's stages run in the tools' own environments, each built by its `setup.sh` as in step 3:
MinerU (`envs/mineru/setup.sh`), DECIMER.ai for segmentation and the classifier, MolGlyph,
MolScribe and MolNexTR for reading, BioMiner for the 32B model, and MolVec, whose JDK runs OPSIN in
the benchmark configs.
`benchmarks/tools/complete_systems/molscout/setup.sh` then installs MolScout, fetches OPSIN and
checks that every environment is there.

On papers of your own:

```bash
uv run molscout run papers/ -o out/                       # the rules joiner
bash src/molscout/pipeline/workers/serve_32b.sh &         # BioMiner-Instruct on 127.0.0.1:8613
curl -s http://127.0.0.1:8613/v1/models                   # wait until this lists the model
uv run molscout run papers/ -o out/ --joiner both --vllm-url http://127.0.0.1:8613/v1
```

The first run uses the `rules` joiner, which needs no H200. The last gives MolScout as scored. A
rerun into the same folder skips each stage whose settings, code and inputs are unchanged, so it
starts at the 32B join and vLLM has the GPU to itself. `serve_32b.sh` needs `MOLSCOUT_STORE`. It
first copies the BioMiner environments and weights to a local work folder; `--no-stage` serves them
from the store.

MolScout's harness configs sit in `benchmarks/configs/` beside the baselines' and run the same way,
so its results land in `benchmarks/results/` with the same files. Every config runs every paper of
its dataset, `biovista` or `internal`. The configs of one dataset share a folder of upstream stage
outputs, so they read the same drawings.

| Config | Setup |
|:--|:--|
| `molscout__<dataset>` | MolScout as scored: the `both` joiner |
| `molscout_rules__<dataset>` | the `rules` joiner |
| `molscout_32b__<dataset>` | the `32b` joiner |
| `molscout_both__<dataset>` | the `both` joiner, the same setup as `molscout__<dataset>` |
| `molscout_rules_review__<dataset>` | the `rules` joiner, exporting `accepted` and `review` |
| `molscout_l0_drawings__<dataset>` to `molscout_l6_junk__<dataset>` | the ladder ([results.md](results.md#the-ladder)) |
| `molscout_no_attachments__<dataset>`, `molscout_no_slot_labels__<dataset>` | the `rules` joiner with one fix off |

The configs with the 32B model pass `--start-vllm`: the run starts BioMiner-Instruct when the 32B
join first has work and stops it at the end.

```bash
uv run molscout bench benchmarks/configs/molscout_rules__biovista.yaml   # the rules joiner
uv run molscout bench benchmarks/configs/molscout__biovista.yaml         # MolScout as scored; one H200
```

To score a MolScout run again, use `molscout score` as in [step 4](#4-run-and-score).

## 7. Charts

```bash
uv run --extra figures python scripts/figures.py
```

The script reads every `scores.json` and writes each chart to `docs/figures/` as a light and a dark
PNG at twice its size. A run or split without a `scores.json` yet is drawn as "pending".
`scripts/pipeline_figure.py` draws the pipeline figure the same way, as two SVGs.

## 8. Weights & Biases (optional)

```bash
uv run --extra report python scripts/wandb_upload.py --entity <entity> --project MolScout
uv run --extra report python scripts/wandb_upload.py --benchmark complete-systems --entity <entity>
```

`--entity` and `--project` fall back to `WANDB_ENTITY` and `WANDB_PROJECT`. There is no default
entity; the project defaults to `MolScout`.

Each result folder becomes one run with its config, metrics and files. Internal runs upload metrics
and counts only.
