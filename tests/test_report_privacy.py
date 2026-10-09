"""The Internal set's privacy on the real results: nothing publish_papers would log for benchmarks/results holds an
Internal SMILES.

Everything publish_papers hands W&B is captured with wandb.init replaced (no run starts, nothing leaves the machine):
init arguments, configs, summaries, the logged HTML pages and tables, every artifact and the echoed lines. The
secrets are the Internal ground truth's SMILES and the Internal runs' predictions, each as written and as RDKit
canonical. A secret that is also public BioVista material (a label or a BioVista run's output) may appear in the
BioVista tables; the allowlist is computed here, never written down.

Failure messages give counts only, and the checks assert on plain ints, so pytest never prints a secret.
Skips where the private ground truth, an Internal run or the BioVista download is missing.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from molscout.data.internal import GROUND_TRUTH_PATH

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
SYSTEMS = ("biominer", "decimer_ai", "openchemie")
SUBSTRING_MIN = 6  # shorter Internal SMILES occur by chance inside text, so text is searched for these and longer
# Internal reference forms that are also public BioVista material (labels or BioVista outputs) when this was written.
PUBLIC_INTERNAL_REFERENCES = 3


def _missing() -> list[str]:
    needed = [
        REPO / GROUND_TRUTH_PATH,
        *(RESULTS / f"{tool}__internal" / "predictions.csv" for tool in SYSTEMS),
        REPO / "data" / "raw" / "biovista",
    ]
    return [str(path.relative_to(REPO)) for path in needed if not path.exists()]


def _forms(values: Iterable[str]) -> set[str]:
    """Each SMILES as written and as RDKit canonical; empty strings left out."""
    from molscout.scoring import canonical_smiles

    forms = set()
    for value in values:
        if value:
            forms.add(value)
            canonical = canonical_smiles(value)
            if canonical:
                forms.add(canonical)
    return forms


def _internal_references() -> set[str]:
    """Every SMILES-like column of the Internal ground truth."""
    with (REPO / GROUND_TRUTH_PATH).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = [column for column in reader.fieldnames or () if "smiles" in column.lower()]
        return _forms(row[column] or "" for row in reader for column in columns)


def _predicted(dataset: str) -> set[str]:
    """Every SMILES the runs on `dataset` output."""
    from molscout.predictions import read_predictions

    paths = sorted(RESULTS.glob(f"*__{dataset}/predictions.csv"))
    return _forms(prediction.smiles for path in paths for prediction in read_predictions(path))


def _public_biovista() -> set[str]:
    from molscout.data.biovista_truth import BIOVISTA_PAPERS_PATH, load_biovista_truth
    from molscout.report.paper_analysis import BIOVISTA_ROOT

    truth = load_biovista_truth(REPO / BIOVISTA_ROOT, REPO / BIOVISTA_PAPERS_PATH)
    labels = _forms(smiles for references in truth.references.values() for smiles in references)
    return labels | _predicted("biovista")


def _published(wandb_papers, monkeypatch, tmp_path) -> dict:
    """What publish_papers would log for the real results, captured from a stand-in for wandb.init."""
    import wandb

    for variable in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR", "WANDB_ARTIFACT_DIR"):
        (tmp_path / variable).mkdir()
        monkeypatch.setenv(variable, str(tmp_path / variable))
    monkeypatch.setenv("WANDB_MODE", "offline")
    benchmark = wandb_papers.prepare_papers(RESULTS, REPO, allow_inconsistent=True, warn=lambda _: None)
    run = MagicMock()
    inits: list[dict] = []
    monkeypatch.setattr(wandb_papers.wandb, "init", lambda **kwargs: inits.append(kwargs) or run)
    echoed: list[str] = []
    wandb_papers.publish_papers(benchmark, entity="e", project="p", repo_root=REPO, offline=True, echo=echoed.append)
    logged = run.__enter__.return_value
    entries = [call.args[0] for call in logged.log.call_args_list]
    values = [value for entry in entries for value in entry.values()]
    inits = [{key: value for key, value in init.items() if key != "settings"} for init in inits]
    return {
        "runs": benchmark.runs,
        "texts": {
            "init": json.dumps(inits, default=str),
            "echo": "\n".join(echoed),
            "configs": json.dumps([call.args[0] for call in logged.config.update.call_args_list], default=str),
            "summaries": json.dumps([call.args[0] for call in logged.summary.update.call_args_list], default=str),
            "log keys": json.dumps([list(entry) for entry in entries]),
        },
        "pages": [Path(value._path).read_text(encoding="utf-8") for value in values if isinstance(value, wandb.Html)],
        "tables": [value for value in values if isinstance(value, wandb.Table)],
        "artifacts": [call.args[0] for call in logged.log_artifact.call_args_list],
    }


def _containing(texts: Iterable[str], secrets: Iterable[str]) -> int:
    """How many of the secrets occur somewhere in the texts."""
    joined = "\n\x00\n".join(texts)
    return sum(secret in joined for secret in secrets)


def test_no_internal_smiles_reaches_anything_published_for_the_real_results(monkeypatch, tmp_path):
    missing = _missing()
    if missing:
        pytest.skip(f"needs the private Internal data and the real results: {len(missing)} path(s) missing")
    pytest.importorskip("wandb")
    pytest.importorskip("plotly")
    from rdkit import rdBase

    from molscout.report import wandb_papers

    with rdBase.BlockLogs():  # RDKit would echo an unreadable SMILES to stderr
        references = _internal_references()
        secrets = references | _predicted("internal")
        public = _public_biovista()
        published = _published(wandb_papers, monkeypatch, tmp_path)

    private = secrets - public
    reference_count, private_count, public_references = len(references), len(private), len(references & public)
    assert reference_count > 0 and private_count > 0, f"{reference_count} references, {private_count} private secrets"
    assert public_references <= PUBLIC_INTERNAL_REFERENCES, (
        f"{public_references} Internal reference forms are also public BioVista material "
        f"(were {PUBLIC_INTERNAL_REFERENCES}); check the allowlist still stays small"
    )

    # (a) Table cells: only BioVista structures, and an Internal one only where BioVista has it too.
    tables = published["tables"]
    table_count = len(tables)
    assert table_count == 3, f"{table_count} tables logged, expected 3"
    cells = {str(cell) for table in tables for row in table.data for cell in row} | {
        str(column) for table in tables for column in table.columns
    }
    reached = len(cells & public)
    assert reached > 0, "no table cell holds a BioVista structure: the search does not reach the molecules"
    leaked_cells = len(cells & private)
    assert leaked_cells == 0, f"{leaked_cells} Internal SMILES are table cells"

    # (b) Text: HTML pages, configs, summaries, init arguments and the Internal artifacts' files.
    runs = published["runs"]
    internal = {run.name for run in runs if run.dataset == "internal"}
    artifacts = [artifact for artifact in published["artifacts"] if artifact.name in internal]
    artifact_count, internal_count = len(artifacts), len(internal)
    assert artifact_count == internal_count >= len(SYSTEMS), f"{artifact_count} of {internal_count} Internal artifacts"
    files = [
        Path(entry.local_path).read_text(encoding="utf-8")
        for artifact in artifacts
        for entry in artifact.manifest.entries.values()
    ]
    pages = published["pages"]
    page_count = len(pages)
    assert page_count == 4, f"{page_count} HTML pages logged, expected 4"
    long_secrets = {secret for secret in secrets if len(secret) >= SUBSTRING_MIN}
    for label, texts in (
        ("HTML pages", pages),
        ("Internal artifact files", files),
        *((label, [text]) for label, text in published["texts"].items()),
    ):
        found = _containing(texts, long_secrets)
        assert found == 0, f"{found} Internal SMILES of {SUBSTRING_MIN}+ characters occur in the {label}"

    # (c) An Internal run's artifact leaves its predictions out.
    with_predictions = sum("predictions.csv" in artifact.manifest.entries for artifact in artifacts)
    assert with_predictions == 0, f"{with_predictions} Internal artifacts hold predictions.csv"
