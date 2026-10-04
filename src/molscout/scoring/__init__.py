"""Canonical SMILES, exact match, and crop and whole-PDF scores with 95% CIs."""

from molscout.scoring.crops import score_crops
from molscout.scoring.papers import PaperCounts, PaperResult, count_paper, score_papers
from molscout.scoring.report import build_report, write_scores
from molscout.scoring.smiles import (
    RDKIT_VERSION,
    canonical_smiles,
    check_rdkit_version,
    is_exact_match,
    stereo_stripped_smiles,
)
from molscout.scoring.speed import seconds_per_item

__all__ = [
    "RDKIT_VERSION",
    "PaperCounts",
    "PaperResult",
    "build_report",
    "canonical_smiles",
    "check_rdkit_version",
    "count_paper",
    "is_exact_match",
    "score_crops",
    "score_papers",
    "seconds_per_item",
    "stereo_stripped_smiles",
    "write_scores",
]
