"""Dataset names used in predictions.csv, and how each one is scored."""

from enum import StrEnum
from types import MappingProxyType


class Kind(StrEnum):
    """Crop datasets have one molecule per image; paper datasets have many per PDF."""

    CROP = "crop"
    PAPER = "paper"


DATASETS = MappingProxyType(
    {
        "uspto": Kind.CROP,
        "uob": Kind.CROP,
        "jpo": Kind.CROP,
        "clef": Kind.CROP,
        "molrecbench_wild": Kind.CROP,
        "biovista": Kind.PAPER,
        "internal": Kind.PAPER,
    }
)


def dataset_kind(name: str) -> Kind:
    """Return how a known dataset is scored; raise ValueError for an unknown name."""
    try:
        return DATASETS[name]
    except KeyError:
        known = ", ".join(sorted(DATASETS))
        raise ValueError(f"unknown dataset {name!r}; expected one of: {known}") from None
