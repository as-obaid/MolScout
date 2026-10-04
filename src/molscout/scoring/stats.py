"""Proportions with 95% confidence intervals: Wilson, and bootstrap over papers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Z_95 = 1.959963984540054
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 6630


def ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator, with 0 / 0 defined as 0.0."""
    return 0.0 if denominator == 0 else numerator / denominator


def wilson_interval(successes: float, trials: float, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval; (0.0, 1.0) when there are no trials."""
    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError(f"need 0 <= successes <= trials, got {successes} of {trials}")
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    z2 = z * z
    scale = 1 + z2 / trials
    centre = (p + z2 / (2 * trials)) / scale
    half = z * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials)) / scale
    return (max(0.0, centre - half), min(1.0, centre + half))


def proportion(successes: float, trials: float) -> dict[str, object]:
    """A proportion as written to scores.json: value, Wilson 95% CI and its counts."""
    low, high = wilson_interval(successes, trials)
    return {"value": ratio(successes, trials), "ci95": [low, high], "successes": successes, "trials": trials}


def bootstrap_mean_interval(
    values: Sequence[float], *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """Percentile 95% CI of the mean, resampling the values (one per paper) with replacement."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("bootstrap needs at least one value")
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(resamples, data.size))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return (float(low), float(high))


def macro(values: Sequence[float]) -> dict[str, object]:
    """Unweighted mean over papers with its bootstrap 95% CI."""
    low, high = bootstrap_mean_interval(values)
    return {"value": float(np.mean(values)), "ci95": [low, high], "papers": len(values)}
