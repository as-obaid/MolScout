"""Proportions with 95% confidence intervals: Wilson, and bootstrap over papers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Z_95 = 1.959963984540054
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 6630
PAPER_BOOTSTRAP = "paper bootstrap"  # ci_method of a proportion whose CI resamples papers


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
    means = data[_draws(data.size, resamples, seed)].mean(axis=1)
    return _percentile_interval(means)


def bootstrap_ratio_interval(
    numerators: Sequence[float],
    denominators: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile 95% CI of sum(numerators) / sum(denominators), resampling the papers (one pair each) with replacement.

    Each draw pools its papers' counts, as the reported value does; a draw whose denominator is 0 counts as 0.0.
    """
    tops, bottoms = np.asarray(numerators, dtype=float), np.asarray(denominators, dtype=float)
    if tops.ndim != 1 or tops.shape != bottoms.shape:
        raise ValueError(f"need one numerator and one denominator per paper, got {tops.shape} and {bottoms.shape}")
    if tops.size == 0:
        raise ValueError("bootstrap needs at least one paper")
    draws = _draws(tops.size, resamples, seed)
    pooled_tops, pooled_bottoms = tops[draws].sum(axis=1), bottoms[draws].sum(axis=1)
    ratios = np.divide(pooled_tops, pooled_bottoms, out=np.zeros(resamples), where=pooled_bottoms != 0)
    return _percentile_interval(ratios)


def macro(values: Sequence[float]) -> dict[str, object]:
    """Unweighted mean over papers with its bootstrap 95% CI."""
    low, high = bootstrap_mean_interval(values)
    return {"value": float(np.mean(values)), "ci95": [low, high], "papers": len(values)}


def paper_proportion(numerators: Sequence[float], denominators: Sequence[float]) -> dict[str, object]:
    """A proportion pooled over papers, as written to scores.json: value, paper-bootstrap 95% CI, counts and method.

    `numerators` and `denominators` hold one count per paper; the value is the ratio of their sums. Pooled counts
    from clustered papers are not independent trials, so the interval resamples papers rather than using Wilson.
    """
    successes, trials = sum(numerators), sum(denominators)
    low, high = bootstrap_ratio_interval(numerators, denominators)
    return {
        "value": ratio(successes, trials),
        "ci95": [low, high],
        "successes": successes,
        "trials": trials,
        "ci_method": PAPER_BOOTSTRAP,
    }


def _draws(papers: int, resamples: int, seed: int) -> np.ndarray:
    """Row i holds the indices of resample i: `papers` draws with replacement, from a generator seeded with `seed`."""
    return np.random.default_rng(seed).integers(0, papers, size=(resamples, papers))


def _percentile_interval(draws: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(draws, [2.5, 97.5])
    return (float(low), float(high))
