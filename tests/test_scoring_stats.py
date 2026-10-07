import pytest

from molscout.scoring.stats import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    bootstrap_mean_interval,
    bootstrap_ratio_interval,
    macro,
    paper_proportion,
    proportion,
    ratio,
    wilson_interval,
)

# statsmodels.stats.proportion.proportion_confint(k, n, alpha=0.05, method="wilson")
REFERENCE = [
    (5, 11, 0.212712716225, 0.719908462591),
    (5, 10, 0.236593090513, 0.763406909487),
    (5, 10.5, 0.224008446579, 0.741127596675),
    (7, 11, 0.353801174508, 0.848335289046),
    (3, 6, 0.187616306483, 0.812383693517),
    (4, 6, 0.299993315138, 0.903228588894),
    (0, 5, 0.0, 0.434482464783),
    (5, 5, 0.565517535217, 1.0),
]


@pytest.mark.parametrize(("k", "n", "low", "high"), REFERENCE)
def test_wilson_matches_statsmodels(k, n, low, high):
    assert wilson_interval(k, n) == pytest.approx((low, high), abs=1e-9)


def test_wilson_with_no_trials_is_uninformative():
    assert wilson_interval(0, 0) == (0.0, 1.0)


@pytest.mark.parametrize(("k", "n"), [(-1, 5), (6, 5), (1, -1)])
def test_wilson_rejects_impossible_counts(k, n):
    with pytest.raises(ValueError):
        wilson_interval(k, n)


def test_ratio_defines_zero_over_zero_as_zero():
    assert ratio(0, 0) == 0.0
    assert ratio(1, 4) == 0.25


def test_proportion_record():
    record = proportion(3, 6)
    assert record["value"] == 0.5
    assert (record["successes"], record["trials"]) == (3, 6)
    assert record["ci95"] == pytest.approx([0.187616306483, 0.812383693517], abs=1e-9)


def test_bootstrap_defaults():
    assert BOOTSTRAP_RESAMPLES == 10_000
    assert BOOTSTRAP_SEED == 6630


def test_bootstrap_is_reproducible_with_the_fixed_seed():
    values = [0.5, 0.5, 0.0, 0.0, 1.0]
    assert bootstrap_mean_interval(values) == bootstrap_mean_interval(values)


def test_bootstrap_seed_changes_the_draw():
    # Off a 0.1 grid: on it, resampled means tie on 1/60 steps and both seeds hit the same percentiles.
    values = [0.13, 0.91, 0.42, 0.77, 0.26, 0.05]
    assert bootstrap_mean_interval(values, seed=1) != bootstrap_mean_interval(values, seed=2)


def test_bootstrap_interval_brackets_the_mean_within_the_data_range():
    low, high = bootstrap_mean_interval([0.5, 0.5, 0.0, 0.0])
    assert 0.0 <= low <= 0.25 <= high <= 0.5


@pytest.mark.parametrize("values", [[0.4, 0.4, 0.4], [0.7]])
def test_bootstrap_of_constant_values_is_a_point(values):
    assert bootstrap_mean_interval(values) == pytest.approx((values[0], values[0]))


def test_bootstrap_needs_values():
    with pytest.raises(ValueError):
        bootstrap_mean_interval([])


def test_macro_record():
    record = macro([0.5, 0.5, 0.0, 0.0])
    assert record["value"] == 0.25
    assert record["papers"] == 4
    assert record["ci95"][0] <= 0.25 <= record["ci95"][1]


def test_macro_interval_is_pinned():
    # Pinned before the paper bootstrap of pooled ratios shared its resampling: the macro draw must not move.
    assert bootstrap_mean_interval([0.13, 0.91, 0.42, 0.77, 0.26, 0.05]) == (0.18166666666666667, 0.6966666666666667)


def test_ratio_bootstrap_pools_each_draw_and_counts_zero_over_zero_as_zero():
    # Two papers, 1/1 and 0/0: a draw of the second twice is 0/0, so 0; any other draw is 1. P(0/0) = 1/4 > 2.5%.
    assert bootstrap_ratio_interval([1, 0], [1, 0]) == (0.0, 1.0)


@pytest.mark.parametrize(("numerators", "denominators"), [([2, 2, 2], [5, 5, 5]), ([3], [7])])
def test_ratio_bootstrap_of_identical_papers_or_one_paper_is_a_point(numerators, denominators):
    point = numerators[0] / denominators[0]
    assert bootstrap_ratio_interval(numerators, denominators) == (point, point)


def test_ratio_bootstrap_is_reproducible_and_seeded():
    # Six uneven papers: with fewer, both seeds' percentiles can land on the same pooled ratio.
    numerators, denominators = [5, 0, 2, 9, 1, 4], [7, 3, 5, 20, 11, 13]
    assert bootstrap_ratio_interval(numerators, denominators) == bootstrap_ratio_interval(numerators, denominators)
    assert bootstrap_ratio_interval(numerators, denominators, seed=1) != bootstrap_ratio_interval(
        numerators, denominators, seed=2
    )


@pytest.mark.parametrize(("numerators", "denominators"), [([], []), ([1, 2], [3])])
def test_ratio_bootstrap_needs_one_count_pair_per_paper(numerators, denominators):
    with pytest.raises(ValueError):
        bootstrap_ratio_interval(numerators, denominators)


def test_paper_proportion_record():
    record = paper_proportion([3, 1, 0], [4, 1.5, 2])
    assert record["value"] == pytest.approx(4 / 7.5)
    assert (record["successes"], record["trials"]) == (4, 7.5)
    assert record["ci95"] == list(bootstrap_ratio_interval([3, 1, 0], [4, 1.5, 2]))
    assert record["ci_method"] == "paper bootstrap"
