"""Binomial thinning turns Poisson data into Poisson data with a lower mean."""

import numpy as np
import pytest

from pet_sim_recon.recon.counts import binomial_thin


def test_thinned_poisson_counts_are_poisson_with_the_scaled_mean():
    rng = np.random.default_rng(0)
    counts = rng.poisson(20.0, 1_000_000)
    thinned = binomial_thin(counts, 0.1, rng)
    assert thinned.mean() == pytest.approx(2.0, rel=0.005)
    assert thinned.var() == pytest.approx(2.0, rel=0.01)  # a Poisson variable's variance equals its mean


def test_thinning_never_adds_counts():
    rng = np.random.default_rng(1)
    counts = rng.poisson(3.0, 1000)
    assert np.all(binomial_thin(counts, 0.5, rng) <= counts)


def test_keep_fraction_outside_zero_to_one_is_rejected():
    with pytest.raises(ValueError, match="keep_fraction"):
        binomial_thin(np.array([1]), 1.5, np.random.default_rng(0))
