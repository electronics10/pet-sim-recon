"""Acquisition settings: the F-18 mean life and the window check."""

import math

import pytest

from pet_sim_recon.acquisition import F18_MEAN_LIFE_S, AcquisitionSettings


def test_f18_mean_life_is_about_9502_s():
    """The value the prototype used: half-life 109.77 min divided by ln 2."""
    assert F18_MEAN_LIFE_S == pytest.approx(9502, abs=1)


def test_window_upside_down_is_rejected():
    with pytest.raises(ValueError, match="low < high"):
        AcquisitionSettings(
            acquisition_time_s=1.0,
            isotope_mean_life_s=F18_MEAN_LIFE_S,
            energy_resolution_fwhm=0.12,
            energy_window_low_keV=600.0,
            energy_window_high_keV=350.0,
        )

def test_decay_weighted_time():
    # a scan of one mean life sees (1 - 1/e) of a mean life at the starting activity; a short scan, its length
    one_mean_life = AcquisitionSettings(100.0, 100.0, 0.15, 358.0, 664.0)
    assert one_mean_life.decay_weighted_time_s == pytest.approx(100.0 * (1 - math.exp(-1)))
    short = AcquisitionSettings(1.0, F18_MEAN_LIFE_S, 0.15, 358.0, 664.0)
    assert short.decay_weighted_time_s == pytest.approx(1.0, abs=1e-4)
