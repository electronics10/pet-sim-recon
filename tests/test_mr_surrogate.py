"""The MR surrogate: labels to intensities, blur of a given FWHM, Rician noise (docs/image-quality.md)."""

import math

import numpy as np
import pytest

from pet_sim_recon.image_grid import ImageGrid
from pet_sim_recon.mr_surrogate import FWHM_PER_SIGMA, NEMA_IQ_MR_INTENSITIES, mr_surrogate
from pet_sim_recon.phantoms import nema_iq_phantom

GRID = ImageGrid(num_voxels_zyx=(56, 40, 40), voxel_size_mm_zyx=(1.0, 1.0, 1.0))


def test_labels_become_intensities_and_the_hidden_rod_looks_like_pmma():
    phantom = nema_iq_phantom(GRID, 20000.0)
    image = mr_surrogate(phantom.label_id, phantom.label_names, NEMA_IQ_MR_INTENSITIES, GRID, blur_fwhm_mm=0, snr=None)
    for name in phantom.label_names:
        assert np.all(image[phantom.label_mask(name)] == np.float32(NEMA_IQ_MR_INTENSITIES[name])), name
    assert np.all(image[phantom.label_mask("rod_4mm")] == image[phantom.label_mask("pmma")][0])
    assert np.all(image[phantom.label_mask("water_insert")] == image[phantom.label_mask("hot")][0])


def test_a_label_without_intensity_is_rejected():
    label_id = np.ones((1, 2, 2), dtype=np.int16)
    with pytest.raises(ValueError, match="pmma"):
        mr_surrogate(label_id, ("air", "pmma"), {"air": 0.0}, GRID, blur_fwhm_mm=0, snr=None)


def test_blur_has_the_requested_fwhm():
    """A single bright voxel spreads into a Gaussian whose variance along x is sigma^2 = (FWHM / 2.355)^2."""
    grid = ImageGrid(num_voxels_zyx=(1, 1, 61), voxel_size_mm_zyx=(1.0, 1.0, 1.0))
    label_id = np.ones((1, 1, 61), dtype=np.int16)
    label_id[0, 0, 30] = 2
    image = mr_surrogate(label_id, ("air", "hot"), {"air": 0.0, "hot": 1.0}, grid, blur_fwhm_mm=3.0, snr=None)
    x_mm = np.arange(61) - 30.0
    profile = image[0, 0].astype(float)
    variance_mm2 = (profile * x_mm**2).sum() / profile.sum()
    assert variance_mm2 == pytest.approx((3.0 / FWHM_PER_SIGMA) ** 2, rel=0.01)


def test_rician_noise_lifts_a_zero_signal_to_sigma_sqrt_pi_over_2():
    """Where the signal is 0, the magnitude of two Gaussian channels is Rayleigh-distributed: mean sigma sqrt(pi/2)."""
    label_id = np.ones((1, 200, 200), dtype=np.int16)
    image = mr_surrogate(label_id, ("air",), {"air": 0.0}, GRID, blur_fwhm_mm=0, snr=10.0, rng=np.random.default_rng(0))
    sigma = 1 / 10.0
    assert image.mean() == pytest.approx(sigma * math.sqrt(math.pi / 2), rel=0.01)
