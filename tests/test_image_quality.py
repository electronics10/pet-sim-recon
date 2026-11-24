"""Image-quality metrics against known answers (docs/image-quality.md)."""

import math

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from pet_sim_recon.image_grid import ImageGrid
from pet_sim_recon.image_quality import crc, ensemble_summary, nema_iq_metrics, rc_mean
from pet_sim_recon.mr_surrogate import FWHM_PER_SIGMA
from pet_sim_recon.phantoms import nema_iq_phantom

ACTIVITY_BQ_PER_ML = 20000.0
# 1 mm voxels with even counts: no voxel center sits on a section boundary.
GRID = ImageGrid(num_voxels_zyx=(56, 40, 40), voxel_size_mm_zyx=(1.0, 1.0, 1.0))


@pytest.fixture(scope="module")
def phantom():
    return nema_iq_phantom(GRID, ACTIVITY_BQ_PER_ML)


def test_the_true_image_scores_perfectly(phantom):
    metrics = nema_iq_metrics(phantom.voxel_object.activity_Bq_per_mL, phantom, ACTIVITY_BQ_PER_ML)
    assert metrics["pct_std"] == 0
    assert metrics["uniform_over_true"] == 1
    for d in (2, 3, 4, 5):  # on 1 mm voxels no voxel center falls inside the 1 mm rod, so it is left out
        assert metrics[f"rc_mean_{d}mm"] == 1
        assert metrics[f"rc_nema_{d}mm"] == 1
    assert metrics["sor_water"] == metrics["sor_air"] == 0


def test_crc_worked_example():
    """docs/image-quality.md: a_H = 4, a_B = 1, measured C_H = 3.0, C_B = 1.1 gives CRC = 0.576."""
    image = np.array([[[3.0, 1.1]]])
    hot, background = image == 3.0, image == 1.1
    assert crc(image, hot, background, 4.0, 1.0) == pytest.approx((3.0 / 1.1 - 1) / 3)
    assert crc(image, hot, background, 4.0, 1.0) == pytest.approx(0.576, abs=0.001)


@pytest.mark.parametrize("diameter_mm", [1.0, 2.0, 3.0, 5.0])
def test_partial_volume_curve_of_a_blurred_disk(diameter_mm):
    """A disk of activity 1, blurred by a 2 mm-FWHM Gaussian, keeps 1 - exp(-r^2 / (2 sigma^2)) at its center.

    At d = FWHM that is exactly 0.5. A fine grid (0.02 mm) makes the disk's pixelated edge negligible.
    """
    fwhm_mm, pixel_mm, half_width = 2.0, 0.02, 300
    coordinate_mm = np.arange(-half_width, half_width + 1) * pixel_mm
    y_mm, x_mm = np.meshgrid(coordinate_mm, coordinate_mm, indexing="ij")
    disk = (x_mm**2 + y_mm**2 <= (diameter_mm / 2) ** 2).astype(float)[None]
    sigma_mm = fwhm_mm / FWHM_PER_SIGMA
    blurred = gaussian_filter(disk, (0, sigma_mm / pixel_mm, sigma_mm / pixel_mm))
    center = np.zeros(blurred.shape, bool)
    center[0, half_width, half_width] = True
    expected = 1 - math.exp(-((diameter_mm / 2) ** 2) / (2 * sigma_mm**2))
    assert rc_mean(blurred, center, 1.0) == pytest.approx(expected, abs=0.005)


def test_nema_maximum_rises_with_noise_and_the_mean_does_not(phantom):
    """Independent Gaussian noise of 20 % on the true image. The rod-core mean stays at 1. NEMA's maximum, over
    the slab-averaged pixels at the rod's level, rises by E[max of n standard normals] x 20 % / sqrt(slices).
    The PMMA pixels around the rod sit near 0 and never win the maximum."""
    truth = phantom.voxel_object.activity_Bq_per_mL
    noise_fraction, num_draws, diameter_mm = 0.2, 200, 3.0
    rng = np.random.default_rng(1)
    rc_means, rc_nemas = [], []
    for _ in range(num_draws):
        noisy = truth + noise_fraction * ACTIVITY_BQ_PER_ML * rng.standard_normal(truth.shape)
        metrics = nema_iq_metrics(noisy, phantom, ACTIVITY_BQ_PER_ML)
        rc_means.append(metrics["rc_mean_3mm"])
        rc_nemas.append(metrics["rc_nema_3mm"])

    standard_error = lambda values: np.std(values, ddof=1) / math.sqrt(len(values))
    assert abs(np.mean(rc_means) - 1) < 4 * standard_error(rc_means)

    slab = phantom.rods_slab()
    rod_pixels = int((phantom.label_mask("rod_3mm")[slab][0] & phantom.rod_roi_yx(diameter_mm)).sum())
    expected_max = rng.standard_normal((200_000, rod_pixels)).max(axis=1).mean()
    expected_rc = 1 + expected_max * noise_fraction / math.sqrt(slab.sum())
    # The uniform mean in the denominator carries its own small noise, so allow 4 standard errors plus 0.5 %.
    assert abs(np.mean(rc_nemas) - expected_rc) < 4 * standard_error(rc_nemas) + 0.005
    assert np.mean(rc_nemas) > 1.05


def test_ensemble_summary():
    summary = ensemble_summary([{"rc": 1.0, "noise": 4.0}, {"rc": 3.0, "noise": 4.0}])
    assert summary["rc"] == {"mean": 2.0, "sd": pytest.approx(math.sqrt(2))}
    assert summary["noise"] == {"mean": 4.0, "sd": 0.0}
