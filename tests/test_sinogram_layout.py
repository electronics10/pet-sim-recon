"""MCGPU-PET's transaxial binning: the kernel port, its inverse table, and each bin's line (theta, s)."""

import numpy as np
import pytest

from pet_sim_recon.mcgpu.input_file import SinogramSettings
from pet_sim_recon.mcgpu.sinogram_layout import (
    check_sinogram_fits_scanner,
    crystal_pair_table,
    kernel_transaxial_bin,
    line_angle_and_distance,
    nominal_view_angles_rad,
    radial_positions_mm,
)
from pet_sim_recon.scanner import Scanner

# The 8-crystal ring of docs/sinogram-layout.md. R = 1 mm, so s is in units of R.
SMALL_SCANNER = Scanner(radius_mm=1.0, axial_fov_mm=1.0, num_rings=1, num_crystals_per_ring=8)
SMALL_SINOGRAM = SinogramSettings(
    num_angular_bins=4, num_radial_bins=7, num_axial_slices=1, max_ring_difference=0, span=1
)

# Upstream's sample scanner (external/MCGPU-PET/sample_simulation/MCGPU-PET.in).
SAMPLE_SCANNER = Scanner(radius_mm=90.5, axial_fov_mm=126.56, num_rings=80, num_crystals_per_ring=336)
SAMPLE_SINOGRAM = SinogramSettings(
    num_angular_bins=168, num_radial_bins=147, num_axial_slices=159, max_ring_difference=79, span=11
)

# The table in docs/sinogram-layout.md: EIGHT_CRYSTAL_PAIRS[view][radial bin] = (a, b).
EIGHT_CRYSTAL_PAIRS = [
    [(5, 6), (5, 7), (4, 7), (0, 4), (0, 3), (1, 3), (1, 2)],
    [(6, 7), (0, 6), (0, 5), (1, 5), (1, 4), (2, 4), (2, 3)],
    [(0, 7), (1, 7), (1, 6), (2, 6), (2, 5), (3, 5), (3, 4)],
    [(0, 1), (0, 2), (2, 7), (3, 7), (3, 6), (4, 6), (4, 5)],
]


def test_worked_examples():
    """Crystals (0, 3) and (2, 6) from the docs. The order of the two crystals doesn't matter."""
    assert kernel_transaxial_bin(0, 3, num_crystals=8, num_angular_bins=4, num_radial_bins=7) == (0, 4)
    assert kernel_transaxial_bin(3, 0, num_crystals=8, num_angular_bins=4, num_radial_bins=7) == (0, 4)
    assert kernel_transaxial_bin(2, 6, num_crystals=8, num_angular_bins=4, num_radial_bins=7) == (2, 3)


def test_eight_crystal_table():
    np.testing.assert_array_equal(crystal_pair_table(SMALL_SCANNER, SMALL_SINOGRAM), EIGHT_CRYSTAL_PAIRS)


def test_sample_scanner_fills_every_bin():
    """crystal_pair_table raises if two pairs share a bin; here no bin is left empty either."""
    table = crystal_pair_table(SAMPLE_SCANNER, SAMPLE_SINOGRAM)
    assert (table >= 0).all()


def test_line_of_crystals_0_and_3():
    """Docs, worked example: a horizontal chord below the center, normal pointing down (-90 degrees)."""
    angles_rad = SMALL_SCANNER.crystal_center_angles_rad()
    theta_rad, s_mm = line_angle_and_distance(angles_rad[0], angles_rad[3], radius_mm=1.0)
    assert theta_rad == pytest.approx(-np.pi / 2)
    assert s_mm == pytest.approx(np.cos(3 * np.pi / 8))  # 0.383 R


def test_radial_positions_are_not_arc_corrected():
    """s = R sin(pi d / N): steps of 0.383 R near the center, 0.217 R at the edge."""
    s_mm = radial_positions_mm(SMALL_SCANNER, SMALL_SINOGRAM)
    np.testing.assert_allclose(s_mm, [-0.924, -0.707, -0.383, 0.0, 0.383, 0.707, 0.924], atol=1e-3)


@pytest.mark.parametrize(
    "scanner, sinogram", [(SMALL_SCANNER, SMALL_SINOGRAM), (SAMPLE_SCANNER, SAMPLE_SINOGRAM)], ids=["8", "336"]
)
def test_every_bin_line_sits_on_the_nominal_grid(scanner, sinogram):
    """The exact line of each bin's crystal pair has
    theta = nominal view angle, + half a view step at an even distance from the center bin (interleaving),
    s = radial_positions_mm, the same in every view."""
    table = crystal_pair_table(scanner, sinogram)
    angles_rad = scanner.crystal_center_angles_rad()
    theta_rad, s_mm = line_angle_and_distance(angles_rad[table[..., 0]], angles_rad[table[..., 1]], scanner.radius_mm)

    distance_from_center = np.arange(sinogram.num_radial_bins) - sinogram.num_radial_bins // 2
    half_view_step_rad = np.pi / sinogram.num_angular_bins / 2
    interleave_rad = np.where(distance_from_center % 2 == 0, half_view_step_rad, 0.0)
    expected_theta_rad = nominal_view_angles_rad(sinogram)[:, None] + interleave_rad[None, :]
    expected_s_mm = np.broadcast_to(radial_positions_mm(scanner, sinogram), s_mm.shape)  # same in every view

    np.testing.assert_allclose(theta_rad, expected_theta_rad, atol=1e-9)
    np.testing.assert_allclose(s_mm, expected_s_mm, atol=1e-9)


def test_angular_bins_must_be_half_the_crystals():
    """The kernel never checks the view index against num_angular_bins; a wrong value mis-bins silently."""
    with pytest.raises(ValueError, match="half"):
        check_sinogram_fits_scanner(SAMPLE_SCANNER, SinogramSettings(84, 147, 159, 79, 11))
