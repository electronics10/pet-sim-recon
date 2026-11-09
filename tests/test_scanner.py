"""Scanner geometry agrees with how the MCGPU-PET kernel bins photon hits."""

import numpy as np
import pytest

from pet_sim_recon.scanner import Scanner

# The prototype's default scanner (mcgpu-pet-wrapper templates/template.json).
SCANNER = Scanner(radius_mm=52.5, axial_fov_mm=150.0, num_rings=75, num_crystals_per_ring=320)


def kernel_crystal_index(x_mm, y_mm, num_crystals):
    """MCGPU-PET_kernel.cu line 1749: ix = floor((phi / 2pi + 0.5) * NCRYSTALS)."""
    phi = np.arctan2(y_mm, x_mm)
    return np.floor((phi / (2 * np.pi) + 0.5) * num_crystals).astype(int)


def kernel_ring_index(z_mm, axial_fov_mm, num_rings):
    """MCGPU-PET_kernel.cu line 1751: iz = floor((z / FOVZ + 0.5) * NROWS)."""
    return np.floor((z_mm / axial_fov_mm + 0.5) * num_rings).astype(int)


def test_crystal_centers_for_four_crystals():
    """The worked example on the whiteboard: centres at -3pi/4, -pi/4, pi/4, 3pi/4."""
    scanner = Scanner(radius_mm=1.0, axial_fov_mm=1.0, num_rings=1, num_crystals_per_ring=4)
    expected = np.pi * np.array([-0.75, -0.25, 0.25, 0.75])
    np.testing.assert_allclose(scanner.crystal_center_angles_rad(), expected)


@pytest.mark.parametrize("offset_in_pitches", [-0.49, 0.0, 0.49])
def test_hits_inside_each_crystal_bin_back_to_it(offset_in_pitches):
    """A hit anywhere within a crystal, not just at its centre, gets that crystal's index."""
    phi = SCANNER.crystal_center_angles_rad() + offset_in_pitches * SCANNER.crystal_pitch_rad
    x_mm = SCANNER.radius_mm * np.cos(phi)
    y_mm = SCANNER.radius_mm * np.sin(phi)
    index = kernel_crystal_index(x_mm, y_mm, SCANNER.num_crystals_per_ring)
    np.testing.assert_array_equal(index, np.arange(SCANNER.num_crystals_per_ring))


@pytest.mark.parametrize("offset_in_pitches", [-0.49, 0.0, 0.49])
def test_hits_inside_each_ring_bin_back_to_it(offset_in_pitches):
    z_mm = SCANNER.ring_center_z_mm() + offset_in_pitches * SCANNER.ring_pitch_mm
    index = kernel_ring_index(z_mm, SCANNER.axial_fov_mm, SCANNER.num_rings)
    np.testing.assert_array_equal(index, np.arange(SCANNER.num_rings))


def test_ring_centers_are_symmetric_about_zero():
    z_mm = SCANNER.ring_center_z_mm()
    np.testing.assert_allclose(z_mm, -z_mm[::-1])


def test_invalid_scanner_is_rejected():
    with pytest.raises(ValueError):
        Scanner(radius_mm=-52.5, axial_fov_mm=150.0, num_rings=75, num_crystals_per_ring=320)