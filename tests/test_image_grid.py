"""Image grid geometry agrees with MCGPU-PET's voxel frame, shifted to the scanner center."""

import warnings

import numpy as np
import pytest

from pet_sim_recon.image_grid import ImageGrid, check_fits_scanner
from pet_sim_recon.scanner import Scanner

# The prototype's default scanner and grid (mcgpu-pet-wrapper templates/template.json), in (z, y, x).
SCANNER = Scanner(radius_mm=52.5, axial_fov_mm=150.0, num_rings=75, num_crystals_per_ring=320)
PROTOTYPE_GRID = ImageGrid(num_voxels_zyx=(149, 80, 80), voxel_size_mm_zyx=(1.0, 1.0, 1.0))

# A deliberately uneven grid, so that a mixed-up axis shows up.
UNEVEN_GRID = ImageGrid(num_voxels_zyx=(5, 4, 3), voxel_size_mm_zyx=(3.0, 2.0, 1.5))


def mcgpu_voxel_centers_mm(num_voxels, voxel_size_mm):
    """MCGPU-PET source(): position = (blockIdx + r) / inv_voxel_size; r = 0.5 gives the center."""
    return (np.arange(num_voxels) + 0.5) * voxel_size_mm


def test_ten_voxels_of_two_mm():
    """The cube from issue #2: centers at -9, -7, ..., 9 mm."""
    grid = ImageGrid(num_voxels_zyx=(10, 10, 10), voxel_size_mm_zyx=(2.0, 2.0, 2.0))
    expected = np.arange(-9.0, 10.0, 2.0)
    for centers in grid.voxel_centers_mm_zyx():
        np.testing.assert_allclose(centers, expected)
    assert grid.first_voxel_center_mm_zyx == (-9.0, -9.0, -9.0)


def test_centers_are_mcgpu_centers_shifted_by_half_the_extent():
    centers_zyx = UNEVEN_GRID.voxel_centers_mm_zyx()
    for centers, n, d, extent in zip(
        centers_zyx, UNEVEN_GRID.num_voxels_zyx, UNEVEN_GRID.voxel_size_mm_zyx, UNEVEN_GRID.extent_mm_zyx
    ):
        np.testing.assert_allclose(centers, mcgpu_voxel_centers_mm(n, d) - extent / 2)


def test_centers_are_symmetric_about_zero():
    for centers in UNEVEN_GRID.voxel_centers_mm_zyx():
        np.testing.assert_allclose(centers, -centers[::-1])


def test_uneven_grid_keeps_its_axes_apart():
    assert UNEVEN_GRID.extent_mm_zyx == (15.0, 8.0, 4.5)
    assert UNEVEN_GRID.first_voxel_center_mm_zyx == (-6.0, -3.0, -1.5)


def test_prototype_grid_warns_about_its_corners():
    with pytest.warns(UserWarning, match="corners"):
        check_fits_scanner(PROTOTYPE_GRID, SCANNER)


def test_grid_inside_the_cylinder_gives_no_warning():
    grid = ImageGrid(num_voxels_zyx=(149, 70, 70), voxel_size_mm_zyx=(1.0, 1.0, 1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # turn any warning into a failure
        check_fits_scanner(grid, SCANNER)


def test_grid_wider_than_the_bore_is_rejected():
    grid = ImageGrid(num_voxels_zyx=(149, 120, 120), voxel_size_mm_zyx=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        check_fits_scanner(grid, SCANNER)


def test_invalid_grid_is_rejected():
    with pytest.raises(ValueError):
        ImageGrid(num_voxels_zyx=(10, 10), voxel_size_mm_zyx=(1.0, 1.0, 1.0))