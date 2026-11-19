"""MLEM: the worked example of docs/mlem.md, and the properties every iteration must have."""

import numpy as np
import pytest

from pet_sim_recon.image_grid import ImageGrid
from pet_sim_recon.mcgpu.input_file import SinogramSettings
from pet_sim_recon.mcgpu.sinogram_layout import summed_bin_lines
from pet_sim_recon.recon.mlem import mlem, sensitivity_image
from pet_sim_recon.recon.projector import Projector
from pet_sim_recon.recon.sinogram_lines import SinogramLines
from pet_sim_recon.scanner import Scanner

# Two voxels side by side along x, 1 mm each. Bin 0 crosses both, bin 1 only voxel 0: A = [[1, 1], [1, 0]].
TWO_VOXELS = ImageGrid(num_voxels_zyx=(1, 1, 2), voxel_size_mm_zyx=(1.0, 1.0, 1.0))
TWO_LINES = SinogramLines(
    start_mm_zyx=np.array([[0.0, 0.0, -9.0], [0.0, -9.0, -0.5]]),
    end_mm_zyx=np.array([[0.0, 0.0, 9.0], [0.0, 9.0, -0.5]]),
    oblique_planes=np.array([], dtype=int),
    sinogram_shape=(2, 1, 1),
)
MEASURED = np.array([3.0, 1.0]).reshape(2, 1, 1)  # from the image (1, 2)

SCANNER = Scanner(radius_mm=10.0, axial_fov_mm=4.0, num_rings=4, num_crystals_per_ring=16)
SINOGRAM = SinogramSettings(8, 9, 7, 3, 1)
GRID = ImageGrid(num_voxels_zyx=(4, 10, 10), voxel_size_mm_zyx=(1.0, 1.0, 1.0))


def small_problem(seed=0):
    """A small scanner, random line factors, and Poisson data from a random image."""
    lines = summed_bin_lines(SCANNER, SINOGRAM)
    projector = Projector(lines, GRID)
    rng = np.random.default_rng(seed)
    factors = rng.uniform(0.5, 1.0, len(lines.start_mm_zyx)).astype(np.float32)
    truth = rng.uniform(1.0, 5.0, GRID.num_voxels_zyx)
    return projector, factors, rng, rng.poisson(projector.forward(truth, factors)).astype(float)


def poisson_log_likelihood(measured, expected):
    return float((measured * np.log(expected) - expected).sum())


def test_worked_example_step_by_step():
    projector = Projector(TWO_LINES, TWO_VOXELS)
    images = []
    mlem(projector, MEASURED, 1.0, num_iterations=2, on_iteration=lambda _, image: images.append(image.copy()))
    np.testing.assert_allclose(images[0][0, 0], [1.25, 1.5], rtol=1e-5)
    np.testing.assert_allclose(images[1][0, 0], [13 / 11, 18 / 11], rtol=1e-5)


def test_worked_example_converges_to_the_answer():
    image = mlem(Projector(TWO_LINES, TWO_VOXELS), MEASURED, 1.0, num_iterations=500)
    np.testing.assert_allclose(image[0, 0], [1.0, 2.0], rtol=1e-3)


def test_every_iteration_matches_the_total_counts():
    projector, factors, _, measured = small_problem()
    def check(_, image):
        assert projector.forward(image, factors).sum() == pytest.approx(measured.sum(), rel=1e-4)
    mlem(projector, measured, factors, num_iterations=5, on_iteration=check)


def test_every_iteration_raises_the_likelihood_also_with_a_background():
    projector, factors, rng, measured = small_problem(seed=1)
    background = rng.uniform(0.1, 1.0, measured.shape).astype(np.float32)
    measured = measured + rng.poisson(background)
    values = []
    def record(_, image):
        values.append(poisson_log_likelihood(measured, projector.forward(image, factors) + background))
    image = mlem(projector, measured, factors, background=background, num_iterations=10, on_iteration=record)
    assert np.all(np.diff(values) > -1e-5 * np.abs(values[:-1]))  # float32 rounding
    assert image.min() >= 0


def test_voxels_the_scanner_cannot_see_stay_zero():
    # only voxel 0 of the two is on a line
    lines = SinogramLines(TWO_LINES.start_mm_zyx[1:], TWO_LINES.end_mm_zyx[1:], np.array([], dtype=int), (1, 1, 1))
    projector = Projector(lines, TWO_VOXELS)
    np.testing.assert_allclose(sensitivity_image(projector, 1.0)[0, 0], [1.0, 0.0])
    image = mlem(projector, np.array([[[4.0]]]), 1.0, num_iterations=3)
    np.testing.assert_allclose(image[0, 0], [4.0, 0.0], rtol=1e-5)
