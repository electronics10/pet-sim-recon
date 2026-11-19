"""MAP-EM: it reduces to MLEM, climbs its objective, and solves a problem worked by hand."""

import math

import numpy as np
import pytest

from pet_sim_recon.image_grid import ImageGrid
from pet_sim_recon.mcgpu.input_file import SinogramSettings
from pet_sim_recon.mcgpu.sinogram_layout import summed_bin_lines
from pet_sim_recon.recon.map_em import map_em, objective
from pet_sim_recon.recon.mlem import mlem, sensitivity_image
from pet_sim_recon.recon.priors import (
    PAIR_OFFSETS,
    PairWeights,
    QuadraticPenalty,
    RelativeDifferencePenalty,
    bowsher_weights,
    reference_beta,
    uniform_weights,
)
from pet_sim_recon.recon.projector import Projector
from pet_sim_recon.recon.sinogram_lines import SinogramLines
from pet_sim_recon.scanner import Scanner

SCANNER = Scanner(radius_mm=10.0, axial_fov_mm=4.0, num_rings=4, num_crystals_per_ring=16)
SINOGRAM = SinogramSettings(8, 9, 7, 3, 1)
GRID = ImageGrid(num_voxels_zyx=(4, 10, 10), voxel_size_mm_zyx=(1.0, 1.0, 1.0))

# Two voxels along x, each seen by its own line along y: A is the identity, so sensitivity is 1 in both.
TWO_VOXELS = ImageGrid(num_voxels_zyx=(1, 1, 2), voxel_size_mm_zyx=(1.0, 1.0, 1.0))
TWO_LINES = SinogramLines(
    start_mm_zyx=np.array([[0.0, -9.0, -0.5], [0.0, -9.0, 0.5]]),
    end_mm_zyx=np.array([[0.0, 9.0, -0.5], [0.0, 9.0, 0.5]]),
    oblique_planes=np.array([], dtype=int),
    sinogram_shape=(2, 1, 1),
)


def small_problem(seed=0):
    projector = Projector(summed_bin_lines(SCANNER, SINOGRAM), GRID)
    rng = np.random.default_rng(seed)
    factors = rng.uniform(0.5, 1.0, len(projector.start_mm_zyx)).astype(np.float32)
    truth = rng.uniform(1.0, 5.0, GRID.num_voxels_zyx)
    background = rng.uniform(0.1, 0.5, projector.sinogram_shape).astype(np.float32)
    measured = rng.poisson(projector.forward(truth, factors) + background).astype(float)
    return projector, factors, background, measured, rng


def penalties(rng):
    mr = rng.random(GRID.num_voxels_zyx)
    return {
        "quadratic": QuadraticPenalty(uniform_weights(GRID)),
        "bowsher": QuadraticPenalty(bowsher_weights(mr, 6, GRID)),
        "relative difference": RelativeDifferencePenalty(uniform_weights(GRID)),
        "bowsher relative difference": RelativeDifferencePenalty(bowsher_weights(mr, 6, GRID)),
    }


@pytest.mark.parametrize("name", ["quadratic", "relative difference"])
def test_beta_zero_is_exactly_mlem(name):
    projector, factors, background, measured, rng = small_problem()
    penalty = penalties(rng)[name]
    image = map_em(projector, measured, factors, penalty, 0.0, background, num_iterations=7)
    expected = mlem(projector, measured, factors, background, num_iterations=7)
    np.testing.assert_allclose(image, expected, rtol=1e-4)  # threaded float32 sums differ in the last digits


@pytest.mark.parametrize("name", ["quadratic", "bowsher", "relative difference", "bowsher relative difference"])
def test_every_iteration_raises_the_objective(name):
    projector, factors, background, measured, rng = small_problem(seed=1)
    penalty = penalties(rng)[name]
    start = mlem(projector, measured, factors, background, num_iterations=3)
    beta = 0.3 * reference_beta(penalty, sensitivity_image(projector, factors), start)
    values = [objective(projector, start, measured, factors, penalty, beta, background)]
    def record(_, image):
        values.append(objective(projector, image, measured, factors, penalty, beta, background))
    image = map_em(projector, measured, factors, penalty, beta, background, 15, start, record)
    assert np.all(np.diff(values) > -1e-6 * np.abs(values[:-1]))  # float32 rounding
    assert values[-1] > values[0]
    assert image.min() >= 0


def test_de_pierro_converges_to_the_answer_worked_by_hand():
    # A = I, y = (1, 9), one pair with omega = 1, beta = 1/4. Stationarity (docs/map-em.md):
    #   1/x_0 - 1 + (x_1 - x_0)/4 = 0   and   9/x_1 - 1 - (x_1 - x_0)/4 = 0.
    # Solved here by Newton's method, independently of the code under test.
    x = np.array([1.5, 6.0])
    for _ in range(50):
        residual = np.array([1 / x[0] - 1 + (x[1] - x[0]) / 4, 9 / x[1] - 1 - (x[1] - x[0]) / 4])
        jacobian = np.array([[-1 / x[0] ** 2 - 0.25, 0.25], [0.25, -9 / x[1] ** 2 - 0.25]])
        x -= np.linalg.solve(jacobian, residual)
    weights = {offset: np.zeros(TWO_VOXELS.num_voxels_zyx, dtype=np.float32) for offset in PAIR_OFFSETS}
    weights[(0, 0, 1)][0, 0, 0] = 1.0
    penalty = QuadraticPenalty(PairWeights(weights, TWO_VOXELS.num_voxels_zyx))
    image = map_em(Projector(TWO_LINES, TWO_VOXELS), np.array([1.0, 9.0]).reshape(2, 1, 1), 1.0, penalty, 0.25,
                   num_iterations=500)
    np.testing.assert_allclose(image[0, 0], x, rtol=1e-3)


def test_first_de_pierro_step_by_hand():
    # The worked example of docs/map-em.md: from x = (1, 3), y = (1, 9), omega = 1, beta = 1/4.
    # c = (1, 9), s = (1, 1). Voxel 0: a = 1/2, b = 1 - (1 + 3) / 4 = 0, so x = sqrt(c / a) = sqrt(2).
    # Voxel 1: a = 1/2, b = 0, so x = sqrt(9 / (1/2)) = sqrt(18).
    weights = {offset: np.zeros(TWO_VOXELS.num_voxels_zyx, dtype=np.float32) for offset in PAIR_OFFSETS}
    weights[(0, 0, 1)][0, 0, 0] = 1.0
    penalty = QuadraticPenalty(PairWeights(weights, TWO_VOXELS.num_voxels_zyx))
    start = np.array([[[1.0, 3.0]]])
    image = map_em(Projector(TWO_LINES, TWO_VOXELS), np.array([1.0, 9.0]).reshape(2, 1, 1), 1.0, penalty, 0.25,
                   num_iterations=1, initial_image=start)
    np.testing.assert_allclose(image[0, 0], [math.sqrt(2), math.sqrt(18)], rtol=1e-5)


def test_a_strong_penalty_flattens_the_image():
    projector, factors, background, measured, rng = small_problem(seed=2)
    penalty = QuadraticPenalty(uniform_weights(GRID))
    start = mlem(projector, measured, factors, background, num_iterations=5)
    beta = reference_beta(penalty, sensitivity_image(projector, factors), start)
    weak = map_em(projector, measured, factors, penalty, 0.01 * beta, background, 30, start)
    strong = map_em(projector, measured, factors, penalty, 30 * beta, background, 30, start)
    support = start > 0
    assert strong[support].std() < weak[support].std()
