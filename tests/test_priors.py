"""Pair weights and penalties: hand-worked values, and gradients against finite differences."""

import numpy as np
import pytest

from pet_sim_recon.image_grid import ImageGrid
from pet_sim_recon.recon.priors import (
    ALL_OFFSETS,
    PAIR_OFFSETS,
    PairWeights,
    QuadraticPenalty,
    RelativeDifferencePenalty,
    bowsher_weights,
    pair_slices,
    uniform_weights,
)

CUBE = ImageGrid(num_voxels_zyx=(3, 4, 5), voxel_size_mm_zyx=(2.0, 1.0, 1.0))
LINE = ImageGrid(num_voxels_zyx=(1, 1, 4), voxel_size_mm_zyx=(1.0, 1.0, 1.0))  # four voxels in a row along x
ALONG_X = (0, 0, 1)


def weights_along_x(values):
    """Weights for the line of 4 voxels: values[j] is omega between voxel j and j + 1."""
    weights = {offset: np.zeros(LINE.num_voxels_zyx, dtype=np.float32) for offset in PAIR_OFFSETS}
    weights[ALONG_X][0, 0, : len(values)] = values
    return PairWeights(weights, LINE.num_voxels_zyx)


def numerical_gradient(penalty, image, step=1e-4):
    gradient = np.zeros_like(image, dtype=float)
    for index in np.ndindex(image.shape):
        up, down = image.astype(float), image.astype(float)
        up[index] += step
        down[index] -= step
        gradient[index] = (penalty.value(up) - penalty.value(down)) / (2 * step)
    return gradient


def test_every_neighbor_pair_is_visited_once():
    assert len(ALL_OFFSETS) == 26 and len(PAIR_OFFSETS) == 13
    assert not any(tuple(-step for step in offset) in PAIR_OFFSETS for offset in PAIR_OFFSETS)
    first, second = pair_slices((0, 1, -1), (3, 4, 5))
    assert first == (slice(0, 3), slice(0, 3), slice(1, 5))
    assert second == (slice(0, 3), slice(1, 4), slice(0, 4))


def test_quadratic_penalty_by_hand():
    # pairs (0,1) with weight 1 and (1,2) with weight 2; x = 1, 3, 6, 0
    penalty = QuadraticPenalty(weights_along_x([1.0, 2.0]))
    image = np.array([[[1.0, 3.0, 6.0, 0.0]]])
    assert penalty.value(image) == pytest.approx(0.5 * (1 * 2**2 + 2 * 3**2))
    np.testing.assert_allclose(penalty.gradient(image)[0, 0], [-2.0, 2.0 - 6.0, 6.0, 0.0])


def test_relative_difference_penalty_by_hand():
    # one pair x = 1, 3, gamma = 2: d^2 / (x_j + x_k + 2 |d|) = 4 / 8
    penalty = RelativeDifferencePenalty(weights_along_x([1.0]), gamma=2.0)
    image = np.array([[[1.0, 3.0, 0.0, 0.0]]])
    assert penalty.value(image) == pytest.approx(0.5)


@pytest.mark.parametrize("make_penalty", [QuadraticPenalty, RelativeDifferencePenalty])
def test_gradient_matches_finite_differences(make_penalty):
    image = np.random.default_rng(0).uniform(1.0, 5.0, CUBE.num_voxels_zyx)
    penalty = make_penalty(uniform_weights(CUBE))
    np.testing.assert_allclose(penalty.gradient(image.astype(np.float32)), numerical_gradient(penalty, image), rtol=1e-3, atol=1e-3)


def test_relative_difference_penalty_handles_zero_voxels():
    penalty = RelativeDifferencePenalty(weights_along_x([1.0, 1.0, 1.0]))
    image = np.zeros((1, 1, 4), dtype=np.float32)
    assert penalty.value(image) == 0
    np.testing.assert_array_equal(penalty.gradient(image), 0)


def test_de_pierro_terms_by_hand():
    penalty = QuadraticPenalty(weights_along_x([1.0, 2.0]))
    total_weight, pair_sum = penalty.de_pierro_terms(np.array([[[1.0, 3.0, 6.0, 0.0]]]))
    np.testing.assert_allclose(total_weight[0, 0], [1.0, 3.0, 2.0, 0.0])
    np.testing.assert_allclose(pair_sum[0, 0], [4.0, 4.0 + 18.0, 18.0, 0.0])


def test_uniform_weights_fall_with_distance():
    weights = uniform_weights(CUBE).weights  # voxels are 2 x 1 x 1 mm (z, y, x)
    assert weights[(0, 0, 1)][0, 0, 0] == pytest.approx(1.0)
    assert weights[(1, 0, 0)][0, 0, 0] == pytest.approx(0.5)
    assert weights[(0, 1, 1)][0, 0, 0] == pytest.approx(1 / np.sqrt(2))


def test_bowsher_weights_by_hand():
    # MR along a line: 0, 0.1, 5, 5.2. With one neighbor: 0 picks 1, 1 picks 0, 2 picks 3, 3 picks 2.
    # So pairs (0,1) and (2,3) get (1 + 1) / 2 = 1; the pair (1,2) across the MR edge gets 0.
    weights = bowsher_weights(np.array([[[0.0, 0.1, 5.0, 5.2]]]), 1, LINE).weights[ALONG_X]
    np.testing.assert_allclose(weights[0, 0, :3], [1.0, 0.0, 1.0])


def test_bowsher_one_way_choice_gives_half_weight():
    # MR 0, 3, 3.5, 10: voxels 1 and 2 pick each other. Voxel 0 picks 1 and voxel 3 picks 2, neither picked back.
    weights = bowsher_weights(np.array([[[0.0, 3.0, 3.5, 10.0]]]), 1, LINE).weights[ALONG_X]
    np.testing.assert_allclose(weights[0, 0, :3], [0.5, 1.0, 0.5])


def test_averaged_weights_give_the_same_penalty_as_the_one_way_sum():
    # R_Bowsher = 1/2 sum_j sum_k w_jk M(x_j, x_k) with one-way w_jk (Schramm et al. 2018 eq. 10, up to a constant)
    rng = np.random.default_rng(3)
    mr = rng.random(CUBE.num_voxels_zyx)
    image = rng.uniform(1.0, 5.0, CUBE.num_voxels_zyx)
    num_neighbors = 4
    ranking = np.full((26,) + CUBE.num_voxels_zyx, np.inf)
    for n, offset in enumerate(ALL_OFFSETS):
        first, second = pair_slices(offset, CUBE.num_voxels_zyx)
        ranking[n][first] = np.abs(mr[first] - mr[second])
    picked = np.zeros(ranking.shape, dtype=bool)
    np.put_along_axis(picked, np.argsort(ranking, axis=0)[:num_neighbors], True, axis=0)

    one_way_sum = 0.0
    nearest = min(CUBE.voxel_size_mm_zyx)
    for n, offset in enumerate(ALL_OFFSETS):
        first, second = pair_slices(offset, CUBE.num_voxels_zyx)
        distance_factor = nearest / np.linalg.norm(np.multiply(offset, CUBE.voxel_size_mm_zyx))
        one_way_sum += (picked[n][first] * distance_factor * 0.5 * (image[first] - image[second]) ** 2).sum()
    penalty = QuadraticPenalty(bowsher_weights(mr, num_neighbors, CUBE))
    assert penalty.value(image) == pytest.approx(0.5 * one_way_sum, rel=1e-6)


def test_flat_mr_ties_are_reproducible():
    flat = np.ones(CUBE.num_voxels_zyx)
    first_run = bowsher_weights(flat, 6, CUBE, seed=1).weights
    second_run = bowsher_weights(flat, 6, CUBE, seed=1).weights
    for offset in PAIR_OFFSETS:
        np.testing.assert_array_equal(first_run[offset], second_run[offset])
