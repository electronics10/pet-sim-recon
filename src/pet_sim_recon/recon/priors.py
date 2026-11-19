"""Penalties on pairs of neighboring voxels, and the weights of those pairs (docs/map-em.md)."""

import itertools
import math

import numpy as np

from pet_sim_recon.image_grid import ImageGrid

ALL_OFFSETS = tuple(offset for offset in itertools.product((-1, 0, 1), repeat=3) if offset != (0, 0, 0))
# One offset of each opposite pair, so every pair of neighboring voxels is visited once.
PAIR_OFFSETS = tuple(offset for offset in ALL_OFFSETS if offset > (0, 0, 0))


def pair_slices(offset, shape):
    """Slices selecting voxels j and their neighbors k = j + offset, wherever both lie in the image."""
    first = tuple(slice(max(0, -step), size - max(0, step)) for step, size in zip(offset, shape))
    second = tuple(slice(max(0, step), size - max(0, -step)) for step, size in zip(offset, shape))
    return first, second


def _distance_mm(offset, grid: ImageGrid) -> float:
    return math.dist(np.multiply(offset, grid.voxel_size_mm_zyx), (0, 0, 0))


class PairWeights:
    """The weight omega of every neighboring pair: weights[offset][j] belongs to the pair (j, j + offset)."""

    def __init__(self, weights: dict, shape: tuple[int, int, int], xp=np):
        self.weights = weights
        self.shape = tuple(shape)
        self.xp = xp

    def pairs(self):
        """For every offset: the slices of j and k, and the weights of those pairs."""
        for offset, weight in self.weights.items():
            first, second = pair_slices(offset, self.shape)
            yield first, second, weight[first]

    def restricted_to(self, support) -> "PairWeights":
        """Drop the pairs with a voxel outside support, so held-at-zero voxels don't pull on their neighbors."""
        restricted = {}
        for offset, weight in self.weights.items():
            first, second = pair_slices(offset, self.shape)
            kept = self.xp.zeros_like(weight)
            kept[first] = weight[first] * (support[first] & support[second])
            restricted[offset] = kept
        return PairWeights(restricted, self.shape, self.xp)

    def total_per_voxel(self):
        """Omega_j: the sum of omega_jk over the neighbors k of each voxel j."""
        total = self.xp.zeros(self.shape, dtype=self.xp.float32)
        for first, second, weight in self.pairs():
            total[first] += weight
            total[second] += weight
        return total


def uniform_weights(grid: ImageGrid, xp=np) -> PairWeights:
    """omega_jk = nearest-neighbor distance / distance(j, k), for all 26 neighbors."""
    nearest_mm = min(grid.voxel_size_mm_zyx)
    shape = grid.num_voxels_zyx
    weights = {
        offset: xp.full(shape, nearest_mm / _distance_mm(offset, grid), dtype=xp.float32) for offset in PAIR_OFFSETS
    }
    return PairWeights(weights, shape, xp)


def bowsher_weights(mr_image: np.ndarray, num_neighbors: int, grid: ImageGrid, xp=np, seed: int = 0) -> PairWeights:
    """Bowsher's weights: each voxel picks its num_neighbors neighbors closest to it in the MR image.

    With w_jk = 1 if j picks k, the pair weight is (w_jk + w_kj) / 2, times the distance factor of uniform_weights.
    Ties (a flat MR image) go to the nearer neighbor, then at random.
    """
    if not 1 <= num_neighbors <= len(ALL_OFFSETS):
        raise ValueError(f"num_neighbors must be in 1..{len(ALL_OFFSETS)}, got {num_neighbors}")
    shape = grid.num_voxels_zyx
    mr_image = np.asarray(mr_image, dtype=float)
    mr_difference = np.full((len(ALL_OFFSETS),) + shape, np.inf)  # inf: that neighbor is outside the image
    for n, offset in enumerate(ALL_OFFSETS):
        first, second = pair_slices(offset, shape)
        mr_difference[n][first] = np.abs(mr_image[first] - mr_image[second])

    # Tie-breaks, far smaller than any real MR difference.
    distances = np.array([_distance_mm(offset, grid) for offset in ALL_OFFSETS])
    distance_rank = distances.argsort(kind="stable").argsort() / len(ALL_OFFSETS)
    random_part = np.random.default_rng(seed).random(mr_difference.shape) / len(ALL_OFFSETS)
    tie_scale = 1e-9 * max(np.ptp(mr_image), 1e-30)
    ranking = mr_difference + tie_scale * (distance_rank[:, None, None, None] + random_part)

    picked = np.zeros(ranking.shape, dtype=bool)
    np.put_along_axis(picked, np.argpartition(ranking, num_neighbors - 1, axis=0)[:num_neighbors], True, axis=0)
    picked &= np.isfinite(mr_difference)

    nearest_mm = min(grid.voxel_size_mm_zyx)
    weights = {}
    for offset in PAIR_OFFSETS:
        first, second = pair_slices(offset, shape)
        j_picks_k = picked[ALL_OFFSETS.index(offset)][first]
        k_picks_j = picked[ALL_OFFSETS.index(tuple(-step for step in offset))][second]
        weight = np.zeros(shape, dtype=np.float32)
        weight[first] = (j_picks_k.astype(np.float32) + k_picks_j) / 2 * nearest_mm / _distance_mm(offset, grid)
        weights[offset] = xp.asarray(weight)
    return PairWeights(weights, shape, xp)


class QuadraticPenalty:
    """R(x) = 1/2 * sum over pairs of omega_jk (x_j - x_k)^2."""

    def __init__(self, pair_weights: PairWeights):
        self.pair_weights = pair_weights
        self.xp = pair_weights.xp

    def restricted_to(self, support):
        return QuadraticPenalty(self.pair_weights.restricted_to(support))

    def value(self, image) -> float:
        return sum(
            0.5 * float(self.xp.sum(weight * (image[first] - image[second]) ** 2, dtype=self.xp.float64))
            for first, second, weight in self.pair_weights.pairs()
        )

    def gradient(self, image):
        gradient = self.xp.zeros_like(image)
        for first, second, weight in self.pair_weights.pairs():
            pull = weight * (image[first] - image[second])
            gradient[first] += pull
            gradient[second] -= pull
        return gradient

    def de_pierro_terms(self, image):
        """Omega_j = sum_k omega_jk, and S_j = sum_k omega_jk (x_j + x_k): docs/map-em.md."""
        pair_sum = self.xp.zeros_like(image)
        for first, second, weight in self.pair_weights.pairs():
            term = weight * (image[first] + image[second])
            pair_sum[first] += term
            pair_sum[second] += term
        return self.pair_weights.total_per_voxel(), pair_sum


class RelativeDifferencePenalty:
    """R(x) = sum over pairs of omega_jk (x_j - x_k)^2 / (x_j + x_k + gamma |x_j - x_k|). Nuyts et al. 2002."""

    def __init__(self, pair_weights: PairWeights, gamma: float = 2.0):
        self.pair_weights = pair_weights
        self.gamma = gamma
        self.xp = pair_weights.xp

    def restricted_to(self, support):
        return RelativeDifferencePenalty(self.pair_weights.restricted_to(support), self.gamma)

    def _denominator(self, x_j, x_k):
        """x_j + x_k + gamma |x_j - x_k|, and whether it is positive. Two zero voxels contribute nothing."""
        denominator = x_j + x_k + self.gamma * self.xp.abs(x_j - x_k)
        positive = denominator > 0
        return self.xp.where(positive, denominator, 1), positive

    def value(self, image) -> float:
        total = 0.0
        for first, second, weight in self.pair_weights.pairs():
            x_j, x_k = image[first], image[second]
            denominator, positive = self._denominator(x_j, x_k)
            terms = self.xp.where(positive, weight * (x_j - x_k) * ((x_j - x_k) / denominator), 0)
            total += float(self.xp.sum(terms, dtype=self.xp.float64))
        return total

    def gradient(self, image):
        """Per pair, with d = x_j - x_k and D the denominator: dR/dx_j = omega d (x_j + 3 x_k + gamma |d|) / D^2."""
        xp = self.xp
        gradient = xp.zeros_like(image)
        for first, second, weight in self.pair_weights.pairs():
            x_j, x_k = image[first], image[second]
            difference = x_j - x_k
            denominator, positive = self._denominator(x_j, x_k)
            ratio = difference / denominator  # dividing twice avoids float32 underflow of D^2
            edge = self.gamma * xp.abs(difference)
            gradient[first] += xp.where(positive, weight * ratio * ((x_j + 3 * x_k + edge) / denominator), 0)
            gradient[second] -= xp.where(positive, weight * ratio * ((3 * x_j + x_k + edge) / denominator), 0)
        return gradient


def reference_beta(penalty, sensitivity, reference_image) -> float:
    """The beta at which penalty and likelihood curve about equally at a typical voxel. A heuristic scale.

    Likelihood curvature ~ s / x; quadratic penalty ~ Omega; relative difference ~ Omega / x. Averages are over
    voxels above 10 % of the reference image's maximum.
    """
    xp = penalty.xp
    typical = reference_image > 0.1 * reference_image.max()
    mean_sensitivity = float(xp.mean(sensitivity[typical]))
    mean_weight = float(xp.mean(penalty.pair_weights.total_per_voxel()[typical]))
    if isinstance(penalty, QuadraticPenalty):
        return mean_sensitivity / (float(xp.mean(reference_image[typical])) * mean_weight)
    return mean_sensitivity / mean_weight