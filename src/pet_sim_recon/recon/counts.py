"""Tools that change the number of counts in a measured sinogram."""

import numpy as np


def binomial_thin(counts: np.ndarray, keep_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Keep each count with probability keep_fraction. Poisson(mean) data become Poisson(keep_fraction * mean)."""
    if not 0 <= keep_fraction <= 1:
        raise ValueError(f"keep_fraction must be in [0, 1], got {keep_fraction}")
    return rng.binomial(counts, keep_fraction)