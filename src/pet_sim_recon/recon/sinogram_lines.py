"""Which lines make up each sinogram bin: what a scanner's layout hands to the projector."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SinogramLines:
    """The lines that make up each bin of a (plane, view, radial bin) sinogram, end points in mm as (z, y, x).

    Every bin has line A. The bins of the planes in oblique_planes also have line B. The first
    prod(sinogram_shape) rows of start/end are the A lines in bin order, then come the B lines, in bin order.
    """

    start_mm_zyx: np.ndarray  # (num_lines, 3)
    end_mm_zyx: np.ndarray  # (num_lines, 3)
    oblique_planes: np.ndarray  # plane indices, increasing
    sinogram_shape: tuple[int, int, int]
