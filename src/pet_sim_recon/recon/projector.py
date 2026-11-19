"""Projector: line integrals of an image along the lines of each sinogram bin (docs/projector.md)."""

import math

import numpy as np
import parallelproj_core

from pet_sim_recon.image_grid import ImageGrid
from pet_sim_recon.recon.sinogram_lines import SinogramLines


class Projector:
    """Image (z, y, x) to sinogram and back. A bin gets the sum, over its lines, of line_factor * line integral.

    xp is numpy or cupy: where the lines live and the projections run.
    """

    def __init__(self, lines: SinogramLines, grid: ImageGrid, xp=np):
        self.xp = xp
        self.start_mm_zyx = xp.asarray(lines.start_mm_zyx, dtype=xp.float32)
        self.end_mm_zyx = xp.asarray(lines.end_mm_zyx, dtype=xp.float32)
        self.oblique_planes = xp.asarray(lines.oblique_planes)
        self.sinogram_shape = tuple(lines.sinogram_shape)
        self.image_shape = tuple(grid.num_voxels_zyx)
        self.image_origin_mm = xp.asarray(grid.first_voxel_center_mm_zyx, dtype=xp.float32)
        self.voxel_size_mm = xp.asarray(grid.voxel_size_mm_zyx, dtype=xp.float32)
        self.num_a_lines = math.prod(self.sinogram_shape)

    def line_integrals(self, image):
        """Integral of the image along every line (image units x mm), Joseph's method."""
        image = self.xp.ascontiguousarray(self.xp.asarray(image, dtype=self.xp.float32))
        values = self.xp.zeros(len(self.start_mm_zyx), dtype=self.xp.float32)
        parallelproj_core.joseph3d_fwd(
            self.start_mm_zyx, self.end_mm_zyx, image, self.image_origin_mm, self.voxel_size_mm, values
        )
        return values

    def forward(self, image, line_factors=1.0):
        """Sinogram: each bin sums line_factors * line integral over its lines."""
        values = self.line_integrals(image) * line_factors
        sinogram = values[: self.num_a_lines].reshape(self.sinogram_shape)
        sinogram[self.oblique_planes] += values[self.num_a_lines :].reshape((-1,) + self.sinogram_shape[1:])
        return sinogram

    def adjoint(self, sinogram, line_factors=1.0):
        """Back-projection, the transpose of forward: each line spreads line_factor * its bin's value."""
        sinogram = self.xp.asarray(sinogram, dtype=self.xp.float32)
        values = self.xp.concatenate([sinogram.ravel(), sinogram[self.oblique_planes].ravel()]) * line_factors
        image = self.xp.zeros(self.image_shape, dtype=self.xp.float32)
        parallelproj_core.joseph3d_back(
            self.start_mm_zyx, self.end_mm_zyx, image, self.image_origin_mm, self.voxel_size_mm,
            self.xp.ascontiguousarray(values, dtype=self.xp.float32),
        )
        return image