"""Image grid: the voxel geometry shared by phantoms, simulations and reconstructions."""

import math
import warnings
from dataclasses import dataclass

import numpy as np

from pet_sim_recon.scanner import Scanner


@dataclass(frozen=True)
class ImageGrid:
    """A regular voxel grid centered on the scanner. Every triple is in (z, y, x) order."""

    num_voxels_zyx: tuple[int, int, int]
    voxel_size_mm_zyx: tuple[float, float, float]

    def __post_init__(self):
        if len(self.num_voxels_zyx) != 3 or len(self.voxel_size_mm_zyx) != 3:
            raise ValueError("num_voxels_zyx and voxel_size_mm_zyx must each have 3 entries (z, y, x)")
        if any(n < 1 for n in self.num_voxels_zyx):
            raise ValueError(f"num_voxels_zyx must all be >= 1, got {self.num_voxels_zyx}")
        if any(d <= 0 for d in self.voxel_size_mm_zyx):
            raise ValueError(f"voxel_size_mm_zyx must all be > 0, got {self.voxel_size_mm_zyx}")

    @property
    def extent_mm_zyx(self) -> tuple[float, ...]:
        """Full size of the grid along z, y and x."""
        return tuple(n * d for n, d in zip(self.num_voxels_zyx, self.voxel_size_mm_zyx))

    @property
    def voxel_volume_mL(self) -> float:
        """Volume of one voxel. 1 mL = 1 cm^3 = 1000 mm^3."""
        dz_mm, dy_mm, dx_mm = self.voxel_size_mm_zyx
        return dz_mm * dy_mm * dx_mm / 1000

    @property
    def first_voxel_center_mm_zyx(self) -> tuple[float, float, float]:
        """Center of voxel [0, 0, 0]. parallelproj calls this image_origin."""
        return tuple(centers[0] for centers in self.voxel_centers_mm_zyx())

    def voxel_centers_mm_zyx(self) -> tuple[np.ndarray, ...]:
        """Voxel-center coordinates along z, y and x, in the scanner-centered frame."""
        return tuple(
            (np.arange(n) - (n - 1) / 2) * d
            for n, d in zip(self.num_voxels_zyx, self.voxel_size_mm_zyx)
        )


def check_fits_scanner(grid: ImageGrid, scanner: Scanner) -> None:
    """Raise if the grid can't sit in the scanner; warn if it fits badly."""
    extent_z_mm, extent_y_mm, extent_x_mm = grid.extent_mm_zyx
    voxel_z_mm = grid.voxel_size_mm_zyx[0]

    if max(extent_x_mm, extent_y_mm) > 2 * scanner.radius_mm:
        raise ValueError(
            f"grid is {extent_x_mm} x {extent_y_mm} mm across, wider than the detector bore "
            f"({2 * scanner.radius_mm} mm)"
        )

    corner_radius_mm = math.hypot(extent_x_mm / 2, extent_y_mm / 2)
    if corner_radius_mm > scanner.radius_mm:
        warnings.warn(
            f"grid corners reach {corner_radius_mm:.1f} mm, beyond the detector radius "
            f"{scanner.radius_mm} mm; harmless only if the corners hold nothing but air"
        )

    if abs(extent_z_mm - scanner.axial_fov_mm) > voxel_z_mm:
        warnings.warn(
            f"grid is {extent_z_mm} mm long but the axial FOV is {scanner.axial_fov_mm} mm; "
            f"activity beyond the rings is lost, and rings beyond the grid see little"
        )
