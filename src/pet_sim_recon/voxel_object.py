"""Voxel object: the material, density and activity that fill an image grid."""

from dataclasses import dataclass

import numpy as np

from pet_sim_recon.image_grid import ImageGrid


@dataclass(frozen=True, eq=False)
class VoxelObject:
    """Material, density and activity on an image grid. Arrays are (z, y, x).

    Material id k means material_names[k - 1].
    """

    grid: ImageGrid
    material_id: np.ndarray  # integers >= 1
    density_g_per_cm3: np.ndarray  # > 0
    activity_Bq_per_mL: np.ndarray  # >= 0
    material_names: tuple[str, ...]  # e.g. ("air", "water")

    def __post_init__(self):
        arrays = {
            "material_id": self.material_id,
            "density_g_per_cm3": self.density_g_per_cm3,
            "activity_Bq_per_mL": self.activity_Bq_per_mL,
        }
        for name, array in arrays.items():
            if array.shape != tuple(self.grid.num_voxels_zyx):
                raise ValueError(f"{name} has shape {array.shape}, grid is {self.grid.num_voxels_zyx}")
        if not np.issubdtype(self.material_id.dtype, np.integer):
            raise ValueError(f"material_id must be integers, got {self.material_id.dtype}")
        if np.any(self.material_id < 1):
            raise ValueError("material_id starts at 1")
        if len(set(self.material_names)) != len(self.material_names):
            raise ValueError(f"material_names must not repeat, got {self.material_names}")
        if self.material_id.max() > len(self.material_names):
            raise ValueError(
                f"material_id goes up to {self.material_id.max()}, "
                f"but only {len(self.material_names)} material names are given"
            )
        if np.any(self.density_g_per_cm3 <= 0):
            raise ValueError("density must be > 0 everywhere (air is about 0.0012 g/cm^3)")
        if np.any(self.activity_Bq_per_mL < 0):
            raise ValueError("activity must be >= 0")

    @property
    def total_activity_Bq(self) -> float:
        return float(self.activity_Bq_per_mL.sum()) * self.grid.voxel_volume_mL