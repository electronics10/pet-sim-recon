"""Scanner geometry: rings of crystals on a cylinder around the z axis."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scanner:
    """A cylindrical PET scanner, centered on the origin with its axis along z"""

    radius_mm: float
    axial_fov_mm: float
    num_rings: int
    num_crystals_per_ring: int

    def __post_init__(self):
        if self.radius_mm <= 0 or self.axial_fov_mm <= 0:
            raise ValueError(
                f"radius_mm and axial_fov_mm must be > 0, got {self.radius_mm} and {self.axial_fov_mm}"
            )
        if self.num_rings <1 or self.num_crystals_per_ring < 1:
            raise ValueError(
                f"num_rings and num_crystals_per_ring must be >= 1, "
                f"got {self.num_rings} and {self.num_crystals_per_ring}"
            )

    @property
    def ring_pitch_mm(self) -> float:
        """Axial length of one ring."""
        return self.axial_fov_mm / self.num_rings

    @property
    def crystal_pitch_rad(self) -> float:
        """Angle covered by one crystal."""
        return 2 * np.pi / self.num_crystals_per_ring

    def crystal_center_angles_rad(self) -> np.ndarray:
        """Angle phi of each crystal center. Crystal 0 starts at phi = -pi (MCGPU-PET kernel)."""
        crystal_index = np.arange(self.num_crystals_per_ring)
        return -np.pi + (crystal_index + 0.5) * self.crystal_pitch_rad

    def ring_center_z_mm(self) -> np.ndarray:
        """Axial position z of each ring centre. Ring 0 starts at z = -axial_fov_mm / 2."""
        ring_index = np.arange(self.num_rings)
        return -self.axial_fov_mm / 2 + (ring_index + 0.5) * self.ring_pitch_mm