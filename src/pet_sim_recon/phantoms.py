"""Standard test objects: the NEMA NU 4 image-quality phantom, its compartments and its regions (docs/image-quality.md)."""

import math
from dataclasses import dataclass

import numpy as np

from pet_sim_recon.image_grid import ImageGrid
from pet_sim_recon.voxel_object import VoxelObject

AIR_DENSITY_G_PER_CM3 = 0.00120479  # dry air at sea level (NIST)
WATER_DENSITY_G_PER_CM3 = 1.0
# PMMA is simulated as water at PMMA's density, an approximation (docs/materials.md): no PMMA material file.
PMMA_DENSITY_G_PER_CM3 = 1.19
MATERIAL_NAMES = ("air", "water")
AIR, WATER = 1, 2

# Measurement regions (NEMA NU 4-2008 as commonly reported; see docs/image-quality.md, "Not yet checked").
UNIFORM_VOI_RADIUS_MM = 11.25
UNIFORM_VOI_LENGTH_MM = 10.0
RODS_SLAB_LENGTH_MM = 10.0
ROD_CORE_MARGIN_MM = 0.5
INSERT_VOI_RADIUS_MM = 2.0
INSERT_VOI_LENGTH_MM = 7.5


@dataclass(frozen=True)
class NemaIqGeometry:
    """The NEMA NU 4-2008 image-quality phantom, in mm, in the scanner-centered frame."""

    body_radius_mm: float = 15.0  # the hot cavity
    wall_mm: float = 3.0  # PMMA shell
    rods_z_mm: tuple[float, float] = (-25.0, -5.0)  # solid PMMA with the five rods
    uniform_z_mm: tuple[float, float] = (-5.0, 10.0)
    inserts_z_mm: tuple[float, float] = (10.0, 25.0)
    rod_diameters_mm: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)
    rod_pitch_radius_mm: float = 7.0  # rod k sits at angle 2 pi k / 5
    insert_radius_mm: float = 4.0
    insert_offset_mm: float = 7.5  # the water insert at x = +7.5, the air insert at x = -7.5

    def rod_center_mm_xy(self, diameter_mm: float) -> tuple[float, float]:
        k = self.rod_diameters_mm.index(diameter_mm)
        angle = 2 * math.pi * k / len(self.rod_diameters_mm)
        return self.rod_pitch_radius_mm * math.cos(angle), self.rod_pitch_radius_mm * math.sin(angle)

    def insert_center_mm_xy(self, name: str) -> tuple[float, float]:
        offsets = {"water": self.insert_offset_mm, "air": -self.insert_offset_mm}
        return offsets[name], 0.0


def rod_label(diameter_mm: float) -> str:
    return f"rod_{diameter_mm:g}mm"


@dataclass(frozen=True, eq=False)
class NemaIqPhantom:
    """The built phantom: its VoxelObject, a compartment label per voxel, and the NEMA measurement regions.

    Label id k means label_names[k - 1], as material_id in VoxelObject. Masks are (z, y, x) boolean arrays.
    """

    geometry: NemaIqGeometry
    voxel_object: VoxelObject
    label_id: np.ndarray
    label_names: tuple[str, ...]

    @property
    def grid(self) -> ImageGrid:
        return self.voxel_object.grid

    def label_mask(self, name: str) -> np.ndarray:
        return self.label_id == self.label_names.index(name) + 1

    def uniform_voi(self) -> np.ndarray:
        """Cylinder of radius 11.25 mm over the central 10 mm of the uniform section."""
        return _cylinder(self.grid, (0.0, 0.0), UNIFORM_VOI_RADIUS_MM, self.geometry.uniform_z_mm, UNIFORM_VOI_LENGTH_MM)

    def rods_slab(self) -> np.ndarray:
        """The central 10 mm of the rod section, as a mask along z only: shape (num_z,)."""
        return _central_slab(self.grid, self.geometry.rods_z_mm, RODS_SLAB_LENGTH_MM)

    def rod_core(self, diameter_mm: float) -> np.ndarray:
        """Circle of radius d/2 - 0.5 mm around the rod's axis, over the central 10 mm."""
        radius_mm = diameter_mm / 2 - ROD_CORE_MARGIN_MM
        if radius_mm <= 0:
            raise ValueError(f"a {diameter_mm} mm rod has no core: d/2 - {ROD_CORE_MARGIN_MM} mm <= 0")
        center = self.geometry.rod_center_mm_xy(diameter_mm)
        return _cylinder(self.grid, center, radius_mm, self.geometry.rods_z_mm, RODS_SLAB_LENGTH_MM)

    def rod_roi_yx(self, diameter_mm: float) -> np.ndarray:
        """NEMA's circle of diameter 2d around the rod, in (y, x) only: shape (num_y, num_x)."""
        return _disk_yx(self.grid, self.geometry.rod_center_mm_xy(diameter_mm), diameter_mm)

    def insert_voi(self, name: str) -> np.ndarray:
        """Cylinder of diameter 4 mm over the central 7.5 mm of the "water" or "air" insert."""
        center = self.geometry.insert_center_mm_xy(name)
        return _cylinder(self.grid, center, INSERT_VOI_RADIUS_MM, self.geometry.inserts_z_mm, INSERT_VOI_LENGTH_MM)


def nema_iq_phantom(grid: ImageGrid, activity_Bq_per_mL: float, geometry: NemaIqGeometry = NemaIqGeometry()) -> NemaIqPhantom:
    """Build the phantom on grid. A voxel belongs to a shape when its center is inside, boundary included.

    Shapes are painted in this order, each overwriting the ones before: air, PMMA shell, hot cavity, PMMA rod
    section, rods, inserts. Where a boundary falls exactly on voxel centers, the later shape owns that slice.
    """
    g = geometry
    body_z_mm = (g.rods_z_mm[0], g.inserts_z_mm[1])
    label_names = ("air", "pmma", "hot", *(rod_label(d) for d in g.rod_diameters_mm), "water_insert", "air_insert")
    shape = grid.num_voxels_zyx
    label_id = np.ones(shape, dtype=np.int16)
    material_id = np.full(shape, AIR, dtype=np.int16)
    density = np.full(shape, AIR_DENSITY_G_PER_CM3)
    activity = np.zeros(shape)

    def paint(mask, label, material, density_g_per_cm3, activity_value):
        label_id[mask] = label_names.index(label) + 1
        material_id[mask] = material
        density[mask] = density_g_per_cm3
        activity[mask] = activity_value

    paint(_cylinder(grid, (0.0, 0.0), g.body_radius_mm + g.wall_mm, body_z_mm), "pmma", WATER, PMMA_DENSITY_G_PER_CM3, 0.0)
    paint(_cylinder(grid, (0.0, 0.0), g.body_radius_mm, body_z_mm), "hot", WATER, WATER_DENSITY_G_PER_CM3, activity_Bq_per_mL)
    paint(_cylinder(grid, (0.0, 0.0), g.body_radius_mm, g.rods_z_mm), "pmma", WATER, PMMA_DENSITY_G_PER_CM3, 0.0)
    for diameter_mm in g.rod_diameters_mm:
        rod = _cylinder(grid, g.rod_center_mm_xy(diameter_mm), diameter_mm / 2, g.rods_z_mm)
        paint(rod, rod_label(diameter_mm), WATER, WATER_DENSITY_G_PER_CM3, activity_Bq_per_mL)
    paint(_cylinder(grid, g.insert_center_mm_xy("water"), g.insert_radius_mm, g.inserts_z_mm),
          "water_insert", WATER, WATER_DENSITY_G_PER_CM3, 0.0)
    paint(_cylinder(grid, g.insert_center_mm_xy("air"), g.insert_radius_mm, g.inserts_z_mm),
          "air_insert", AIR, AIR_DENSITY_G_PER_CM3, 0.0)

    voxel_object = VoxelObject(grid, material_id, density, activity, MATERIAL_NAMES)
    return NemaIqPhantom(geometry, voxel_object, label_id, label_names)


def _disk_yx(grid: ImageGrid, center_mm_xy, radius_mm) -> np.ndarray:
    _, y_mm, x_mm = grid.voxel_centers_mm_zyx()
    x0, y0 = center_mm_xy
    return (x_mm[None, :] - x0) ** 2 + (y_mm[:, None] - y0) ** 2 <= radius_mm**2


def _z_range(grid: ImageGrid, z_range_mm) -> np.ndarray:
    z_mm = grid.voxel_centers_mm_zyx()[0]
    return (z_mm >= z_range_mm[0]) & (z_mm <= z_range_mm[1])


def _central_slab(grid: ImageGrid, z_range_mm, length_mm) -> np.ndarray:
    z_mm = grid.voxel_centers_mm_zyx()[0]
    return np.abs(z_mm - sum(z_range_mm) / 2) <= length_mm / 2


def _cylinder(grid: ImageGrid, center_mm_xy, radius_mm, z_range_mm, central_length_mm=None) -> np.ndarray:
    """Cylinder along z. With central_length_mm, only that length around the middle of z_range_mm."""
    along_z = _z_range(grid, z_range_mm) if central_length_mm is None else _central_slab(grid, z_range_mm, central_length_mm)
    return along_z[:, None, None] & _disk_yx(grid, center_mm_xy, radius_mm)[None]
