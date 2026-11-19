"""The material table matches upstream's ready-made files."""

import math
import zipfile
from pathlib import Path

import numpy as np
import pytest

from pet_sim_recon.image_grid import ImageGrid
from pet_sim_recon.mcgpu.materials import (
    MATERIAL_FILES,
    attenuation_map_per_mm,
    mass_attenuation_cm2_per_g,
    material_files,
)
from pet_sim_recon.mcgpu.runner import DEFAULT_MATERIALS_DIR
from pet_sim_recon.voxel_object import VoxelObject

MATERIALS_ZIP = (
    Path(__file__).parents[1]
    / "external" / "MCGPU-PET" / "sample_simulation" / "materials"
    / "other_materials_ICRP110_5-515keV__MCGPU-PET.zip"
)


@pytest.mark.skipif(not MATERIALS_ZIP.exists(), reason="run `pixi run build-mcgpu` first")
def test_table_lists_exactly_the_files_in_upstreams_zip():
    files_in_zip = {name for name in zipfile.ZipFile(MATERIALS_ZIP).namelist() if name.endswith(".mcgpu.gz")}
    assert set(MATERIAL_FILES.values()) == files_in_zip


def test_files_come_back_in_id_order():
    assert material_files(("water", "air")) == ["water_5-515keV.mcgpu.gz", "air_5-515keV.mcgpu.gz"]


def test_unknown_material_is_rejected():
    with pytest.raises(ValueError, match="pmma"):
        material_files(("water", "pmma"))


def test_more_than_15_materials_is_rejected():
    sixteen = tuple(sorted(MATERIAL_FILES))[:16]
    with pytest.raises(ValueError, match="at most 15"):
        material_files(sixteen)

@pytest.mark.skipif(not (DEFAULT_MATERIALS_DIR / "water_5-515keV.mcgpu.gz").exists(), reason="run `pixi run build-mcgpu`")
def test_water_attenuation_matches_nist():
    # NIST XCOM table for liquid water: 0.09687 cm^2/g at 500 keV and 0.08956 at 600 keV. Log-log interpolation
    # to 511 keV gives 0.0960; PENELOPE's cross sections, in the material file, should agree within 0.5 %.
    fraction = math.log(511 / 500) / math.log(600 / 500)
    nist = math.exp((1 - fraction) * math.log(0.09687) + fraction * math.log(0.08956))
    value = mass_attenuation_cm2_per_g(DEFAULT_MATERIALS_DIR / "water_5-515keV.mcgpu.gz")
    assert value == pytest.approx(nist, rel=0.005)


@pytest.mark.skipif(not (DEFAULT_MATERIALS_DIR / "water_5-515keV.mcgpu.gz").exists(), reason="run `pixi run build-mcgpu`")
def test_attenuation_map_uses_each_voxels_material_and_density():
    grid = ImageGrid(num_voxels_zyx=(1, 1, 3), voxel_size_mm_zyx=(1.0, 1.0, 1.0))
    material_id = np.array([[[1, 2, 2]]])
    density = np.array([[[0.0012, 1.0, 2.0]]])
    voxel_object = VoxelObject(grid, material_id, density, np.zeros((1, 1, 3)), ("air", "water"))
    mu_per_mm = attenuation_map_per_mm(voxel_object, DEFAULT_MATERIALS_DIR)[0, 0]
    water = mass_attenuation_cm2_per_g(DEFAULT_MATERIALS_DIR / "water_5-515keV.mcgpu.gz")
    air = mass_attenuation_cm2_per_g(DEFAULT_MATERIALS_DIR / "air_5-515keV.mcgpu.gz")
    np.testing.assert_allclose(mu_per_mm, [air * 0.0012 / 10, water / 10, 2 * water / 10])
