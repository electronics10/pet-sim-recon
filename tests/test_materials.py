"""The material table matches upstream's ready-made files."""

import zipfile
from pathlib import Path

import pytest

from pet_sim_recon.mcgpu.materials import MATERIAL_FILES, material_files

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