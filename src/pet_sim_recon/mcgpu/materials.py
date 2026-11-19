"""MCGPU-PET material files: which file each material name means, and what they say about attenuation."""

import gzip
from pathlib import Path

import numpy as np

from pet_sim_recon.voxel_object import VoxelObject

MAX_MATERIALS = 15  # per run, MCGPU-PET.h line 96
ANNIHILATION_PHOTON_ENERGY_EV = 510998.95  # MCGPU-PET.h line 128
MM_PER_CM = 10.0

# The 17 ready-made files in upstream's
# sample_simulation/materials/other_materials_ICRP110_5-515keV__MCGPU-PET.zip
# (air, water, and ICRP 110 reference tissues), unpacked by `pixi run build-mcgpu` into external/mcgpu-materials/.
MATERIAL_FILES = {
    "adipose": "adipose_ICRP110_5-515keV.mcgpu.gz",
    "air": "air_5-515keV.mcgpu.gz",
    "blood_spleen": "blood_spleen_ICRP110_5-515keV.mcgpu.gz",
    "brain": "brain_ICRP110_5-515keV.mcgpu.gz",
    "breast_glandular": "breast_glandular_ICRP110_5-515keV.mcgpu.gz",
    "cartilage": "cartilage_ICRP110_5-515keV.mcgpu.gz",
    "eyes": "eyes_ICRP110_5-515keV.mcgpu.gz",
    "glands_others": "glands_others_ICRP110_5-515keV.mcgpu.gz",
    "kidneys": "kidneys_ICRP110_5-515keV.mcgpu.gz",
    "liver": "liver_ICRP110_5-515keV.mcgpu.gz",
    "lung": "lung_ICRP110_5-515keV.mcgpu.gz",
    "muscle": "muscle_ICRP110_5-515keV.mcgpu.gz",
    "skin": "skin_ICRP110_5-515keV.mcgpu.gz",
    "soft_tissue": "soft_tissue_ICRP110_5-515keV.mcgpu.gz",
    "spongiosa": "spongiosa_ICRP110_5-515keV.mcgpu.gz",
    "stomach_intestines": "stomach_intestines_ICRP110_5-515keV.mcgpu.gz",
    "water": "water_5-515keV.mcgpu.gz",
}


def material_files(material_names: tuple[str, ...]) -> list[str]:
    """MCGPU-PET file for each material name, in the same order (so material id k gets the k-th file)."""
    unknown = [name for name in material_names if name not in MATERIAL_FILES]
    if unknown:
        raise ValueError(
            f"no MCGPU-PET material file for {unknown}; available: {sorted(MATERIAL_FILES)} (docs/materials.md)"
        )
    if len(material_names) > MAX_MATERIALS:
        raise ValueError(f"MCGPU-PET allows at most {MAX_MATERIALS} materials per run, got {len(material_names)}")
    return [MATERIAL_FILES[name] for name in material_names]


def mass_attenuation_cm2_per_g(material_file: Path) -> float:
    """Total interaction probability per g/cm^2 of a 511 keV photon, from a material file.

    The file lists mean free paths (cm) at its nominal density. The kernel interpolates 1 / (mean free path *
    nominal density) linearly in energy (MCGPU-PET.cu lines 1688, 1731, 1743), and so does this.
    """
    with gzip.open(material_file, "rt") as file:
        lines = file.read().splitlines()
    nominal_density_g_per_cm3 = float(lines[lines.index("#[NOMINAL DENSITY (g/cm^3)]") + 1].lstrip("# "))
    first_row = next(i for i, line in enumerate(lines) if line.startswith("#[Energy (eV)")) + 1
    last_row = next(i for i in range(first_row, len(lines)) if lines[i].startswith("#"))
    table = np.loadtxt(lines[first_row:last_row])
    energy_eV, total_mean_free_path_cm = table[:, 0], table[:, 4]
    inverse_path = 1 / (total_mean_free_path_cm * nominal_density_g_per_cm3)
    return float(np.interp(ANNIHILATION_PHOTON_ENERGY_EV, energy_eV, inverse_path))


def attenuation_map_per_mm(voxel_object: VoxelObject, materials_dir: Path) -> np.ndarray:
    """Linear attenuation coefficient of each voxel at 511 keV, in 1/mm, as the simulator sees it."""
    files = material_files(voxel_object.material_names)
    mass_attenuation = np.array([mass_attenuation_cm2_per_g(Path(materials_dir) / name) for name in files])
    per_cm = mass_attenuation[voxel_object.material_id - 1] * voxel_object.density_g_per_cm3
    return per_cm / MM_PER_CM