"""MCGPU-PET material files: which file each material name means."""

MAX_MATERIALS = 15  # per run, MCGPU-PET.h line 96

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