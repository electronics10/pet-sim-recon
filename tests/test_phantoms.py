"""The NEMA NU 4 image-quality phantom: geometry, voxelization rule and regions (docs/image-quality.md)."""

import math

import numpy as np
import pytest

from pet_sim_recon.image_grid import ImageGrid
from pet_sim_recon.mcgpu.runner import check_peak_density
from pet_sim_recon.phantoms import (
    AIR,
    AIR_DENSITY_G_PER_CM3,
    PMMA_DENSITY_G_PER_CM3,
    WATER,
    WATER_DENSITY_G_PER_CM3,
    NemaIqGeometry,
    nema_iq_phantom,
    rod_label,
)

ACTIVITY_BQ_PER_ML = 20000.0
# Even voxel counts put every boundary between voxel centers, so no slice is shared by two sections.
FINE_GRID = ImageGrid(num_voxels_zyx=(112, 80, 80), voxel_size_mm_zyx=(0.5, 0.5, 0.5))
# The talk's grid: 149 slices of 1 mm put voxel centers exactly on the boundaries z = -5, 10 and 25 mm.
TALK_GRID = ImageGrid(num_voxels_zyx=(149, 80, 80), voxel_size_mm_zyx=(1.0, 1.0, 1.0))


@pytest.fixture(scope="module")
def phantom():
    return nema_iq_phantom(FINE_GRID, ACTIVITY_BQ_PER_ML)


def test_total_activity_matches_the_geometry(phantom):
    """Hot volume: the 30 mm cavity over the uniform and insert sections, plus the rods, minus the two inserts."""
    g = NemaIqGeometry()
    cavity_mm3 = math.pi * g.body_radius_mm**2 * (g.inserts_z_mm[1] - g.uniform_z_mm[0])
    rods_mm3 = sum(math.pi * (d / 2) ** 2 for d in g.rod_diameters_mm) * (g.rods_z_mm[1] - g.rods_z_mm[0])
    inserts_mm3 = 2 * math.pi * g.insert_radius_mm**2 * (g.inserts_z_mm[1] - g.inserts_z_mm[0])
    expected_Bq = ACTIVITY_BQ_PER_ML * (cavity_mm3 + rods_mm3 - inserts_mm3) / 1000
    # 0.5 mm voxels cut the curved edges; the error is well under 1 %.
    assert phantom.voxel_object.total_activity_Bq == pytest.approx(expected_Bq, rel=0.01)


def test_simulator_accepts_the_phantom(phantom):
    """MCGPU-PET needs one voxel of at least 0.1 g/cm^3 somewhere (Woodcock steps, docs/mcgpu-simulation.md)."""
    check_peak_density(phantom.voxel_object)
    assert phantom.voxel_object.density_g_per_cm3.max() == PMMA_DENSITY_G_PER_CM3


def test_regions_of_the_true_activity(phantom):
    """Every rod core and the uniform VOI hold exactly the fill activity; the insert VOIs hold none."""
    truth = phantom.voxel_object.activity_Bq_per_mL
    assert truth[phantom.uniform_voi()].min() == truth[phantom.uniform_voi()].max() == ACTIVITY_BQ_PER_ML
    for diameter_mm in (2.0, 3.0, 4.0, 5.0):
        core = truth[phantom.rod_core(diameter_mm)]
        assert core.size > 0 and core.min() == ACTIVITY_BQ_PER_ML
    for name in ("water", "air"):
        assert truth[phantom.insert_voi(name)].max() == 0


def test_labels_have_the_right_material(phantom):
    voxel_object = phantom.voxel_object
    hot_labels = ("hot", *(rod_label(d) for d in NemaIqGeometry().rod_diameters_mm))
    expected = {  # label: (material id, density, activity)
        "air": (AIR, AIR_DENSITY_G_PER_CM3, 0.0),
        "pmma": (WATER, PMMA_DENSITY_G_PER_CM3, 0.0),
        "water_insert": (WATER, WATER_DENSITY_G_PER_CM3, 0.0),
        "air_insert": (AIR, AIR_DENSITY_G_PER_CM3, 0.0),
        **{label: (WATER, WATER_DENSITY_G_PER_CM3, ACTIVITY_BQ_PER_ML) for label in hot_labels},
    }
    for label, (material, density, activity) in expected.items():
        mask = phantom.label_mask(label)
        assert mask.any(), label
        assert np.all(voxel_object.material_id[mask] == material), label
        assert np.all(voxel_object.density_g_per_cm3[mask] == density), label
        assert np.all(voxel_object.activity_Bq_per_mL[mask] == activity), label


def test_a_slice_on_a_boundary_belongs_to_the_later_shape():
    """On the talk grid, z = -5 mm is in both the rod and the uniform section, and z = 10 mm in both the uniform
    section and the inserts. The rod section and the inserts are painted later, so they own those slices."""
    phantom = nema_iq_phantom(TALK_GRID, ACTIVITY_BQ_PER_ML)
    z_mm, y_mm, x_mm = TALK_GRID.voxel_centers_mm_zyx()
    near_axis = (np.argmin(np.abs(y_mm)), np.argmin(np.abs(x_mm)))
    at_water_insert = (np.argmin(np.abs(y_mm)), np.argmin(np.abs(x_mm - 7.5)))
    labels = lambda z, yx: phantom.label_names[phantom.label_id[np.argmin(np.abs(z_mm - z)), yx[0], yx[1]] - 1]
    assert labels(-5.0, near_axis) == "pmma"
    assert labels(-4.0, near_axis) == "hot"
    assert labels(10.0, at_water_insert) == "water_insert"
    assert labels(9.0, at_water_insert) == "hot"
    # The same rule makes the rods' "central 10 mm" 11 slices here (z = -20 ... -10, both ends included).
    assert phantom.rods_slab().sum() == 11
