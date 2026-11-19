"""End to end: a simulated uniform water cylinder against the system model and MLEM (docs/mlem.md).

Nothing is fitted or rescaled: the model predicts counts, and MLEM returns Bq/mL. Runs MCGPU-PET (about 20 s).
"""

import numpy as np
import pytest

from pet_sim_recon.acquisition import F18_MEAN_LIFE_S, AcquisitionSettings
from pet_sim_recon.image_grid import ImageGrid
from pet_sim_recon.mcgpu.input_file import SinogramSettings
from pet_sim_recon.mcgpu.materials import attenuation_map_per_mm
from pet_sim_recon.mcgpu.output_files import read_counts, read_sinograms
from pet_sim_recon.mcgpu.runner import DEFAULT_MATERIALS_DIR, DEFAULT_MCGPU_DIR, run_simulation
from pet_sim_recon.mcgpu.sinogram_layout import summed_bin_lines, summed_plane_ring_pairs, sum_mirror_planes
from pet_sim_recon.recon.mlem import mlem, sensitivity_image
from pet_sim_recon.recon.projector import Projector
from pet_sim_recon.recon.system_model import attenuation_factors, line_sensitivity
from pet_sim_recon.scanner import Scanner
from pet_sim_recon.voxel_object import VoxelObject

pytestmark = pytest.mark.skipif(
    not (DEFAULT_MCGPU_DIR / "MCGPU-PET.x").exists(), reason="run `pixi run build-mcgpu` first"
)

SCANNER = Scanner(radius_mm=52.5, axial_fov_mm=150.0, num_rings=75, num_crystals_per_ring=320)
SINOGRAM = SinogramSettings(160, 193, 149, 74, 1)
GRID = ImageGrid(num_voxels_zyx=(75, 36, 36), voxel_size_mm_zyx=(2.0, 2.0, 2.0))
ACQUISITION = AcquisitionSettings(10.0, F18_MEAN_LIFE_S, 0.15, 358.0, 664.0)
ACTIVITY_BQ_PER_ML = 1e4
CYLINDER_RADIUS_MM, CYLINDER_HALF_LENGTH_MM = 25.0, 50.0
CENTER_RADIAL_BIN = SINOGRAM.num_radial_bins // 2
# Bins within 40 of the center see s <= 20 mm, inside the cylinder. At its edge the line model is too sharp.
RADIAL_BINS_INSIDE = range(CENTER_RADIAL_BIN - 40, CENTER_RADIAL_BIN + 41)
MIN_COUNTS_PER_GROUP = 50_000  # Poisson noise below 0.45 %


def voxel_radius_and_z_mm():
    z_mm, y_mm, x_mm = np.meshgrid(*GRID.voxel_centers_mm_zyx(), indexing="ij")
    return np.hypot(x_mm, y_mm), z_mm


@pytest.fixture(scope="module")
def cylinder():
    radius_mm, z_mm = voxel_radius_and_z_mm()
    inside = (radius_mm <= CYLINDER_RADIUS_MM) & (np.abs(z_mm) <= CYLINDER_HALF_LENGTH_MM)
    material_id = np.where(inside, 2, 1)
    density = np.where(inside, 1.0, 0.0012)
    activity = np.where(inside, ACTIVITY_BQ_PER_ML, 0.0)
    return VoxelObject(GRID, material_id, density, activity, ("air", "water"))


@pytest.fixture(scope="module")
def run_dir(cylinder, tmp_path_factory):
    path = tmp_path_factory.mktemp("cylinder") / "run"
    run_simulation(path, cylinder, SCANNER, SINOGRAM, ACQUISITION, random_seed=11)
    return path


@pytest.fixture(scope="module")
def summed_data(run_dir):
    trues, scatter = read_sinograms(run_dir)
    return sum_mirror_planes(trues, SCANNER, SINOGRAM)[0], sum_mirror_planes(scatter, SCANNER, SINOGRAM)[0]


@pytest.fixture(scope="module")
def model(cylinder):
    cupy = pytest.importorskip("cupy")
    projector = Projector(summed_bin_lines(SCANNER, SINOGRAM), GRID, xp=cupy)
    sensitivity = line_sensitivity(projector.start_mm_zyx, projector.end_mm_zyx, SCANNER, ACQUISITION)
    attenuation = attenuation_factors(projector, attenuation_map_per_mm(cylinder, DEFAULT_MATERIALS_DIR))
    return projector, sensitivity * attenuation


@pytest.fixture(scope="module")
def expected_trues(cylinder, model):
    projector, line_factors = model
    return projector.xp.asnumpy(projector.forward(cylinder.activity_Bq_per_mL, line_factors))


def core_mean_over_truth(image):
    """Mean of the image well inside the cylinder, relative to the true activity."""
    radius_mm, z_mm = voxel_radius_and_z_mm()
    core = (radius_mm <= 18.0) & (np.abs(z_mm) <= 40.0)
    return image[core].mean() / ACTIVITY_BQ_PER_ML


def test_model_predicts_the_total_trues(summed_data, expected_trues):
    assert summed_data[0].sum() / expected_trues.sum() == pytest.approx(1.0, abs=0.005)


def test_ratio_is_flat_across_ring_differences(summed_data, expected_trues):
    # the check that exposed the prototype's missing cos^4 (REWRITE_NOTES section 1)
    trues = summed_data[0]
    ring_pairs = summed_plane_ring_pairs(SCANNER, SINOGRAM)
    ring_difference = ring_pairs[:, 1] - ring_pairs[:, 0]
    checked = 0
    for difference in np.unique(ring_difference):
        planes = ring_difference == difference
        if trues[planes].sum() >= MIN_COUNTS_PER_GROUP:
            assert trues[planes].sum() / expected_trues[planes].sum() == pytest.approx(1.0, abs=0.02)
            checked += 1
    assert checked >= 40


def test_ratio_is_flat_across_radial_bins_inside_the_object(summed_data, expected_trues):
    # the etendue has no radial factor
    trues = summed_data[0]
    for radial in RADIAL_BINS_INSIDE:
        assert trues[:, :, radial].sum() / expected_trues[:, :, radial].sum() == pytest.approx(1.0, abs=0.02)


def test_sensitivity_image_matches_detections_per_slice(cylinder, model, run_dir):
    # image_Trues counts detected trues by emitting voxel: the model's activity * sensitivity, slice by slice
    projector, line_factors = model
    sensitivity = projector.xp.asnumpy(sensitivity_image(projector, line_factors))
    detected = read_counts(run_dir / "image_Trues.raw.gz", GRID.num_voxels_zyx)
    expected_per_slice = (cylinder.activity_Bq_per_mL * sensitivity).sum(axis=(1, 2))
    filled = np.nonzero(cylinder.activity_Bq_per_mL.any(axis=(1, 2)))[0]
    np.testing.assert_allclose(detected.sum(axis=(1, 2))[filled] / expected_per_slice[filled], 1.0, atol=0.02)


def test_mlem_on_trues_returns_the_activity_in_bq_per_ml(model, summed_data):
    projector, line_factors = model
    image = projector.xp.asnumpy(mlem(projector, summed_data[0], line_factors, num_iterations=30))
    assert core_mean_over_truth(image) == pytest.approx(1.0, abs=0.01)


def test_mlem_with_the_scatter_as_background_returns_the_activity(model, summed_data):
    projector, line_factors = model
    trues, scatter = summed_data
    background = projector.xp.asarray(scatter, dtype=projector.xp.float32)
    image = mlem(projector, trues + scatter, line_factors, background=background, num_iterations=30)
    assert core_mean_over_truth(projector.xp.asnumpy(image)) == pytest.approx(1.0, abs=0.01)
