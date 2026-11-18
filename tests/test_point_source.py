"""End to end: a simulated point source lands in the bins docs/sinogram-layout.md predicts.

Runs MCGPU-PET (about 10 s), so it is skipped until `pixi run build-mcgpu` has been run.
"""

import numpy as np
import pytest

from pet_sim_recon.acquisition import F18_MEAN_LIFE_S, AcquisitionSettings
from pet_sim_recon.image_grid import ImageGrid
from pet_sim_recon.mcgpu.input_file import SinogramSettings
from pet_sim_recon.mcgpu.output_files import read_sinograms
from pet_sim_recon.mcgpu.runner import DEFAULT_MCGPU_DIR, run_simulation
from pet_sim_recon.mcgpu.sinogram_layout import (
    crystal_pair_table,
    line_angle_and_distance,
    ring_pair_planes,
    sum_mirror_planes,
)
from pet_sim_recon.scanner import Scanner
from pet_sim_recon.voxel_object import VoxelObject

pytestmark = pytest.mark.skipif(
    not (DEFAULT_MCGPU_DIR / "MCGPU-PET.x").exists(), reason="run `pixi run build-mcgpu` first"
)

# The prototype's scanner, span 1. 2 mm voxels line up with the 2 mm rings.
SCANNER = Scanner(radius_mm=52.5, axial_fov_mm=150.0, num_rings=75, num_crystals_per_ring=320)
SINOGRAM = SinogramSettings(
    num_angular_bins=160, num_radial_bins=193, num_axial_slices=149, max_ring_difference=74, span=1
)
GRID = ImageGrid(num_voxels_zyx=(75, 35, 35), voxel_size_mm_zyx=(2.0, 2.0, 2.0))
SOURCE_VOXEL = (47, 22, 25)  # center (z, y, x) = (20, 10, 16) mm: off-center in all three axes
SOURCE_ACTIVITY_BQ = 3e5

# How far a detected LOR can pass from the source center: half the source voxel's diagonal (sqrt(3) mm), plus
# how far a line between crystal centers can be from the photons' true line (half a crystal, 0.52 mm, and half
# a ring, 1 mm, at the ends: 1.13 mm). Acollinearity adds about 0.2 mm. 3 mm covers all of it.
MAX_MISS_MM = 3.0


@pytest.fixture(scope="module")
def point_source_run(tmp_path_factory):
    """A point source in air, plus one water voxel in the far corner (the Woodcock overshoot, see the runner)."""
    shape = GRID.num_voxels_zyx
    material_id = np.full(shape, 1)
    density = np.full(shape, 0.0012)
    material_id[0, 0, 0], density[0, 0, 0] = 2, 1.0
    activity = np.zeros(shape)
    activity[SOURCE_VOXEL] = SOURCE_ACTIVITY_BQ / GRID.voxel_volume_mL
    point_source = VoxelObject(GRID, material_id, density, activity, ("air", "water"))
    acquisition = AcquisitionSettings(1.0, F18_MEAN_LIFE_S, 0.15, 358.0, 664.0)

    run_dir = tmp_path_factory.mktemp("point-source") / "run"
    run_simulation(run_dir, point_source, SCANNER, SINOGRAM, acquisition, random_seed=7)
    trues, _ = read_sinograms(run_dir)
    return trues


@pytest.fixture(scope="module")
def counted_bins(point_source_run):
    """Every bin with trues: its counts, its crystal pair (ix1 < ix2) and its ring pair in photon order."""
    trues = point_source_run
    planes = ring_pair_planes(SCANNER, SINOGRAM)
    rings_of_plane = np.full((trues.shape[0], 2), -1)
    for ring_1, ring_2 in zip(*np.nonzero(planes >= 0)):
        rings_of_plane[planes[ring_1, ring_2]] = ring_1, ring_2
    crystal_pairs = crystal_pair_table(SCANNER, SINOGRAM)

    plane, view, radial = np.nonzero(trues)
    return {
        "counts": trues[plane, view, radial].astype(float),
        "ix1": crystal_pairs[view, radial, 0],
        "ix2": crystal_pairs[view, radial, 1],
        "ring_1": rings_of_plane[plane, 0],
        "ring_2": rings_of_plane[plane, 1],
    }


def source_center_mm_zyx():
    return np.array([centers[i] for centers, i in zip(GRID.voxel_centers_mm_zyx(), SOURCE_VOXEL)])


def miss_distance_mm(start, end, point):
    """Distance from a point to the infinite line through start and end."""
    direction = end - start
    return np.linalg.norm(np.cross(point - start, direction), axis=-1) / np.linalg.norm(direction, axis=-1)


def mirror_miss_distances_mm(bins):
    """Miss distance of the two LORs each bin mixes: A = (ix1 in ring_1, ix2 in ring_2), B = its mirror."""
    center = SCANNER.crystal_center_mm_zyx
    source = source_center_mm_zyx()
    miss_a = miss_distance_mm(center(bins["ix1"], bins["ring_1"]), center(bins["ix2"], bins["ring_2"]), source)
    miss_b = miss_distance_mm(center(bins["ix1"], bins["ring_2"]), center(bins["ix2"], bins["ring_1"]), source)
    return miss_a, miss_b


def test_every_count_has_a_lor_through_the_source(counted_bins):
    """Catches wrong signs, flipped axes, a wrong angle convention and a wrong ring order: each would put
    counts tens of mm away."""
    miss_a, miss_b = mirror_miss_distances_mm(counted_bins)
    counts = counted_bins["counts"]
    near = np.minimum(miss_a, miss_b) < MAX_MISS_MM
    assert counts[near].sum() / counts.sum() > 0.999


def test_no_half_bin_offset_transaxially(counted_bins):
    """The count-weighted mean of s(bin) - s(source) is ~0. A half-bin offset would make it about 0.26 mm."""
    angles_rad = SCANNER.crystal_center_angles_rad()
    theta_rad, s_mm = line_angle_and_distance(
        angles_rad[counted_bins["ix1"]], angles_rad[counted_bins["ix2"]], SCANNER.radius_mm
    )
    _, y_mm, x_mm = source_center_mm_zyx()
    residual_mm = s_mm - (x_mm * np.cos(theta_rad) + y_mm * np.sin(theta_rad))
    assert abs(np.average(residual_mm, weights=counted_bins["counts"])) < 0.05


def test_no_half_ring_offset_axially(counted_bins):
    """In the direct planes (both photons in one ring), the count-weighted mean ring z is the source z."""
    direct = counted_bins["ring_1"] == counted_bins["ring_2"]
    ring_z_mm = SCANNER.ring_center_z_mm()[counted_bins["ring_1"][direct]]
    mean_z_mm = np.average(ring_z_mm, weights=counted_bins["counts"][direct])
    assert abs(mean_z_mm - source_center_mm_zyx()[0]) < 0.05


def test_mirror_mix_is_half_and_half(counted_bins):
    """Oblique bins where only one of A and B passes the source: A has about half of their counts.
    Without the mix (segment sign from geometry), it would have all of them."""
    miss_a, miss_b = mirror_miss_distances_mm(counted_bins)
    counts = counted_bins["counts"]
    only_a = (miss_a < MAX_MISS_MM) & (miss_b >= MAX_MISS_MM)
    only_b = (miss_b < MAX_MISS_MM) & (miss_a >= MAX_MISS_MM)
    fraction_a = counts[only_a].sum() / (counts[only_a].sum() + counts[only_b].sum())
    assert 0.48 < fraction_a < 0.52


def test_summing_mirror_planes_keeps_every_count(point_source_run):
    summed, kept_planes = sum_mirror_planes(point_source_run, SCANNER, SINOGRAM)
    assert len(kept_planes) == 75 * 76 // 2  # one plane per unordered ring pair
    assert summed.sum() == point_source_run.sum()
