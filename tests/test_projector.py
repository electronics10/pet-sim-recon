"""The projector and each line's physics factor: known answers, the adjoint, and bins of two lines."""

import math

import numpy as np
import pytest

from pet_sim_recon.acquisition import AcquisitionSettings
from pet_sim_recon.image_grid import ImageGrid
from pet_sim_recon.mcgpu.input_file import SinogramSettings
from pet_sim_recon.mcgpu.sinogram_layout import summed_bin_lines
from pet_sim_recon.recon.projector import Projector
from pet_sim_recon.recon.sinogram_lines import SinogramLines
from pet_sim_recon.recon.system_model import MM3_PER_ML, attenuation_factors, etendue_mm2, line_sensitivity
from pet_sim_recon.scanner import Scanner

GRID = ImageGrid(num_voxels_zyx=(4, 6, 8), voxel_size_mm_zyx=(1.0, 1.0, 1.0))  # 4 x 6 x 8 mm

# A small scanner, span 1: 4 rings of 16 crystals.
SCANNER = Scanner(radius_mm=10.0, axial_fov_mm=4.0, num_rings=4, num_crystals_per_ring=16)
SINOGRAM = SinogramSettings(
    num_angular_bins=8, num_radial_bins=9, num_axial_slices=7, max_ring_difference=3, span=1
)


def lines_of(*bins):
    """SinogramLines for a sinogram of one view and one radial bin: one plane per entry of bins.

    Each entry is a list of one line (start, end) or two (line A, line B).
    """
    a_lines = [plane_lines[0] for plane_lines in bins]
    b_lines = [plane_lines[1] for plane_lines in bins if len(plane_lines) == 2]
    oblique_planes = [plane for plane, plane_lines in enumerate(bins) if len(plane_lines) == 2]
    start, end = np.array(a_lines + b_lines, dtype=float).transpose(1, 0, 2)
    return SinogramLines(start, end, np.array(oblique_planes, dtype=int), (len(bins), 1, 1))


def test_a_line_along_x_measures_the_grid_width():
    # through voxel centers in z and y, so there is nothing to interpolate: 8 voxels of 1 mm
    projector = Projector(lines_of([((0.5, 0.5, -20.0), (0.5, 0.5, 20.0))]), GRID)
    assert projector.forward(np.ones((4, 6, 8)))[0, 0, 0] == pytest.approx(8.0, rel=1e-5)


def test_a_slanted_line_measures_its_length_in_the_grid():
    # y = x / 2: across the 8 mm in x it travels 8 * sqrt(1 + 1/4) mm, and stays inside the grid in y
    projector = Projector(lines_of([((0.5, -10.0, -20.0), (0.5, 10.0, 20.0))]), GRID)
    assert projector.forward(np.ones((4, 6, 8)))[0, 0, 0] == pytest.approx(8 * math.sqrt(1.25), rel=1e-5)


def test_line_integral_scales_with_the_voxel_value():
    image = np.zeros((4, 6, 8))
    image[2, 3, :] = 5.0  # a rod along x, through the voxel centers the line passes
    projector = Projector(lines_of([((0.5, 0.5, -20.0), (0.5, 0.5, 20.0))]), GRID)
    assert projector.forward(image)[0, 0, 0] == pytest.approx(5.0 * 8, rel=1e-5)


def test_a_bin_of_two_lines_sums_them_with_their_factors():
    line_0 = ((0.5, 0.5, -20.0), (0.5, 0.5, 20.0))
    line_a = ((-1.5, -10.0, -20.0), (1.5, 10.0, 20.0))
    line_b = ((1.5, -10.0, -20.0), (-1.5, 10.0, 20.0))
    image = np.random.default_rng(0).random((4, 6, 8))
    single = {line: Projector(lines_of([line]), GRID).forward(image)[0, 0, 0] for line in (line_0, line_a, line_b)}

    projector = Projector(lines_of([line_0], [line_a, line_b]), GRID)
    factors = np.array([2.0, 3.0, 5.0])  # line_0, line_a, line_b: A lines first, then B lines
    sinogram = projector.forward(image, factors)
    assert sinogram[0, 0, 0] == pytest.approx(2.0 * single[line_0], rel=1e-5)
    assert sinogram[1, 0, 0] == pytest.approx(3.0 * single[line_a] + 5.0 * single[line_b], rel=1e-5)


def test_adjoint_is_the_transpose():
    # <A x, y> = <x, A^T y> for random x, y and random line factors
    lines = summed_bin_lines(SCANNER, SINOGRAM)
    grid = ImageGrid(num_voxels_zyx=(4, 10, 10), voxel_size_mm_zyx=(1.0, 1.4, 1.4))
    projector = Projector(lines, grid)
    rng = np.random.default_rng(1)
    image = rng.random(grid.num_voxels_zyx)
    sinogram = rng.random(lines.sinogram_shape)
    factors = rng.random(len(lines.start_mm_zyx))

    left = np.vdot(projector.forward(image, factors).astype(float), sinogram)
    right = np.vdot(image, projector.adjoint(sinogram, factors).astype(float))
    assert left == pytest.approx(right, rel=1e-5)


def test_etendue_of_a_diameter():
    # a line through the axis hits both faces head on (cos = 1) and is 2R long: A^2 / (2R)^2
    start, end = np.array([[0.0, 0.0, -10.0]]), np.array([[0.0, 0.0, 10.0]])
    face_area_mm2 = 10.0 * (2 * math.pi / 16) * 1.0
    assert etendue_mm2(start, end, SCANNER)[0] == pytest.approx(face_area_mm2**2 / 20.0**2)


def test_etendue_is_cos4_of_the_axial_angle_on_a_ring():
    # docs/projector.md: on a ring, A^2 cos(alpha)^2 / L^2 = A^2 cos^4(theta) / (4 R^2), whatever the radial bin
    lines = summed_bin_lines(SCANNER, SINOGRAM)
    start, end = lines.start_mm_zyx.astype(float), lines.end_mm_zyx.astype(float)
    direction = end - start
    cos_theta = np.hypot(direction[:, 1], direction[:, 2]) / np.linalg.norm(direction, axis=1)
    face_area_mm2 = SCANNER.radius_mm * SCANNER.crystal_pitch_rad * SCANNER.ring_pitch_mm
    expected = face_area_mm2**2 * cos_theta**4 / (4 * SCANNER.radius_mm**2)
    np.testing.assert_allclose(etendue_mm2(start, end, SCANNER), expected, rtol=1e-5)


def test_line_sensitivity_units():
    # 1 Bq/mL along 1 mm of line, for the decay-weighted time, caught with probability etendue / (2 pi)
    acquisition = AcquisitionSettings(10.0, 1e9, 0.15, 358.0, 664.0)  # mean life so long there is no decay
    start, end = np.array([[0.0, 0.0, -10.0]]), np.array([[0.0, 0.0, 10.0]])
    expected = 10.0 * etendue_mm2(start, end, SCANNER) / (2 * math.pi) / MM3_PER_ML
    np.testing.assert_allclose(line_sensitivity(start, end, SCANNER, acquisition), expected, rtol=1e-6)


def test_gpu_and_cpu_give_the_same_projections():
    cupy = pytest.importorskip("cupy")
    lines = summed_bin_lines(SCANNER, SINOGRAM)
    grid = ImageGrid(num_voxels_zyx=(4, 10, 10), voxel_size_mm_zyx=(1.0, 1.4, 1.4))
    rng = np.random.default_rng(2)
    image = rng.random(grid.num_voxels_zyx)
    sinogram = rng.random(lines.sinogram_shape)

    cpu, gpu = Projector(lines, grid), Projector(lines, grid, xp=cupy)
    np.testing.assert_allclose(cupy.asnumpy(gpu.forward(image)), cpu.forward(image), rtol=1e-5)
    np.testing.assert_allclose(cupy.asnumpy(gpu.adjoint(sinogram)), cpu.adjoint(sinogram), rtol=1e-5)


def test_attenuation_is_per_line():
    # two lines along x through different rows: one crosses water-like voxels, the other nothing
    mu_per_mm = np.zeros((4, 6, 8))
    mu_per_mm[2, 3, :] = 0.01
    line_in_water = ((0.5, 0.5, -20.0), (0.5, 0.5, 20.0))
    line_in_air = ((-1.5, -2.5, -20.0), (-1.5, -2.5, 20.0))
    projector = Projector(lines_of([line_in_water, line_in_air]), GRID)
    np.testing.assert_allclose(attenuation_factors(projector, mu_per_mm), [math.exp(-0.01 * 8), 1.0], rtol=1e-6)
