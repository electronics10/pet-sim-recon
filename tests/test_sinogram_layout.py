"""MCGPU-PET's sinogram binning: the kernel ports, the tables built from them, and each bin's line (theta, s)."""

import numpy as np
import pytest

from pet_sim_recon.mcgpu.input_file import SinogramSettings
from pet_sim_recon.mcgpu.sinogram_layout import (
    check_sinogram_fits_scanner,
    crystal_pair_table,
    kernel_axial_plane,
    kernel_transaxial_bin,
    line_angle_and_distance,
    nominal_view_angles_rad,
    num_planes,
    planes_before_segment,
    radial_positions_mm,
    ring_pair_planes,
    sum_mirror_planes,
)
from pet_sim_recon.scanner import Scanner

# The 8-crystal ring of docs/sinogram-layout.md. R = 1 mm, so s is in units of R.
SMALL_SCANNER = Scanner(radius_mm=1.0, axial_fov_mm=1.0, num_rings=1, num_crystals_per_ring=8)
SMALL_SINOGRAM = SinogramSettings(
    num_angular_bins=4, num_radial_bins=7, num_axial_slices=1, max_ring_difference=0, span=1
)

# 4 rings of 8 crystals: the axial worked examples in docs/sinogram-layout.md.
FOUR_RING_SCANNER = Scanner(radius_mm=1.0, axial_fov_mm=4.0, num_rings=4, num_crystals_per_ring=8)

# Upstream's sample scanner (external/MCGPU-PET/sample_simulation/MCGPU-PET.in).
SAMPLE_SCANNER = Scanner(radius_mm=90.5, axial_fov_mm=126.56, num_rings=80, num_crystals_per_ring=336)
SAMPLE_SINOGRAM = SinogramSettings(
    num_angular_bins=168, num_radial_bins=147, num_axial_slices=159, max_ring_difference=79, span=11
)

# The table in docs/sinogram-layout.md: EIGHT_CRYSTAL_PAIRS[view][radial bin] = (a, b).
EIGHT_CRYSTAL_PAIRS = [
    [(5, 6), (5, 7), (4, 7), (0, 4), (0, 3), (1, 3), (1, 2)],
    [(6, 7), (0, 6), (0, 5), (1, 5), (1, 4), (2, 4), (2, 3)],
    [(0, 7), (1, 7), (1, 6), (2, 6), (2, 5), (3, 5), (3, 4)],
    [(0, 1), (0, 2), (2, 7), (3, 7), (3, 6), (4, 6), (4, 5)],
]


def test_worked_examples():
    """Crystals (0, 3) and (2, 6) from the docs. The order of the two crystals doesn't matter."""
    assert kernel_transaxial_bin(0, 3, num_crystals=8, num_angular_bins=4, num_radial_bins=7) == (0, 4)
    assert kernel_transaxial_bin(3, 0, num_crystals=8, num_angular_bins=4, num_radial_bins=7) == (0, 4)
    assert kernel_transaxial_bin(2, 6, num_crystals=8, num_angular_bins=4, num_radial_bins=7) == (2, 3)


def test_eight_crystal_table():
    np.testing.assert_array_equal(crystal_pair_table(SMALL_SCANNER, SMALL_SINOGRAM), EIGHT_CRYSTAL_PAIRS)


def test_sample_scanner_fills_every_bin():
    """crystal_pair_table raises if two pairs share a bin; here no bin is left empty either."""
    table = crystal_pair_table(SAMPLE_SCANNER, SAMPLE_SINOGRAM)
    assert (table >= 0).all()


def test_line_of_crystals_0_and_3():
    """Docs, worked example: a horizontal chord below the center, normal pointing down (-90 degrees)."""
    angles_rad = SMALL_SCANNER.crystal_center_angles_rad()
    theta_rad, s_mm = line_angle_and_distance(angles_rad[0], angles_rad[3], radius_mm=1.0)
    assert theta_rad == pytest.approx(-np.pi / 2)
    assert s_mm == pytest.approx(np.cos(3 * np.pi / 8))  # 0.383 R


def test_radial_positions_are_not_arc_corrected():
    """s = R sin(pi d / N): steps of 0.383 R near the center, 0.217 R at the edge."""
    s_mm = radial_positions_mm(SMALL_SCANNER, SMALL_SINOGRAM)
    np.testing.assert_allclose(s_mm, [-0.924, -0.707, -0.383, 0.0, 0.383, 0.707, 0.924], atol=1e-3)


@pytest.mark.parametrize(
    "scanner, sinogram", [(SMALL_SCANNER, SMALL_SINOGRAM), (SAMPLE_SCANNER, SAMPLE_SINOGRAM)], ids=["8", "336"]
)
def test_every_bin_line_sits_on_the_nominal_grid(scanner, sinogram):
    """The exact line of each bin's crystal pair has
    theta = nominal view angle, + half a view step at an even distance from the center bin (interleaving),
    s = radial_positions_mm, the same in every view."""
    table = crystal_pair_table(scanner, sinogram)
    angles_rad = scanner.crystal_center_angles_rad()
    theta_rad, s_mm = line_angle_and_distance(angles_rad[table[..., 0]], angles_rad[table[..., 1]], scanner.radius_mm)

    distance_from_center = np.arange(sinogram.num_radial_bins) - sinogram.num_radial_bins // 2
    half_view_step_rad = np.pi / sinogram.num_angular_bins / 2
    interleave_rad = np.where(distance_from_center % 2 == 0, half_view_step_rad, 0.0)
    expected_theta_rad = nominal_view_angles_rad(sinogram)[:, None] + interleave_rad[None, :]
    expected_s_mm = np.broadcast_to(radial_positions_mm(scanner, sinogram), s_mm.shape)  # same in every view

    np.testing.assert_allclose(theta_rad, expected_theta_rad, atol=1e-9)
    np.testing.assert_allclose(s_mm, expected_s_mm, atol=1e-9)


def test_angular_bins_must_be_half_the_crystals():
    """The kernel never checks the view index against num_angular_bins; a wrong value mis-bins silently."""
    with pytest.raises(ValueError, match="half"):
        check_sinogram_fits_scanner(SAMPLE_SCANNER, SinogramSettings(84, 147, 159, 79, 11))


# --- Axial: ring pairs -> planes ---


def four_ring_sinogram(span, max_ring_difference=3, num_axial_slices=7):
    return SinogramSettings(4, 7, num_axial_slices, max_ring_difference, span)


def test_four_ring_planes_span_1():
    """docs/sinogram-layout.md: planes[ring of photon 1, ring of photon 2], 25 planes, 16 of them used."""
    planes = ring_pair_planes(FOUR_RING_SCANNER, four_ring_sinogram(span=1))
    expected = [
        [0, 12, 20, 24],
        [7, 2, 14, 22],
        [17, 9, 4, 16],
        [23, 19, 11, 6],
    ]
    np.testing.assert_array_equal(planes, expected)
    assert num_planes(four_ring_sinogram(span=1)) == 25


def test_four_ring_planes_span_3():
    """Span 3: ring differences 0 and 1 share segment 0, so both photon orders land in one plane."""
    planes = ring_pair_planes(FOUR_RING_SCANNER, four_ring_sinogram(span=3))
    expected = [
        [0, 1, 10, 11],
        [1, 2, 3, 12],
        [7, 3, 4, 5],
        [8, 9, 5, 6],
    ]
    np.testing.assert_array_equal(planes, expected)
    assert num_planes(four_ring_sinogram(span=3)) == 13


def kernel_segment_start(segment, num_axial_slices, span):
    """The kernel's own loop for where a segment starts (MCGPU-PET_kernel.cu lines 1764-1769), verbatim."""
    ofseg = segment * num_axial_slices
    for kka in range(segment + 1):
        ofk = (kka - 2) * 0.5
        if kka > 1:
            ofseg -= span + 1
        if kka > 3:
            ofseg -= int(np.floor(ofk)) * 2 * span
    return ofseg


@pytest.mark.parametrize("span", range(1, 13))
def test_segment_start_is_read_inputs_plane_count(span):
    """kernel_axial_plane uses read_input's count of the planes before a segment as the segment's start.
    That is the same number as the kernel's own loop."""
    for segment in range(30):
        assert planes_before_segment(segment, 31, span) == kernel_segment_start(segment, 31, span)


@pytest.mark.parametrize(
    "num_rings, sinogram, expected_planes, expected_used",
    [
        (75, SinogramSettings(160, 147, 149, 74, 1), 11101, 5625),  # the prototype (span 1)
        (80, SAMPLE_SINOGRAM, 1293, 1293),  # upstream's sample (span 11)
    ],
    ids=["prototype", "sample"],
)
def test_plane_counts(num_rings, sinogram, expected_planes, expected_used):
    """Every ring pair within the max ring difference gets a plane; span 1 uses only about half of them."""
    scanner = Scanner(radius_mm=100.0, axial_fov_mm=100.0, num_rings=num_rings, num_crystals_per_ring=320)
    planes = ring_pair_planes(scanner, sinogram)
    assert num_planes(sinogram) == expected_planes
    assert len(np.unique(planes[planes >= 0])) == expected_used


def test_span_1_mirror_ring_pairs_get_different_planes():
    """The segment sign comes from photon order, so (a, b) and (b, a) land in different planes (b != a)."""
    for ring_1, ring_2 in [(0, 1), (2, 0), (3, 1)]:
        plane_ab, segment_ab = kernel_axial_plane(ring_1, ring_2, num_axial_slices=7, span=1)
        plane_ba, segment_ba = kernel_axial_plane(ring_2, ring_1, num_axial_slices=7, span=1)
        assert plane_ab != plane_ba
        assert {segment_ab, segment_ba} == {2 * abs(ring_2 - ring_1) - 1, 2 * abs(ring_2 - ring_1)}


@pytest.mark.parametrize(
    "sinogram, message",
    [
        (four_ring_sinogram(span=1, num_axial_slices=5), "num_axial_slices"),  # too few: segments overlap
        (four_ring_sinogram(span=1, max_ring_difference=4), "max_ring_difference"),  # more than 4 rings allow
    ],
    ids=["axial-slices", "max-ring-difference"],
)
def test_bad_axial_settings_are_rejected(sinogram, message):
    with pytest.raises(ValueError, match=message):
        check_sinogram_fits_scanner(FOUR_RING_SCANNER, sinogram)


def test_span_whose_last_segment_is_cut_short_is_rejected():
    """16 rings, span 4, max ring difference 15: the kernel computes planes past the end of the file for some
    ring pairs within the max ring difference, and drops them silently."""
    scanner = Scanner(radius_mm=1.0, axial_fov_mm=16.0, num_rings=16, num_crystals_per_ring=8)
    with pytest.raises(ValueError, match="drop"):
        check_sinogram_fits_scanner(scanner, SinogramSettings(4, 7, 31, 15, 4))


# --- The mirror mix: summing each plane with its mirror ---


def test_sum_mirror_planes_four_rings_span_1():
    """Each plane's count is its own index. Odd segments (photon 2 in the lower ring) are added onto their
    mirrors: plane 7 = (1, 0) onto plane 12 = (0, 1), and so on (the table in test_four_ring_planes_span_1).
    10 summed planes remain, one per unordered ring pair."""
    sinogram = four_ring_sinogram(span=1)
    used = np.unique(ring_pair_planes(FOUR_RING_SCANNER, sinogram))
    used = used[used >= 0]
    counts = np.zeros((25, 1, 1), dtype=np.int32)
    counts[used, 0, 0] = used

    summed, kept_planes = sum_mirror_planes(counts, FOUR_RING_SCANNER, sinogram)
    np.testing.assert_array_equal(kept_planes, [0, 2, 4, 6, 12, 14, 16, 20, 22, 24])
    np.testing.assert_array_equal(summed[:, 0, 0], [0, 2, 4, 6, 12 + 7, 14 + 9, 16 + 11, 20 + 17, 22 + 19, 24 + 23])
    assert summed.sum() == counts.sum()


def test_sum_mirror_planes_four_rings_span_3():
    """Span 3: segment 0 already holds both photon orders, so only segments 1 and 2 merge."""
    counts = np.ones((13, 1, 1), dtype=np.int32)
    summed, kept_planes = sum_mirror_planes(counts, FOUR_RING_SCANNER, four_ring_sinogram(span=3))
    np.testing.assert_array_equal(kept_planes, [0, 1, 2, 3, 4, 5, 6, 10, 11, 12])
    np.testing.assert_array_equal(summed[:, 0, 0], [1, 1, 1, 1, 1, 1, 1, 2, 2, 2])


def test_counts_in_an_unreachable_plane_are_rejected():
    """Span 1 leaves plane 1 empty (no ring pair reaches it); counts there mean the layout is wrong."""
    counts = np.zeros((25, 1, 1), dtype=np.int32)
    counts[1] = 5
    with pytest.raises(ValueError, match="no ring pair"):
        sum_mirror_planes(counts, FOUR_RING_SCANNER, four_ring_sinogram(span=1))
