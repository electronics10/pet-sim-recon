"""MCGPU-PET's transaxial sinogram layout: which crystal pair lands in which (view, radial) bin.

Conventions and derivations: docs/sinogram-layout.md.
"""

import numpy as np

from pet_sim_recon.mcgpu.input_file import SinogramSettings
from pet_sim_recon.scanner import Scanner

# Line angles of crystal pairs lie on a grid of half a crystal pitch (about 1e-2 rad), so this tolerance
# only absorbs floating-point rounding at exactly theta = +-90 degrees.
ANGLE_TOLERANCE_RAD = 1e-9


def check_sinogram_fits_scanner(scanner: Scanner, sinogram: SinogramSettings) -> None:
    """Raise for sinogram settings the kernel would mis-bin or drop silently."""
    num_crystals = scanner.num_crystals_per_ring
    if num_crystals % 2 != 0 or sinogram.num_angular_bins != num_crystals // 2:
        raise ValueError(
            f"num_angular_bins must be half of num_crystals_per_ring ({num_crystals}), got "
            f"{sinogram.num_angular_bins}; the kernel doesn't check this and mis-bins silently"
        )
    num_rings = scanner.num_rings
    if sinogram.num_axial_slices != 2 * num_rings - 1:
        raise ValueError(
            f"num_axial_slices must be 2 * num_rings - 1 = {2 * num_rings - 1}, got {sinogram.num_axial_slices}; "
            f"with fewer, segments overlap"
        )
    if not 0 <= sinogram.max_ring_difference <= num_rings - 1 or sinogram.span < 1:
        raise ValueError(
            f"need 0 <= max_ring_difference <= num_rings - 1 = {num_rings - 1} and span >= 1, "
            f"got {sinogram.max_ring_difference} and {sinogram.span}"
        )
    ring_pair_planes(scanner, sinogram)  # raises if the kernel would drop or mix ring pairs


def kernel_transaxial_bin(
    crystal_1: int, crystal_2: int, num_crystals: int, num_angular_bins: int, num_radial_bins: int
) -> tuple[int, int]:
    """(view, radial bin) of a crystal pair: tally_Sinogram, MCGPU-PET_kernel.cu lines 1777-1782.

    The radial bin can fall outside [0, num_radial_bins); the kernel then drops the event.
    """
    ix1, ix2 = min(crystal_1, crystal_2), max(crystal_1, crystal_2)
    view = ((ix1 + ix2 + num_angular_bins + 1) % num_crystals) // 2
    radial = abs(ix2 - ix1 - num_angular_bins)
    if ix1 < view or ix2 >= view + num_angular_bins:
        radial = -radial
    return view, radial + num_radial_bins // 2


def crystal_pair_table(scanner: Scanner, sinogram: SinogramSettings) -> np.ndarray:
    """Crystal pair (a, b), a < b, in each bin. Shape (views, radial bins, 2); -1 where no pair lands.

    Built by sending every pair through kernel_transaxial_bin, so it is the kernel's binning by construction.
    """
    check_sinogram_fits_scanner(scanner, sinogram)
    num_crystals = scanner.num_crystals_per_ring
    num_views, num_radial_bins = sinogram.num_angular_bins, sinogram.num_radial_bins

    table = np.full((num_views, num_radial_bins, 2), -1, dtype=int)
    for a in range(num_crystals):
        for b in range(a + 1, num_crystals):
            view, radial = kernel_transaxial_bin(a, b, num_crystals, num_views, num_radial_bins)
            if not 0 <= radial < num_radial_bins:
                continue  # a near-tangent line, dropped by the kernel
            if table[view, radial, 0] != -1:
                raise RuntimeError(f"pairs {tuple(table[view, radial])} and {(a, b)} share bin {(view, radial)}")
            table[view, radial] = a, b
    return table


def line_angle_and_distance(phi_a_rad, phi_b_rad, radius_mm):
    """Line through two points on a circle as x cos(theta) + y sin(theta) = s, theta in [-pi/2, pi/2).

    Works on scalars and arrays. Returns (theta_rad, s_mm).
    """
    theta_rad = (phi_a_rad + phi_b_rad) / 2  # the normal points halfway between the two points
    s_mm = radius_mm * np.cos((phi_b_rad - phi_a_rad) / 2)  # height of the isosceles triangle

    # (theta, s) and (theta -+ pi, -s) are the same line: fold theta into [-pi/2, pi/2).
    too_high = theta_rad >= np.pi / 2 - ANGLE_TOLERANCE_RAD
    too_low = theta_rad < -np.pi / 2 - ANGLE_TOLERANCE_RAD
    theta_rad = np.where(too_high, theta_rad - np.pi, np.where(too_low, theta_rad + np.pi, theta_rad))
    s_mm = np.where(too_high | too_low, -s_mm, s_mm)
    return theta_rad, s_mm


def nominal_view_angles_rad(sinogram: SinogramSettings) -> np.ndarray:
    """Nominal angle of each view: theta_k = -pi/2 + k * pi / num_angular_bins.

    Bins at an even distance from the center radial bin sit half a view step higher (interleaving).
    """
    view_index = np.arange(sinogram.num_angular_bins)
    return -np.pi / 2 + view_index * np.pi / sinogram.num_angular_bins


def radial_positions_mm(scanner: Scanner, sinogram: SinogramSettings) -> np.ndarray:
    """Signed distance s of each radial bin: R sin(pi d / N), d = bins from the center bin. Same in every view.

    Not arc-corrected: evenly spaced in crystal difference, not in s.
    """
    distance_from_center = np.arange(sinogram.num_radial_bins) - sinogram.num_radial_bins // 2
    return scanner.radius_mm * np.sin(np.pi * distance_from_center / scanner.num_crystals_per_ring)


def planes_before_segment(segment: int, num_axial_slices: int, span: int) -> int:
    """Number of planes in segments 0 .. segment - 1: read_input, MCGPU-PET.cu lines 1177-1187.

    Segment 0 has num_axial_slices planes; higher tilts have fewer ring sums, so fewer planes.
    """
    num_planes = segment * num_axial_slices
    for k in range(1, segment + 1):  # the read_input loop's aaa
        if k > 1:
            num_planes -= span + 1
        if k > 3:
            num_planes -= 2 * ((k - 2) // 2) * span
    return num_planes


def num_planes(sinogram: SinogramSettings) -> int:
    """Planes in the sinogram file (read_input's NSINOS): 2 * floor(MRD / span) + 1 segments."""
    num_segments = 2 * (sinogram.max_ring_difference // sinogram.span) + 1
    return planes_before_segment(num_segments, sinogram.num_axial_slices, sinogram.span)


def kernel_axial_plane(ring_1: int, ring_2: int, num_axial_slices: int, span: int) -> tuple[int, int]:
    """(plane, segment) of a ring pair in photon order: tally_Sinogram, MCGPU-PET_kernel.cu lines 1760-1773.

    Before this, the kernel drops pairs with |ring_2 - ring_1| > max_ring_difference (line 1759).
    """
    ring_difference = abs(ring_2 - ring_1)
    tilt = (ring_difference - 1 + (span + 1) // 2) // span  # the kernel's incl: ring differences near tilt * span
    segment = 2 * tilt
    if ring_2 < ring_1 and segment > 0:
        segment -= 1  # odd segments: photon 2 in the lower ring
    smallest_difference = 0 if tilt == 0 else (span + 1) // 2 + (tilt - 1) * span  # the kernel's ofz
    plane = planes_before_segment(segment, num_axial_slices, span) + (ring_1 + ring_2) - smallest_difference
    return plane, segment


def ring_pair_planes(scanner: Scanner, sinogram: SinogramSettings) -> np.ndarray:
    """Plane of each ring pair in photon order, planes[ring_1, ring_2]; -1 where the kernel drops the pair.

    Built by sending every ring pair through kernel_axial_plane. Raises if the kernel would lose a pair within
    max_ring_difference, or put two segments or two axial positions into one plane.
    """
    num_rings = scanner.num_rings
    total_planes = num_planes(sinogram)
    planes = np.full((num_rings, num_rings), -1, dtype=int)
    plane_contents = {}  # plane -> (segment, ring_1 + ring_2)
    for ring_1 in range(num_rings):
        for ring_2 in range(num_rings):
            if abs(ring_2 - ring_1) > sinogram.max_ring_difference:
                continue  # dropped by the kernel, by design
            plane, segment = kernel_axial_plane(ring_1, ring_2, sinogram.num_axial_slices, sinogram.span)
            if not 0 <= plane < total_planes:
                raise ValueError(f"ring pair {(ring_1, ring_2)} falls outside the {total_planes} planes; "
                                 f"the kernel would drop it silently (span {sinogram.span}, "
                                 f"max_ring_difference {sinogram.max_ring_difference})")
            contents = (segment, ring_1 + ring_2)
            if plane_contents.setdefault(plane, contents) != contents:
                raise ValueError(f"plane {plane} would mix {plane_contents[plane]} and {contents} "
                                 f"(segment, ring sum)")
            planes[ring_1, ring_2] = plane
    return planes


def mirror_partner_planes(scanner: Scanner, sinogram: SinogramSettings) -> np.ndarray:
    """For each plane of the file, the plane it is summed into; -1 for planes no ring pair reaches.

    Segment 0 and the even segments (photon 2 in the higher ring) keep their planes; each odd-segment plane
    goes to its mirror, the plane of the same ring pairs in the other photon order.
    """
    planes = ring_pair_planes(scanner, sinogram)
    partner = np.full(num_planes(sinogram), -1, dtype=int)
    for ring_1, ring_2 in zip(*np.nonzero(planes >= 0)):
        partner[planes[ring_1, ring_2]] = planes[min(ring_1, ring_2), max(ring_1, ring_2)]
    return partner


def sum_mirror_planes(
    counts: np.ndarray, scanner: Scanner, sinogram: SinogramSettings
) -> tuple[np.ndarray, np.ndarray]:
    """Add each plane onto its mirror partner: the mirror mix of docs/sinogram-layout.md.

    counts: (plane, view, radial bin), as read_sinograms returns it. Returns the summed counts, shaped
    (summed plane, view, radial bin), and for each summed plane the file plane it keeps.
    """
    partner = mirror_partner_planes(scanner, sinogram)
    if counts[partner < 0].any():
        raise ValueError("counts in a plane no ring pair reaches: the file and the layout disagree")
    kept_planes = np.unique(partner[partner >= 0])
    summed = np.zeros((len(kept_planes),) + counts.shape[1:], dtype=counts.dtype)
    for plane in np.nonzero(partner >= 0)[0]:
        summed[np.searchsorted(kept_planes, partner[plane])] += counts[plane]
    return summed, kept_planes