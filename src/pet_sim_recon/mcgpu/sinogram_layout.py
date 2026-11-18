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
    """Raise unless num_angular_bins is half of num_crystals_per_ring, which the kernel assumes."""
    num_crystals = scanner.num_crystals_per_ring
    if num_crystals % 2 != 0 or sinogram.num_angular_bins != num_crystals // 2:
        raise ValueError(
            f"num_angular_bins must be half of num_crystals_per_ring ({num_crystals}), got "
            f"{sinogram.num_angular_bins}; the kernel doesn't check this and mis-bins silently"
        )


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