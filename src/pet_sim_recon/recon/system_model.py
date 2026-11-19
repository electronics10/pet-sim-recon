"""The physics factor of each line: how many coincidences one unit of activity along it gives (docs/projector.md)."""

import math

from pet_sim_recon.acquisition import AcquisitionSettings
from pet_sim_recon.scanner import Scanner

MM3_PER_ML = 1000.0


def etendue_mm2(start_mm_zyx, end_mm_zyx, scanner: Scanner):
    """Etendue of each line's crystal pair: A cos(alpha_1) A cos(alpha_2) / L^2. numpy or cupy, shape (num_lines,).

    A is a crystal's face, a patch of the detector cylinder. alpha is the angle between the line and the face's
    normal, which points along the radius. Both end points must be crystal centers on the cylinder.
    """
    face_area_mm2 = scanner.radius_mm * scanner.crystal_pitch_rad * scanner.ring_pitch_mm
    direction = end_mm_zyx - start_mm_zyx
    length_mm = (direction**2).sum(axis=1) ** 0.5
    # the face normal at an end point (z, y, x) is (0, y, x) / R
    cos_start = abs((direction[:, 1:] * start_mm_zyx[:, 1:]).sum(axis=1)) / (length_mm * scanner.radius_mm)
    cos_end = abs((direction[:, 1:] * end_mm_zyx[:, 1:]).sum(axis=1)) / (length_mm * scanner.radius_mm)
    return face_area_mm2**2 * cos_start * cos_end / length_mm**2


def line_sensitivity(start_mm_zyx, end_mm_zyx, scanner: Scanner, acquisition: AcquisitionSettings):
    """Expected coincidences per unit line integral (Bq/mL x mm), for each line, without attenuation.

    A pair emitted at a point on the line is caught by both faces with probability 2 * solid angle / (4 pi).
    Integrated over the tube between the faces, that is etendue / (2 pi) times the line integral.
    """
    etendue = etendue_mm2(start_mm_zyx, end_mm_zyx, scanner)
    return acquisition.decay_weighted_time_s * etendue / (2 * math.pi) / MM3_PER_ML


def attenuation_factors(projector, attenuation_per_mm):
    """Probability that both photons of a pair cross each line without interacting: exp(-integral of mu)."""
    return projector.xp.exp(-projector.line_integrals(attenuation_per_mm))