"""The .in file gives MCGPU-PET's read_input the values we mean, in its units."""

from pathlib import Path

import pytest

from pet_sim_recon.acquisition import AcquisitionSettings
from pet_sim_recon.mcgpu.input_file import SinogramSettings, write_input_file
from pet_sim_recon.scanner import Scanner

UPSTREAM_SAMPLE_IN = Path(__file__).parents[1] / "external" / "MCGPU-PET" / "sample_simulation" / "MCGPU-PET.in"

# Upstream's sample (external/MCGPU-PET/sample_simulation/MCGPU-PET.in), in toolkit units.
SAMPLE_SCANNER = Scanner(radius_mm=90.5, axial_fov_mm=126.56, num_rings=80, num_crystals_per_ring=336)
SAMPLE_SINOGRAM = SinogramSettings(
    num_angular_bins=168, num_radial_bins=147, num_axial_slices=159, max_ring_difference=79, span=11
)
SAMPLE_ACQUISITION = AcquisitionSettings(
    acquisition_time_s=1.0,
    isotope_mean_life_s=70000.0,
    energy_resolution_fwhm=0.12,
    energy_window_low_keV=350.0,
    energy_window_high_keV=600.0,
)


def read_values(text):
    """Value lines per section, read as read_input does: text before '#', blank lines skipped."""
    values = {}
    section = None
    for line in text.splitlines():
        if "[SECTION " in line:
            section = line.split("[SECTION ")[1].split(" v.")[0]
            values[section] = []
            continue
        value = line.split("#")[0].strip()
        if section is not None and value:
            values[section].append(value)
    return values


def numbers(value_lines):
    """All numbers in some value lines, as floats."""
    return [float(token) for line in value_lines for token in line.split()]


def write_sample(tmp_path, random_seed=12345):
    path = tmp_path / "MCGPU-PET.in"
    write_input_file(
        path, SAMPLE_SCANNER, SAMPLE_SINOGRAM, SAMPLE_ACQUISITION, random_seed, "object.vox.gz", ("air", "water")
    )
    return read_values(path.read_text())


@pytest.mark.skipif(not UPSTREAM_SAMPLE_IN.exists(), reason="run `pixi run build-mcgpu` first")
def test_physics_and_geometry_match_upstream_sample(tmp_path):
    ours = write_sample(tmp_path)
    upstream = read_values(UPSTREAM_SAMPLE_IN.read_text())

    for section in ("ENERGY PARAMETERS", "SINOGRAM PARAMETERS"):
        assert numbers(ours[section]) == numbers(upstream[section]), section
    # Source: acquisition time and mean life (the activity table after them differs on purpose).
    assert numbers(ours["SOURCE PET SCAN"][:2]) == numbers(upstream["SOURCE PET SCAN"][:2])
    # Detector line (x y z H R): the second value line of the phase-space section.
    assert numbers(ours["PHASE SPACE FILE"][1:2]) == numbers(upstream["PHASE SPACE FILE"][1:2])
    assert [Path(name).name for name in ours["MATERIAL FILE LIST"]] == [
        Path(name).name for name in upstream["MATERIAL FILE LIST"]
    ]


def test_units_are_cm_and_eV(tmp_path):
    ours = write_sample(tmp_path)
    assert numbers(ours["PHASE SPACE FILE"][1:2]) == [0.0, 0.0, 0.0, 12.656, -9.05]
    assert numbers(ours["ENERGY PARAMETERS"]) == [0.12, 350000.0, 600000.0]


def test_seed_zero_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="random_seed"):
        write_sample(tmp_path, random_seed=0)