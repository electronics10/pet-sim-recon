"""MCGPU-PET input file (.in): the toolkit's settings in MCGPU-PET's layout and units (cm, eV)."""

from dataclasses import dataclass
from pathlib import Path

from pet_sim_recon.acquisition import AcquisitionSettings
from pet_sim_recon.mcgpu.materials import material_files
from pet_sim_recon.scanner import Scanner

GPU_NUMBER = 0
THREADS_PER_BLOCK = 32  # read_input requires a multiple of 32
PHASE_SPACE_SIZE = 1000  # allocated but never filled: we ask for sinograms only
UNUSED_IMAGE_RESOLUTION = 128  # read, but only used in commented-out kernel code
NUM_ENERGY_BINS = 700  # bins of Energy_Sinogram_Spectrum.dat, upstream's value


@dataclass(frozen=True)
class SinogramSettings:
    """MCGPU-PET's sinogram tally numbers. Their meaning: docs/sinogram-layout.md."""

    num_angular_bins: int
    num_radial_bins: int
    num_axial_slices: int
    max_ring_difference: int
    span: int


def write_input_file(
    path: Path,
    scanner: Scanner,
    sinogram: SinogramSettings,
    acquisition: AcquisitionSettings,
    random_seed: int,
    vox_filename: str,
    material_names: tuple[str, ...],
) -> None:
    """Write an MCGPU-PET .in file. Section headers must match read_input exactly."""
    if random_seed < 1:
        raise ValueError("random_seed must be >= 1; MCGPU-PET's 0 means 'seed from the clock'")

    axial_fov_cm = _number(scanner.axial_fov_mm / 10)
    lines = [
        "# MCGPU-PET input file, written by pet_sim_recon.mcgpu.input_file",
        "",
        "#[SECTION SIMULATION CONFIG v.2016-07-05]",
        f"{random_seed}  # RANDOM SEED",
        f"{GPU_NUMBER}  # GPU NUMBER",
        f"{THREADS_PER_BLOCK}  # GPU THREADS PER CUDA BLOCK",
        "1.0  # FACTOR TO SCALE THE INPUT MATERIAL DENSITY",
        "",
        "#[SECTION SOURCE PET SCAN v.2017-03-14]",
        f"{_number(acquisition.acquisition_time_s)}  # ACQUISITION TIME [s]",
        f"{_number(acquisition.isotope_mean_life_s)}  # ISOTOPE MEAN LIFE [s]",
        "0 0.0  # per-material activity table: empty, activity comes from the .vox file",
        "",
        "#[SECTION PHASE SPACE FILE v.2016-07-05]",
        "MCGPU_PET.psf  # PHASE SPACE FILE NAME (not written: sinogram-only mode)",
        f"0.0 0.0 0.0 {axial_fov_cm} {_number(-scanner.radius_mm / 10)}  # DETECTOR X Y Z, HEIGHT, RADIUS [cm]",
        f"{PHASE_SPACE_SIZE}  # PHASE SPACE FILE SIZE",
        "0  # REPORT TRUES AND SCATTER (0 = both)",
        "2  # REPORT SINOGRAM ONLY",
        "",
        "#[SECTION DOSE DEPOSITION v.2012-12-12]",
        "NO  # TALLY MATERIAL DOSE",
        "NO  # TALLY 3D VOXEL DOSE",
        "",
        "#[SECTION ENERGY PARAMETERS v.2019-04-25]",
        f"{_number(acquisition.energy_resolution_fwhm)}  # ENERGY RESOLUTION (FWHM fraction)",
        f"{_number(acquisition.energy_window_low_keV * 1000)}  # ENERGY WINDOW LOW [eV]",
        f"{_number(acquisition.energy_window_high_keV * 1000)}  # ENERGY WINDOW HIGH [eV]",
        "",
        "#[SECTION SINOGRAM PARAMETERS v.2019-04-25]",
        f"{axial_fov_cm}  # AXIAL FIELD OF VIEW [cm]",
        f"{scanner.num_rings}  # NUMBER OF ROWS (rings)",
        f"{scanner.num_crystals_per_ring}  # TOTAL NUMBER OF CRYSTALS (per ring)",
        f"{sinogram.num_angular_bins}  # NUMBER OF ANGULAR BINS",
        f"{sinogram.num_radial_bins}  # NUMBER OF RADIAL BINS",
        f"{sinogram.num_axial_slices}  # NUMBER OF Z SLICES",
        f"{UNUSED_IMAGE_RESOLUTION}  # IMAGE RESOLUTION (unused)",
        f"{NUM_ENERGY_BINS}  # NUMBER OF ENERGY BINS",
        f"{sinogram.max_ring_difference}  # MAXIMUM RING DIFFERENCE",
        f"{sinogram.span}  # SPAN",
        "",
        "#[SECTION VOXELIZED GEOMETRY FILE v.2009-11-30]",
        vox_filename,
        "",
        "#[SECTION MATERIAL FILE LIST v.2009-11-30]",
    ]
    # Material id k in the .vox file means the k-th file listed here.
    for material_id, filename in enumerate(material_files(material_names), start=1):
        lines.append(f"materials/{filename}  # {material_id}")
    Path(path).write_text("\n".join(lines) + "\n")


def _number(value: float) -> str:
    """At most 10 significant digits, so 126.56 / 10 prints as 12.656, not 12.656000000000001."""
    return f"{value:.10g}"