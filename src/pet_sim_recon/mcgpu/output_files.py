"""Read a finished MCGPU-PET run folder: the settings it was made with, and its sinograms."""

import gzip
import json
from pathlib import Path

import numpy as np

from pet_sim_recon.mcgpu.input_file import SinogramSettings
from pet_sim_recon.mcgpu.sinogram_layout import num_planes
from pet_sim_recon.scanner import Scanner

SINOGRAM_FILES = ("sinogram_Trues.raw.gz", "sinogram_Scatter.raw.gz")


def read_run_settings(run_dir: Path) -> tuple[Scanner, SinogramSettings]:
    """The scanner and sinogram settings of a run, from the settings.json the runner wrote."""
    settings = json.loads((Path(run_dir) / "settings.json").read_text())
    return Scanner(**settings["scanner"]), SinogramSettings(**settings["sinogram"])


def read_sinograms(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Trues and scatter counts of a run, each shaped (plane, view, radial bin). Layout: docs/sinogram-layout.md."""
    _, sinogram = read_run_settings(run_dir)
    shape = (num_planes(sinogram), sinogram.num_angular_bins, sinogram.num_radial_bins)
    trues, scatter = (read_counts(Path(run_dir) / name, shape) for name in SINOGRAM_FILES)
    return trues, scatter


def read_counts(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    """One gzipped sinogram file. MCGPU-PET.cu lines 548-553 write it as int32, bin = (plane * views + view) *
    radial bins + radial bin, so C order gives `shape` directly. Little-endian, as on every x86 host."""
    with gzip.open(path, "rb") as file:
        counts = np.frombuffer(file.read(), dtype="<i4")
    if counts.size != np.prod(shape):
        raise ValueError(f"{path} holds {counts.size} counts, but the settings give {shape}")
    return counts.reshape(shape).copy()  # copy: frombuffer's array is read-only