"""Run MCGPU-PET: stage a run folder, launch the binary, check its outputs."""

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from pet_sim_recon.acquisition import AcquisitionSettings
from pet_sim_recon.image_grid import check_fits_scanner
from pet_sim_recon.mcgpu.input_file import SinogramSettings, write_input_file
from pet_sim_recon.mcgpu.materials import material_files
from pet_sim_recon.mcgpu.sinogram_layout import check_sinogram_fits_scanner
from pet_sim_recon.mcgpu.vox_file import write_vox
from pet_sim_recon.scanner import Scanner
from pet_sim_recon.voxel_object import VoxelObject

# Built by `pixi run build-mcgpu` (docs/mcgpu-build.md). parents[3] is the repo root in the editable install.
DEFAULT_MCGPU_DIR = Path(__file__).resolve().parents[3] / "external" / "MCGPU-PET"
DEFAULT_MATERIALS_DIR = Path(__file__).resolve().parents[3] / "external" / "mcgpu-materials"
VOX_FILENAME = "object.vox.gz"
OUTPUT_FILES = (
    "sinogram_Trues.raw.gz",
    "sinogram_Scatter.raw.gz",
    "image_Trues.raw.gz",
    "image_Scatter.raw.gz",
)


def run_simulation(
    run_dir: Path,
    voxel_object: VoxelObject,
    scanner: Scanner,
    sinogram: SinogramSettings,
    acquisition: AcquisitionSettings,
    random_seed: int,
    mcgpu_dir: Path = DEFAULT_MCGPU_DIR,
    materials_dir: Path = DEFAULT_MATERIALS_DIR,
) -> Path:
    """Simulate voxel_object in a new folder run_dir and return run_dir. Never overwrites a run."""
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists; a run is never overwritten")
    binary = Path(mcgpu_dir) / "MCGPU-PET.x"
    if not binary.exists():
        raise FileNotFoundError(f"{binary} not found; run `pixi run build-mcgpu`")
    missing_materials = [
        name for name in material_files(voxel_object.material_names) if not (Path(materials_dir) / name).exists()
    ]
    if missing_materials:
        raise FileNotFoundError(f"{missing_materials} not in {materials_dir}; run `pixi run build-mcgpu`")
    check_fits_scanner(voxel_object.grid, scanner)
    check_sinogram_fits_scanner(scanner, sinogram)

    run_dir.mkdir(parents=True)
    settings = {
        "scanner": asdict(scanner),
        "grid": asdict(voxel_object.grid),
        "sinogram": asdict(sinogram),
        "acquisition": asdict(acquisition),
        "material_names": list(voxel_object.material_names),
        "random_seed": random_seed,
        "mcgpu_binary": str(binary),
    }
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2))
    write_vox(voxel_object, run_dir / VOX_FILENAME)
    write_input_file(
        run_dir / "MCGPU-PET.in", scanner, sinogram, acquisition, random_seed, VOX_FILENAME,
        voxel_object.material_names,
    )
    (run_dir / "materials").symlink_to(Path(materials_dir).resolve())

    with open(run_dir / "MCGPU-PET.out", "w") as log:
        result = subprocess.run([str(binary), "MCGPU-PET.in"], cwd=run_dir, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"MCGPU-PET exited with code {result.returncode}; see {run_dir / 'MCGPU-PET.out'}")
    missing = [name for name in OUTPUT_FILES if not (run_dir / name).exists()]
    if missing:
        raise RuntimeError(f"MCGPU-PET finished but did not write {missing}; see {run_dir / 'MCGPU-PET.out'}")
    return run_dir