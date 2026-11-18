"""The runner never overwrites a run, and refuses objects MCGPU-PET would mis-simulate."""

import numpy as np
import pytest

from pet_sim_recon.acquisition import F18_MEAN_LIFE_S, AcquisitionSettings
from pet_sim_recon.image_grid import ImageGrid
from pet_sim_recon.mcgpu.input_file import SinogramSettings
from pet_sim_recon.mcgpu.runner import check_peak_density, run_simulation
from pet_sim_recon.scanner import Scanner
from pet_sim_recon.voxel_object import VoxelObject


def test_existing_run_folder_is_refused(tmp_path):
    grid = ImageGrid(num_voxels_zyx=(4, 4, 4), voxel_size_mm_zyx=(1.0, 1.0, 1.0))
    shape = grid.num_voxels_zyx
    water = VoxelObject(grid, np.full(shape, 1), np.full(shape, 1.0), np.zeros(shape), ("water",))
    acquisition = AcquisitionSettings(1.0, F18_MEAN_LIFE_S, 0.12, 350.0, 650.0)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "result.txt").write_text("an earlier run")

    with pytest.raises(FileExistsError):
        run_simulation(
            run_dir,
            water,
            Scanner(radius_mm=10.0, axial_fov_mm=4.0, num_rings=2, num_crystals_per_ring=8),
            SinogramSettings(4, 3, 3, 1, 1),
            acquisition,
            random_seed=1,
            mcgpu_dir=tmp_path / "no-mcgpu-needed",  # the folder check comes first
        )
    assert (run_dir / "result.txt").read_text() == "an earlier run"

def test_object_without_a_dense_voxel_is_refused():
    """All air (0.0012 g/cm^3): MCGPU-PET's Woodcock steps get so long that photons are mis-binned
    (docs/mcgpu-simulation.md). One water voxel in a corner is enough."""
    grid = ImageGrid(num_voxels_zyx=(4, 4, 4), voxel_size_mm_zyx=(1.0, 1.0, 1.0))
    shape = grid.num_voxels_zyx
    material_id = np.full(shape, 1)
    density = np.full(shape, 0.0012)
    air = VoxelObject(grid, material_id, density, np.zeros(shape), ("air", "water"))
    with pytest.raises(ValueError, match="densest voxel"):
        check_peak_density(air)

    material_id[0, 0, 0], density[0, 0, 0] = 2, 1.0
    check_peak_density(VoxelObject(grid, material_id, density, np.zeros(shape), ("air", "water")))
