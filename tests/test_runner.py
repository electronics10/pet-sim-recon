"""The runner never overwrites an existing run."""

import numpy as np
import pytest

from pet_sim_recon.acquisition import F18_MEAN_LIFE_S, AcquisitionSettings
from pet_sim_recon.image_grid import ImageGrid
from pet_sim_recon.mcgpu.input_file import SinogramSettings
from pet_sim_recon.mcgpu.runner import run_simulation
from pet_sim_recon.scanner import Scanner
from pet_sim_recon.voxel_object import VoxelObject


def test_existing_run_folder_is_refused(tmp_path):
    grid = ImageGrid(num_voxels_zyx=(4, 4, 4), voxel_size_mm_zyx=(1.0, 1.0, 1.0))
    shape = grid.num_voxels_zyx
    water = VoxelObject(grid, np.full(shape, 2), np.full(shape, 1.0), np.zeros(shape))
    acquisition = AcquisitionSettings(1.0, F18_MEAN_LIFE_S, 0.12, 350.0, 650.0)

    # A fake binary: never run, because the folder check stops first.
    fake_mcgpu = tmp_path / "fake-mcgpu"
    fake_mcgpu.mkdir()
    (fake_mcgpu / "MCGPU-PET.x").write_text("")

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
            mcgpu_dir=fake_mcgpu,
        )
    assert (run_dir / "result.txt").read_text() == "an earlier run"