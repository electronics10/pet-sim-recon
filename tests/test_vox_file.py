"""The .vox file matches what MCGPU-PET's load_voxels reads."""

import numpy as np
import pytest

from pet_sim_recon.image_grid import ImageGrid
from pet_sim_recon.mcgpu.vox_file import read_vox, write_vox
from pet_sim_recon.voxel_object import VoxelObject

# Uneven, so that a mixed-up axis shows up. Voxel volume = 3 x 2 x 1.5 mm^3 = 0.009 mL.
UNEVEN_GRID = ImageGrid(num_voxels_zyx=(2, 3, 4), voxel_size_mm_zyx=(3.0, 2.0, 1.5))


def labeled_object():
    """Each voxel's activity tells where it is: 100*z + 10*y + x Bq per voxel."""
    z, y, x = np.meshgrid(np.arange(2), np.arange(3), np.arange(4), indexing="ij")
    label = 100 * z + 10 * y + x
    return VoxelObject(
        grid=UNEVEN_GRID,
        material_id=np.full(label.shape, 2),
        density_g_per_cm3=np.full(label.shape, 1.0),
        activity_Bq_per_mL=label / UNEVEN_GRID.voxel_volume_mL,
    )


def test_voxels_are_written_in_the_kernel_read_order(tmp_path):
    path = tmp_path / "labeled.vox"
    write_vox(labeled_object(), path)
    activity_column = np.loadtxt(path, skiprows=7)[:, 2]

    # MCGPU-PET.cu load_voxels: for k (z) { for j (y) { for i (x) { read one line } } }
    expected = [100 * k + 10 * j + i for k in range(2) for j in range(3) for i in range(4)]
    np.testing.assert_allclose(activity_column, expected, rtol=1e-6)


def test_header_lists_x_y_z_in_cm(tmp_path):
    path = tmp_path / "labeled.vox"
    write_vox(labeled_object(), path)
    lines = path.read_text().splitlines()
    assert lines[1].split()[:3] == ["4", "3", "2"]
    np.testing.assert_allclose([float(v) for v in lines[2].split()[:3]], [0.15, 0.2, 0.3])


@pytest.mark.parametrize("filename", ["object.vox", "object.vox.gz"])
def test_write_then_read_gives_the_same_object(tmp_path, filename):
    original = labeled_object()
    write_vox(original, tmp_path / filename)
    copy = read_vox(tmp_path / filename)

    assert copy.grid.num_voxels_zyx == original.grid.num_voxels_zyx
    np.testing.assert_allclose(copy.grid.voxel_size_mm_zyx, original.grid.voxel_size_mm_zyx)
    np.testing.assert_array_equal(copy.material_id, original.material_id)
    np.testing.assert_allclose(copy.density_g_per_cm3, original.density_g_per_cm3)
    np.testing.assert_allclose(copy.activity_Bq_per_mL, original.activity_Bq_per_mL, rtol=1e-6)


def test_more_than_15_materials_is_rejected(tmp_path):
    obj = labeled_object()
    material_id = obj.material_id.copy()
    material_id[0, 0, 0] = 16
    too_many = VoxelObject(obj.grid, material_id, obj.density_g_per_cm3, obj.activity_Bq_per_mL)
    with pytest.raises(ValueError, match="material"):
        write_vox(too_many, tmp_path / "bad.vox")