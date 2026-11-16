"""VoxelObject rejects physically impossible contents."""

import numpy as np
import pytest

from pet_sim_recon.image_grid import ImageGrid
from pet_sim_recon.voxel_object import VoxelObject

GRID = ImageGrid(num_voxels_zyx=(2, 3, 4), voxel_size_mm_zyx=(2.0, 2.0, 2.0))  # voxel = 0.008 mL


def make_object(material_id=None, density=None, activity=None, grid=GRID):
    """A valid water object, with any one array swapped in."""
    shape = grid.num_voxels_zyx
    return VoxelObject(
        grid=grid,
        material_id=np.full(shape, 2) if material_id is None else material_id,
        density_g_per_cm3=np.full(shape, 1.0) if density is None else density,
        activity_Bq_per_mL=np.full(shape, 1000.0) if activity is None else activity,
    )


def test_total_activity():
    """24 voxels x 0.008 mL x 1000 Bq/mL = 192 Bq."""
    assert make_object().total_activity_Bq == pytest.approx(192.0)


def test_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        make_object(activity=np.zeros((4, 3, 2)))


def test_zero_density_is_rejected():
    density = np.full(GRID.num_voxels_zyx, 1.0)
    density[0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="density"):
        make_object(density=density)


def test_material_id_zero_is_rejected():
    with pytest.raises(ValueError, match="starts at 1"):
        make_object(material_id=np.zeros(GRID.num_voxels_zyx, dtype=int))


def test_negative_activity_is_rejected():
    with pytest.raises(ValueError, match="activity"):
        make_object(activity=np.full(GRID.num_voxels_zyx, -1.0))