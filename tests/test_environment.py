"""Checks that the environment works: the package is installed from src/, and the GPU
stack (cupy + parallelproj) runs a forward projection with a known answer."""

from pathlib import Path

import cupy
import numpy as np
import parallelproj

import pet_sim_recon


def test_package_is_imported_from_src():
    """The editable install points at src/, so tests run against the code you edit."""
    package_dir = Path(pet_sim_recon.__file__).parent
    assert package_dir.parent.name == "src"


def test_cupy_sees_a_gpu():
    assert cupy.cuda.runtime.getDeviceCount() >= 1


def test_forward_projection_of_uniform_cube_on_gpu():
    """A line integral through a uniform cube of ones equals the path length inside it. The cube is cubic on purpose. It sidesteps axis order (z, y, x vs x, y, z) and the image frame. Those are conventions with their own chunk later."""
    num_voxels = 10
    voxel_size_mm = 2.0
    cube_side_mm = num_voxels * voxel_size_mm  # 20 mm

    image = cupy.ones((num_voxels, num_voxels, num_voxels), dtype=cupy.float32)
    voxel_size = cupy.full(3, voxel_size_mm, dtype=cupy.float32)
    # parallelproj wants the centre of voxel [0, 0, 0]; this puts the cube's centre at the origin.
    image_origin = cupy.full(3, -(num_voxels - 1) / 2 * voxel_size_mm, dtype=cupy.float32)

    # Two LORs, both starting and ending outside the cube, both through voxel centres.
    start = cupy.asarray([[-50.0, 1.0, 1.0],     # straight along axis 0
                          [-50.0, -50.0, 1.0]],  # diagonal in the axis-0 / axis-1 plane
                         dtype=cupy.float32)
    end = cupy.asarray([[50.0, 1.0, 1.0],
                        [50.0, 50.0, 1.0]],
                       dtype=cupy.float32)

    projection = parallelproj.joseph3d_fwd(start, end, image, image_origin, voxel_size)

    assert isinstance(projection, cupy.ndarray)  # it ran on the GPU path
    expected_mm = [cube_side_mm, cube_side_mm * np.sqrt(2)]
    np.testing.assert_allclose(cupy.asnumpy(projection), expected_mm, rtol=1e-5)
