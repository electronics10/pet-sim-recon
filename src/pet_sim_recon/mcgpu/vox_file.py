"""MCGPU-PET .vox file (penEasy 2008 voxel format plus an activity column)."""

import gzip
from pathlib import Path

import numpy as np

from pet_sim_recon.image_grid import ImageGrid
from pet_sim_recon.voxel_object import VoxelObject

MAX_MATERIALS = 15  # MCGPU-PET.h line 96
HEADER_START = "[SECTION VOXELS HEADER v.2008-04-13]"
HEADER_END = "[END OF VXH SECTION]"


def write_vox(voxel_object: VoxelObject, path: Path) -> None:
    """Write a .vox file; numpy gzips it when the name ends in .gz."""
    if voxel_object.material_id.max() > MAX_MATERIALS:
        raise ValueError(f"MCGPU-PET allows material ids up to {MAX_MATERIALS}")

    grid = voxel_object.grid
    nz, ny, nx = grid.num_voxels_zyx
    dz_cm, dy_cm, dx_cm = (d / 10 for d in grid.voxel_size_mm_zyx)
    header = "\n".join([
        HEADER_START,
        f"{nx} {ny} {nz}   No. OF VOXELS IN X,Y,Z",
        f"{dx_cm:.6g} {dy_cm:.6g} {dz_cm:.6g}   VOXEL SIZE (cm) ALONG X,Y,Z",
        "1   COLUMN NUMBER WHERE MATERIAL ID IS LOCATED",
        "2   COLUMN NUMBER WHERE THE MASS DENSITY IS LOCATED",
        "0   BLANK LINES AT END OF X,Y-CYCLES (1=YES,0=NO)",
        HEADER_END,
    ])

    activity_Bq_per_voxel = voxel_object.activity_Bq_per_mL * grid.voxel_volume_mL
    # ravel() flattens (z, y, x) with x fastest: the kernel's read order.
    columns = np.column_stack([
        voxel_object.material_id.ravel(),
        voxel_object.density_g_per_cm3.ravel(),
        activity_Bq_per_voxel.ravel(),
    ])
    np.savetxt(path, columns, fmt=["%d", "%.6g", "%.6g"], header=header, comments="")


def read_vox(path: Path) -> VoxelObject:
    """Read a .vox or .vox.gz file back into a VoxelObject."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as file:
        lines = file.read().splitlines()

    start = next(i for i, line in enumerate(lines) if line.startswith(HEADER_START))
    end = next(i for i, line in enumerate(lines) if line.startswith(HEADER_END))
    nx, ny, nz = (int(v) for v in lines[start + 1].split()[:3])
    dx_cm, dy_cm, dz_cm = (float(v) for v in lines[start + 2].split()[:3])

    grid = ImageGrid(num_voxels_zyx=(nz, ny, nx), voxel_size_mm_zyx=(dz_cm * 10, dy_cm * 10, dx_cm * 10))
    columns = np.loadtxt(lines[end + 1:], comments="#", ndmin=2)
    shape = (nz, ny, nx)
    return VoxelObject(
        grid=grid,
        material_id=columns[:, 0].astype(int).reshape(shape),
        density_g_per_cm3=columns[:, 1].reshape(shape),
        activity_Bq_per_mL=columns[:, 2].reshape(shape) / grid.voxel_volume_mL,
    )