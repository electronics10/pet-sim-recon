# Coordinate frame

Every position in the toolkit is in one frame, defined here. Code and other docs point to this page.

## The toolkit frame

- **Origin**: the scanner center, i.e. the axis of the detector cylinder, halfway along its axial FOV.
- **Axes**: z along the scanner axis; x and y transaxial, right-handed. Looking from +z toward the origin,
  +x points right and +y up.
- **Units**: mm (and mL for activity concentration). MCGPU-PET works in cm and eV; the conversion happens
  only in `mcgpu/vox_file.py` and `mcgpu/input_file.py`.
- **Order**: every triple, and every array's axes, are **(z, y, x)**. x varies fastest in memory (C order).

## Voxels

Along one axis with n voxels of size d, voxel i is centered at

  c_i = (i − (n − 1)/2) · d.

So the grid is centered on the scanner and spans ±n·d/2. Example: n = 4, d = 2 mm gives centers
−3, −1, 1, 3 mm, and the grid spans −4 to 4 mm. Code: `ImageGrid.voxel_centers_mm_zyx`.

parallelproj places an image by the center of voxel [0, 0, 0], which it calls `img_origin`:
c_0 = −(n − 1)/2 · d per axis (`ImageGrid.first_voxel_center_mm_zyx`). How parallelproj interpolates
between voxel centers (Joseph's method) is documented with the projector.

## Crystals and rings

Angles φ are measured from +x toward +y, in [−π, π). With N crystals per ring, crystal i covers
[−π + i·2π/N, −π + (i + 1)·2π/N) and is centered at

  φ_i = −π + (i + ½) · 2π/N.

With R rings over an axial FOV L, ring k covers [−L/2 + k·L/R, −L/2 + (k + 1)·L/R) and is centered at

  z_k = −L/2 + (k + ½) · L/R.

Source: the kernel's crystal and ring indices, `MCGPU-PET_kernel.cu` lines 1749–1752,
`ix = floor((φ/2π + ½)·N)` and `iz = floor((z/L + ½)·R)`, in the kernel's detector-centered coordinates
(below). Code: `Scanner.crystal_center_angles_rad`, `Scanner.ring_center_z_mm`. Tested by porting those
kernel lines (`tests/test_scanner.py`).

The scanner is an ideal cylinder: a crystal is a point on it, with no depth (an MCGPU-PET approximation, see
`docs/mcgpu-simulation.md`).

## Lines in a transaxial plane

A line is written x cos θ + y sin θ = s: θ is the direction of its normal and s its signed distance from
the z axis. Each line has two such descriptions, (θ, s) and (θ ± π, −s); the toolkit keeps
**θ in [−π/2, π/2)**. How MCGPU-PET's sinogram bins map onto (θ, s): `docs/sinogram-layout.md`.

## MCGPU-PET's frame, and how it maps to ours

- MCGPU-PET puts the voxel grid's **corner** at its origin: the grid spans [0, n·d] on each axis (cm).
- The `.in` file gives the detector center as `0 0 0` with a **negative** radius. A negative radius makes
  MCGPU-PET move the detector center to the grid's center (`MCGPU-PET.cu` lines 316–322).
- The kernel then measures every photon position from that center before computing crystal and ring
  (`MCGPU-PET_kernel.cu` line 1677: "Move to coordinate system with cylinder center at origin").

So, per axis: **toolkit position = 10 × MCGPU-PET position (cm) − n·d/2** (mm). Tested in
`tests/test_image_grid.py` (MCGPU-PET's voxel centers minus half the extent equal ours).