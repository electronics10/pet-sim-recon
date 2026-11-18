# Running an MCGPU-PET simulation

```python
from pet_sim_recon.mcgpu.runner import run_simulation

run_simulation(run_dir, voxel_object, scanner, sinogram, acquisition, random_seed=12345)
```

`run_simulation` needs the binary from `pixi run build-mcgpu` (see `mcgpu-build.md`). It creates
`run_dir`, and refuses if the folder already exists, so a run is never overwritten. It then writes:

| File | What it is |
| --- | --- |
| `settings.json` | Scanner, grid, sinogram and acquisition settings, material names, the seed, and the binary path |
| `object.vox.gz` | The voxel object, in MCGPU-PET's units (activity in Bq per voxel, sizes in cm) |
| `MCGPU-PET.in` | The simulator's input file |
| `materials/` | A link to `external/mcgpu-materials/` (see `materials.md`) |
| `MCGPU-PET.out` | Everything the simulator printed |
| `sinogram_Trues.raw.gz`, `sinogram_Scatter.raw.gz` | The sinograms, read in the sinogram-layout chunk |
| `image_Trues.raw.gz`, `image_Scatter.raw.gz` | Detected coincidences per voxel of origin |
| `Energy_Sinogram_Spectrum.dat` | Energy spectrum of the detected photons |

## What the .in file sets

`pet_sim_recon/mcgpu/input_file.py` writes the values in the order `read_input` (`MCGPU-PET.cu`,
lines 905–1235) reads them. Sections are found by exact header text, version date included. Within a
section, values are read one per line, text after `#` is ignored, and blank lines are skipped.

Units at the boundary: lengths in cm (toolkit: mm), energy window in **eV** (toolkit: keV). Upstream's own
comment on the window lines says "keV", but the photon energies the kernel compares against are in eV
(510998.95 eV at emission).

Values the toolkit fixes, and why:

| Value | Setting | Why |
| --- | --- | --- |
| Random seed | required, ≥ 1 | MCGPU-PET's 0 means "seed from the clock", which can't be reproduced |
| Report mode | trues and scatter, sinogram only | The phase-space file lists events in GPU arrival order, which varies between runs |
| Phase-space size | 1000 | Still allocated by `read_input`, but never filled in sinogram-only mode |
| Per-material activity table | empty (`0 0.0`) | The old activity model; activity comes from the `.vox` file |
| Dose tallies | off | Not needed |
| Image resolution | 128 | Read, but used only in commented-out kernel code |
| Energy bins | 700 | Bins of `Energy_Sinogram_Spectrum.dat`, upstream's value |
| Materials | the object's `material_names`, in ID order | ID k means the k-th name; see `materials.md` |

Approximations in the simulator's physics:

- **Energy resolution:** each photon's energy is blurred by a Gaussian with FWHM = R × (its own
  energy) (kernel line 1803): the same fraction R at every energy. In a real scintillator the fraction
  grows at lower energy, roughly as 1/√E (assumption: counting statistics of the scintillation light
  dominate). So MCGPU-PET *under*-blurs low-energy photons. With R = 12 % at 511 keV, a real detector
  would be at about 12 % × √(511/340) ≈ 15 % at 340 keV, while MCGPU-PET stays at 12 %. Scattered photons
  are the low-energy ones, so this changes how many of them cross the lower edge of the window. Likely
  effect (reasoning, not yet checked): there are more scattered photons just below the edge than just
  above it, so a wider blur pushes more of them in than out. MCGPU-PET would then accept slightly less
  scatter than a real scanner with the same resolution quoted at 511 keV.
- **Energy window:** both photons must be strictly inside (low, high).
- **Decay:** activity decays as exp(−t/τ) during the scan (kernel line 286), with τ the mean life. For F-18
  (τ = 9502 s), a 100 s scan loses about 1 %.

## Woodcock overshoot: why every object needs one dense voxel

MCGPU-PET moves photons with Woodcock tracking: steps drawn with the *shortest* mean free path in the whole
grid (`MCGPU-PET_kernel.cu` line 320). When a step leaves the grid, the photon is tallied where that step
left it (line 328), not at the grid's edge. `tally_Sinogram` then intersects the photon's line with the
detector cylinder in float32 (lines 1645–1687). Far from the cylinder, the quadratic's discriminant is the
difference of two numbers ~10⁸ cm², and float32 keeps only ~7 digits of them, so the hit point and the
crystal come out wrong.

The step length is what matters. With water anywhere in the grid, the mean free path at 511 keV is about
10 cm and the error is negligible. In an all-air grid (0.0012 g/cm³), it is about 90 m (μ/ρ ≈ 0.087 cm²/g,
NIST XCOM).

Measured with a point source in an all-air grid (10⁶ decays): 21 % of trues fell more than 3 mm from the
source, the mean |transaxial residual| was 1.3 mm instead of 0.52 mm, and 11 % of trues were lost. Adding a
single voxel of 0.1, 0.3 or 1.0 g/cm³ in a far corner made all of it disappear: the results then matched an
exact numpy replica of the kernel geometry.

So `run_simulation` refuses objects whose densest voxel is below 0.1 g/cm³ (`check_peak_density`), the
lowest density tested. Any real phantom passes. For an object in air, put one water voxel in a corner.

## Verification (2026-11-16)

Upstream's `sample_simulation` was run through the toolkit:

- the sample `.vox` read with `read_vox` and written again with `write_vox`;
- the `.in` written by `write_input_file`;
- the run launched by `run_simulation`, with seed 12345.

The outputs (`sinogram_Trues`, `sinogram_Scatter`, `image_Trues`, `image_Scatter`, decompressed, and
`Energy_Sinogram_Spectrum.dat`) are byte-identical to the build-check run in `mcgpu-build.md`. That run used
upstream's own `.in` and `.vox` with the same seed. Both runs simulated 23,866 histories.

So for this sample, the toolkit's `.vox` writer, `.in` writer and runner give the simulator exactly the
input that upstream's own files give it. The unit test `tests/test_input_file.py` checks the same thing
without a GPU, for the physics and geometry sections. The run is kept in `data/runs/sample-reproduction/`
(not in git).