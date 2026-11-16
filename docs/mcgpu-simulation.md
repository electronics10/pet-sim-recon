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