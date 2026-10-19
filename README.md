# pet-sim-recon

The BIPI lab's shared toolkit for simulated preclinical PET: simulate scans with MCGPU-PET, build
datasets, reconstruct, correct, and measure. Used by [deep-scatter-pet](https://github.com/electronics10/deep-scatter-pet) and [mr-pet-synergy](https://github.com/electronics10/mr-pet-synergy); any future
simulated-PET study can use it too. It replaces three earlier prototypes (a simulator wrapper, a dataset
generator and a reconstructor). Their code may be migrated after review (see Engineering); the
findings below are kept.

Status: **draft README, 2026-10-19**. Everything under "Decided" was confirmed by the project lead;
everything else is open.

## What it provides

- **Simulation:** run MCGPU-PET on a voxel object (material, density and activity per voxel). Output:
  trues and scatter sinograms stored separately, the emission image, and a record of every input.
- **Datasets:** many random objects drawn from user-set ranges (ellipsoids and elliptic cylinders,
  optionally inside a larger body that also has activity). Generation resumes after a crash, and every
  run can be rebuilt exactly from its recorded object list and seed.
- **Sinogram tools:** read sinograms in the simulator's exact bin order; merge mirror planes; rebin
  (SSRB, FORE) and arc-correct for analytic methods.
- **Scanner model:** forward and back projection, attenuation factors, and an adjoint test.
- **Count tools:** binomial thinning (lower count level) and Poisson splitting (two independent halves).
- **Reconstruction:** MLEM with an additive scatter term; MAP-EM with PET-only penalties and an
  MR-guided penalty (Bowsher weights); the relative difference prior (RDP).
- **Classical scatter estimation:** SSS (single scatter simulation).
- **Metrics:** generic image-quality measures shared by the studies.
- **Real images:** read the scanner's DICOM images (PET and MR), place them on the simulation grid, and
  forward-project them into synthetic raw data.

Learned models (networks, training) belong to the studies, not here.

**Real raw data** (sinograms or list-mode from a real scanner) are not available now: the Bruker scanner
gives only images. The design must not rule them out, so they can be added if a source appears.

## Decided

**Simulator facts.** Each was checked in the prototypes against the MCGPU-PET source code and against
simulated point sources. Much of the prototype code was AI-written and not fully reviewed, so each fact
gets a test here before anything relies on it.

- **Always simulate at span 1** (one sinogram plane per ordered ring pair). Compression at simulation time
  cannot be undone, and reconstruction needs the uncompressed planes. Rebinning to a higher span is
  allowed afterwards, as a view of the data.
- **File layout:** sinograms are `(plane, angle, radial)` with the radial index fastest. At span 1 the
  simulator fills only every other plane slot; reading strips the empty slots.
- **Angles** run from +90° to −90° across the angular bins.
- **Radial bins are not arc-corrected:** bin position `s = R · cos(π m / N_det)` (R = scanner radius,
  m = crystal-index separation). Iterative reconstruction models this exactly; analytic methods (FBP,
  FORE) arc-correct first.
- **Oblique planes are mixed:** the simulator decides a plane's sign from photon tracking order, which
  is random. So plane (a, b) holds a 50/50 mix of lines (a, b) and (b, a). Its forward model is the mean
  of the two mirror lines' projections, and mirror planes can be added exactly.
- **The scanner model is built from the exact line endpoints** (parallelproj's low-level projectors),
  not from a library-defined scanner geometry. The library's own bin order differs from the
  simulator's; the mismatch gave near-black MLEM images.

**Modelling choices.**

- **Scatter is added** to the forward model (`A x + s`), never subtracted from the data. Subtraction
  makes negative data and breaks the Poisson model.
- **The simulated scanner matches the real Bruker 7 T PET** in the physically real quantities: radius
  52.5 mm, axial field of view 150 mm, transaxial field of view 80 mm, energy window and energy
  resolution. The real scanner's monolithic crystals cannot be modelled in MCGPU-PET, so the discrete
  crystal layout is a stand-in.
- **Water at a chosen density stands in for soft tissue:** at 511 keV, attenuation follows electron
  density, which follows mass density for light elements.
- **Simulate at the highest count level needed;** lower levels come from thinning, which is free.
- **Reconstructions have a free overall scale;** compare absolute values only after matching scale on a
  known region.

**Engineering.**

- **One Python package**, split into clear internal parts. The GPU simulator is an optional extra, so
  everything else installs on any machine. Split into several packages only if someone outside the lab
  needs one part alone.
- **Environment: pixi**, because parallelproj and CuPy need non-Python libraries from conda-forge.
- **Public** on GitHub.
- **Studies pin a toolkit version.**
- **Prototype code may be migrated** one module at a time, but only once the lead has reviewed and
  understood it and tests cover it.
- **Reference speed (prototype):** 80 × 80 × 149 voxels of 1 mm, about 174 M lines of response, about
  2.6 s per MLEM iteration on one GPU.
- **Cite** Badal et al. 2024, *MCGPU-PET*, Comput. Phys. Commun. 295:109008, in any publication that uses
  the simulator.

## Open questions

1. File format for simulation runs and reconstructions (e.g. HDF5, NumPy files + JSON, NIfTI).
2. Does MCGPU-PET model detector-efficiency differences between lines? If yes, how to compute
   normalization factors.
3. How to ship MCGPU-PET in a public repository: a bundled compiled binary breaks on other GPU types,
   and MCGPU-PET's own licence needs checking. Building from source may be better.
4. Licence for the toolkit (the prototypes used MIT).
5. Which metrics belong here and which stay study-specific.
6. Whether convolution subtraction is needed (depends on [deep-scatter-pet](https://github.com/electronics10/deep-scatter-pet) open question 1).
7. Where the MR-like image generator lives: here, or in [mr-pet-synergy](https://github.com/electronics10/mr-pet-synergy).
8. The split between this toolkit and the two studies is a first draft and may change.
9. How real raw data would enter, if a source appears: a reader that turns it into the same sinogram
   layout as simulated data.

## Previous works (may serve as prototypes)

- [mcgpu-pet-wrapper](https://github.com/electronics10/mcgpu-pet-wrapper)
- [mcgpu-recon](https://github.com/electronics10/mcgpu-recon)
- [pet-sim-gen](https://github.com/electronics10/pet-sim-gen)
- [pet-scatter-correction](https://github.com/electronics10/pet-scatter-correction) (SSS and real-image tools)
