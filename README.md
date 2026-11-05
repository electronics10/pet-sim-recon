# pet-sim-recon

This is a toolkit for simulating PET by Monte Carlo simulation. It also provides the corresponding reconstruction and some other features such as scatter correction. 

The Monte Carlo simulation is based on [MCGPU-PET](https://github.com/DIDSR/MCGPU-PET) (Herraiz et al., Comput. Phys. Commun. 295 (2024) 109008, doi:10.1016/j.cpc.2023.109008, developed at the US FDA and in the public domain (17 U.S.C. §105)).

## What it (should) provides

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
- **Real raw data:** the scanner operator agreed to provide list-mode data from the 2026-11-02 phantom
session (record in `experiments/2026-11-02-phantom-pilot/` of [mr-pet-synergy](https://github.com/electronics10/mr-pet-synergy) and [deep-scatter-pet](https://github.com/electronics10/deep-scatter-pet)).
Using it needs the file format, normalization data and the detector geometry. The real detector
(monolithic LYSO blocks with depth of interaction) differs from the simulated discrete-crystal stand-in,
so real data need their own scanner model. The design must leave room for this.

## Open questions

1. How to ship MCGPU-PET in a public repository: a bundled compiled binary breaks on other GPU types,
   and MCGPU-PET's own licence needs checking. Building from source may be better.
2. Which metrics belong here and which stay study-specific.
3. Whether convolution subtraction is needed (depends on [deep-scatter-pet](https://github.com/electronics10/deep-scatter-pet) open question 1).
4. Where the MR-like image generator lives: here, or in [mr-pet-synergy](https://github.com/electronics10/mr-pet-synergy).
5. How real list-mode data would enter: a reader for the Bruker list-mode format, and a scanner model for the real detector geometry, separate from the simulated one.

## Note for developer

This is mainly a scientific research project/tool. There is no clear-cut goal. Every design decision may be changed since we are not sure what to expect later on.

This toolkit would later be used by [deep-scatter-pet](https://github.com/electronics10/deep-scatter-pet) and [mr-pet-synergy](https://github.com/electronics10/mr-pet-synergy), both of which are my downstream research projects; any future simulated-PET study can use it too. It replaces my earlier prototypes, of which the code may be migrated after review. The earlier prototypes:

- [mcgpu-pet-wrapper](https://github.com/electronics10/mcgpu-pet-wrapper)
- [mcgpu-recon](https://github.com/electronics10/mcgpu-recon)
- [pet-sim-gen](https://github.com/electronics10/pet-sim-gen)
- [pet-scatter-correction](https://github.com/electronics10/pet-scatter-correction)



