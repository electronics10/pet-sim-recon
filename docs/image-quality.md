# Image-quality metrics

"Recovery coefficient" means different things in different papers. This document defines each number we report, under
its own name, with its source. All of them are computed on the NEMA NU 4-2008 image-quality phantom unless stated.

## The phantom and its regions

The NEMA NU 4 image-quality phantom is a 30 mm-diameter cylinder in three axial sections (scanner-centered frame,
`docs/coordinate-frame.md`):

| Section | z (mm) | Contents |
| --- | --- | --- |
| Rods | −25 to −5 | Solid PMMA with five fillable rods, 1–5 mm in diameter, 7 mm from the axis |
| Uniform | −5 to 10 | Activity everywhere |
| Cold inserts | 10 to 25 | Activity around two cold cylinders, one of water and one of air |

The regions we measure in:

| Region | Size |
| --- | --- |
| Uniform VOI | cylinder, 22.5 mm diameter, central 10 mm of the uniform section |
| Rod ROI (NEMA) | circle twice the rod's diameter, on the central 10 mm of the rods, averaged along z |
| Rod core (ours) | circle of radius d/2 − 0.5 mm, central 10 mm |
| Cold-insert VOI | cylinder, 4 mm diameter, central 7.5 mm of the insert |

*Assumption:* the NEMA sizes in this table come from the prototype's code and common usage. We have not checked them
against the text of NEMA NU 4-2008.

Boundaries are inclusive: a voxel whose center lies exactly on a boundary counts as inside. On a 1 mm grid with an
odd number of slices (the talk's 149), the section boundaries fall on voxel centers, so the rods' "central 10 mm"
is 11 slices, and the later-painted section owns each boundary slice (`phantoms.nema_iq_phantom`).

## Noise: %STD

    %STD = 100 × SD / mean, over the voxels of the uniform VOI

It measures voxel-to-voxel noise in one image. Lower is better. It is not the spread between realizations, which we
report as ± SD over an ensemble (`docs/map-em.md`).

## Recovery of a hot object

### RC_mean: partial-volume recovery (our primary measure)

    RC_mean = mean in the rod core / true activity concentration

This is the original meaning of "recovery coefficient" (Hoffman, Huang & Phelps 1979). Limited resolution spreads a
small object's activity into its surroundings, so the object looks less concentrated than it is: the partial-volume
effect. RC_mean against object size is the recovery curve used for partial-volume correction.

- 1 is correct. Below 1 is lost contrast; above 1 is overshoot (ringing, `docs/map-em.md`).
- A mean over many voxels averages the noise away, so RC_mean measures bias, not noise.
- The core stays 0.5 mm inside the rod's edge, so voxels cut by the edge are left out.
- It needs the true activity. In simulation we know it. On real data, use the measured concentration of the filling
  solution.

### RC_NEMA: the NEMA NU 4-2008 recovery coefficient

    RC_NEMA = max over the rod ROI of the 10 mm-averaged image / mean of the uniform VOI

The standard averages the central 10 mm of the rods along z, finds the pixel with the largest average, and takes the
mean along the axial line profile through it (Harteveld et al. 2011). That is the same as the maximum of the averaged
image.

Two biases, both in the direction of a larger RC:

- **The maximum rises with noise** (Hallen et al. 2020). Only the pixels at the rod's level compete for the maximum;
  the cold PMMA around the rod never wins. With independent noise of SD σ per voxel, averaging S slices leaves
  σ/√S, and the largest of the n rod pixels sits about E[max of n standard normals] × σ/√S above their mean. For the
  3 mm rod on 1 mm voxels, n = 7 and E[max] = 1.35: at 20 % noise over 10 slices, +8.5 % (`tests/test_image_quality.py`
  measures +8.4 %). Reconstructed noise is correlated between neighboring voxels and slices, so on reconstructions
  this is only a rough guide. Unfiltered MLEM at 23 iterations (realization 0, 19.4 % noise) gives RC_NEMA = 1.19
  against RC_mean = 1.05.
- **The denominator is a measured mean.** For the uniform priors it rises by edge ringing (`docs/map-em.md`, issue #25),
  which lowers RC_NEMA instead.

We report RC_NEMA only to compare with published NEMA results. Its standard deviation, as the standard defines it,
propagates the profile's SD and the uniform %STD; Hallen et al. note that this is not the SD of a maximum.

### CRC: contrast recovery (NEMA NU 2)

For a hot object in a warm background:

    CRC = (C_H / C_B − 1) / (a_H / a_B − 1)

C is the measured mean, a the true concentration, H the hot object and B the background (SNMMI phantom-report guide;
arXiv 2402.06742). It is the background-subtracted contrast, rescaled by how well the background itself came out:

    CRC = (C_H − C_B) / (a_H − a_B) × a_B / C_B

**Example.** a_H = 4, a_B = 1; measured C_H = 3.0, C_B = 1.1.
RC_mean = 3.0 / 4 = 0.75. CRC = (3.0 / 1.1 − 1) / (4 − 1) = 1.727 / 3 = 0.576.
Check: (3.0 − 1.1) / 3 × 1 / 1.1 = 0.633 / 1.1 = 0.576.

CRC asks "how much of the contrast against the surroundings survives", which is what a reader sees. RC_mean asks "is
the number right". They differ when the background is biased.

**It does not apply to the NU 4 rods**: they sit in cold PMMA, so a_B = 0 and a_H / a_B is undefined. CRC needs a
phantom with hot objects in a warm background, which is also how lesions look in real tissue.

## Cold objects: spill-over ratio

    SOR = mean in the cold-insert VOI / mean of the uniform VOI

0 is correct (NEMA NU 4). NEMA NU 2 reports the same thing as a cold contrast, Q_C = 1 − C_C / C_B. Like RC_NEMA, SOR
divides by a measured mean, so edge ringing in the uniform region lowers it slightly for the uniform priors.

## What we report

| Name | Formula | Use |
| --- | --- | --- |
| `pct_std` | SD / mean in the uniform VOI | Noise axis of every comparison |
| `rc_mean` | rod-core mean / true | Primary recovery measure: bias |
| `rc_nema` | max of the 10 mm-averaged ROI / uniform mean | Only to compare with published NEMA results |
| `sor` | cold-insert mean / uniform mean | Spill-over, NEMA NU 4 |
| `crc` | NU 2 form | Hot objects in a warm background (a future phantom) |

Methods are compared at equal `pct_std`, with means and SDs over an ensemble of noise realizations.

## Not yet checked

- The NEMA NU 4 ROI and VOI sizes above, against the standard's text.
- The exact %STD formula for RC_NEMA in the standard.

## References

- Hoffman, Huang & Phelps, J Comput Assist Tomogr 3(3):299–308, 1979.
- NEMA NU 4-2008, Performance Measurements of Small Animal Positron Emission Tomographs.
- Harteveld et al., J Nucl Med 52(10):1646–1653, 2011.
- Hallen, Schug & Schulz, EJNMMI Phys 7:12, 2020.
- SNMMI, "Understanding your NEMA IQ phantom report", v1.0.
