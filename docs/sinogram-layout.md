# Sinogram layout

What each bin of MCGPU-PET's sinograms (`sinogram_Trues.raw.gz`, `sinogram_Scatter.raw.gz`) means.
Positions and the line convention (θ, s) are in `docs/coordinate-frame.md`. Code:
`src/pet_sim_recon/mcgpu/sinogram_layout.py`.

## Transaxial: crystal pair → (view, radial bin)

With N crystals per ring, `NANGLES` views and `NRAD` radial bins, the kernel bins crystals ix1 < ix2
(swapped if needed) as follows (`tally_Sinogram`, `MCGPU-PET_kernel.cu` lines 1777–1782):

```c
ith = ((ix1 + ix2 + NANGLES + 1) % NCRYSTALS) / 2;    // view
ir  = abs(ix2 - ix1 - NANGLES);                       // distance from the center bin
if ((ix1 < ith) || (ix2 >= ith + NANGLES)) ir = -ir;  // which side of the center
ir  = ir + NRAD / 2;                                  // radial bin; dropped if outside [0, NRAD)
```

Why it works: draw the radii to the two crystals. The chord between them has its normal halfway between
them, at (φ_a + φ_b)/2, and lies at distance R |cos(π (b − a)/N)| from the center. So **the sum a + b
decides the view, and the difference b − a decides the radial bin.**

**Requirement: NANGLES = N/2.** The kernel's view index runs 0 … N/2 − 1 and is never checked against
NANGLES (lines 1778–1787). With other values, events land in the wrong place silently.
`check_sinogram_fits_scanner` enforces it, and the runner calls it.

### Worked example: 8 crystals (4 views, 7 radial bins)

Each cell: crystal pair, θ, s/R.

| View | bin 0 | bin 1 | bin 2 | bin 3 | bin 4 | bin 5 | bin 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | (5,6) −90°, −0.924 | (5,7) −67.5°, −0.707 | (4,7) −90°, −0.383 | (0,4) −67.5°, 0 | (0,3) −90°, +0.383 | (1,3) −67.5°, +0.707 | (1,2) −90°, +0.924 |
| 1 | (6,7) −45°, −0.924 | (0,6) −22.5°, −0.707 | (0,5) −45°, −0.383 | (1,5) −22.5°, 0 | (1,4) −45°, +0.383 | (2,4) −22.5°, +0.707 | (2,3) −45°, +0.924 |
| 2 | (0,7) 0°, −0.924 | (1,7) 22.5°, −0.707 | (1,6) 0°, −0.383 | (2,6) 22.5°, 0 | (2,5) 0°, +0.383 | (3,5) 22.5°, +0.707 | (3,4) 0°, +0.924 |
| 3 | (0,1) 45°, −0.924 | (0,2) 67.5°, −0.707 | (2,7) 45°, −0.383 | (3,7) 67.5°, 0 | (3,6) 45°, +0.383 | (4,6) 67.5°, +0.707 | (4,5) 45°, +0.924 |

Example: crystals 0 and 3 (−157.5°, −22.5°) are 135° apart, so |s| = R cos 67.5° = 0.383 R. The normal
points at −90°, so the chord is horizontal at y = −0.383 R, and s = −y = +0.383 R. View 0, bin 4.

## The convention

- **The source of truth for a bin is its exact crystal pair** (`crystal_pair_table`, built by sending
  every pair through a port of the kernel lines). θ and s below follow from it.
- **View k** has nominal angle **θ_k = −π/2 + k·π/NANGLES**, increasing with k.
- **Interleaving:** bins at an even distance from the center bin sit at θ_k + half a view step. Neighboring
  radial bins come from neighboring sums a + b, and the `/ 2` puts two sums into one view.
- **Radial bin r** with d = r − NRAD/2 lies at **s = R sin(π d / N)**, in every view: s increases with r,
  and the center bin is s = 0. This follows from |s| = R |cos(π (b − a)/N)| with b − a = N/2 ± d.
- **Not arc-corrected:** bins are evenly spaced in crystal difference, not in s. For 8 crystals the steps
  are 0.383 R near the center and 0.217 R at the edge.
- One pair per bin: every bin holds exactly one crystal pair. Pairs beyond the outermost radial bin
  (near-tangent lines) are dropped by the kernel.

Your prototype's "+90° → −90°" describes the same data with the angle measured differently (line direction
instead of normal, or s signed the other way). That ambiguity is why the convention is pinned by tests.

Checked by `tests/test_sinogram_layout.py`: the 8-crystal table above, every bin's exact line on the
nominal (θ_k, s) grid with interleaving, for 8 and 336 crystals (upstream's sample scanner), and one pair
per bin. The same checks also passed for 320 crystals (your prototype's scanner) while drafting.

**Floating-point note:** folding θ into [−π/2, π/2) uses a 1e-9 rad tolerance
(`ANGLE_TOLERANCE_RAD`). Without it, lines at exactly ±90° fold the wrong way through rounding.

## Who uses which

- The projector (our module, from the projector chunk) turns each bin's exact crystal pair into LOR end
  points and has parallelproj integrate along them, so interleaving and the radial spacing are modeled
  automatically. parallelproj's own scanner and sinogram classes use a different bin order and are not used.
- Rebinning, arc correction and FBP use (θ, s) and must handle the interleaving and spacing explicitly.

## Not yet verified

That a simulated point source lands in the bin this predicts. That end-to-end check needs the sinogram
reader and comes with the axial layout.