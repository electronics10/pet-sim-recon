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

## Axial: ring pair → plane

The kernel names the two photons' rings `iz1` and `iz2` in **photon order**, the order it tracked them,
which is random. Not crystal order. With R rings (`MCGPU-PET_kernel.cu` lines 1758–1773):

- **Segment.** A segment collects ring differences δ = |iz2 − iz1| near a multiple of the span:
  tilt = floor((δ − 1 + (span+1)/2) / span), so tilt = δ for span 1, and tilt 1 is δ 6–16 for span 11.
  Tilt 0 is segment 0. Each tilt t > 0 gives segment **2t − 1 when iz2 < iz1** and **2t when iz2 > iz1**.
- **Plane in the segment:** set by the ring sum iz1 + iz2 (twice the LOR's axial midpoint, in rings),
  counted from the segment's smallest ring difference.
- **Segment start:** the total size of the earlier segments, the same count `read_input` uses for the
  file's number of planes (`MCGPU-PET.cu` lines 1176–1187). Segment 0 has `num_axial_slices` = 2R − 1
  planes, and higher tilts have fewer.

Code: `kernel_axial_plane` (the port) and `ring_pair_planes` (the table, built by sending every ring pair
through the port).

Worked example, 4 rings, span 1, with `planes[iz1, iz2]`:

|  | iz2 = 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| **iz1 = 0** | 0 | 12 | 20 | 24 |
| **1** | 7 | 2 | 14 | 22 |
| **2** | 17 | 9 | 4 | 16 |
| **3** | 23 | 19 | 11 | 6 |

For span 1, only every other plane of a segment is used, because the ring sum and δ have the same parity.
So 16 of these 25 planes hold counts, and 5625 of 11101 on the prototype's scanner. For span 3, δ 0 and 1
share segment 0, so (0, 1) and (1, 0) land in the same plane: axial compression.

**Settings the kernel would mis-bin silently**, checked by `check_sinogram_fits_scanner`, which the runner
calls:

- `num_angular_bins` ≠ N/2 (see the transaxial section);
- `num_axial_slices` ≠ 2R − 1: with fewer, segments overlap;
- span and max ring difference such that the kernel's tilt for some δ ≤ MRD exceeds the
  2 floor(MRD/span) + 1 segments `read_input` allocates. Those events land past the end of the file and
  are dropped (e.g. 16 rings, span 4, MRD 15). `ring_pair_planes` replays every ring pair and raises.

## The mirror mix

The kernel computes the plane (line 1773) **before** it swaps the photons into crystal order (line 1777).
So for an LOR A = (crystal ix1 in ring a, ix2 in ring b), a ≠ b, the photon order decides the plane:
P(a, b) when photon 1 hits ix1, P(b, a) when it hits ix2. Photon order doesn't depend on which crystal
has the lower index, so each LOR's counts split half and half between P(a, b) and P(b, a), in the same
(view, radial) bin. Each of those planes holds half of A and half of its mirror B = (ix1 in ring b, ix2 in
ring a).

The physics is unaffected. Every coincidence is counted once, in the right transaxial bin, at the right
axial midpoint and |δ|. What's lost is which of the two tilts A and B it had. A and B cross at the middle of
the chord, so the loss is small near the axis and grows for sources far from the chord's middle. Reading
P(a, b) as "A only", the usual meaning on a real scanner, would be wrong: it blurs off-center sources along
z. My reading of the code is that this is probably unintended: line 1777 swaps the rings along with the
crystals, as if they were meant to stay paired, but the plane is already fixed by then.

**Measured** (`tests/test_point_source.py`, and 10⁶ decays while drafting): in oblique bins where only one of
A and B passes the source, A holds 0.4994 of the counts (267,581 against 268,242). Without the mix it would
hold all of them.

**What the toolkit does:** `sum_mirror_planes` adds every plane onto its mirror. The ±δ planes have the
same mean, so nothing is lost, and the oblique planes halve. A **summed bin holds every LOR whose crystal
pair is the bin's and whose two rings, in either assignment to the two crystals, are a ring pair of the
plane, each with weight 1** (a direct pair a = a is one LOR). A real scanner's sinogram summed the same way
has exactly this meaning too, so summed simulated and summed real data match in structure.

## End to end: a point source

`tests/test_point_source.py` simulates a point source at (z, y, x) = (20, 10, 16) mm on the prototype's
scanner (span 1) and checks the layout against it:

- **> 99.9 % of trues** fall in bins with an LOR (A or B) passing within 3 mm of the source. 3 mm =
  half the 2 mm source voxel's diagonal (1.73 mm), plus the gap between a line through crystal centers and
  the photons' true line (half a crystal and half a ring at the ends: 1.13 mm), plus acollinearity
  (~0.2 mm). A wrong sign, axis or ring order would put counts tens of mm away.
- **Transaxial:** the count-weighted mean of s(bin) − s(source) is below 0.05 mm (measured −0.001 mm). A
  half-bin offset would give about 0.26 mm.
- **Axial:** in direct planes, the count-weighted mean ring z equals the source z within 0.05 mm (measured
  −0.0002 mm).
- **Mirror mix:** 0.48 < fraction A < 0.52 (above).
- Summing mirror planes keeps every count.

While drafting, the transaxial residuals also matched a numpy replica of the kernel geometry (emission,
acollinearity, cylinder hit, crystal index), to within 0.002 per 1 mm bin of the residual histogram. The object needs one dense voxel for this to hold:
see "Woodcock overshoot" in `docs/mcgpu-simulation.md`.