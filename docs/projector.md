# The projector: what each sinogram bin measures

Each bin of the mirror-summed sinogram (`docs/sinogram-layout.md`) holds the coincidences of one or two
lines between crystal centers: line A for every bin, and line B, its mirror, for oblique planes. The
expected number of trues in a bin is

    ȳ = Σ over the bin's lines ℓ of   T_dec · E_ℓ / (2π) · (1 / 1000) · exp(−∫_ℓ μ dl) · ∫_ℓ x dl

with x the activity in Bq/mL, ∫_ℓ x dl its line integral in (Bq/mL)·mm, E_ℓ the étendue of the line's
crystal pair in mm², μ the attenuation coefficient in 1/mm, T_dec the decay-weighted scan time in s, and
1000 mm³ per mL.

Code: `recon/projector.py` (the line integrals), `recon/system_model.py` (E and T_dec),
`mcgpu/sinogram_layout.py` (`summed_bin_lines`: the lines of each bin).

## Line integrals: Joseph's method

parallelproj 2 computes ∫ x dl with Joseph's method (Joseph 1982, IEEE TMI 1(3):192–196). Step through the
image along the axis the line runs most along (the *principal* axis), one voxel plane at a time. In each
plane, read the image at the point where the line crosses it, by linear interpolation between the
neighboring voxel centers (bilinear in 3D). Multiply the sum by the step length along the line: the voxel
size divided by the cosine between the line and the principal axis.

**Worked example (2D, 1 mm pixels, centers at −1, 0, 1 in x and y).** The line y = x/2 + 1/4 runs more
along x than along y, so x is the principal axis. The step along the line is √(1 + 1/4) = 1.118 mm.

| Column x | Line's y there | Interpolation weights |
| --- | --- | --- |
| −1 | −0.25 | 0.25 to y = −1, 0.75 to y = 0 |
| 0 | 0.25 | 0.75 to y = 0, 0.25 to y = 1 |
| 1 | 0.75 | 0.25 to y = 0, 0.75 to y = 1 |

- A uniform image of ones gives 3 × 1.118 = 3.354 mm: the length of the line across the 3 mm wide grid.
- An image with only the center pixel at 1 gives 0.75 × 1.118 = **0.839 mm**. The line's exact path through
  that pixel's square is 1.118 mm (it enters at (−0.5, 0) and leaves at (0.5, 0.5)). Joseph gives
  0.28 mm of it to the pixel above, which the line passes close to.

So Joseph's method is an *interpolation*, not the exact path length per voxel (that is Siddon's method,
Siddon 1985, Med Phys 12(2):252–255). It is smoother and costs the same. Both give the same total over a
uniform region. All three numbers above were computed with parallelproj.

At the grid's edge, interpolation partners outside the grid count as zero, so a line skimming the edge
sees part of an edge voxel.

**parallelproj version.** We use parallelproj ≥ 2.0.2. In 1.10.2 the forward and back-projection kernels broke
the tie |dy| = |dx| (a line at exactly 45° in the transaxial plane) toward different axes
(`projector_kernels.cu` lines 195–203 vs 531–543), so the back-projection was not the transpose of the
forward one on those lines: up to 12 % per line, 192 of 30,880 bins on the prototype scanner. In parallelproj
2 both share one function (`ray_cube_intersection_joseph` in libparallelproj).

## Sensitivity: the étendue of a crystal pair

**The setup.** MCGPU-PET's detector is the bare cylinder: a photon is assigned to the crystal (angle) and ring
(z) where its line crosses the cylinder (`tally_Sinogram`, kernel lines 1645–1687). So a crystal is a patch
of the cylinder, of area A = R · (2π/N) · (ring pitch), facing the axis.

**Step 1: one emission point.** A positron annihilation emits two photons back to back, along a direction u
uniform on the sphere. The pair is caught by faces 1 and 2 if the line through the point along u crosses
both. Either photon can go to face 1, so the probability is 2Ω/(4π) = Ω/(2π), where Ω is the solid angle of
directions from the point that pass through both faces.

**Step 2: a thin tube.** The lines joining the two faces form a thin tube. Cut it across at some position
along the line, and integrate Ω over the cut:

  ∫ Ω dA = E = A cos α₁ · A cos α₂ / L²

where L is the distance between the faces and α₁, α₂ are the angles between the line and each face's normal.
E is the *étendue* of the pair (radiometry's measure of a bundle of lines). Check it at face 1 itself: the cut
is face 1 seen along the line, area A cos α₁, and every point on it sees face 2 under the solid angle
A cos α₂ / L². E is the same at every cut, because it just counts the lines in the bundle.

**Step 3: activity along the tube.** Expected counts = ∫ x(P) Ω(P)/(2π) dV. Write dV = dA dl, along and across
the tube. If x doesn't change across the tube's width (about one crystal):

  counts per second = (E / 2π) · ∫ x dl.

So the line integral is the right quantity, and E/(2π) is its weight. The longer path of a slanted line
through the activity is real; it is part of ∫ x dl.

**On a ring, E is cos⁴θ.** Both faces of a chord make the same angle α with their normals (the chord is
symmetric). Let θ be the line's angle to the transverse plane, and β the angle between its transverse
projection and the radius at the face. Then

- cos α = cos θ · cos β;
- the transverse chord is D = 2R cos β, so L = D / cos θ = 2R cos β / cos θ;
- E = A² cos²α / L² = A² cos²θ cos²β · cos²θ / (4R² cos²β) = **A² cos⁴θ / (4R²)**.

β cancels: **no dependence on the radial bin**, only on the ring difference. A chord far from the center
hits the faces at a slant (smaller cos β) but is shorter by the same factor.

The prototype found the cos⁴θ weight by fitting (REWRITE_NOTES §1, exponent 4.03–4.08). Its explanation (a
cos³ detection probability, times a Joseph integral that "grows as 1/cos") lands on the same weight. Here the
whole cos⁴ is geometry, and the 1/cos of the path is physics, not an error. The prototype's leftover deficit near the radial edge is not a sensitivity error. On the NEMA phantom (radius 15 mm)
the simulated/model ratio is 1.01, 0.97 and 1.39 at 29, 30 and 31 bins from the center (s = 14.8, 15.2, 15.7 mm), and
0.9996 summed over bins 24 to 36 (0.9979 on the opposite side). The simulated edge is slightly wider than the model's. That is what a model without
crystal width or acollinearity should give, though we haven't tested the two causes separately.

**Scale.** On the prototype scanner (R = 52.5 mm, 320 crystals, 2 mm rings), A = 2.06 mm². A direct line
through the center has E = A²/(2R)² = 3.86 × 10⁻⁴ mm². A 100 s scan of 10 kBq/mL along 50 mm of that line
gives 100 × 3.86 × 10⁻⁴/(2π) × 10⁴ × 50 / 1000 ≈ 3 counts.

## Attenuation

A true needs both photons to cross the object with no interaction: Compton and Rayleigh events mark a photon as
scattered (kernel lines 366–370 and 383–386), and a photoelectric event absorbs it. Together the two photons
cover the whole line, wherever the pair was emitted, so the survival probability is exp(−∫_ℓ μ dl), with μ from the
total mean free path in the material file at 511 keV, scaled by the voxel's density (`attenuation_map_per_mm`).

**The factor is per line, not per bin.** The two lines of a summed bin cross different tissue, so a bin's expected
count is a_A J_A + a_B J_B, which in general isn't a common factor times (J_A + J_B). (The prototype used
exp(−½(∫_A μ + ∫_B μ)) on the mean of the two lines, an approximation.)

## Decay

MCGPU-PET decays the activity as exp(−t/τ) during the scan (kernel line 286). A scan of length t then counts
T_dec = τ(1 − e^(−t/τ)) seconds of the starting activity (`AcquisitionSettings.decay_weighted_time_s`).

## Check: absolute counts from a point source

With no fitted constant, the model predicts how many trues MCGPU-PET detects from a point source in air:

| Run (10⁶ decays) | Simulated trues | Predicted | Ratio |
| --- | --- | --- | --- |
| water voxel in a corner | 729,758 | 730,055 | 0.9996 |
| 0.1 g/cm³ voxel in a corner | 730,128 | 730,055 | 1.0001 |

Poisson noise on these counts is 0.12 %. `tests/test_point_source.py` checks the ratio to 1 % on every run.
So reconstructed images come out in Bq/mL.

## Approximations

- **A line, not a tube.** Activity is taken as constant across a tube about one crystal wide.
- **Joseph's interpolation** instead of exact path lengths.
- **Small faces.** The étendue formula treats each face as small compared with L (2 mm against ~100 mm).
- **Every unscattered photon is accepted.** With 15 % FWHM at 511 keV, the 358–664 keV window sits at
  ±4.7σ.
- **Acollinearity is ignored.** It moves counts between neighboring bins but keeps the total. MCGPU-PET has
  no positron range (kernel 690–692).
- **Attenuation along the central line.** exp(−∫μ dl) is evaluated on the line, not averaged across the tube.