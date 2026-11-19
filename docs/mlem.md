# MLEM: from the Poisson model to the update

Code: `recon/mlem.py`. The system model it inverts: `docs/projector.md`.

## The model

Bin i counts y_i ~ Poisson(ȳ_i), with

    ȳ_i = Σ_j a_ij x_j + b_i

- x_j is the activity of voxel j (Bq/mL), and x_j ≥ 0.
- a_ij is the expected counts in bin i per Bq/mL in voxel j: `projector.forward(image, line_factors)` is A x.
- b_i is an additive background, here the scatter.

Up to a constant, the log-likelihood is

    L(x) = Σ_i ( y_i log ȳ_i − ȳ_i )

and Σ_i ȳ_i = Σ_j s_j x_j + Σ_i b_i, with the **sensitivity** s_j = Σ_i a_ij: the expected counts per Bq/mL in
voxel j, over the whole scanner. In code, s = Aᵀ1 (`sensitivity_image`).

## The update

Maximizing L directly is hard, because the log holds a sum over all voxels. MLEM replaces L by a simpler function
that touches it at the current image xⁿ and lies below it everywhere (a *surrogate*; Shepp & Vardi 1982, Lange &
Carson 1984).

**Step 1: split each bin.** At the current image, give each voxel its share of bin i, and the background its share:

    r_ij = a_ij x_jⁿ / ȳ_iⁿ        r_ib = b_i / ȳ_iⁿ        Σ_j r_ij + r_ib = 1

**Step 2: Jensen's inequality.** The log is concave, so the log of a weighted average is at least the weighted
average of the logs:

    log ȳ_i = log( Σ_j r_ij · (a_ij x_j / r_ij) + r_ib · (b_i / r_ib) )
            ≥ Σ_j r_ij log(a_ij x_j / r_ij) + r_ib log(b_i / r_ib)

with equality at x = xⁿ.

**Step 3: collect the terms in x.** Everything that doesn't depend on x is a constant:

    Q(x | xⁿ) = Σ_j ( c_j log x_j − s_j x_j ) + constant,    c_j = Σ_i y_i r_ij = x_jⁿ Σ_i a_ij y_i / ȳ_iⁿ

Q separates: each voxel is its own one-dimensional problem.

**Step 4: maximize.** c_j / x_j − s_j = 0 gives

    x_jⁿ⁺¹ = c_j / s_j = (x_jⁿ / s_j) Σ_i a_ij y_i / ȳ_iⁿ

In operators: **x ← x / (Aᵀ1) · Aᵀ( y / (A x + b) )**.

This is also the EM algorithm: r_ij y_i is the expected number of bin i's counts that came from voxel j, and c_j
adds them up over the bins. MAP-EM keeps Steps 1–3 and adds a penalty to Q.

## What follows from it

- **The likelihood never decreases.** L(xⁿ⁺¹) ≥ Q(xⁿ⁺¹ | xⁿ) ≥ Q(xⁿ | xⁿ) = L(xⁿ).
- **The image stays non-negative**, since every factor is.
- **With b = 0, the model's total equals the measured total after every update:** Σ_j s_j xⁿ⁺¹_j = Σ_j c_j = Σ_i y_i.
- **Noise grows with iterations.** MLEM fits the noise as it converges. Stopping early is a form of regularization;
  MAP-EM replaces it with an explicit penalty.

## Worked example

Two voxels, two bins: bin 1 sees both voxels, bin 2 only voxel 1, so A = [[1, 1], [1, 0]]. The true image is (1, 2),
so y = (3, 1). The sensitivity is s = Aᵀ1 = (2, 1). Start from x = (1, 1).

| Iteration | A x | y / (A x) | Aᵀ(y / A x) | new x |
| --- | --- | --- | --- | --- |
| 1 | (2, 1) | (3/2, 1) | (5/2, 3/2) | (1 · 5/2 / 2, 1 · 3/2 / 1) = (1.25, 1.5) |
| 2 | (2.75, 1.25) | (12/11, 4/5) | (104/55, 12/11) | (13/11, 18/11) = (1.182, 1.636) |

The iterates approach (1, 2). After each one, s·x = 4 = y₁ + y₂. `tests/test_mlem.py` runs this example through
the real projector.

## Choices and approximations

- **Support.** Voxels with sensitivity below 2.5 % of the maximum stay at zero, because the update divides by s_j and
  would amplify noise there. The 2.5 % is the prototype's default, not a derived value.
- **Start.** A uniform image of 1 Bq/mL inside the support.
- **Background.** So far b is MCGPU-PET's own scatter sinogram, the same noisy realization that is in y. This is an
  oracle: it shows what perfect scatter knowledge gives. Scatter estimation comes later.

## Checks against MCGPU-PET

Nothing is fitted or rescaled: the model predicts counts, and MLEM returns Bq/mL.

**Uniform water cylinder** (radius 25 mm, length 100 mm, 3.9 × 10⁷ decays, 2 mm voxels, 30 iterations):

| Quantity | Result |
| --- | --- |
| Simulated / predicted trues, total | 0.9990 |
| Same, every sixth ring difference from 0 to 54 | 0.995 to 1.003 |
| Same, every eighth radial bin inside the cylinder | 0.995 to 1.002 |
| Detected trues / (activity × sensitivity), every sixth filled slice | 0.997 to 1.001 |
| MLEM on trues: mean in the core / true activity | 0.9995 |
| MLEM on trues + scatter, with the scatter as background | 0.9993 |
| MLEM on trues + scatter, no scatter correction | 1.174 |

**NEMA NU 4 image-quality phantom** (the prototype's run, 2.25 × 10⁷ trues, 1 mm voxels):

| Quantity | Result |
| --- | --- |
| Simulated / predicted trues, total | 0.9997 |
| Same, every eighth ring difference from 0 to 64 | 1.000 to 1.004 |
| Detected / predicted, every sixth filled slice | 0.994 to 1.007 |
| MLEM on trues + scatter, 30 iterations: total activity / true | 0.9995 |

`tests/test_cylinder.py` repeats the cylinder checks with half the counts.