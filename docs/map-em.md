# MAP-EM: MLEM plus what we believe about images

Code: `recon/map_em.py`, `recon/priors.py`. It starts where `docs/mlem.md` ends.

## Why a penalty

MLEM fits the data, noise included: with more iterations the image gets noisier. MAP-EM adds what we expect of an
image, that neighboring voxels are usually similar, and maximizes

    Φ(x) = L(x) − β R(x),    x ≥ 0

- L is the Poisson log-likelihood of `docs/mlem.md`.
- R is a penalty, large when neighbors differ.
- β sets how much we trust R against the data. β = 0 is MLEM.

(Read as Bayes: exp(−βR) is a prior on images, and x is its maximum a posteriori estimate.) Unlike MLEM, MAP-EM is
meant to run close to convergence: β controls the smoothness, not the number of iterations.

## The penalties

Pairs {j, k} of neighboring voxels (the 26 around each voxel), each with a weight ω_jk ≥ 0:

| Penalty | Term of a pair, with d = x_j − x_k | Behavior |
| --- | --- | --- |
| Quadratic | ½ ω_jk d² | Large differences cost a lot, so edges blur |
| Relative difference (RDP; Nuyts et al. 2002) | ω_jk d² / (x_j + x_k + γ\|d\|) | Grows like d²/(2x) for small d, but never exceeds \|d\|/(1 + γ), so edges cost less; the penalty scales with the local activity |

Uniform weights are ω_jk = (nearest-neighbor distance) / (distance from j to k). We use γ = 2.

## Bowsher's prior: let the MR image choose the neighbors

The idea (Bowsher et al. 2004): smooth a voxel only with voxels that look like it in the MR image.

1. For each voxel j, take its B neighbors with the smallest |m_j − m_k|, with m the MR image. Set w_jk = 1 for them and
   0 for the others. Voxel j may pick k without k picking j.
2. Penalize with those weights: R(x) = ½ Σ_j Σ_k w_jk M(x_j, x_k), where M is the quadratic or relative-difference term.

The weights are computed once, from the MR image, before reconstruction. So R stays an ordinary convex penalty in x, and
every algorithm below applies unchanged.

**Example.** MR along a row of four voxels: 0, 3, 3.5, 10, with B = 1. Voxels 1 and 2 pick each other. Voxel 0 can
only pick 1, and voxel 3 only 2, and neither is picked back.

| Pair | w one way | w other way | ω = average |
| --- | --- | --- | --- |
| (0, 1) | 1 | 0 | ½ |
| (1, 2) | 1 | 1 | 1 |
| (2, 3) | 0 | 1 | ½ |

**Averaging changes nothing.** M is symmetric, so in the double sum the terms (j, k) and (k, j) are the same M, and they
add up to (w_jk + w_kj) M. Hence

    ½ Σ_j Σ_k w_jk M(x_j, x_k) = Σ over pairs of  ½ (w_jk + w_kj) M(x_j, x_k)

So storing one symmetric weight ω_jk = ½(w_jk + w_kj) per pair is the same penalty. `tests/test_priors.py` checks this
numerically. The derivative of this penalty with respect to x_l has both terms, the neighbors l picked and the voxels
that picked l (Schramm et al. 2018, eq. 13). It is called the **symmetric** Bowsher prior, and it is what we implement.

**The asymmetric variant.** A heuristic modification keeps only the first of those terms when updating x_l: the
neighbors l itself picked (Schramm et al. 2018, eq. 14). That is no longer the gradient of any known function, so there is no Φ to
climb. Schramm et al. report that it is stable in practice and performs slightly better than the symmetric prior. We
have not implemented it.

**Choices that differ between papers.** Schramm et al. use B = 4 of the 18 nearest neighbors. We follow the prototype's B
= 6 of 26 with distance weights, and sweep B. Ties (a perfectly flat MR image) go to the nearer neighbor, then at random
with a fixed seed.

**What it can't know.** The MR image says nothing about a PET-only lesion. There, the neighbors are picked by MR noise,
and the prior acts as an ordinary smoother on a thinner set of neighbors. Schramm et al. found little difference from
post-smoothed OSEM in a stand-alone lesion, and that at very high β all anatomical priors transfer non-shared gradients.

## De Pierro's update, for quadratic penalties

MLEM worked because its surrogate separates into one problem per voxel. The penalty couples voxels through (x_j − x_k)².
De Pierro (1995) separates that too.

**Step 1: split the difference.** With xⁿ the current image,

    x_j − x_k = ½ (2x_j − x_jⁿ − x_kⁿ) − ½ (2x_k − x_jⁿ − x_kⁿ)

**Step 2: convexity.** For any u and v, (½u − ½v)² ≤ ½u² + ½v². So

    (x_j − x_k)² ≤ ½ (2x_j − x_jⁿ − x_kⁿ)² + ½ (2x_k − x_jⁿ − x_kⁿ)²

Each term holds a single unknown, and at x = xⁿ both sides are equal.

**Step 3: the surrogate of Φ.** Take the MLEM surrogate and subtract this bound. For voxel j alone:

    Q_j(x_j) = c_j log x_j − s_j x_j − (β/4) Σ_k ω_jk (2x_j − x_jⁿ − x_kⁿ)²

Since we subtract an upper bound of R, ΣQ_j still lies below Φ and touches it at xⁿ, so Φ never decreases.

**Step 4: maximize.** Setting dQ_j/dx_j = 0 and multiplying by x_j gives a quadratic equation:

    a x_j² + b x_j − c_j = 0,    a = 2β Ω_j,    b = s_j − β S_j

with Ω_j = Σ_k ω_jk and S_j = Σ_k ω_jk (x_jⁿ + x_kⁿ). Its positive root is the update:

    x_jⁿ⁺¹ = 2 c_j / ( b + √(b² + 4 a c_j) )

At β = 0 it is c_j / s_j, the MLEM update. For b < 0 the code uses the equal form (√(b² + 4ac) − b) / (2a), which
avoids subtracting nearly equal numbers.

**Worked example.** Two voxels, each seen by its own bin (A = I, so s = (1, 1)), y = (1, 9), one pair with ω = 1, β = ¼.
Start from xⁿ = (1, 3).

- c = xⁿ · y / xⁿ = (1, 9).
- Ω = 1 and S = 1 + 3 = 4 for both voxels, so a = ½ and b = 1 − ¼ · 4 = 0.
- With b = 0 the root is √(c / a): x = (√2, √18) = (1.414, 4.243).

| Iteration | x₀ | x₁ | Φ |
| --- | --- | --- | --- |
| start | 1 | 3 | 5.388 |
| 1 | 1.414 | 4.243 | 6.696 |
| 2 | 1.888 | 4.677 | 6.982 |
| 3 | 2.194 | 4.932 | 7.084 |
| converged | 2.856 | 5.455 | 7.163 |

MLEM alone would go to y = (1, 9). The penalty pulls the two voxels toward each other.

## The update for RDP

RDP has no simple surrogate of this kind, so we climb Φ directly. The direction is the gradient, scaled as in EM:

    p = (x / s) ∇Φ(x) = ( c − s x − β x ∇R(x) ) / s

At β = 0, p = c/s − x, so a step of 1 is exactly the MLEM update. We then choose the step α ≥ 0 that maximizes Φ(x + αp)
while keeping x ≥ 0. L is concave and RDP is convex (Nuyts et al. 2002), so Φ is concave along the line, and bisection
on the sign of dΦ/dα finds its maximum. Since A(x + αp) = Ax + αAp, the whole search costs one extra forward projection.

## Choosing β

`reference_beta` sets a scale β₀ at which penalty and likelihood have about the same curvature at a typical voxel: the
likelihood curves like s/x, the quadratic penalty like Ω, and RDP like Ω/x. We then sweep a dimensionless β̃ = β/β₀. This
is a heuristic. **The same β̃ doesn't mean the same noise for different priors**, so methods are compared on
contrast-vs-noise curves, at equal noise, never at equal β̃.

## Results: the NEMA NU 4 phantom at 10 % dose

**Setup.** The full-dose MCGPU-PET run (2.25 × 10⁷ trues, 3.3 × 10⁶ scattered) is thinned to 10 %, five times with
different seeds, about 2.6 × 10⁶ counts each. Scatter enters as a known background (an oracle). MLEM runs 23 iterations,
and each MAP reconstruction runs 60 more from that image. The MR image is made from the phantom's own materials, with the
4 mm rod removed from it, so that rod is invisible in MR. So is the water insert, which looks like the water around it.
The air insert and the other rods are visible.

Rod values are the mean in the rod's core (radius d/2 − 0.5 mm, central 10 mm) over the uniform-region mean, so 1 is
correct. Cold-insert values are spill-over ratios, so 0 is correct. Values are mean ± standard deviation over the five
realizations.

**At about 5 % noise**

| Method | %STD | 3 mm rod (in MR) | 4 mm rod (not in MR) | 5 mm rod (in MR) | Water insert (not in MR) | Air insert (in MR) |
| --- | --- | --- | --- | --- | --- | --- |
| MLEM + 2 mm Gaussian | 5.5 | 0.669 ± 0.012 | 0.814 ± 0.012 | 0.828 ± 0.009 | 0.173 ± 0.003 | 0.177 ± 0.008 |
| Quadratic, β̃ = 0.3 | 4.4 | 0.831 ± 0.016 | 0.915 ± 0.013 | 0.917 ± 0.010 | 0.069 ± 0.004 | 0.074 ± 0.010 |
| RDP, β̃ = 0.3 | 5.1 | 0.956 ± 0.028 | 0.983 ± 0.022 | 0.972 ± 0.014 | 0.118 ± 0.003 | 0.117 ± 0.009 |
| Bowsher (B = 6), β̃ = 0.3 | 6.2 | 0.988 ± 0.018 | 0.963 ± 0.015 | 0.987 ± 0.015 | 0.065 ± 0.004 | 0.066 ± 0.010 |

**At about 3 % noise**

| Method | %STD | 3 mm rod (in MR) | 4 mm rod (not in MR) | 5 mm rod (in MR) | Water insert (not in MR) | Air insert (in MR) |
| --- | --- | --- | --- | --- | --- | --- |
| MLEM + 3 mm Gaussian | 3.2 | 0.440 ± 0.005 | 0.627 ± 0.006 | 0.684 ± 0.006 | 0.214 ± 0.003 | 0.215 ± 0.007 |
| Quadratic, β̃ = 1 | 2.9 | 0.617 ± 0.007 | 0.742 ± 0.007 | 0.776 ± 0.006 | 0.194 ± 0.005 | 0.185 ± 0.012 |
| RDP, β̃ = 1 | 2.8 | 0.779 ± 0.016 | 0.870 ± 0.012 | 0.888 ± 0.010 | 0.250 ± 0.003 | 0.240 ± 0.008 |
| Bowsher (B = 6), β̃ = 1 | 2.9 | 0.992 ± 0.012 | 0.825 ± 0.008 | 0.992 ± 0.013 | 0.143 ± 0.005 | 0.077 ± 0.009 |

**What this shows.**

- **Bowsher keeps what the MR shows.** At about 3 % noise the MR-visible rods stay at 0.99, against 0.62–0.89 for the
  uniform priors.
- **It helps less where the MR is blind.** The MR-invisible 4 mm rod falls to 0.825. That is better than quadratic
  (0.742) but below RDP (0.870). Spill-over in the invisible water insert is 0.143, nearly twice that of the visible air
  insert (0.077).
- **So the prior's benefit follows the MR**, as its construction predicts. It didn't erase the MR-invisible rod here.
- **The uniform-region mean stays within 0.2 % of the truth with Bowsher at every β̃ tested.** At β̃ = 1 it rises by 3.1 %
  with quadratic and 2.3 % with RDP. We have not investigated why.

**Limits of this comparison.**

- The MR image comes from the phantom: perfect alignment, same voxels, same edges. Real PET-MR is harder.
- The five realizations are thinned from one full-dose run, so their spread slightly understates the true noise.
- B = 6 and the other settings are not tuned. A sweep of B = 3, 6, 12 was run on one realization only.
- This is one phantom. It says nothing about clinical images.

**Agreement with the prototype.** On the same full-dose run, 114 of the 120 values (%STD, max-voxel RC of the 3, 4 and 5 mm
rods, both spill-over ratios) fall within 3 standard deviations of this ensemble. The two largest differences are RDP's %STD
at β̃ = 0.01 (44.7 against 42.4) and its water spill-over at β̃ = 1 (0.234 against 0.250). The two codes differ in
how they attenuate the mirror lines, and we did not trace these differences further.

## Approximations and limits

- **Iterations.** MAP runs a finite number of iterations from an MLEM start, so it is close to converged, not exactly.
- **Support.** Pairs with a voxel outside the reconstruction support are dropped, so the voxels held at zero don't pull
  on their neighbors.
- **The MR image in our simulations** comes from the phantom itself: perfectly aligned, same edges, same voxels. Real
  PET-MR is less kind.

## References

- Shepp & Vardi, IEEE Trans Med Imaging 1(2):113–122, 1982.
- De Pierro, IEEE Trans Med Imaging 14(1):132–137, 1995.
- Nuyts, Bequé, Dupont & Mortelmans, IEEE Trans Nucl Sci 49(1):56–60, 2002.
- Bowsher et al., IEEE Nucl Sci Symp Conf Rec 4:2488–2492, 2004.
- Vunckx et al., IEEE Trans Med Imaging 31(3):599–612, 2012.
- Schramm et al., IEEE Trans Med Imaging 37(2):590–603, 2018.