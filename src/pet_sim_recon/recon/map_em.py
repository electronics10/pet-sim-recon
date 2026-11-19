"""MAP-EM: maximize the Poisson log-likelihood minus beta times a penalty, x >= 0 (docs/map-em.md)."""

from pet_sim_recon.recon.mlem import em_numerator, reconstruction_support, sensitivity_image
from pet_sim_recon.recon.priors import QuadraticPenalty

MAX_BISECTIONS = 30
STEP_TOLERANCE = 1e-4


def objective(projector, image, measured, line_factors, penalty, beta, background=0.0) -> float:
    """Phi(x) = sum_i (y_i log ybar_i - ybar_i) - beta R(x)."""
    xp = projector.xp
    measured = xp.asarray(measured, dtype=xp.float32)
    expected = projector.forward(image, line_factors) + background
    has_counts = measured > 0
    log_expected = xp.log(xp.where(has_counts, expected, 1))
    likelihood = xp.sum(measured * log_expected, dtype=xp.float64) - xp.sum(expected, dtype=xp.float64)
    return float(likelihood) - beta * penalty.value(image)


def de_pierro_step(image, numerator, sensitivity, penalty, beta):
    """The positive root of a x^2 + b x - c = 0 in every voxel: a = 2 beta Omega, b = s - beta S, c = numerator."""
    xp = penalty.xp
    total_weight, pair_sum = penalty.de_pierro_terms(image)
    a = 2 * beta * total_weight
    b = sensitivity - beta * pair_sum
    root = xp.sqrt(b * b + 4 * a * numerator)
    # Two forms of one root, each free of cancellation for its sign of b.
    return xp.where(b > 0, 2 * numerator / xp.where(b + root > 0, b + root, 1), (root - b) / xp.where(a > 0, 2 * a, 1))


def line_search_step(projector, image, numerator, sensitivity, in_support, measured, expected, line_factors, penalty, beta):
    """Move along the EM-preconditioned gradient (x / s) grad Phi, to the maximum of Phi on that line."""
    xp = projector.xp
    penalty_gradient = penalty.gradient(image)
    direction = xp.where(in_support, (numerator - sensitivity * image - beta * image * penalty_gradient) / sensitivity, 0)
    direction_forward = projector.forward(direction, line_factors)

    def slope(step):
        """d Phi / d step along the direction. Phi is concave on the line, so the slope only falls."""
        expected_at_step = expected + step * direction_forward
        positive = expected_at_step > 0
        safe_expected = xp.where(positive, expected_at_step, 1)
        likelihood_terms = xp.where(measured > 0, measured * direction_forward / safe_expected, 0) - direction_forward
        penalty_at_step = penalty.gradient(xp.maximum(image + step * direction, 0))
        return float(xp.sum(likelihood_terms, dtype=xp.float64)) - beta * float(
            xp.sum(penalty_at_step * direction, dtype=xp.float64)
        )

    falling = direction < 0
    max_step = float(xp.min(image[falling] / -direction[falling])) if bool(falling.any()) else float("inf")
    if slope(0.0) <= 0:
        return image
    low, high = 0.0, min(1.0, max_step)
    while slope(high) > 0:  # step 1 is the MLEM step; grow until the maximum is bracketed
        if high >= max_step:
            return xp.maximum(image + max_step * direction, 0)
        low, high = high, min(2 * high, max_step)
    for _ in range(MAX_BISECTIONS):
        middle = (low + high) / 2
        if slope(middle) > 0:
            low = middle
        else:
            high = middle
        if high - low < STEP_TOLERANCE * high:
            break
    return xp.maximum(image + low * direction, 0)


def map_em(
    projector, measured, line_factors, penalty, beta, background=0.0, num_iterations=60, initial_image=None,
    on_iteration=None,
):
    """Maximize objective() over x >= 0. Quadratic penalties use De Pierro's update, others the line search."""
    xp = projector.xp
    measured = xp.asarray(measured, dtype=xp.float32)
    sensitivity = sensitivity_image(projector, line_factors)
    in_support = reconstruction_support(sensitivity)
    safe_sensitivity = xp.where(in_support, sensitivity, 1)
    penalty = penalty.restricted_to(in_support)
    start = xp.ones_like(sensitivity) if initial_image is None else xp.asarray(initial_image, dtype=xp.float32)
    image = xp.where(in_support, start, xp.zeros_like(start))

    for iteration in range(num_iterations):
        expected = projector.forward(image, line_factors) + background
        numerator = em_numerator(projector, image, measured, expected, line_factors)
        if beta == 0:
            updated = numerator / safe_sensitivity
        elif isinstance(penalty, QuadraticPenalty):
            updated = de_pierro_step(image, numerator, safe_sensitivity, penalty, beta)
        else:
            updated = line_search_step(
                projector, image, numerator, safe_sensitivity, in_support, measured, expected, line_factors, penalty, beta
            )
        image = xp.where(in_support, updated, xp.zeros_like(updated))
        if on_iteration is not None:
            on_iteration(iteration, image)
    return image