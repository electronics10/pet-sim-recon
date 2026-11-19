"""MLEM reconstruction (Shepp & Vardi 1982; docs/mlem.md)."""

MIN_SENSITIVITY_FRACTION = 0.025  # voxels seen less than this, relative to the best seen, stay at zero


def sensitivity_image(projector, line_factors):
    """Expected coincidences per unit activity in each voxel: the back-projection of a sinogram of ones."""
    xp = projector.xp
    return projector.adjoint(xp.ones(projector.sinogram_shape, dtype=xp.float32), line_factors)


def mlem(projector, measured, line_factors, background=0.0, num_iterations=50, on_iteration=None):
    """Maximize the Poisson likelihood of measured ~ Poisson(A x + background), x >= 0.

    A x is projector.forward(x, line_factors). Returns the image, in the units line_factors assume.
    on_iteration(iteration, image), if given, runs after every update.
    """
    xp = projector.xp
    measured = xp.asarray(measured, dtype=xp.float32)
    sensitivity = sensitivity_image(projector, line_factors)
    in_support = sensitivity > MIN_SENSITIVITY_FRACTION * sensitivity.max()
    image = xp.where(in_support, xp.ones_like(sensitivity), xp.zeros_like(sensitivity))

    for iteration in range(num_iterations):
        expected = projector.forward(image, line_factors) + background
        ratio = xp.where(expected > 0, measured / xp.where(expected > 0, expected, 1), 0)
        update = projector.adjoint(ratio, line_factors) / xp.where(in_support, sensitivity, 1)
        image = xp.where(in_support, image * update, 0)
        if on_iteration is not None:
            on_iteration(iteration, image)
    return image