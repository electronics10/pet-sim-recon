"""MLEM reconstruction (Shepp & Vardi 1982; docs/mlem.md)."""

MIN_SENSITIVITY_FRACTION = 0.025  # voxels seen less than this, relative to the best seen, stay at zero


def sensitivity_image(projector, line_factors):
    """Expected coincidences per unit activity in each voxel: the back-projection of a sinogram of ones."""
    xp = projector.xp
    return projector.adjoint(xp.ones(projector.sinogram_shape, dtype=xp.float32), line_factors)


def reconstruction_support(sensitivity):
    """The voxels we reconstruct: those the scanner sees well enough."""
    return sensitivity > MIN_SENSITIVITY_FRACTION * sensitivity.max()


def em_numerator(projector, image, measured, expected, line_factors):
    """c_j of docs/mlem.md: image_j times the back-projection of measured / expected."""
    xp = projector.xp
    ratio = xp.where(expected > 0, measured / xp.where(expected > 0, expected, 1), 0)
    return image * projector.adjoint(ratio, line_factors)


def mlem(projector, measured, line_factors, background=0.0, num_iterations=50, on_iteration=None):
    """Maximize the Poisson likelihood of measured ~ Poisson(A x + background), x >= 0.

    A x is projector.forward(x, line_factors). Returns the image, in the units line_factors assume.
    on_iteration(iteration, image), if given, runs after every update.
    """
    xp = projector.xp
    measured = xp.asarray(measured, dtype=xp.float32)
    sensitivity = sensitivity_image(projector, line_factors)
    in_support = reconstruction_support(sensitivity)
    image = xp.where(in_support, xp.ones_like(sensitivity), xp.zeros_like(sensitivity))

    for iteration in range(num_iterations):
        expected = projector.forward(image, line_factors) + background
        numerator = em_numerator(projector, image, measured, expected, line_factors)
        image = xp.where(in_support, numerator / xp.where(in_support, sensitivity, 1), 0)
        if on_iteration is not None:
            on_iteration(iteration, image)
    return image