"""A simulated MR image of a phantom: compartment labels -> intensities, blur, Rician noise (docs/image-quality.md)."""

import math

import numpy as np
from scipy.ndimage import gaussian_filter

from pet_sim_recon.image_grid import ImageGrid

FWHM_PER_SIGMA = 2 * math.sqrt(2 * math.log(2))  # about 2.355

# The talk's contrast: water 1, PMMA 0.35, air 0. The 4 mm rod and the water insert are hidden: each gets the
# intensity of its surroundings, so the MR shows no edge there.
NEMA_IQ_MR_INTENSITIES = {
    "air": 0.0, "pmma": 0.35, "hot": 1.0, "water_insert": 1.0, "air_insert": 0.0,
    "rod_1mm": 1.0, "rod_2mm": 1.0, "rod_3mm": 1.0, "rod_4mm": 0.35, "rod_5mm": 1.0,
}


def mr_surrogate(label_id, label_names, intensity_by_label: dict, grid: ImageGrid, blur_fwhm_mm: float,
                 snr: float | None, rng: np.random.Generator | None = None) -> np.ndarray:
    """MR-like image: each label's intensity, a Gaussian blur of blur_fwhm_mm, then Rician noise.

    snr = None gives a noise-free image. Noise has standard deviation 1 / snr on each of the two channels.
    """
    missing = [name for name in label_names if name not in intensity_by_label]
    if missing:
        raise ValueError(f"no MR intensity given for {missing}")
    intensities = np.array([intensity_by_label[name] for name in label_names], dtype=np.float32)
    image = intensities[label_id - 1]
    if blur_fwhm_mm > 0:
        sigma_voxels = tuple(blur_fwhm_mm / FWHM_PER_SIGMA / size for size in grid.voxel_size_mm_zyx)
        image = gaussian_filter(image, sigma_voxels)
    if snr is None:
        return image
    # The magnitude of a complex signal with Gaussian noise on both channels (Gudbjartsson & Patz 1995).
    noise = 1 / snr
    real = image + noise * rng.standard_normal(image.shape)
    imaginary = noise * rng.standard_normal(image.shape)
    return np.sqrt(real**2 + imaginary**2)
