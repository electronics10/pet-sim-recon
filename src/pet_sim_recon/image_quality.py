"""Image-quality metrics (docs/image-quality.md). Images and masks are numpy arrays in (z, y, x)."""

import numpy as np

from pet_sim_recon.phantoms import NemaIqPhantom


def pct_std(image, voi) -> float:
    """Noise: 100 x SD / mean over the voxels of voi (SD over voxels, ddof = 0)."""
    values = image[voi]
    return 100 * float(values.std()) / float(values.mean())


def rc_mean(image, core, true_activity) -> float:
    """Partial-volume recovery: mean in the core over the true activity."""
    return float(image[core].mean()) / true_activity


def rc_nema(image, roi_yx, slab, uniform_mean) -> float:
    """NEMA NU 4: the largest pixel of the slab-averaged image inside the (y, x) ROI, over the uniform mean."""
    averaged = image[slab].mean(axis=0)
    return float(averaged[roi_yx].max()) / uniform_mean


def sor(image, insert_voi, uniform_mean) -> float:
    """Spill-over ratio: mean in a cold insert over the uniform mean."""
    return float(image[insert_voi].mean()) / uniform_mean


def crc(image, hot_voi, background_voi, true_hot, true_background) -> float:
    """Contrast recovery, NEMA NU 2 form: (C_H / C_B - 1) / (a_H / a_B - 1)."""
    measured_ratio = float(image[hot_voi].mean()) / float(image[background_voi].mean())
    return (measured_ratio - 1) / (true_hot / true_background - 1)


def nema_iq_metrics(image, phantom: NemaIqPhantom, true_activity) -> dict:
    """The NEMA NU 4 set: noise, uniform mean, rc_mean (rods >= 2 mm), rc_nema (all rods), both spill-over ratios."""
    image = np.asarray(image)
    uniform = phantom.uniform_voi()
    uniform_mean = float(image[uniform].mean())
    result = {"pct_std": pct_std(image, uniform), "uniform_over_true": uniform_mean / true_activity}
    slab = phantom.rods_slab()
    for diameter_mm in phantom.geometry.rod_diameters_mm:
        result[f"rc_nema_{diameter_mm:g}mm"] = rc_nema(image, phantom.rod_roi_yx(diameter_mm), slab, uniform_mean)
        if diameter_mm >= 2:
            result[f"rc_mean_{diameter_mm:g}mm"] = rc_mean(image, phantom.rod_core(diameter_mm), true_activity)
    for name in ("water", "air"):
        result[f"sor_{name}"] = sor(image, phantom.insert_voi(name), uniform_mean)
    return result


def ensemble_summary(records: list[dict]) -> dict:
    """Mean and SD (ddof = 1) of every key over noise realizations: {key: {"mean": ..., "sd": ...}}."""
    return {key: {"mean": float(np.mean([r[key] for r in records])), "sd": float(np.std([r[key] for r in records], ddof=1))}
            for key in records[0]}
