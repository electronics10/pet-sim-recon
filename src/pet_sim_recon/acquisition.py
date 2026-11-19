"""Acquisition settings: how long, which isotope, which photon energies are accepted."""

import math
from dataclasses import dataclass

# F-18 half-life is 109.77 min; the mean life is half-life / ln 2.
F18_MEAN_LIFE_S = 109.77 * 60 / math.log(2)  # about 9502 s


@dataclass(frozen=True)
class AcquisitionSettings:
    """Settings of one PET acquisition. Energies in keV."""

    acquisition_time_s: float
    isotope_mean_life_s: float
    energy_resolution_fwhm: float  # FWHM as a fraction of the photon energy, e.g. 0.12
    energy_window_low_keV: float
    energy_window_high_keV: float

    def __post_init__(self):
        if self.acquisition_time_s <= 0 or self.isotope_mean_life_s <= 0:
            raise ValueError("acquisition_time_s and isotope_mean_life_s must be > 0")
        if not 0 < self.energy_resolution_fwhm < 1:
            raise ValueError(f"energy_resolution_fwhm is a fraction, got {self.energy_resolution_fwhm}")
        if not 0 <= self.energy_window_low_keV < self.energy_window_high_keV:
            raise ValueError(
                f"need 0 <= low < high, got {self.energy_window_low_keV} and {self.energy_window_high_keV} keV"
            )

    @property
    def decay_weighted_time_s(self) -> float:
        """Integral of exp(-t / mean life) over the scan: the scan time, counted at the starting activity."""
        mean_life_s = self.isotope_mean_life_s
        return mean_life_s * (1 - math.exp(-self.acquisition_time_s / mean_life_s))