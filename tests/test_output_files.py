"""Reading a finished run folder: shapes from its settings, totals from MCGPU-PET's own log."""

import gzip
import re
from pathlib import Path

import numpy as np
import pytest

from pet_sim_recon.mcgpu.output_files import read_counts, read_run_settings, read_sinograms

# The end-to-end reproduction of upstream's sample (docs/mcgpu-simulation.md). Data stays out of git.
SAMPLE_RUN = Path(__file__).parents[1] / "data" / "runs" / "sample-reproduction-materials"


def logged_count(run_dir, kind):
    """MCGPU-PET.out reports e.g. 'Number of true coincidences in the sinogram = 3734'."""
    log = (run_dir / "MCGPU-PET.out").read_text()
    return int(re.search(rf"Number of {kind} coincidences in the sinogram = (\d+)", log).group(1))


@pytest.mark.skipif(not SAMPLE_RUN.exists(), reason="needs the sample reproduction run in data/runs/")
def test_sample_run_shapes_and_totals():
    trues, scatter = read_sinograms(SAMPLE_RUN)
    assert trues.shape == scatter.shape == (1293, 168, 147)  # (plane, view, radial bin)
    assert trues.sum() == logged_count(SAMPLE_RUN, "true")
    assert scatter.sum() == logged_count(SAMPLE_RUN, "scatter")


@pytest.mark.skipif(not SAMPLE_RUN.exists(), reason="needs the sample reproduction run in data/runs/")
def test_settings_are_read_back():
    scanner, sinogram = read_run_settings(SAMPLE_RUN)
    assert scanner.num_crystals_per_ring == 336
    assert sinogram.span == 11


def test_bin_order_is_plane_view_radial(tmp_path):
    """A file whose value is its own bin index, (plane * views + view) * radial bins + radial bin (the kernel's
    ibin), reads back with value 100 * plane + 10 * view + radial bin for 2 planes, 3 views, 4 radial bins."""
    plane, view, radial = np.meshgrid(np.arange(2), np.arange(3), np.arange(4), indexing="ij")
    ibin = (plane * 3 + view) * 4 + radial
    values = np.zeros(24, dtype="<i4")
    values[ibin.ravel()] = (100 * plane + 10 * view + radial).ravel()
    path = tmp_path / "sinogram.raw.gz"
    with gzip.open(path, "wb") as file:
        file.write(values.tobytes())

    np.testing.assert_array_equal(read_counts(path, (2, 3, 4)), 100 * plane + 10 * view + radial)
