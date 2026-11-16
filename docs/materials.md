# Materials

## How an object names its materials

A `VoxelObject` stores a material ID per voxel and a tuple `material_names`: ID k means
`material_names[k - 1]`. For example, `("air", "water", "lung")` gives air = 1, water = 2, lung = 3. The
names are the toolkit's own. Only `pet_sim_recon/mcgpu/materials.py` knows which MCGPU-PET file each name
means, and it lists exactly the object's materials in the `.in` file, in ID order.

MCGPU-PET reads at most 15 material files per run (`MAX_MATERIALS`, `MCGPU-PET.h` line 96).

## The ready-made set

`pixi run build-mcgpu` unpacks upstream's
`sample_simulation/materials/other_materials_ICRP110_5-515keV__MCGPU-PET.zip` into
`external/mcgpu-materials/` (git-ignored). It holds 17 files tabulated from 5 to 515 keV: air, water, and
15 ICRP 110 reference tissues.

| Name | Name | Name |
| --- | --- | --- |
| `adipose` | `cartilage` | `lung` |
| `air` | `eyes` | `muscle` |
| `blood_spleen` | `glands_others` | `skin` |
| `brain` | `kidneys` | `soft_tissue` |
| `breast_glandular` | `liver` | `spongiosa` (trabecular bone) |
| | | `stomach_intestines` |
| | | `water` |

The prototypes' `adipose` and `spongiosa` files are byte-identical to these. So are the `air` and `water`
files in upstream's `sample_simulation/materials/`.

Density is set per voxel in the `.vox` file, separately from the material. A material file provides the
cross-sections *per unit mass*. The same file can therefore describe a material at another density, which
is an approximation when the composition differs. Example: the prototypes simulated PMMA as water at
1.19 g/cm³. Water's mass attenuation coefficient at 511 keV is about 0.096 cm²/g, PMMA's about
0.093 cm²/g, so attenuation through PMMA was about 3 % too high per unit mass (`mcgpu-recon`
`REWRITE_NOTES.md`).

## Making a new material file

A material that isn't in the set (e.g. PMMA) needs its own file, made in two steps.

1. **A PENELOPE 2006 material file.** PENELOPE's `material` program builds it from the composition and
   density (e.g. `pmma.mat`). PENELOPE is distributed by the OECD Nuclear Energy Agency Data Bank and needs
   registration. For that reason, material files and PENELOPE are never committed to this repository.
2. **Convert it** with upstream's `MC-GPU_create_material_data.f` (in `external/mcgpu-materials/`, from
   the same zip). It's compiled together with PENELOPE's `penelope.f`, and asks four questions. Upstream's
   answers, from `MC-GPU_create_material_data.out`:

   ```
   $ gfortran -O MC-GPU_create_material_data.f penelope.f -o MC-GPU_create_material_data.x
   $ ./MC-GPU_create_material_data.x
   Energy range:        5000 515000      (eV; must reach above 511 keV)
   Number of bins:      10200            (50 eV wide)
   PENELOPE file:       water.mat
   Output file:         water_5-515keV.mcgpu
   ```

   Then gzip the output (MCGPU-PET reads `.gz` through zlib).

To use a new file, put it in `external/mcgpu-materials/` (or pass another `materials_dir` to
`run_simulation`), and add its name to `MATERIAL_FILES` in `pet_sim_recon/mcgpu/materials.py`. Record
where it came from (composition, density, PENELOPE version) next to the entry. Note that
`tests/test_materials.py` checks the table against upstream's zip, so it will then need to allow the extra
name.