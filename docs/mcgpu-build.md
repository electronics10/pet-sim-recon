# Building MCGPU-PET

```bash
pixi run build-mcgpu
```

This runs `scripts/build_mcgpu.sh`, which:

1. clones MCGPU-PET (https://github.com/DIDSR/MCGPU-PET) into `external/MCGPU-PET/` and checks out the
   pinned commit `1ae73df`;
2. clones NVIDIA's cuda-samples at tag `v12.0` into `external/cuda-samples/`, for the two headers
   MCGPU-PET includes: `helper_functions.h` and `helper_cuda.h`;
3. compiles `external/MCGPU-PET/MCGPU-PET.x`.

`external/` is git-ignored. Running the script again doesn't download again, and always builds the pinned
commit.

## Why the flags look like this

The flags are those of the upstream `Makefile`:

- `-DUSING_CUDA -O3 -use_fast_math -m64`;
- `-gencode=arch=compute_75,code=sm_75`;
- `-lcudart -lm -lz`.

Two changes:

- The Makefile's include and library paths assume `/usr/local/cuda`, which doesn't exist on this machine.
  The script passes only the cuda-samples include path, and `nvcc` finds CUDA itself.
- `--ptxas-options=-v` is dropped. It only prints register usage.

`compute_75`/`sm_75` is the TITAN RTX's compute capability. For another GPU, check it with
`nvidia-smi --query-gpu=compute_cap --format=csv,noheader` and change the `-gencode` flag.

## Compiler

The system `nvcc` is 12.0 (Ubuntu's `nvidia-cuda-toolkit`). nvcc 12.0 doesn't support `gcc` 13, the system
default. Ubuntu's package handles this: its `nvcc.profile` puts `/usr/lib/nvidia-cuda-toolkit/bin` first on
nvcc's PATH, and the `gcc` there is `gcc-12`. So no extra flag is needed. The pixi environment has no
`nvcc`, so `pixi run` uses the system one.

Compilation prints warnings; they are expected.

## Verification (2026-11-16)

Question: does this build behave like the binary the prototypes used
(`mcgpu-pet-wrapper/mcgpu_pet_wrapper/MCGPU-PET.x`)?

Both come from the same source. MCGPU-PET's code (`MCGPU-PET.cu`, `MCGPU-PET_kernel.cu`, `MCGPU-PET.h`,
`Makefile`) hasn't changed since upstream commit `80016d5` (2022-02-05). With a fixed seed, each GPU
thread's random-number stream depends only on the seed and the thread's index. Counts are added with
integer atomic adds, so their order doesn't matter. Two binaries of the same code should therefore give
byte-identical tallies.

Test: upstream's `sample_simulation` (9 cm water cube, 1 s acquisition), with two changes. The seed is set
to 12345, because 0 means "seed from the clock". The report mode is set to 2, sinogram only, because the
phase-space file lists events in GPU arrival order, which varies between runs.

Result: `sinogram_Trues`, `sinogram_Scatter`, `image_Trues`, `image_Scatter` (compared decompressed) and
`Energy_Sinogram_Spectrum.dat` were byte-identical. Both runs simulated 23,866 histories.

| Output | Total counts | Nonzero bins |
| --- | --- | --- |
| `sinogram_Trues` | 3,734 | 3,733 |
| `sinogram_Scatter` | 734 | 734 |

Redo this check after changing the GPU, the driver, the compiler or the pinned commit. Both runs are kept
in `data/build-check/` (not in git) as the reference.