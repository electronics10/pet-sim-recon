#!/usr/bin/env bash
# Build MCGPU-PET from pinned upstream sources into external/MCGPU-PET/MCGPU-PET.x.
# The compiler flags are the upstream Makefile's; only the include paths differ (docs/mcgpu-build.md).
set -euo pipefail

MCGPU_URL="https://github.com/DIDSR/MCGPU-PET.git"
MCGPU_COMMIT="1ae73df2bd3c854fb12164244fdd07c5fc0ddf82"
SAMPLES_URL="https://github.com/NVIDIA/cuda-samples.git"
SAMPLES_TAG="v12.0"  # matches the system nvcc 12.0

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXTERNAL_DIR="$REPO_ROOT/external"
mkdir -p "$EXTERNAL_DIR"

if [ ! -d "$EXTERNAL_DIR/MCGPU-PET" ]; then
    git clone "$MCGPU_URL" "$EXTERNAL_DIR/MCGPU-PET"
fi
git -C "$EXTERNAL_DIR/MCGPU-PET" checkout --quiet "$MCGPU_COMMIT"

if [ ! -d "$EXTERNAL_DIR/cuda-samples" ]; then
    git clone --depth 1 --branch "$SAMPLES_TAG" "$SAMPLES_URL" "$EXTERNAL_DIR/cuda-samples"
fi

# The 17 ready-made material files and upstream's converter (docs/materials.md).
MATERIALS_ZIP="$EXTERNAL_DIR/MCGPU-PET/sample_simulation/materials/other_materials_ICRP110_5-515keV__MCGPU-PET.zip"
mkdir -p "$EXTERNAL_DIR/mcgpu-materials"
python -m zipfile -e "$MATERIALS_ZIP" "$EXTERNAL_DIR/mcgpu-materials"
echo "Unpacked materials into $EXTERNAL_DIR/mcgpu-materials"

echo "Using $(command -v nvcc): $(nvcc --version | tail -1)"
cd "$EXTERNAL_DIR/MCGPU-PET"
nvcc -DUSING_CUDA -O3 -use_fast_math -m64 \
    -gencode=arch=compute_75,code=sm_75 \
    -I. -I"$EXTERNAL_DIR/cuda-samples/Common" \
    MCGPU-PET.cu -o MCGPU-PET.x \
    -lcudart -lm -lz
echo "Built $EXTERNAL_DIR/MCGPU-PET/MCGPU-PET.x"