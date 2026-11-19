#!/usr/bin/env bash
# Builds the Rodinia 3.1 CUDA applications that tools/app_energy.py measures (the ones that make their own input), for the
# GPU's architecture, each linked with apps/device_select.cpp so that APP_CUDA_DEVICE picks the GPU.
#
#   apps/build_rodinia.sh RODINIA_CUDA_DIR BUILD_DIR [ARCH]
#
# RODINIA_CUDA_DIR is the cuda/ folder of https://github.com/yuhc/gpu-rodinia (Rodinia 3.1; the sources are not changed),
# ARCH the nvcc architecture (default sm_86). Needs nvcc (CUDA_PATH, default /usr/local/cuda).
set -euo pipefail
SRC=$(realpath "$1")
OUT=$(realpath -m "$2")
ARCH=${3:-sm_86}
NVCC=${CUDA_PATH:-/usr/local/cuda}/bin/nvcc
HERE=$(dirname "$(realpath "$0")")
mkdir -p "$OUT"
FLAGS=(-O3 -arch="$ARCH" -w -Xlinker --wrap=cudaSetDevice,--wrap=main)
"$NVCC" -O3 -arch="$ARCH" -c "$HERE/device_select.cpp" -o "$OUT/device_select.o"
SEL="$OUT/device_select.o"

build() {  # name, then nvcc arguments
    local name=$1; shift
    echo "building $name"
    "$NVCC" "${FLAGS[@]}" "$@" "$SEL" -o "$OUT/$name" -lm -lcuda
}

R=$SRC
build backprop $R/backprop/backprop.c $R/backprop/facetrain.c $R/backprop/imagenet.c $R/backprop/backprop_cuda.cu
build gaussian $R/gaussian/gaussian.cu -I$R/gaussian
build lud -I$R/lud/common -DGPU_TIMER $R/lud/cuda/lud.cu $R/lud/cuda/lud_kernel.cu $R/lud/common/common.c
build needle $R/nw/needle.cu -I$R/nw
build pathfinder $R/pathfinder/pathfinder.cu
build srad_v2 $R/srad/srad_v2/srad.cu -I$R/srad/srad_v2
build lavaMD $R/lavaMD/main.c $R/lavaMD/kernel/kernel_gpu_cuda_wrapper.cu $R/lavaMD/util/num/num.c \
    $R/lavaMD/util/timer/timer.c $R/lavaMD/util/device/device.cu -Xcompiler -fopenmp -lgomp
build particlefilter $R/particlefilter/ex_particle_CUDA_float_seq.cu -use_fast_math
build streamcluster $R/streamcluster/streamcluster_cuda_cpu.cpp $R/streamcluster/streamcluster_cuda.cu -I$R/streamcluster -Xcompiler -pthread
build hotspot $R/hotspot/hotspot.cu
build hotspot3D $R/hotspot3D/3D.cu

# The probes of idle gaps and copies (tools/app_energy.py probes), not part of Rodinia.
"$NVCC" -O3 -arch="$ARCH" -w -Xlinker --wrap=cudaSetDevice,--wrap=main "$HERE/AppProbes.cu" "$SEL" -o "$OUT/app_probes"
echo "built app_probes"
echo "built into $OUT"
