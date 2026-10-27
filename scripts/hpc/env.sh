#!/usr/bin/env bash
# Edit once per cluster. Sourced by the Slurm scripts.
# Example for an NVIDIA A100 system with CUDA 12 modules (names differ per centre: check `module avail cuda python`).
# module load python/3.11 cuda/12.2
# One-time setup (login node):
#   python -m venv $WORK/venvs/wq && source $WORK/venvs/wq/bin/activate
#   pip install "westquant-cudaq[batched,gpu]"        # or: pip install -e /path/to/westquant-cudaq[batched,gpu]
#   python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"   # run inside a GPU job to verify
source "${WQ_VENV:-$WORK/venvs/wq}/bin/activate"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export CUPY_CACHE_DIR=${CUPY_CACHE_DIR:-$WORK/.cupy_cache}   # kernel cache on shared FS, not $HOME quota
