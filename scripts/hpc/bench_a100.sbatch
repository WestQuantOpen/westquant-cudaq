#!/usr/bin/env bash
# Validate and benchmark on one A100 before a campaign: GPU test suite, then throughput per qubit count.
#   sbatch scripts/hpc/bench_a100.sbatch /path/to/westquant-cudaq
#SBATCH --job-name=wq-bench
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=wq-bench_%j.out
set -euo pipefail
REPO=${1:-.}
HERE=${WQ_HPC_DIR:-$SLURM_SUBMIT_DIR/scripts/hpc}
source "$HERE/env.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
cd "$REPO"
WQ_TEST_BACKEND=cupy python -m pytest -q tests/test_batched.py        # same tests as on CPU, now through CuPy
for dt in complex64 complex128; do
  echo "== heterogeneous random circuits, $dt"
  wq-batch bench --backend cupy --dtype $dt --nmin 4 --nmax 20 --nstep 2 --circuits 4096 --depth 40 --compare-qiskit 32
  echo "== same structure (parameter-sweep regime), $dt"
  wq-batch bench --backend cupy --dtype $dt --nmin 4 --nmax 20 --nstep 4 --circuits 4096 --depth 40 --same-structure
done
