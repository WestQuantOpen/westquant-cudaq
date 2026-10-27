#!/usr/bin/env bash
# Slurm job array: one A100 per array task, each task works through every N-th shard.
#
#   python scripts/hpc/make_manifest.py --circuits circuits.parquet --out $WORK/wqbench/shards
#   sbatch --array=0-15 scripts/hpc/run_array_a100.sbatch $WORK/wqbench/shards $WORK/wqbench/out \
#          --noise noise.json --trajectories 200 --shots 1000
#
# --array=0-(K-1) starts K GPU tasks; shard i goes to task i % K. Choose K <= number of shards.
# Adapt account/partition/QoS and the environment block to your centre (Leonardo Booster example in comments).
#SBATCH --job-name=wq-batch
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION        # Leonardo: boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                           # one A100 per array task
#SBATCH --cpus-per-task=8                      # QASM parsing runs in parallel on these cores
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=wq-batch_%A_%a.out
set -euo pipefail
SHARD_DIR=$1; OUT_DIR=$2; shift 2

# Slurm runs a copy of this script from its spool dir, so locate the repo via the submit dir:
HERE=${WQ_HPC_DIR:-$SLURM_SUBMIT_DIR/scripts/hpc}   # submit from the repo root, or export WQ_HPC_DIR
source "$HERE/env.sh"                               # loads modules and activates the venv (edit env.sh once)

K=${SLURM_ARRAY_TASK_COUNT:-1}
RANK=${SLURM_ARRAY_TASK_ID:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
bash "$HERE/worker.sh" "$SHARD_DIR" "$OUT_DIR" "$RANK" "$K" --workers "${SLURM_CPUS_PER_TASK:-8}" "$@"
