#!/usr/bin/env bash
# Whole nodes with 4 x A100 each (e.g. Leonardo Booster). One process per GPU; ranks split the shards.
#
#   sbatch --nodes=2 scripts/hpc/run_node_4xa100.sbatch $WORK/wqbench/shards $WORK/wqbench/out --dtype complex64
#
# 2 nodes -> 8 ranks; shard i goes to rank i % 8. Resubmit the same command after a timeout: done shards are skipped.
#SBATCH --job-name=wq-batch-node
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION        # Leonardo: boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0                                # all node memory
#SBATCH --time=08:00:00
#SBATCH --output=wq-batch-node_%j.out
set -euo pipefail
SHARD_DIR=$1; OUT_DIR=$2; shift 2
HERE=${WQ_HPC_DIR:-$SLURM_SUBMIT_DIR/scripts/hpc}   # submit from the repo root, or export WQ_HPC_DIR
source "$HERE/env.sh"
STRIDE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
# srun gives every task its own GPU (CUDA_VISIBLE_DEVICES has one entry per task, so --device 0 is correct)
srun --kill-on-bad-exit=0 bash -c "bash '$HERE/worker.sh' '$SHARD_DIR' '$OUT_DIR' \$SLURM_PROCID $STRIDE --workers $SLURM_CPUS_PER_TASK $*"
