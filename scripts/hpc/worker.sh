#!/usr/bin/env bash
# Run a strided subset of manifest shards on ONE GPU.
#   worker.sh SHARD_DIR OUT_DIR RANK STRIDE [extra wq-batch run args...]
# Shard i is processed by the rank with i % STRIDE == RANK. Finished shards (output exists) are skipped,
# so a job that hits its time limit can simply be resubmitted.
set -euo pipefail
SHARD_DIR=$1; OUT_DIR=$2; RANK=$3; STRIDE=$4; shift 4
mkdir -p "$OUT_DIR" "$OUT_DIR/logs"
mapfile -t SHARDS < <(ls "$SHARD_DIR"/shard_*.jsonl | sort)
echo "[worker $RANK/$STRIDE] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} shards=${#SHARDS[@]}"
for ((i = RANK; i < ${#SHARDS[@]}; i += STRIDE)); do
  shard=${SHARDS[$i]}
  name=$(basename "$shard" .jsonl)
  out="$OUT_DIR/$name.parquet"
  if [[ -f "$out" ]]; then echo "skip $name"; continue; fi
  echo "[$(date +%T)] start $name"
  wq-batch run --manifest "$shard" --out "$out" --backend cupy --device 0 "$@" \
      > "$OUT_DIR/logs/$name.log" 2>&1 || echo "FAILED $name (see logs/$name.log)"
  echo "[$(date +%T)] done  $name"
done
