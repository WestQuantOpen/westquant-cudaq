#!/usr/bin/env python3
"""Merge shard outputs into one dataset and check the campaign is complete.

  python merge_shards.py --shards $WORK/wqbench/shards --out-dir $WORK/wqbench/out --dataset $WORK/wqbench/dataset

* checks every shard in index.json has a Parquet output and the record count equals the manifest minus parse errors;
* writes a Hive-partitioned dataset to DATASET/data (partition: n_qubits) with zstd compression;
* writes DATASET/campaign_summary.json (throughput per shard, parse errors, missing shards, provenance set).
Exit code 1 if shards are missing or counts disagree, so it can gate the next pipeline step.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="directory with index.json from make_manifest.py")
    ap.add_argument("--out-dir", required=True, help="directory with shard_*.parquet from the Slurm jobs")
    ap.add_argument("--dataset", required=True, help="output dataset directory")
    a = ap.parse_args()

    index = json.loads((Path(a.shards) / "index.json").read_text())
    out_dir = Path(a.out_dir)
    tables, report, missing, mismatched = [], [], [], []
    for sh in index["shards"]:
        stem = Path(sh["file"]).stem
        pq_path, sm_path = out_dir / f"{stem}.parquet", out_dir / f"{stem}.parquet.summary.json"
        if not pq_path.exists():
            missing.append(stem)
            continue
        t = pq.read_table(pq_path)
        summ = json.loads(sm_path.read_text()) if sm_path.exists() else {}
        expected = sh["circuits"] - summ.get("parse_errors", 0)
        if t.num_rows != expected:
            mismatched.append({"shard": stem, "rows": t.num_rows, "expected": expected})
        tables.append(t)
        report.append({"shard": stem, "rows": t.num_rows, "parse_errors": summ.get("parse_errors"),
                       "circuits_per_s": summ.get("circuits_per_s"), "gpu": summ.get("gpu_name")})

    if tables:
        full = pa.concat_tables(tables, promote_options="default")
        dup = full.num_rows - len(set(full.column("circuit_id").to_pylist()))
        ds.write_dataset(full, str(Path(a.dataset) / "data"), format="parquet", partitioning=["n_qubits"], partitioning_flavor="hive",
                         existing_data_behavior="overwrite_or_ignore",
                         file_options=ds.ParquetFileFormat().make_write_options(compression="zstd"))
        prov = {k: sorted(set(map(str, full.column(k).to_pylist()))) for k in
                ("run_id", "westquant_cudaq", "qiskit", "gpu_name", "dtype", "noise_digest") if k in full.column_names}
    else:
        dup, prov = 0, {}
    summary = {"shards_total": len(index["shards"]), "shards_done": len(tables), "missing": missing,
               "count_mismatch": mismatched, "duplicate_circuit_ids": dup,
               "records": sum(r["rows"] for r in report), "provenance": prov, "per_shard": report}
    Path(a.dataset).mkdir(parents=True, exist_ok=True)
    (Path(a.dataset) / "campaign_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_shard"}, indent=1))
    return 1 if (missing or mismatched or dup) else 0


if __name__ == "__main__":
    sys.exit(main())
