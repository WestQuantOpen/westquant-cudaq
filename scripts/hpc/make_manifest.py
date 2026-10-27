#!/usr/bin/env python3
"""Split a circuit collection into GPU-sized, cost-balanced manifest shards.

Input (one of):
  --qasm-dir DIR        every *.qasm file is one circuit (circuit_id = file stem)
  --circuits FILE       JSONL or Parquet with at least circuit_id and qasm (other columns are kept)

Shards are sorted by qubit count, so a shard holds circuits of similar size and the simulator fills its batches well.
Cost per circuit ~ 2**n * (number of gate lines); shards are cut when they reach --target-cost or --max-circuits.

Writes OUT/shard_00000.jsonl ... and OUT/index.json (used by the Slurm scripts and merge_shards.py).

  python make_manifest.py --circuits all_circuits.parquet --out shards --max-circuits 20000
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

QREG = re.compile(r"qreg\s+\w+\s*\[\s*(\d+)\s*\]")
QUBIT3 = re.compile(r"qubit\s*\[\s*(\d+)\s*\]")


def n_qubits(qasm: str) -> int:
    regs = [int(m) for m in QREG.findall(qasm)] or [int(m) for m in QUBIT3.findall(qasm)]
    if regs:
        return sum(regs)
    from qiskit import qasm2
    return qasm2.loads(qasm, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS).num_qubits


def gate_lines(qasm: str) -> int:
    return sum(1 for line in qasm.splitlines()
               if line.strip() and not line.lstrip().startswith(("OPENQASM", "include", "qreg", "creg", "qubit", "bit",
                                                                  "measure", "barrier", "//")))


def load(args) -> list[dict]:
    if args.qasm_dir:
        return [{"circuit_id": p.stem, "qasm": p.read_text()} for p in sorted(Path(args.qasm_dir).glob("*.qasm"))]
    path = args.circuits
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--qasm-dir")
    src.add_argument("--circuits")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-circuits", type=int, default=20000)
    ap.add_argument("--target-cost", type=float, default=2e9, help="sum of 2**n * gates per shard")
    ap.add_argument("--max-qubits", type=int, default=26, help="skip larger circuits (state-vector limit per row)")
    args = ap.parse_args()

    items = load(args)
    seen, rows, skipped = set(), [], []
    for it in items:
        cid = str(it["circuit_id"])
        if cid in seen:
            raise SystemExit(f"duplicate circuit_id {cid}")
        seen.add(cid)
        n = n_qubits(it["qasm"])
        if n > args.max_qubits:
            skipped.append(cid)
            continue
        rows.append((n, (2 ** n) * max(1, gate_lines(it["qasm"])), it))
    rows.sort(key=lambda r: (r[0], r[1]))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    shards, cur, cost = [], [], 0.0
    for n, c, it in rows:
        if cur and (len(cur) >= args.max_circuits or cost + c > args.target_cost):
            shards.append(cur); cur, cost = [], 0.0
        cur.append((n, c, it)); cost += c
    if cur:
        shards.append(cur)

    index = {"n_circuits": len(rows), "skipped_too_large": skipped, "shards": []}
    for i, sh in enumerate(shards):
        name = f"shard_{i:05d}.jsonl"
        with open(out / name, "w") as f:
            for _n, _c, it in sh:
                f.write(json.dumps(it) + "\n")
        index["shards"].append({"file": name, "circuits": len(sh), "n_min": sh[0][0], "n_max": sh[-1][0],
                                "cost": sum(c for _n, c, _it in sh)})
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{len(rows)} circuits -> {len(shards)} shards in {out} ({len(skipped)} skipped > {args.max_qubits} qubits)")


if __name__ == "__main__":
    main()
