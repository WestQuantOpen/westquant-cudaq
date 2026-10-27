"""Command line: ``wq-batch run | plan | bench``.

Manifest = JSON Lines or Parquet with one circuit per row::

    {"circuit_id": "qaoa-8-000123", "qasm": "OPENQASM 2.0; ...", "family": "qaoa", "expected": "01011010", "meta": {...}}

``expected`` (optional) is a classical bitstring (Qiskit order, clbit 0 rightmost) or a list of them; it enables
``*_success_prob``. Everything in ``meta`` plus ``family`` is copied to the record's ``meta`` column.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def _load_manifest(path: str) -> list[dict]:
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _parse_one(args):
    item, fusion = args
    from .ir import from_qasm

    meta = dict(item.get("meta") or {})
    for k in ("family", "expected", "source"):
        if item.get(k) is not None:
            meta[k] = item[k]
    try:
        return from_qasm(item["qasm"], fusion=fusion, circuit_id=str(item["circuit_id"]), meta=meta), None
    except Exception as exc:  # noqa: BLE001  -- one bad circuit must not kill a shard; it is reported
        return None, f"{item.get('circuit_id')}: {type(exc).__name__}: {exc}"


def parse_manifest(path: str, fusion: str, workers: int):
    items = _load_manifest(path)
    jobs = [(it, fusion) for it in items]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(_parse_one, jobs, chunksize=64))
    else:
        out = [_parse_one(j) for j in jobs]
    circuits = [c for c, e in out if c is not None]
    errors = [e for c, e in out if e is not None]
    return circuits, errors


def _seed_for(path: str, base: int) -> int:
    return base ^ int(hashlib.sha256(os.path.basename(path).encode()).hexdigest()[:8], 16)


def cmd_run(a) -> int:
    from .backend import get_backend
    from .noise import PauliNoiseModel
    from .runner import BatchRunner

    if os.path.exists(a.out) and not a.overwrite:
        print(f"[wq-batch] {a.out} exists, skipping (use --overwrite)")
        return 0
    noise = PauliNoiseModel.from_json(a.noise) if a.noise else None
    fusion = a.fusion if a.fusion != "auto" else ("1q" if noise and a.trajectories > 0 else "full")
    t0 = time.perf_counter()
    circuits, errors = parse_manifest(a.manifest, fusion, a.workers)
    t_parse = time.perf_counter() - t0
    backend = get_backend(a.backend, a.device)
    runner = BatchRunner(backend=backend, dtype=a.dtype, noise=noise, trajectories=a.trajectories, shots=a.shots,
                         topk=a.topk, seed=_seed_for(a.manifest, a.seed), safety=a.safety)
    info = runner.run_to_parquet(circuits, a.out)
    summary = {"manifest": a.manifest, "out": a.out, "circuits": len(circuits), "parse_errors": len(errors),
               "parse_s": round(t_parse, 2), "sim_s": round(info["wall_s"], 2),
               "circuits_per_s": round(len(circuits) / max(info["wall_s"], 1e-9), 1), "backend": backend.name,
               **backend.info}
    with open(a.out + ".summary.json", "w") as f:
        json.dump({**summary, "errors": errors[:1000]}, f, indent=2)
    print(json.dumps(summary))
    return 0 if not errors or not a.strict else 2


def cmd_plan(a) -> int:
    from .backend import Backend
    from .runner import BYTES_FACTOR, plan_batches

    circuits, errors = parse_manifest(a.manifest, "full", a.workers)
    fake = Backend(np, "plan", None, {})
    budget = int(a.gpu_mem_gb * 2**30 * a.safety)
    rows_per = 1 + a.trajectories

    def cap(n):
        return max(1, min(1 << 16, budget // ((2 ** n) * np.dtype(a.dtype).itemsize * BYTES_FACTOR)))
    batches = plan_batches(circuits, rows_per, cap)
    by_n: dict[int, list[int]] = {}
    for b in batches:
        by_n.setdefault(circuits[b[0]].num_qubits, []).append(len(b))
    print(f"{len(circuits)} circuits, {len(errors)} parse errors, {len(batches)} batches")
    for n in sorted(by_n):
        sizes = by_n[n]
        print(f"  n={n:2d}: {len(sizes):5d} batches, circuits/batch max {max(sizes)}, rows/batch max {max(sizes) * rows_per}"
              f", state bytes/batch {max(sizes) * rows_per * 2**n * np.dtype(a.dtype).itemsize / 2**30:.2f} GiB")
    del fake
    return 0


def cmd_bench(a) -> int:
    from qiskit.circuit.random import random_circuit
    from qiskit.quantum_info import Statevector

    from .backend import get_backend
    from .ir import from_qiskit
    from .runner import BatchRunner

    backend = get_backend(a.backend, a.device)
    print(json.dumps({"backend": backend.name, **backend.info}))
    for n in range(a.nmin, a.nmax + 1, a.nstep):
        qcs = [random_circuit(n, a.depth, max_operands=2, seed=1000 * n + i) for i in range(a.circuits)]
        if a.same_structure:          # parameter-sweep regime: one layout, different angles
            base = qcs[0]
            qcs = [base.assign_parameters([]) for _ in range(a.circuits)]
        irs = [from_qiskit(qc, fusion="full", circuit_id=f"b{n}-{i}") for i, qc in enumerate(qcs)]
        runner = BatchRunner(backend=backend, dtype=a.dtype, topk=4)
        t0 = time.perf_counter()
        recs = list(runner.run(irs))
        dt = time.perf_counter() - t0
        line = {"n": n, "circuits": len(recs), "sim_s": round(dt, 3), "circuits_per_s": round(len(recs) / dt, 1),
                "batches": len({r["batch_index"] for r in recs})}
        if a.compare_qiskit:
            k = min(a.compare_qiskit, len(qcs))
            t1 = time.perf_counter()
            for qc in qcs[:k]:
                Statevector(qc.remove_final_measurements(inplace=False)).probabilities()
            line["qiskit_statevector_circuits_per_s"] = round(k / (time.perf_counter() - t1), 1)
        print(json.dumps(line), flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="wq-batch", description="Batched GPU simulation of many small circuits")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="simulate one manifest shard and write Parquet")
    r.add_argument("--manifest", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--noise", help="noise JSON (PauliNoiseModel dict or calibration with gate_error_1q/2q)")
    r.add_argument("--trajectories", type=int, default=0)
    r.add_argument("--shots", type=int, default=0)
    r.add_argument("--dtype", default="complex64", choices=["complex64", "complex128"])
    r.add_argument("--backend", default="auto", choices=["auto", "cupy", "numpy"])
    r.add_argument("--device", type=int, default=None)
    r.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    r.add_argument("--fusion", default="auto", choices=["auto", "full", "1q", "none"])
    r.add_argument("--topk", type=int, default=16)
    r.add_argument("--seed", type=int, default=20261027)
    r.add_argument("--safety", type=float, default=0.6, help="fraction of free device memory to use")
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--strict", action="store_true", help="exit 2 if any circuit failed to parse")
    r.set_defaults(fn=cmd_run)

    pl = sub.add_parser("plan", help="show batch plan for a manifest without a GPU")
    pl.add_argument("--manifest", required=True)
    pl.add_argument("--gpu-mem-gb", type=float, default=64.0)
    pl.add_argument("--dtype", default="complex64")
    pl.add_argument("--trajectories", type=int, default=0)
    pl.add_argument("--safety", type=float, default=0.6)
    pl.add_argument("--workers", type=int, default=4)
    pl.set_defaults(fn=cmd_plan)

    b = sub.add_parser("bench", help="throughput on random circuits")
    b.add_argument("--nmin", type=int, default=4)
    b.add_argument("--nmax", type=int, default=16)
    b.add_argument("--nstep", type=int, default=2)
    b.add_argument("--circuits", type=int, default=2048)
    b.add_argument("--depth", type=int, default=30)
    b.add_argument("--dtype", default="complex64")
    b.add_argument("--backend", default="auto", choices=["auto", "cupy", "numpy"])
    b.add_argument("--device", type=int, default=None)
    b.add_argument("--same-structure", action="store_true")
    b.add_argument("--compare-qiskit", type=int, default=0, help="also time Qiskit Statevector on this many circuits")
    b.set_defaults(fn=cmd_bench)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
