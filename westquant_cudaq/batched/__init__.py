"""Batched GPU simulation of many small circuits (throughput-oriented data generation).

Quick start::

    from westquant_cudaq.batched import BatchRunner, from_qiskit, get_backend

    irs = [from_qiskit(qc, circuit_id=f"c{i}") for i, qc in enumerate(circuits)]
    runner = BatchRunner(backend=get_backend("auto"))          # CuPy on GPU, NumPy otherwise
    runner.run_to_parquet(irs, "out.parquet")

Command line: ``wq-batch run --manifest shard.jsonl --out shard.parquet`` (see ``docs/BATCHED.md``).
"""
from .backend import Backend, get_backend
from .engine import BatchedStatevector, BatchResult, simulate_batch
from .ir import CircuitIR, Op, from_qasm, from_qiskit, fuse_ops
from .noise import PauliNoiseModel
from .runner import BatchRunner, hellinger_fidelity, max_rows, plan_batches, record_schema

__all__ = ["Backend", "BatchResult", "BatchRunner", "BatchedStatevector", "CircuitIR", "Op", "PauliNoiseModel",
           "from_qasm", "from_qiskit", "fuse_ops", "get_backend", "hellinger_fidelity", "max_rows", "plan_batches",
           "record_schema", "simulate_batch"]
