"""Batch planning and record generation for data campaigns.

``BatchRunner`` takes many :class:`CircuitIR`, groups them into device-sized batches (same qubit count and measured
register; sorted by structure so identical layouts share a batch), simulates, and yields one flat record per circuit,
ready for Parquet. Records carry the provenance needed to rebuild the run (versions, device, seeds, noise digest,
Slurm ids).
"""
from __future__ import annotations

import json
import os
import platform
import socket
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

import numpy as np

from .backend import Backend, get_backend
from .engine import simulate_batch
from .ir import CircuitIR
from .noise import PauliNoiseModel

BYTES_FACTOR = 5          # psi + gathered rows + two transposed temporaries + probabilities, per row


def max_rows(backend: Backend, n: int, dtype: str = "complex64", safety: float = 0.6, cap: int = 1 << 16) -> int:
    per_row = (2 ** n) * np.dtype(dtype).itemsize * BYTES_FACTOR
    return int(max(1, min(cap, backend.free_bytes() * safety // per_row)))


def plan_batches(circuits: Sequence[CircuitIR], rows_per_circuit: int, cap_rows) -> list[list[int]]:
    """Indices grouped by (n, measured qubits), sorted by structure key, chunked to ``cap_rows(n) // rows_per_circuit``."""
    groups: dict[tuple, list[int]] = {}
    for i, c in enumerate(circuits):
        groups.setdefault((c.num_qubits, c.measured), []).append(i)
    batches: list[list[int]] = []
    for (n, _meas), idx in sorted(groups.items(), key=lambda kv: kv[0][0]):
        idx.sort(key=lambda i: (circuits[i].structure_key(), len(circuits[i].ops)))
        per = max(1, cap_rows(n) // rows_per_circuit)
        batches += [idx[k:k + per] for k in range(0, len(idx), per)]
    return batches


def record_schema():
    """Fixed Parquet schema so every shard merges cleanly (absent optional fields are null)."""
    import pyarrow as pa

    s, i32, i64, f64 = pa.string(), pa.int32(), pa.int64(), pa.float64()
    fields = [("circuit_id", s), ("n_qubits", i32), ("n_measured", i32), ("n_ops", i32), ("n_2q", i32),
              ("structure_key", s), ("fusion", s), ("batch_index", i32), ("batch_size", i32), ("batch_rows", i32),
              ("batch_kernel_calls", i32), ("wall_s_share", f64), ("meta", s), ("run_id", s), ("westquant_cudaq", s),
              ("qiskit", s), ("array_backend", s), ("gpu_name", s), ("gpu_total_bytes", i64), ("cupy", s), ("numpy", s),
              ("dtype", s), ("noise_digest", s), ("noise_label", s), ("trajectories", i32), ("shots", i32),
              ("seed", i64), ("host", s), ("python", s), ("slurm_job_id", s), ("slurm_array_task_id", s),
              ("slurm_procid", s)]
    for tag in ("ideal", "noisy"):
        fields += [(f"{tag}_top_bitstrings", pa.list_(s)), (f"{tag}_top_probs", pa.list_(f64)),
                   (f"{tag}_probs", pa.list_(pa.float32())), (f"{tag}_success_prob", f64)]
    fields += [("counts", s), ("hellinger_fidelity", f64)]
    return pa.schema(fields)


def _bitstring(index: int, m: int) -> str:
    return format(index, f"0{m}b") if m else ""


def hellinger_fidelity(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.sum(np.sqrt(np.clip(p, 0, None) * np.clip(q, 0, None)), axis=-1) ** 2


@dataclass
class BatchRunner:
    backend: Backend = field(default_factory=get_backend)
    dtype: str = "complex64"
    noise: PauliNoiseModel | None = None
    trajectories: int = 0
    shots: int = 0
    topk: int = 16
    store_full_up_to: int = 10           # store the full distribution when measured bits <= this
    seed: int = 1234
    safety: float = 0.6
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def provenance(self) -> dict:
        from .. import __version__ as wqc_version
        try:
            import qiskit
            qv = qiskit.__version__
        except Exception:  # noqa: BLE001
            qv = None
        return {"run_id": self.run_id, "westquant_cudaq": wqc_version, "qiskit": qv, "array_backend": self.backend.name,
                **{k: v for k, v in self.backend.info.items()}, "dtype": self.dtype,
                "noise_digest": self.noise.digest() if self.noise else None,
                "noise_label": self.noise.label if self.noise else None, "trajectories": self.trajectories,
                "shots": self.shots, "seed": self.seed, "host": socket.gethostname(), "python": platform.python_version(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "slurm_procid": os.environ.get("SLURM_PROCID")}

    def run(self, circuits: Sequence[CircuitIR]) -> Iterator[dict]:
        rng = np.random.default_rng(self.seed)
        noisy = self.noise is not None and self.trajectories > 0
        R = 1 + (self.trajectories if noisy else 0)
        prov = self.provenance()
        batches = plan_batches(circuits, R, lambda n: max_rows(self.backend, n, self.dtype, self.safety))
        for b_i, idx in enumerate(batches):
            batch = [circuits[i] for i in idx]
            res = simulate_batch(batch, self.backend, dtype=self.dtype, noise=self.noise if noisy else None,
                                 trajectories=self.trajectories if noisy else 0, rng=rng)
            m = res.n_measured
            share = res.stats["wall_s"] / len(batch)
            fid = hellinger_fidelity(res.ideal, res.noisy) if res.noisy is not None else None
            for j, c in enumerate(batch):
                rec = {"circuit_id": c.circuit_id, "n_qubits": c.num_qubits, "n_measured": m, "n_ops": len(c.ops),
                       "n_2q": c.n_two_qubit, "structure_key": c.structure_key(), "fusion": c.fusion,
                       "batch_index": b_i, "batch_size": len(batch), "batch_rows": res.stats["rows"],
                       "batch_kernel_calls": res.stats["kernel_calls"], "wall_s_share": share,
                       "meta": json.dumps(c.meta, sort_keys=True, default=str), **prov}
                for tag, dist in (("ideal", res.ideal[j]), ("noisy", None if res.noisy is None else res.noisy[j])):
                    if dist is None:
                        continue
                    k = min(self.topk, dist.size)
                    top = np.argsort(-dist)[:k]
                    rec[f"{tag}_top_bitstrings"] = [_bitstring(int(t), m) for t in top]
                    rec[f"{tag}_top_probs"] = [float(dist[t]) for t in top]
                    if m <= self.store_full_up_to:
                        rec[f"{tag}_probs"] = dist.astype(np.float32).tolist()
                    exp = c.meta.get("expected")
                    if exp:
                        exps = [exp] if isinstance(exp, str) else list(exp)
                        rec[f"{tag}_success_prob"] = float(sum(dist[int(e, 2)] for e in exps))
                    if self.shots and (tag == "noisy" or res.noisy is None):
                        p = np.clip(dist.astype(np.float64), 0, None)
                        counts = rng.multinomial(self.shots, p / p.sum())
                        nz = np.nonzero(counts)[0]
                        rec["counts"] = json.dumps({_bitstring(int(t), m): int(counts[t]) for t in nz})
                if fid is not None:
                    rec["hellinger_fidelity"] = float(fid[j])
                yield rec

    def run_to_parquet(self, circuits: Sequence[CircuitIR], out_path: str, flush_every: int = 20000) -> dict:
        """Stream records to ``out_path`` (written to ``out_path + '.tmp'`` then renamed: restart-safe)."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        tmp = out_path + ".tmp"
        schema = record_schema()
        unknown = set()
        writer = pq.ParquetWriter(tmp, schema, compression="zstd")
        buf, n_rec, t0 = [], 0, time.perf_counter()
        for rec in self.run(circuits):
            unknown |= set(rec) - set(schema.names)
            buf.append(rec)
            if len(buf) >= flush_every:
                writer.write_table(pa.Table.from_pylist(buf, schema=schema)); n_rec += len(buf); buf = []
        if buf:
            writer.write_table(pa.Table.from_pylist(buf, schema=schema)); n_rec += len(buf)
        writer.close()
        os.replace(tmp, out_path)
        if unknown:
            raise KeyError(f"record fields missing from record_schema(): {sorted(unknown)}")
        return {"records": n_rec, "wall_s": time.perf_counter() - t0, "out": out_path}
