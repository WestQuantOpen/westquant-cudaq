"""Batched state-vector simulation of many small circuits on one device.

All circuits in a batch have the same number of qubits n. The state of the batch is one array ``psi`` of shape
``(rows, 2**n)``. Each row is one circuit (ideal) or one noise trajectory of a circuit. A gate step applies, for every
group of rows that share the same target qubits, a *batched* matrix product

    psi[rows] <- psi[rows] @ U[rows]^T      (per-row 2x2 or 4x4 matrices, one cuBLAS batched GEMM on GPU)

so the cost of a step is one kernel launch per distinct target tuple, not one per circuit. Rows that share their whole
structure (same targets at every step: parameter sweeps, trajectories, transpilation variants of a fixed layout) are
updated without any gather or scatter.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .backend import Backend
from .ir import CircuitIR
from .noise import PAULI_1Q, PAULI_2Q, PauliNoiseModel


@dataclass
class BatchResult:
    """Device-side results reduced to host arrays, one row per *circuit* (not per trajectory)."""
    ideal: np.ndarray                      # (C, 2**m) ideal output distribution over measured bits
    noisy: np.ndarray | None               # (C, 2**m) trajectory-averaged + readout noise, or None
    n_measured: int
    stats: dict = field(default_factory=dict)


class BatchedStatevector:
    def __init__(self, backend: Backend, num_qubits: int, rows: int, dtype: str = "complex64"):
        xp = backend.xp
        self.be, self.xp, self.n, self.rows = backend, xp, num_qubits, rows
        self.cdtype = np.dtype(dtype)
        self.psi = xp.zeros((rows, 2 ** num_qubits), dtype=self.cdtype)
        self.psi[:, 0] = 1.0
        self.kernel_calls = 0

    def apply(self, qubits: Sequence[int], mats, idx=None) -> None:
        """Apply per-row matrices ``mats`` (rows_sel, D, D) or one matrix (D, D) to ``qubits`` on rows ``idx``."""
        xp, n, k = self.xp, self.n, len(qubits)
        sub = self.psi if idx is None else self.psi[idx]
        b = sub.shape[0]
        if b == 0:
            return
        t = sub.reshape((b,) + (2,) * n)
        src = [1 + (n - 1 - q) for q in reversed(qubits)]          # most significant target first
        dst = list(range(n + 1 - k, n + 1))
        t = xp.moveaxis(t, src, dst)
        shp = t.shape
        t = t.reshape(b, -1, 2 ** k)
        if mats.ndim == 2:
            t = t @ mats.T
        else:
            t = xp.matmul(t, xp.swapaxes(mats, 1, 2))
        t = xp.moveaxis(t.reshape(shp), dst, src).reshape(b, 2 ** n)
        if idx is None:
            self.psi = xp.ascontiguousarray(t)
        else:
            self.psi[idx] = t
        self.kernel_calls += 1


def _marginal_axes(n: int, measured: Sequence[int] | None):
    """Axes of the (rows, 2,...,2) probability tensor to keep, ordered so the flattened index equals the classical
    register value (clbit i = measured[i], clbit 0 least significant)."""
    meas = list(range(n)) if measured is None else list(measured)
    keep = [1 + (n - 1 - q) for q in reversed(meas)]
    drop = tuple(a for a in range(1, n + 1) if a not in keep)
    return meas, keep, drop


def simulate_batch(circuits: Sequence[CircuitIR], backend: Backend, *, dtype: str = "complex64",
                   noise: PauliNoiseModel | None = None, trajectories: int = 0,
                   rng: np.random.Generator | None = None) -> BatchResult:
    """Simulate ``circuits`` (same qubit count and same measured-qubit count) in one batch.

    With ``noise`` and ``trajectories = T > 0`` each circuit occupies ``1 + T`` rows: row 0 ideal, rows 1..T noisy.
    """
    if not circuits:
        raise ValueError("empty batch")
    xp = backend.xp
    n = circuits[0].num_qubits
    if any(c.num_qubits != n for c in circuits):
        raise ValueError("all circuits in a batch must have the same number of qubits")
    noisy = noise is not None and trajectories > 0
    if noisy and any(c.fusion == "full" for c in circuits) and not noise.is_trivial:
        raise ValueError("noisy simulation needs circuits converted with fusion='1q' or 'none' "
                         "(full fusion moves 1q gates across 2q gates and would misplace errors)")
    rng = rng or np.random.default_rng()
    T = trajectories if noisy else 0
    R = 1 + T
    C = len(circuits)
    sv = BatchedStatevector(backend, n, C * R, dtype)
    t0 = time.perf_counter()

    # ---- one pass over all ops on the host: target codes, matrices and error probabilities as dense arrays
    S = max(len(c.ops) for c in circuits)
    code = np.full((C, S), -1, dtype=np.int64)                  # -1 = no op at this step (circuit finished)
    M1 = np.zeros((C, S, 2, 2), dtype=sv.cdtype)
    M2 = np.zeros((C, S, 4, 4), dtype=sv.cdtype)
    perr = np.zeros((C, S), dtype=np.float64)
    for ci, c in enumerate(circuits):
        for s, op in enumerate(c.ops):
            if len(op.qubits) == 1:
                code[ci, s] = op.qubits[0]
                M1[ci, s] = op.matrix
                if T:
                    perr[ci, s] = noise.p1_of(op.qubits[0], op.count)
            else:
                a, b = op.qubits
                code[ci, s] = 1024 + a * 512 + b                   # unique code per ordered pair
                M2[ci, s] = op.matrix
                if T:
                    perr[ci, s] = noise.p2_of(a, b)
    M1d, M2d = xp.asarray(M1), xp.asarray(M2)                    # one host-to-device copy per batch
    rows_of = np.arange(C * R).reshape(C, R)
    noisy_rows = rows_of[:, 1:]
    P1d = xp.asarray(PAULI_1Q.astype(sv.cdtype)) if T else None
    P2d = xp.asarray(PAULI_2Q.astype(sv.cdtype)) if T else None

    for s in range(S):
        col = code[:, s]
        uniq, inv = np.unique(col, return_inverse=True)
        for g, cval in enumerate(uniq):
            if cval < 0:
                continue
            cis = np.nonzero(inv == g)[0]
            qubits = (int(cval),) if cval < 1024 else ((int(cval) - 1024) // 512, (int(cval) - 1024) % 512)
            src = M1d if len(qubits) == 1 else M2d
            if cis.size == C:
                mats = src[:, s]
                idx = None
            else:
                cis_d = xp.asarray(cis)
                mats = src[cis_d, s]
                idx = xp.asarray(rows_of[cis].reshape(-1))
            if R > 1:
                mats = xp.repeat(mats, R, axis=0)
            sv.apply(qubits, mats, idx)

            if T:                                                # Pauli errors on the noisy rows of these circuits
                ps = perr[cis, s]
                if np.any(ps > 0):
                    hit = rng.random((cis.size, T)) < ps[:, None]
                    if hit.any():
                        sel = noisy_rows[cis][hit]
                        table = P1d if len(qubits) == 1 else P2d
                        which = rng.integers(0, table.shape[0], size=sel.size)
                        sv.apply(qubits, table[xp.asarray(which)], xp.asarray(sel))

    # ---- reduce on device: probabilities -> marginal over measured qubits -> average trajectories
    meas_sets = {c.measured for c in circuits}
    if len(meas_sets) != 1:
        raise ValueError("all circuits in a batch must measure the same qubits; plan_batches groups them")
    meas, keep, drop = _marginal_axes(n, circuits[0].measured)
    m = len(meas)
    probs = (sv.psi.real ** 2 + sv.psi.imag ** 2).reshape((C * R,) + (2,) * n)
    if drop:
        probs = probs.sum(axis=drop)
    kept_sorted = sorted(keep)
    order = [kept_sorted.index(a) + 1 for a in keep]
    probs = xp.transpose(probs, [0] + order).reshape(C, R, 2 ** m)
    ideal = probs[:, 0, :]
    noisy_p = None
    if T:
        noisy_p = probs[:, 1:, :].mean(axis=1)
        ro = [noise.readout_of(q) for q in meas]
        if any(a or b for a, b in ro):
            t = noisy_p.reshape((C,) + (2,) * m)
            for i, (p01, p10) in enumerate(ro):                  # clbit i sits on axis m - i
                if not (p01 or p10):
                    continue
                conf = xp.asarray(np.array([[1 - p01, p10], [p01, 1 - p10]], dtype=t.dtype))
                ax = m - i
                t = xp.moveaxis(xp.tensordot(t, conf, axes=([ax], [1])), -1, ax)
            noisy_p = t.reshape(C, 2 ** m)
    backend.synchronize()
    stats = {"wall_s": time.perf_counter() - t0, "kernel_calls": sv.kernel_calls, "rows": C * R, "steps": S,
             "n_qubits": n}
    return BatchResult(backend.asnumpy(ideal), None if noisy_p is None else backend.asnumpy(noisy_p), m, stats)
