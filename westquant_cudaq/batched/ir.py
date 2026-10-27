"""Minimal circuit IR for batched simulation: a list of dense 1- and 2-qubit unitaries.

Conventions (identical to Qiskit):
* qubit 0 is the least significant bit of a basis-state index;
* the matrix of a gate on ``qubits=(a, b)`` acts on the index ``bit(a) + 2*bit(b)``, i.e. ``b`` is the more
  significant qubit and the matrix equals ``kron(U_b, U_a)`` for a product gate.

``from_qiskit`` fuses gates (see :func:`fuse_ops`). On a GPU every op is one batched kernel call for the whole batch,
so fusion directly raises throughput.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

_I2 = np.eye(2, dtype=np.complex128)


@dataclass(frozen=True)
class Op:
    qubits: tuple[int, ...]          # 1 or 2 qubits; matrix index ordering as in the module docstring
    matrix: np.ndarray               # (2**k, 2**k) complex128
    name: str = "u"
    count: int = 1                   # number of original gates fused into this op (used to compose noise)


@dataclass
class CircuitIR:
    num_qubits: int
    ops: list[Op]
    measured: tuple[int, ...] | None = None   # qubits read out (in classical-bit order); None = all qubits
    circuit_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    fusion: str = "full"                       # "full" | "1q" | "none" (see fuse_ops)

    @property
    def n_two_qubit(self) -> int:
        return sum(1 for op in self.ops if len(op.qubits) == 2)

    def structure_key(self) -> str:
        """Hash of (n, sequence of target tuples). Circuits with equal keys batch without any gather/scatter."""
        h = hashlib.blake2b(digest_size=12)
        h.update(str(self.num_qubits).encode())
        for op in self.ops:
            h.update(b"|" + ",".join(map(str, op.qubits)).encode())
        return h.hexdigest()


def fuse_ops(num_qubits: int, ops: Iterable[Op], mode: str = "full") -> list[Op]:
    """Exact gate fusion.

    * ``"1q"``: merge consecutive single-qubit gates on the same qubit (``count`` = number merged). Safe with
      depolarizing noise, because a depolarizing channel commutes with single-qubit unitaries on its qubit.
    * ``"full"``: additionally absorb pending single-qubit gates into the next two-qubit gate on that qubit. Fewest
      kernel launches; use for ideal (noise-free) simulation only.
    * ``"none"``: no fusion.
    """
    ops = list(ops)
    if mode == "none":
        return ops
    pending: dict[int, tuple[np.ndarray, int]] = {}
    out: list[Op] = []

    def flush(q: int) -> None:
        if q in pending:
            m, c = pending.pop(q)
            out.append(Op((q,), m, "fused1", c))

    for op in ops:
        if len(op.qubits) == 1:
            q = op.qubits[0]
            m, c = pending.get(q, (_I2, 0))
            pending[q] = (op.matrix @ m, c + op.count)
        elif len(op.qubits) == 2:
            a, b = op.qubits
            if mode == "1q":
                flush(a); flush(b)
                out.append(op)
                continue
            pa, pb = pending.pop(a, None), pending.pop(b, None)
            if pa is None and pb is None:
                out.append(op)
            else:
                pre = np.kron(_I2 if pb is None else pb[0], _I2 if pa is None else pa[0])
                out.append(Op((a, b), op.matrix @ pre, "fused2", op.count))
        else:
            raise ValueError(f"only 1- and 2-qubit ops are supported, got {op.qubits}")
    for q in sorted(pending):
        flush(q)
    return out


def from_qiskit(qc, *, fusion: str = "full", circuit_id: str = "", meta: dict | None = None,
                basis_gates: Sequence[str] = ("u", "cx")) -> CircuitIR:
    """Convert a Qiskit ``QuantumCircuit`` (final measurements allowed) to :class:`CircuitIR`.

    Gates on more than two qubits are decomposed with ``transpile(..., basis_gates, optimization_level=0)``.
    Mid-circuit measurements, resets and classical control are rejected: use a trajectory simulator for those.
    """
    from qiskit import transpile
    from qiskit.quantum_info import Operator

    measured: list[tuple[int, int]] = []
    for inst in qc.data:
        if inst.operation.name == "measure":
            measured.append((qc.find_bit(inst.clbits[0]).index, qc.find_bit(inst.qubits[0]).index))
    base = qc.remove_final_measurements(inplace=False)
    if any(i.operation.name in ("measure", "reset") or getattr(i.operation, "condition", None) is not None
           for i in base.data):
        raise ValueError("mid-circuit measurement, reset or classical control is not supported")
    if any(i.operation.num_qubits > 2 for i in base.data if i.operation.name != "barrier"):
        base = transpile(base, basis_gates=list(basis_gates), optimization_level=0)

    ops: list[Op] = []
    for inst in base.data:
        name = inst.operation.name
        if name in ("barrier", "delay") or name.startswith("save_"):
            continue
        qs = tuple(base.find_bit(q).index for q in inst.qubits)
        if name == "id":
            continue
        mat = np.asarray(Operator(inst.operation).data, dtype=np.complex128)
        ops.append(Op(qs, mat, name))
    ops = fuse_ops(base.num_qubits, ops, fusion)
    meas = tuple(q for _c, q in sorted(measured)) if measured else None
    return CircuitIR(base.num_qubits, ops, meas, circuit_id, dict(meta or {}), fusion)


def from_qasm(text: str, **kw) -> CircuitIR:
    """Parse OpenQASM 2 (built-in) or OpenQASM 3 (needs ``qiskit-qasm3-import``) and convert."""
    head = text.lstrip()[:40].upper()
    if head.startswith("OPENQASM 3"):
        from qiskit import qasm3

        qc = qasm3.loads(text)
    else:
        from qiskit import qasm2

        qc = qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    return from_qiskit(qc, **kw)
