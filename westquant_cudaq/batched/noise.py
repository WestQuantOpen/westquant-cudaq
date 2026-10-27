"""Pauli (depolarizing) trajectory noise for the batched simulator.

Model, per gate location:
* 1-qubit gate on q: with probability p1(q) a uniformly random Pauli in {X, Y, Z} is applied after the gate;
* 2-qubit gate on (a, b): with probability p2(a, b) a uniformly random non-identity two-qubit Pauli (15 choices);
* readout: bit flips with p(1|0) and p(0|1) per measured qubit, applied to the averaged distribution (exact, linear).

Averaging many trajectories converges to the depolarizing-channel density matrix. A fused single-qubit op that
replaced ``m`` gates receives the exactly composed error probability (depolarizing channels commute with
single-qubit unitaries on their own qubit, so the composition is exact).

``from_calibration`` converts average gate *infidelities* r (what most backends report) to Pauli error probabilities
via p = r (d + 1) / d with d = 2 (one qubit) or d = 4 (two qubits).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_I = np.eye(2, dtype=np.complex128)
PAULI_1Q = np.stack([_X, _Y, _Z])                                                 # (3, 2, 2)
PAULI_2Q = np.stack([np.kron(pb, pa) for pb in (_I, _X, _Y, _Z) for pa in (_I, _X, _Y, _Z)][1:])  # (15, 4, 4)


def _lookup(table: float | Mapping, key, default: float = 0.0) -> float:
    if isinstance(table, (int, float)):
        return float(table)
    for k in (key, str(key), ",".join(map(str, key)) if isinstance(key, tuple) else None):
        if k is not None and k in table:
            return float(table[k])
    if isinstance(key, tuple) and len(key) == 2:
        rev = (key[1], key[0])
        for k in (rev, ",".join(map(str, rev))):
            if k in table:
                return float(table[k])
    return float(table.get("default", default)) if isinstance(table, Mapping) else default


@dataclass
class PauliNoiseModel:
    p1: float | dict = 0.0
    p2: float | dict = 0.0
    readout: float | dict = 0.0      # symmetric flip prob, or {q: [p(1|0), p(0|1)]}
    label: str = "pauli"
    extra: dict = field(default_factory=dict)

    # ---- per-location probabilities
    def p1_of(self, q: int, count: int = 1) -> float:
        p = _lookup(self.p1, q)
        if count <= 1 or p == 0.0:
            return p
        f = (1.0 - 4.0 * p / 3.0) ** count             # depolarizing shrink factor composes multiplicatively
        return 0.75 * (1.0 - f)

    def p2_of(self, a: int, b: int) -> float:
        return _lookup(self.p2, (a, b))

    def readout_of(self, q: int) -> tuple[float, float]:
        if isinstance(self.readout, (int, float)):
            return float(self.readout), float(self.readout)
        v = self.readout.get(q, self.readout.get(str(q), self.readout.get("default", 0.0)))
        if isinstance(v, (int, float)):
            return float(v), float(v)
        return float(v[0]), float(v[1])

    @property
    def is_trivial(self) -> bool:
        def zero(t):
            return (t == 0) if isinstance(t, (int, float)) else all(
                (np.all(np.asarray(v, dtype=float) == 0)) for v in t.values())
        return zero(self.p1) and zero(self.p2)

    # ---- (de)serialisation
    def to_dict(self) -> dict[str, Any]:
        def enc(t):
            if isinstance(t, (int, float)):
                return t
            return {(",".join(map(str, k)) if isinstance(k, tuple) else str(k)): v for k, v in t.items()}
        return {"p1": enc(self.p1), "p2": enc(self.p2), "readout": enc(self.readout), "label": self.label,
                "extra": self.extra}

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PauliNoiseModel":
        return cls(d.get("p1", 0.0), d.get("p2", 0.0), d.get("readout", 0.0), d.get("label", "pauli"),
                   dict(d.get("extra", {})))

    @classmethod
    def from_json(cls, path: str) -> "PauliNoiseModel":
        with open(path) as f:
            d = json.load(f)
        return cls.from_calibration(d) if "gate_error_1q" in d or "gate_error_2q" in d else cls.from_dict(d)

    @classmethod
    def from_calibration(cls, cal: Mapping[str, Any], label: str = "calibration") -> "PauliNoiseModel":
        """``cal`` keys: ``gate_error_1q`` {q: r}, ``gate_error_2q`` {"a,b": r}, ``readout_error`` {q: r or [p01, p10]}.

        Scalars are accepted for any key (uniform error)."""
        def conv(t, d):
            k = (d + 1) / d
            if isinstance(t, (int, float)):
                return min(1.0, float(t) * k)
            return {kk: min(1.0, float(v) * k) for kk, v in t.items()}
        return cls(conv(cal.get("gate_error_1q", 0.0), 2), conv(cal.get("gate_error_2q", 0.0), 4),
                   cal.get("readout_error", 0.0), label, {"source": cal.get("source", "")})
