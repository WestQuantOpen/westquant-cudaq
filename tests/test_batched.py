"""Correctness of the batched simulator against Qiskit and against an exact density-matrix reference.

Runs on CPU (NumPy backend). The GPU path executes the same array code through CuPy; run this file on the A100 with
``WQ_TEST_BACKEND=cupy pytest tests/test_batched.py`` to validate it there.
"""
import json
import os

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit.random import random_circuit  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402

from westquant_cudaq.batched import (BatchRunner, PauliNoiseModel, from_qiskit, get_backend,  # noqa: E402
                                     simulate_batch)
from westquant_cudaq.batched.noise import PAULI_1Q, PAULI_2Q  # noqa: E402

BACKEND = get_backend(os.environ.get("WQ_TEST_BACKEND", "numpy"))
TOL = 2e-5   # complex64


def _reference(qc: QuantumCircuit, measured):
    sv = Statevector(qc.remove_final_measurements(inplace=False))
    return sv.probabilities(qargs=list(measured) if measured is not None else None)


def _measured_qubits(qc):
    pairs = sorted((qc.find_bit(i.clbits[0]).index, qc.find_bit(i.qubits[0]).index)
                   for i in qc.data if i.operation.name == "measure")
    return tuple(q for _c, q in pairs) if pairs else None


@pytest.mark.parametrize("fusion", ["none", "1q", "full"])
def test_heterogeneous_batch_matches_qiskit(fusion):
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 5):
        qcs = []
        for i in range(24):
            qc = random_circuit(n, depth=12, max_operands=min(3, n), seed=int(rng.integers(1 << 30)))
            if i % 3 == 0 and n > 1:                               # permuted, partial measurement
                perm = rng.permutation(n)[: max(1, n - 1)]
                qc2 = QuantumCircuit(n, len(perm)); qc2.compose(qc, inplace=True)
                for c, q in enumerate(perm):
                    qc2.measure(int(q), c)
                qc = qc2
            qcs.append(qc)
        irs = [from_qiskit(qc, fusion=fusion, circuit_id=str(i)) for i, qc in enumerate(qcs)]
        groups = {}
        for qc, ir in zip(qcs, irs):
            groups.setdefault(ir.measured, []).append((qc, ir))
        for meas, items in groups.items():
            res = simulate_batch([ir for _q, ir in items], BACKEND)
            for j, (qc, ir) in enumerate(items):
                np.testing.assert_allclose(res.ideal[j], _reference(qc, meas), atol=TOL)


def test_same_structure_fast_path_and_kernel_count():
    base = random_circuit(6, depth=20, max_operands=2, seed=7)
    ir = from_qiskit(base, fusion="full")
    res = simulate_batch([ir] * 64, BACKEND)
    ref = _reference(base, None)
    np.testing.assert_allclose(res.ideal, np.tile(ref, (64, 1)), atol=TOL)
    assert res.stats["kernel_calls"] == len(ir.ops)              # one batched call per op, independent of batch size


def test_zero_noise_trajectories_equal_ideal():
    qc = random_circuit(4, depth=10, max_operands=2, seed=3)
    ir = from_qiskit(qc, fusion="1q")
    res = simulate_batch([ir, ir], BACKEND, noise=PauliNoiseModel(0.0, 0.0), trajectories=50)
    np.testing.assert_allclose(res.noisy, res.ideal, atol=TOL)


def test_full_fusion_rejected_with_noise():
    ir = from_qiskit(random_circuit(3, 6, max_operands=2, seed=1), fusion="full")
    with pytest.raises(ValueError):
        simulate_batch([ir], BACKEND, noise=PauliNoiseModel(0.01, 0.02), trajectories=4)


def _density_reference(ir, noise):
    n = ir.num_qubits
    d = 2 ** n
    rho = np.zeros((d, d), complex); rho[0, 0] = 1

    def embed(op_mat, qubits):
        # full operator via Statevector-free tensor construction
        full = np.zeros((d, d), complex)
        for col in range(d):
            vec = np.zeros(d, complex); vec[col] = 1
            t = vec.reshape((2,) * n)
            axes = [n - 1 - q for q in reversed(qubits)]
            t = np.moveaxis(t, axes, list(range(n - len(qubits), n)))
            shp = t.shape
            t = (t.reshape(-1, 2 ** len(qubits)) @ op_mat.T).reshape(shp)
            full[:, col] = np.moveaxis(t, list(range(n - len(qubits), n)), axes).reshape(d)
        return full

    for op in ir.ops:
        U = embed(op.matrix, op.qubits)
        rho = U @ rho @ U.conj().T
        if len(op.qubits) == 1:
            p = noise.p1_of(op.qubits[0], op.count); table = PAULI_1Q
        else:
            p = noise.p2_of(*op.qubits); table = PAULI_2Q
        if p:
            mix = sum(embed(P, op.qubits) @ rho @ embed(P, op.qubits).conj().T for P in table) / len(table)
            rho = (1 - p) * rho + p * mix
    return np.real(np.diag(rho))


def test_depolarizing_trajectories_converge_to_density_matrix():
    qc = QuantumCircuit(3)
    qc.h(0); qc.rx(0.7, 1); qc.cx(0, 1); qc.ry(1.1, 2); qc.cz(1, 2); qc.t(0); qc.s(0); qc.cx(2, 0); qc.h(1)
    ir = from_qiskit(qc, fusion="1q")
    noise = PauliNoiseModel(p1=0.04, p2={"0,1": 0.12, "1,2": 0.08, "default": 0.1})
    ref = _density_reference(ir, noise)
    res = simulate_batch([ir], BACKEND, noise=noise, trajectories=40000, rng=np.random.default_rng(5))
    np.testing.assert_allclose(res.noisy[0], ref, atol=6e-3)


def test_readout_error_is_applied_per_clbit():
    qc = QuantumCircuit(2, 2); qc.x(0); qc.measure(0, 0); qc.measure(1, 1)     # true outcome "01"
    ir = from_qiskit(qc)
    noise = PauliNoiseModel(0.0, 0.0, readout={0: [0.1, 0.2], 1: [0.3, 0.0]})
    res = simulate_batch([ir], BACKEND, noise=noise, trajectories=2)
    # clbit0: true 1 -> P(1)=0.8 ; clbit1: true 0 -> P(1)=0.3 ; index = c0 + 2 c1
    expect = np.array([0.2 * 0.7, 0.8 * 0.7, 0.2 * 0.3, 0.8 * 0.3])
    np.testing.assert_allclose(res.noisy[0], expect, atol=1e-6)


def test_runner_records_and_parquet(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    qcs = []
    for i in range(10):
        qc = QuantumCircuit(3, 3); qc.h(0); qc.cx(0, 1); qc.cx(1, 2); qc.measure([0, 1, 2], [0, 1, 2]); qcs.append(qc)
    irs = [from_qiskit(qc, fusion="1q", circuit_id=f"ghz{i}", meta={"expected": ["000", "111"]}) for i, qc in enumerate(qcs)]
    runner = BatchRunner(backend=BACKEND, noise=PauliNoiseModel(0.01, 0.03, 0.02), trajectories=200, shots=500)
    out = tmp_path / "s.parquet"
    info = runner.run_to_parquet(irs, str(out))
    t = pq.read_table(out).to_pylist()
    assert info["records"] == 10 and len(t) == 10
    assert abs(t[0]["ideal_success_prob"] - 1.0) < 1e-5
    assert 0.8 < t[0]["noisy_success_prob"] < 1.0
    assert sum(json.loads(t[0]["counts"]).values()) == 500


def test_cli_run_on_manifest(tmp_path):
    from qiskit import qasm2

    from westquant_cudaq.batched.cli import main
    man = tmp_path / "m.jsonl"
    with open(man, "w") as f:
        for i in range(6):
            qc = random_circuit(4, 8, max_operands=2, measure=True, seed=i)
            f.write(json.dumps({"circuit_id": f"r{i}", "qasm": qasm2.dumps(qc), "family": "random"}) + "\n")
        f.write(json.dumps({"circuit_id": "bad", "qasm": "not qasm"}) + "\n")
    out = tmp_path / "o.parquet"
    assert main(["run", "--manifest", str(man), "--out", str(out), "--backend", "numpy", "--workers", "1"]) == 0
    summary = json.loads((tmp_path / "o.parquet.summary.json").read_text())
    assert summary["circuits"] == 6 and summary["parse_errors"] == 1
