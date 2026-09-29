import json

import pytest

from westquant_cudaq import (
    BackendCapability,
    Engine,
    Objective,
    RepresentationCandidate,
    WorkloadProfile,
    allocate_resources,
    create_passport,
    default_cuquantum_capabilities,
    discover_capabilities,
    minimize_qpu,
    normalize_workload,
    search_representations,
)


def profile(**overrides):
    values = dict(workload_id="qaoa-20", qubits=20, depth=30, one_qubit_gates=40, two_qubit_gates=60, observables=8, parameters=4, shots=1024)
    values.update(overrides)
    return WorkloadProfile(**values)


def test_normalize_framework_aliases_and_metadata():
    item = normalize_workload({"num_qubits": 8, "depth": 4, "n_2q_gates": 6, "vendor_field": 3}, framework="qiskit")
    assert item.qubits == 8
    assert item.two_qubit_gates == 6
    assert item.framework == "qiskit"
    assert item.metadata["vendor_field"] == 3


def test_invalid_profile_is_rejected():
    with pytest.raises(ValueError):
        profile(clifford_fraction=1.1)


def test_capability_discovery_has_cpu_fallback():
    capabilities = discover_capabilities()
    assert capabilities[0].name == "qpp-cpu"
    assert capabilities[0].available


def test_cuquantum_catalog_is_domain_neutral():
    names = {item.name for item in default_cuquantum_capabilities()}
    assert names == {"custatevec", "cutensornet", "cudensitymat", "cupauliprop", "custabilizer"}


def test_search_rejects_unverified_and_unavailable_candidates():
    workload = profile(qubits=8)
    candidates = [
        RepresentationCandidate("bad", workload, verified=False),
        RepresentationCandidate("good", workload, ("group_observables",)),
    ]
    backends = [
        BackendCapability("missing", Engine.STATEVECTOR, available=False, gpu=True),
        BackendCapability("cpu", Engine.CPU),
    ]
    plan = search_representations(candidates, backends)
    assert plan.representation.name == "good"
    assert plan.backend.name == "cpu"


def test_statevector_memory_constraint_selects_tensor_network():
    workload = profile(qubits=28, estimated_entanglement=0.2)
    candidates = [RepresentationCandidate("compact", workload)]
    backends = [
        BackendCapability("statevector", Engine.STATEVECTOR, gpu=True, memory_bytes=1_000_000),
        BackendCapability("tensor", Engine.TENSOR_NETWORK, gpu=True, memory_bytes=1_000_000),
    ]
    plan = search_representations(candidates, backends)
    assert plan.backend.engine == Engine.TENSOR_NETWORK
    assert "limited_entanglement" in plan.reasons


def test_noise_requirement_selects_compatible_backend():
    workload = profile(qubits=6, noise_required=True)
    plan = search_representations(
        [RepresentationCandidate("noisy", workload)],
        [
            BackendCapability("gpu", Engine.STATEVECTOR, gpu=True),
            BackendCapability("density", Engine.DENSITY_MATRIX, supports_noise=True),
        ],
    )
    assert plan.backend.name == "density"


def test_allocation_and_qpu_minimization():
    workload = profile()
    plan = search_representations(
        [RepresentationCandidate("qaoa", workload)],
        [BackendCapability("qpu", Engine.QPU, qpu=True)],
        objective=Objective(qpu_weight=0.0),
    )
    allocation = allocate_resources(plan, candidate_count=100, qpu_budget_jobs=1, shot_budget=512)
    minimized = minimize_qpu(workload, allocation, baseline_evaluations=30)
    assert allocation.classical_screening_candidates == 99
    assert allocation.shots == 512
    assert minimized.reduction_factor == 60
    assert "adaptive_shots" in minimized.strategies


def test_passport_is_deterministic_except_timestamp():
    workload = profile(qubits=8)
    plan = search_representations([RepresentationCandidate("base", workload, ("normalize",))], [BackendCapability("cpu", Engine.CPU)])
    allocation = allocate_resources(plan)
    first = create_passport(plan, allocation)
    second = create_passport(plan, allocation)
    assert first.run_id == second.run_id
    decoded = json.loads(first.to_json())
    assert decoded["schema_version"] == "1.0"
    assert decoded["representation"]["trace"] == ["normalize"]
