"""Domain-neutral representation and execution planning for CUDA-Q workloads."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import time
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

__version__ = "0.2.0"


class Engine(str, Enum):
    CPU = "cpu"
    STATEVECTOR = "statevector"
    TENSOR_NETWORK = "tensor_network"
    DENSITY_MATRIX = "density_matrix"
    PAULI_PROPAGATION = "pauli_propagation"
    STABILIZER = "stabilizer"
    QPU = "qpu"


@dataclass(frozen=True)
class WorkloadProfile:
    workload_id: str
    qubits: int
    depth: int
    one_qubit_gates: int = 0
    two_qubit_gates: int = 0
    measurements: int = 0
    observables: int = 1
    parameters: int = 0
    shots: int = 0
    clifford_fraction: float = 0.0
    estimated_entanglement: float = 0.5
    noise_required: bool = False
    gradients_required: bool = False
    precision: str = "fp64"
    framework: str = "generic"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.qubits < 1 or self.depth < 0:
            raise ValueError("qubits must be positive and depth must be non-negative")
        for value in (self.clifford_fraction, self.estimated_entanglement):
            if not 0.0 <= value <= 1.0:
                raise ValueError("fractions must be between zero and one")
        if self.precision not in {"fp32", "fp64"}:
            raise ValueError("precision must be fp32 or fp64")

    @property
    def statevector_bytes(self) -> int:
        bytes_per_amplitude = 8 if self.precision == "fp32" else 16
        return bytes_per_amplitude * (1 << self.qubits)


@dataclass(frozen=True)
class RepresentationCandidate:
    name: str
    profile: WorkloadProfile
    transformation_trace: tuple[str, ...] = ()
    verified: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendCapability:
    name: str
    engine: Engine
    available: bool = True
    gpu: bool = False
    qpu: bool = False
    multi_gpu: bool = False
    max_qubits: int | None = None
    supports_noise: bool = False
    supports_gradients: bool = False
    memory_bytes: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Objective:
    runtime_weight: float = 1.0
    memory_weight: float = 0.2
    quality_weight: float = 2.0
    qpu_weight: float = 4.0
    cost_weight: float = 1.0
    minimum_quality: float = 0.0
    maximum_runtime_s: float | None = None
    maximum_memory_bytes: int | None = None


@dataclass(frozen=True)
class Prediction:
    runtime_s: float
    memory_bytes: int
    quality: float
    monetary_cost: float = 0.0
    qpu_jobs: int = 0
    shots: int = 0
    confidence: float = 0.5


@dataclass(frozen=True)
class Plan:
    representation: RepresentationCandidate
    backend: BackendCapability
    prediction: Prediction
    score: float
    alternatives: tuple[tuple[str, str, float], ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResourceAllocation:
    gpu_count: int
    qpu_jobs: int
    shots: int
    classical_screening_candidates: int
    promoted_candidates: int
    reuse_measurements: bool
    adaptive_stopping: bool


@dataclass(frozen=True)
class MinimizationPlan:
    baseline_qpu_jobs: int
    planned_qpu_jobs: int
    baseline_shots: int
    planned_shots: int
    reduction_factor: float
    strategies: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionPassport:
    schema_version: str
    run_id: str
    created_unix_s: float
    workload: Mapping[str, Any]
    representation: Mapping[str, Any]
    backend: Mapping[str, Any]
    prediction: Mapping[str, Any]
    allocation: Mapping[str, Any]
    environment: Mapping[str, Any]
    result: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=indent, default=str)


class Predictor(Protocol):
    def predict(self, workload: WorkloadProfile, backend: BackendCapability) -> Prediction: ...


class HeuristicPredictor:
    """Transparent baseline intended to be replaced by calibrated predictors."""

    _speed = {
        Engine.CPU: 2.0e6,
        Engine.STATEVECTOR: 2.0e8,
        Engine.TENSOR_NETWORK: 8.0e7,
        Engine.DENSITY_MATRIX: 2.0e7,
        Engine.PAULI_PROPAGATION: 1.2e8,
        Engine.STABILIZER: 5.0e8,
        Engine.QPU: 2.0e4,
    }

    def predict(self, workload: WorkloadProfile, backend: BackendCapability) -> Prediction:
        gates = max(workload.one_qubit_gates + 4 * workload.two_qubit_gates, workload.depth, 1)
        if backend.engine in {Engine.CPU, Engine.STATEVECTOR}:
            work = gates * (2**min(workload.qubits, 32))
            memory = workload.statevector_bytes
        elif backend.engine == Engine.DENSITY_MATRIX:
            work = gates * (4**min(workload.qubits, 16))
            memory = workload.statevector_bytes**2
        elif backend.engine == Engine.TENSOR_NETWORK:
            exponent = max(4.0, workload.qubits * max(workload.estimated_entanglement, 0.1) * 0.65)
            work = gates * (2**min(exponent, 32.0))
            memory = int(min(workload.statevector_bytes, 16 * (2**min(exponent, 30.0))))
        elif backend.engine == Engine.STABILIZER:
            work = gates * workload.qubits**2
            memory = max(4096, workload.qubits**2 * 16)
        elif backend.engine == Engine.PAULI_PROPAGATION:
            work = gates * max(workload.observables, 1) * (1 + 8 * workload.estimated_entanglement)
            memory = max(4096, workload.observables * workload.qubits * 64)
        else:
            work = gates * max(workload.shots, 1)
            memory = 0
        runtime = max(1e-6, work / self._speed[backend.engine])
        quality = 1.0
        if backend.engine == Engine.TENSOR_NETWORK:
            quality -= 0.01 * workload.estimated_entanglement
        if backend.engine == Engine.PAULI_PROPAGATION:
            quality -= 0.02 * workload.estimated_entanglement
        if backend.engine == Engine.QPU:
            quality = 0.9
        qpu_jobs = 1 if backend.qpu else 0
        return Prediction(
            runtime_s=runtime,
            memory_bytes=memory,
            quality=max(0.0, min(1.0, quality)),
            monetary_cost=runtime * (0.02 if backend.gpu else 0.002) + qpu_jobs,
            qpu_jobs=qpu_jobs,
            shots=workload.shots if backend.qpu else 0,
            confidence=0.35,
        )


def normalize_workload(data: WorkloadProfile | Mapping[str, Any], *, framework: str | None = None) -> WorkloadProfile:
    if isinstance(data, WorkloadProfile):
        return replace(data, framework=framework) if framework else data
    aliases = {
        "n_qubits": "qubits",
        "num_qubits": "qubits",
        "n_1q_gates": "one_qubit_gates",
        "n_2q_gates": "two_qubit_gates",
        "n_observables": "observables",
        "n_parameters": "parameters",
    }
    normalized = {aliases.get(key, key): value for key, value in data.items()}
    known = set(WorkloadProfile.__dataclass_fields__)
    extras = {key: value for key, value in normalized.items() if key not in known}
    values = {key: value for key, value in normalized.items() if key in known}
    values.setdefault("workload_id", _stable_hash(normalized)[:16])
    values.setdefault("framework", framework or str(data.get("framework", "generic")))
    values["metadata"] = {**dict(values.get("metadata", {})), **extras}
    return WorkloadProfile(**values)


def discover_capabilities(*, gpu_memory_bytes: int | None = None) -> list[BackendCapability]:
    capabilities = [
        BackendCapability("qpp-cpu", Engine.CPU, supports_noise=True, supports_gradients=True),
    ]
    try:
        import cudaq  # type: ignore

        version = getattr(cudaq, "__version__", _package_version("cudaq"))
        capabilities.extend([
            BackendCapability("nvidia", Engine.STATEVECTOR, gpu=True, multi_gpu=True, supports_gradients=True, memory_bytes=gpu_memory_bytes, metadata={"cudaq": version}),
            BackendCapability("tensornet", Engine.TENSOR_NETWORK, gpu=True, multi_gpu=True, memory_bytes=gpu_memory_bytes, metadata={"cudaq": version}),
            BackendCapability("density-matrix-cpu", Engine.DENSITY_MATRIX, supports_noise=True, max_qubits=16, metadata={"cudaq": version}),
        ])
        if hasattr(cudaq, "get_targets"):
            existing = {item.name for item in capabilities}
            for target in cudaq.get_targets():
                name = getattr(target, "name", str(target))
                if name not in existing:
                    capabilities.append(BackendCapability(name, Engine.QPU, qpu=True, metadata={"cudaq": version, "discovered": True}))
    except ImportError:
        capabilities.extend([
            BackendCapability("nvidia", Engine.STATEVECTOR, available=False, gpu=True, memory_bytes=gpu_memory_bytes),
            BackendCapability("tensornet", Engine.TENSOR_NETWORK, available=False, gpu=True, memory_bytes=gpu_memory_bytes),
        ])
    return capabilities


def default_cuquantum_capabilities(*, available: bool = False, gpu_memory_bytes: int | None = None) -> list[BackendCapability]:
    return [
        BackendCapability("custatevec", Engine.STATEVECTOR, available=available, gpu=True, multi_gpu=True, supports_gradients=True, memory_bytes=gpu_memory_bytes),
        BackendCapability("cutensornet", Engine.TENSOR_NETWORK, available=available, gpu=True, multi_gpu=True, memory_bytes=gpu_memory_bytes),
        BackendCapability("cudensitymat", Engine.DENSITY_MATRIX, available=available, gpu=True, multi_gpu=True, supports_noise=True, memory_bytes=gpu_memory_bytes),
        BackendCapability("cupauliprop", Engine.PAULI_PROPAGATION, available=available, gpu=True, memory_bytes=gpu_memory_bytes),
        BackendCapability("custabilizer", Engine.STABILIZER, available=available, gpu=True, memory_bytes=gpu_memory_bytes),
    ]


def search_representations(
    candidates: Sequence[RepresentationCandidate],
    backends: Sequence[BackendCapability],
    *,
    objective: Objective = Objective(),
    predictor: Predictor | None = None,
) -> Plan:
    if not candidates:
        raise ValueError("at least one representation candidate is required")
    predictor = predictor or HeuristicPredictor()
    evaluated: list[tuple[float, RepresentationCandidate, BackendCapability, Prediction, tuple[str, ...]]] = []
    for candidate in candidates:
        if not candidate.verified:
            continue
        for backend in backends:
            reasons = _incompatibilities(candidate.profile, backend, objective)
            if reasons:
                continue
            prediction = predictor.predict(candidate.profile, backend)
            if prediction.quality < objective.minimum_quality:
                continue
            if objective.maximum_runtime_s is not None and prediction.runtime_s > objective.maximum_runtime_s:
                continue
            if objective.maximum_memory_bytes is not None and prediction.memory_bytes > objective.maximum_memory_bytes:
                continue
            score = _score(prediction, objective)
            evaluated.append((score, candidate, backend, prediction, _selection_reasons(candidate.profile, backend)))
    if not evaluated:
        raise RuntimeError("no verified representation/backend pair satisfies the objective")
    evaluated.sort(key=lambda item: (item[0], item[1].name, item[2].name))
    score, representation, backend, prediction, reasons = evaluated[0]
    alternatives = tuple((item[1].name, item[2].name, item[0]) for item in evaluated[1:6])
    return Plan(representation, backend, prediction, score, alternatives, reasons)


def allocate_resources(
    plan: Plan,
    *,
    candidate_count: int = 1,
    gpu_count: int = 1,
    qpu_budget_jobs: int = 1,
    shot_budget: int | None = None,
) -> ResourceAllocation:
    promoted = 1 if plan.backend.qpu else 0
    qpu_jobs = min(promoted, max(qpu_budget_jobs, 0))
    requested_shots = plan.representation.profile.shots
    shots = min(requested_shots, shot_budget) if shot_budget is not None else requested_shots
    return ResourceAllocation(
        gpu_count=max(0, gpu_count if plan.backend.gpu else 0),
        qpu_jobs=qpu_jobs,
        shots=max(0, shots if qpu_jobs else 0),
        classical_screening_candidates=max(candidate_count - promoted, 0),
        promoted_candidates=promoted,
        reuse_measurements=plan.representation.profile.observables > 1,
        adaptive_stopping=plan.representation.profile.parameters > 0,
    )


def minimize_qpu(
    workload: WorkloadProfile,
    allocation: ResourceAllocation,
    *,
    baseline_evaluations: int = 30,
) -> MinimizationPlan:
    baseline_jobs = max(1, baseline_evaluations)
    baseline_shots = baseline_jobs * max(workload.shots, 1)
    planned_jobs = allocation.qpu_jobs
    planned_shots = allocation.shots
    strategies = ["gpu_preselection"]
    if workload.parameters:
        strategies.extend(["classical_parameter_optimization", "adaptive_stopping"])
    if allocation.reuse_measurements:
        strategies.append("measurement_reuse")
    if planned_shots < workload.shots:
        strategies.append("adaptive_shots")
    reduction = math.inf if planned_shots == 0 else baseline_shots / planned_shots
    return MinimizationPlan(baseline_jobs, planned_jobs, baseline_shots, planned_shots, reduction, tuple(strategies))


def create_passport(
    plan: Plan,
    allocation: ResourceAllocation,
    *,
    result: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> ExecutionPassport:
    payload = {
        "workload": asdict(plan.representation.profile),
        "representation": {"name": plan.representation.name, "trace": plan.representation.transformation_trace, "metadata": dict(plan.representation.metadata)},
        "backend": asdict(plan.backend),
        "prediction": asdict(plan.prediction),
        "allocation": asdict(allocation),
    }
    run_id = _stable_hash(payload)[:24]
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "westquant_cudaq": __version__,
        "cudaq": _package_version("cudaq"),
        "cuquantum_python": _package_version("cuquantum-python"),
        **dict(environment or {}),
    }
    return ExecutionPassport("1.0", run_id, time.time(), payload["workload"], payload["representation"], payload["backend"], payload["prediction"], payload["allocation"], env, dict(result or {}))


def execute_cudaq(
    kernel: Callable[..., Any],
    *args: Any,
    target: str = "nvidia",
    observe_operator: Any | None = None,
    shots: int = 0,
    **kwargs: Any,
) -> Any:
    try:
        import cudaq  # type: ignore
    except ImportError as exc:
        raise RuntimeError("CUDA-Q is optional; install westquant-cudaq[cudaq]") from exc
    cudaq.set_target(target)
    if observe_operator is not None:
        return cudaq.observe(kernel, observe_operator, *args, **kwargs)
    if shots > 0:
        return cudaq.sample(kernel, *args, shots_count=shots, **kwargs)
    return kernel(*args, **kwargs)


def _incompatibilities(workload: WorkloadProfile, backend: BackendCapability, objective: Objective) -> tuple[str, ...]:
    reasons: list[str] = []
    if not backend.available:
        reasons.append("unavailable")
    if backend.max_qubits is not None and workload.qubits > backend.max_qubits:
        reasons.append("qubit_limit")
    if workload.noise_required and not backend.supports_noise:
        reasons.append("noise_unsupported")
    if workload.gradients_required and not backend.supports_gradients:
        reasons.append("gradients_unsupported")
    if backend.memory_bytes is not None and workload.statevector_bytes > backend.memory_bytes and backend.engine == Engine.STATEVECTOR:
        reasons.append("insufficient_gpu_memory")
    if objective.maximum_memory_bytes is not None and workload.statevector_bytes > objective.maximum_memory_bytes and backend.engine == Engine.STATEVECTOR:
        reasons.append("memory_objective")
    if backend.engine == Engine.STABILIZER and workload.clifford_fraction < 0.98:
        reasons.append("non_clifford_workload")
    return tuple(reasons)


def _selection_reasons(workload: WorkloadProfile, backend: BackendCapability) -> tuple[str, ...]:
    reasons = [f"selected_{backend.engine.value}"]
    if backend.engine == Engine.TENSOR_NETWORK and workload.estimated_entanglement < 0.5:
        reasons.append("limited_entanglement")
    if backend.engine == Engine.STABILIZER:
        reasons.append("clifford_dominant")
    if backend.engine == Engine.PAULI_PROPAGATION:
        reasons.append("observable_centric")
    if backend.gpu:
        reasons.append("gpu_accelerated")
    if backend.qpu:
        reasons.append("qpu_required")
    return tuple(reasons)


def _score(prediction: Prediction, objective: Objective) -> float:
    quality_penalty = max(0.0, 1.0 - prediction.quality)
    return (
        objective.runtime_weight * math.log1p(prediction.runtime_s)
        + objective.memory_weight * math.log1p(prediction.memory_bytes) / 32.0
        + objective.quality_weight * quality_penalty
        + objective.qpu_weight * prediction.qpu_jobs
        + objective.cost_weight * prediction.monetary_cost
    )


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


__all__ = [
    "BackendCapability",
    "Engine",
    "ExecutionPassport",
    "HeuristicPredictor",
    "MinimizationPlan",
    "Objective",
    "Plan",
    "Prediction",
    "Predictor",
    "RepresentationCandidate",
    "ResourceAllocation",
    "WorkloadProfile",
    "allocate_resources",
    "create_passport",
    "default_cuquantum_capabilities",
    "discover_capabilities",
    "execute_cudaq",
    "minimize_qpu",
    "normalize_workload",
    "search_representations",
]


def __getattr__(name: str) -> Any:
    """Lazy access to the GPU batch simulator (``westquant_cudaq.batched``).

    Importing :mod:`westquant_cudaq` stays dependency-free; the batched engine (NumPy, optional CuPy,
    optional Qiskit) is only imported when it is first used.
    """
    if name == "batched":
        import importlib

        return importlib.import_module(".batched", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

