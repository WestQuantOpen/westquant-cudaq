# WestQuant CUDA-Q

Domain-neutral representation search and execution intelligence for CUDA-Q and cuQuantum.

> CUDA-Q executes efficiently. WestQuant decides what representation to run, where to run it, and how much quantum work is justified.

## Scope

This package is intentionally industry-neutral. It contains no chemistry, life-science, finance, logistics, or other vertical-specific methods.

It provides:

- Representation candidate search.
- Cross-framework workload normalization.
- CUDA-Q and cuQuantum capability discovery.
- Backend and simulator selection.
- Pluggable cost/quality prediction.
- GPU/QPU resource allocation.
- Reproducibility passports.
- QPU-minimization planning.
- Optional CUDA-Q execution.
- Cross-vendor backend descriptions.

## Install

```bash
pip install westquant-cudaq
```

Install CUDA-Q integration where NVIDIA CUDA-Q is supported:

```bash
pip install 'westquant-cudaq[cudaq]'
```

The planner and tests work without an NVIDIA GPU or CUDA-Q installation.

## Quick start

```python
from westquant_cudaq import (
    BackendCapability,
    Engine,
    RepresentationCandidate,
    WorkloadProfile,
    allocate_resources,
    create_passport,
    default_cuquantum_capabilities,
    minimize_qpu,
    search_representations,
)

workload = WorkloadProfile(
    workload_id="generic-qaoa-24",
    qubits=24,
    depth=40,
    one_qubit_gates=48,
    two_qubit_gates=96,
    observables=12,
    parameters=8,
    shots=1024,
    estimated_entanglement=0.3,
)

representations = [
    RepresentationCandidate("direct", workload),
    RepresentationCandidate(
        "grouped-observables",
        WorkloadProfile(**{
            **workload.__dict__,
            "observables": 5,
            "depth": 36,
        }),
        transformation_trace=("commute", "group_observables"),
    ),
]

backends = [
    BackendCapability("qpp-cpu", Engine.CPU),
    *default_cuquantum_capabilities(available=True, gpu_memory_bytes=24 * 2**30),
]

plan = search_representations(representations, backends)
allocation = allocate_resources(plan, candidate_count=100, gpu_count=1, qpu_budget_jobs=1)
minimization = minimize_qpu(workload, allocation)
passport = create_passport(plan, allocation)

print(plan.backend.name, plan.representation.name)
print(minimization.reduction_factor)
print(passport.to_json())
```

## Policy model

`HeuristicPredictor` is a transparent, deterministic baseline. Production users can implement the `Predictor` protocol with calibrated measurements or learned models:

```python
class MyPredictor:
    def predict(self, workload, backend):
        ...
```

WestQuant never executes an unverified representation candidate. Constraints for noise, gradients, qubit capacity, and memory are enforced before scoring.

## CUDA-Q execution

`execute_cudaq` imports CUDA-Q lazily and supports kernel execution, sampling, and observation. No CUDA dependency is imported when using only planning and passport APIs.

## Relationship to WestQuant Open

- `westquant-core`: WQIR, RepGraph, plugin contracts, and generic search.
- `westquant-bridges`: static representation bridges, including CUDA-Q text/kernel bridges.
- `westquant-orchestrator`: cross-framework run normalization and merged datasets.
- `westquant-qcsc`: semantic QPU minimization and workflow planning.
- `westquant-cudaq`: NVIDIA CUDA-Q/cuQuantum execution intelligence and capability-aware planning.

## License

Apache-2.0
