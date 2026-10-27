# Changelog

## 0.2.0 — 2026-10-27

- Added `westquant_cudaq.batched`: batched state-vector simulation of many small
  circuits (NumPy on CPU, CuPy on GPU) with Pauli-trajectory noise, readout
  error, shots and Parquet output with provenance. Validated against Qiskit
  `Statevector` and an exact density-matrix reference.
- Added the `wq-batch` command (`run`, `plan`, `bench`) and Slurm scripts for
  A100 clusters in `scripts/hpc/`. See `docs/BATCHED.md`.
- New optional extras `batched` and `gpu`. `import westquant_cudaq` stays
  dependency-free; `westquant_cudaq.batched` is imported lazily.
- The module became a package (`westquant_cudaq/`); the public 0.1.0 API is
  unchanged and its tests still pass.
- CI: tests workflow and a test-gated PyPI Trusted Publishing workflow.

## 0.1.0 — 2026-09-29

- Added framework-neutral workload profiles and representation candidates.
- Added CUDA-Q and cuQuantum capability discovery with no-CUDA fallback.
- Added representation search and backend/simulator policy selection.
- Added pluggable cost/quality prediction interfaces.
- Added GPU/QPU resource allocation and QPU-minimization plans.
- Added deterministic execution passports and JSON serialization.
- Added an optional CUDA-Q executor without a mandatory CUDA dependency.
