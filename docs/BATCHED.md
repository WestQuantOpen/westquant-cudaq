# Batched GPU simulation (`westquant_cudaq.batched`)

Throughput-oriented state-vector simulation of **many small circuits** (about 2–24 qubits) for dataset generation.
One batch holds thousands of circuits of the same width; each gate step is one batched matrix product for every
row that shares the same target qubits, so the GPU stays busy even when each circuit is tiny.

Status: v0.2.0. Validated against Qiskit `Statevector` and an exact density-matrix reference on CPU (NumPy).
The GPU path runs the same array code through CuPy (cuBLAS batched GEMM); **run `bench_a100.sbatch` once on the
cluster before a campaign** – it runs the test suite on the GPU and measures throughput per qubit count.

## Install

```bash
pip install "westquant-cudaq[batched,gpu]"     # CUDA 12 + CuPy; CPU-only: [batched]
```

## Python

```python
from westquant_cudaq.batched import BatchRunner, PauliNoiseModel, from_qiskit, get_backend

irs = [from_qiskit(qc, fusion="1q", circuit_id=f"c{i}", meta={"expected": "0101"}) for i, qc in enumerate(circuits)]
noise = PauliNoiseModel.from_calibration({"gate_error_1q": 5e-4, "gate_error_2q": 8e-3, "readout_error": 0.015})
runner = BatchRunner(backend=get_backend("cupy"), noise=noise, trajectories=200, shots=1000)
runner.run_to_parquet(irs, "out.parquet")
```

`fusion`: `"full"` for ideal runs (fewest kernel calls), `"1q"` when noise is used (exact error placement).
The CLI picks this automatically.

## HPC workflow (Slurm, A100)

Submit everything **from the repository root** (the scripts locate themselves via `$SLURM_SUBMIT_DIR`).

1. **Edit `scripts/hpc/env.sh` once** (modules, venv path) and fill in `--account/--partition` in the `.sbatch` files.
2. **Validate and benchmark on one GPU:** `sbatch scripts/hpc/bench_a100.sbatch $PWD`
3. **Shard the circuits:**
   `python scripts/hpc/make_manifest.py --circuits circuits.parquet --out $WORK/wq/shards --max-circuits 20000`
   (input: JSONL/Parquet with `circuit_id`, `qasm`, optional `family`, `expected`, `meta`; or `--qasm-dir`).
   Check the plan without a GPU: `wq-batch plan --manifest $WORK/wq/shards/shard_00000.jsonl --gpu-mem-gb 64`.
4. **Run**, either as an array (one GPU per task)
   `sbatch --array=0-15 scripts/hpc/run_array_a100.sbatch $WORK/wq/shards $WORK/wq/out --noise noise.json --trajectories 200 --shots 1000`
   or on whole 4-GPU nodes
   `sbatch --nodes=4 scripts/hpc/run_node_4xa100.sbatch $WORK/wq/shards $WORK/wq/out --noise noise.json --trajectories 200`.
   Shard *i* goes to rank *i mod K*. Finished shards are skipped, so after a timeout you resubmit the same command.
5. **Merge and check:**
   `python scripts/hpc/merge_shards.py --shards $WORK/wq/shards --out-dir $WORK/wq/out --dataset $WORK/wq/dataset`
   → `dataset/data/n_qubits=*/…parquet` and `dataset/campaign_summary.json`; exit code 1 if anything is missing.

## Tuning on A100

* **dtype:** `complex64` (default) halves memory and, since the kernels are memory-bound, typically gives close to 2× throughput versus `complex128`; the tests use
  2e-5 tolerance. Use `complex128` for validation runs.
* **Batch size** is set from free GPU memory: rows ≈ 0.6 × free / (2ⁿ × 8 B × 5). `--safety` changes the 0.6.
  Noisy runs use `1 + trajectories` rows per circuit, so fewer circuits fit per batch.
* **Where batching pays:** small n (≲ 14) is limited by per-step host work and kernel launches – batching gives the
  largest gain there. Somewhere around n ≈ 20–24 a single circuit fills the GPU by itself; there, per-circuit simulation with CUDA-Q/cuStateVec is the better tool (measure the crossover with `bench_a100.sbatch`).
* **Homogeneous batches are fastest.** Circuits with identical layouts (parameter sweeps, transpilation variants of
  the same layout, trajectories) need one kernel call per gate and no gather/scatter. `make_manifest.py` sorts by
  size; the runner sorts by structure within a batch.
* **CPU cores** parse QASM in parallel (`--workers`, default `$SLURM_CPUS_PER_TASK`). If parsing dominates
  (`parse_s` in the summary), raise `--cpus-per-task` or pre-convert circuits once.
* If GPU utilisation is low for small n, run 2 processes per GPU (CUDA MPS) so one parses while the other simulates.

## Noise model and limits

Pauli trajectories: after each gate a random Pauli error with the gate's depolarizing probability; readout flips
per clbit applied exactly to the averaged distribution. Average gate infidelity r is converted with
p = r(d+1)/d. Not modelled (yet): idle decoherence (T1/T2), crosstalk, leakage, coherent errors, mid-circuit
measurement and reset. Statistical error of the noisy distribution ~ 1/√trajectories.

## Output columns (one row per circuit)

`circuit_id, n_qubits, n_measured, n_ops, n_2q, structure_key, fusion, meta` · `ideal_/noisy_top_bitstrings`,
`*_top_probs`, `*_probs` (full distribution when ≤ 10 measured bits), `*_success_prob` (if `expected` given),
`hellinger_fidelity`, `counts` (if `--shots`) · provenance: `run_id, westquant_cudaq, qiskit, gpu_name, dtype,
noise_digest, trajectories, seed, host, slurm_*`. Schema: `westquant_cudaq.batched.record_schema()`.

## Roadmap

* v0.3: cuStateVec batched backend (`custatevecApplyMatrixBatched`) behind the same API, compared against the CuPy
  path on A100; device-side random numbers for trajectories; CUDA Graphs for repeated layouts.
* v0.3: idle T1/T2 noise (Pauli-twirled thermal relaxation) from backend calibration.
* Records map 1:1 onto the WQBench `hardware_runs`/`transitions` tables; link each campaign to an
  `ExecutionPassport`.
