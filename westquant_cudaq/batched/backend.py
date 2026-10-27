"""Array backend selection for the batched simulator.

The engine is written against the NumPy array API and runs unchanged on CuPy. On an NVIDIA GPU the heavy
operation (one batched matrix product per gate) is dispatched by CuPy to cuBLAS batched GEMM, so thousands of
small state vectors are updated in a single kernel launch.

``prefer="auto"`` uses CuPy when it imports and sees a device, otherwise NumPy (useful for tests and for laptops).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Backend:
    xp: Any                      # numpy or cupy module
    name: str                    # "numpy" | "cupy"
    device: int | None = None
    info: dict = field(default_factory=dict)

    def asnumpy(self, a):
        return self.xp.asnumpy(a) if self.name == "cupy" else np.asarray(a)

    def asarray(self, a, dtype=None):
        return self.xp.asarray(a, dtype=dtype)

    def synchronize(self) -> None:
        if self.name == "cupy":
            self.xp.cuda.Stream.null.synchronize()

    def free_bytes(self) -> int:
        """Free device memory (GPU) or a conservative host budget (CPU)."""
        if self.name == "cupy":
            free, _total = self.xp.cuda.runtime.memGetInfo()
            # CuPy's memory pool may hold freed blocks; count them as available.
            pool = self.xp.get_default_memory_pool()
            return int(free + pool.free_bytes())
        env = os.environ.get("WQ_BATCH_HOST_BYTES")
        return int(env) if env else 4 * 2**30


def get_backend(prefer: str = "auto", device: int | None = None) -> Backend:
    """Return a CuPy backend if possible (``prefer`` in {"auto", "cupy"}), else NumPy.

    ``prefer="cupy"`` raises if CuPy or a GPU is missing, so HPC jobs fail loudly instead of silently running on CPU.
    """
    if prefer in ("auto", "cupy"):
        try:
            import cupy as cp  # noqa: WPS433

            n_dev = cp.cuda.runtime.getDeviceCount()
            if n_dev == 0:
                raise RuntimeError("no CUDA device visible")
            dev = 0 if device is None else int(device)
            cp.cuda.Device(dev).use()
            props = cp.cuda.runtime.getDeviceProperties(dev)
            name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
            return Backend(cp, "cupy", dev, {"gpu_name": name, "gpu_total_bytes": int(props["totalGlobalMem"]),
                                             "cupy": cp.__version__})
        except Exception as exc:  # noqa: BLE001
            if prefer == "cupy":
                raise RuntimeError(f"CuPy GPU backend requested but unavailable: {exc}") from exc
    return Backend(np, "numpy", None, {"numpy": np.__version__})
