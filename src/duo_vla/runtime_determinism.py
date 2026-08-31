"""Strict PyTorch determinism helpers shared by training and serving."""

from __future__ import annotations

import os
from typing import Any

CUBLAS_WORKSPACE_CONFIG = ":4096:8"
PREFERRED_BLAS_LIBRARY = "_BlasBackend.Cublas"
PREFERRED_LINALG_LIBRARY = "_LinalgBackend.Default"


def configure_strict_cuda_determinism(torch: Any) -> None:
    """Enable and verify the deterministic CUDA controls used by Duo-VLA."""

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(f"CUBLAS_WORKSPACE_CONFIG must be {CUBLAS_WORKSPACE_CONFIG!r} before importing CUDA")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.preferred_blas_library("cublas")
    torch.backends.cuda.preferred_linalg_library("default")
    torch.set_float32_matmul_precision("highest")

    checks = {
        "deterministic algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic algorithms warn-only disabled": not torch.is_deterministic_algorithms_warn_only_enabled(),
        "cuDNN benchmarking disabled": not torch.backends.cudnn.benchmark,
        "cuDNN deterministic mode": torch.backends.cudnn.deterministic,
        "cuDNN TF32 disabled": not torch.backends.cudnn.allow_tf32,
        "CUDA matmul TF32 disabled": not torch.backends.cuda.matmul.allow_tf32,
        "float32 matmul precision highest": torch.get_float32_matmul_precision() == "highest",
        "preferred BLAS library cuBLAS": (str(torch.backends.cuda.preferred_blas_library()) == PREFERRED_BLAS_LIBRARY),
        "preferred linalg library default": (
            str(torch.backends.cuda.preferred_linalg_library()) == PREFERRED_LINALG_LIBRARY
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(f"strict CUDA determinism could not be enabled: {failures}")


def deterministic_torch_runtime(torch: Any) -> dict[str, Any]:
    """Return the complete, hashable determinism state after configuration."""

    nccl_version = torch.cuda.nccl.version()
    return {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "nccl": list(nccl_version) if nccl_version is not None else None,
        "preferred_blas_library": str(torch.backends.cuda.preferred_blas_library()),
        "preferred_linalg_library": str(torch.backends.cuda.preferred_linalg_library()),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    }
