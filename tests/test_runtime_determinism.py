from __future__ import annotations

from types import SimpleNamespace

import pytest

from duo_vla.runtime_determinism import configure_strict_cuda_determinism, deterministic_torch_runtime


def _fake_torch() -> tuple[SimpleNamespace, dict[str, object]]:
    state: dict[str, object] = {
        "deterministic": False,
        "preferred_blas": "_BlasBackend.Cublaslt",
        "preferred_linalg": "_LinalgBackend.Cusolver",
        "precision": "high",
        "warn_only": True,
    }
    cudnn = SimpleNamespace(allow_tf32=True, benchmark=True, deterministic=False)
    matmul = SimpleNamespace(allow_tf32=True)

    def preferred_blas_library(backend: str | None = None) -> str:
        if backend is not None:
            state["preferred_blas"] = {
                "cublas": "_BlasBackend.Cublas",
                "cublaslt": "_BlasBackend.Cublaslt",
            }[backend]
        return str(state["preferred_blas"])

    def preferred_linalg_library(backend: str | None = None) -> str:
        if backend is not None:
            state["preferred_linalg"] = {
                "default": "_LinalgBackend.Default",
                "cusolver": "_LinalgBackend.Cusolver",
            }[backend]
        return str(state["preferred_linalg"])

    cuda = SimpleNamespace(
        matmul=matmul,
        nccl=SimpleNamespace(version=lambda: (2, 29, 3)),
        preferred_blas_library=preferred_blas_library,
        preferred_linalg_library=preferred_linalg_library,
    )

    def use_deterministic_algorithms(enabled: bool, *, warn_only: bool) -> None:
        state["deterministic"] = enabled
        state["warn_only"] = warn_only

    fake = SimpleNamespace(
        are_deterministic_algorithms_enabled=lambda: state["deterministic"],
        backends=SimpleNamespace(cuda=cuda, cudnn=cudnn),
        cuda=cuda,
        get_float32_matmul_precision=lambda: state["precision"],
        is_deterministic_algorithms_warn_only_enabled=lambda: state["warn_only"],
        set_float32_matmul_precision=lambda value: state.__setitem__("precision", value),
        use_deterministic_algorithms=use_deterministic_algorithms,
    )
    return fake, state


def test_strict_cuda_determinism_configures_and_reports_every_numeric_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, _ = _fake_torch()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("PYTHONHASHSEED", "2")

    configure_strict_cuda_determinism(fake)

    assert deterministic_torch_runtime(fake) == {
        "cublas_workspace_config": ":4096:8",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_tf32": False,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "float32_matmul_precision": "highest",
        "matmul_tf32": False,
        "nccl": [2, 29, 3],
        "preferred_blas_library": "_BlasBackend.Cublas",
        "preferred_linalg_library": "_LinalgBackend.Default",
        "python_hash_seed": "2",
    }


def test_strict_cuda_determinism_fails_before_configuration_without_cublas_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, state = _fake_torch()
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        configure_strict_cuda_determinism(fake)

    assert state["deterministic"] is False
