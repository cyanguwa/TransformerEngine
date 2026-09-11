# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Additive JAX bridge for the Python fused-attention blob path (Stage 5).

This wires the framework-neutral ``fused_attn_py`` core to JAX's FFI *without*
disturbing the production fused-attention primitive. It is opt-in (off by
default, gated by ``NVTE_FUSED_ATTN_PY``) and reversible: nothing here is
registered as the default lowering, and no CUDA pre-kernel or workspace
workaround is removed yet -- those are separate, later Stage 5 steps.

Design (the ``score_mod`` pattern already used by ``flex_attention.py``): at
lowering time the neutral core builds the cuDNN graph and serializes it to a
blob (:func:`transformer_engine.common.fused_attn_py.build_plan`); that blob and
its UID lists ship as FFI attributes to the existing C++
``FusedAttnScoreMod{Forward,Backward}Handler``, which deserializes once per
``(blob, device)`` and executes it against a ``{uid: ptr}`` variant pack. The
builder->executor contract is therefore pure data -- the same :class:`Plan`
both frameworks consume.

Everything JAX/cuDNN-specific is imported lazily so this module (and its gate)
can be imported without JAX or the cuDNN Python package.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, Tuple

import numpy as np

from transformer_engine.common.fused_attn_py import Plan, build_plan
from transformer_engine.common.fused_attn_py.config import FusedAttnConfig, Pass, RuntimeInfo
from transformer_engine.common.fused_attn_py.serialize import ordered_input_operands

# The C++ FFI handler symbols registered in jax/csrc/extensions/pybind.cpp.
_FWD_FFI = "te_fused_attn_score_mod_forward_ffi"
_BWD_FFI = "te_fused_attn_score_mod_backward_ffi"


def fused_attn_py_enabled() -> bool:
    """True when the Python fused-attention blob path is opted in via env var."""
    return os.getenv("NVTE_FUSED_ATTN_PY", "0") not in ("0", "", "false", "False")


def _import_cudnn():
    try:
        return importlib.import_module("cudnn")
    except ImportError as exc:  # pragma: no cover - requires the package
        raise ImportError(
            "The Python fused-attention blob path needs the cuDNN frontend package. "
            "Install it with: pip install nvidia-cudnn-frontend"
        ) from exc


def _parse_sm_arch(compute_capability: str) -> int:
    """Turn a device compute capability like ``"9.0"`` into ``sm_arch`` 90."""
    major, _, minor = str(compute_capability).partition(".")
    return int(major) * 10 + (int(minor) if minor else 0)


def make_runtime_info(cudnn: Any, *, device: Any = None) -> RuntimeInfo:
    """Collect the device/library facts ``FusedAttnConfig.derive()`` needs on JAX.

    ``sm_arch`` comes from the target device's compute capability, the cuDNN
    backend version from ``cudnn.backend_version()``, and the frontend versions
    from the Python package (``cudnn.__version__``) and the C++ build
    (``transformer_engine_jax.get_cudnn_frontend_version()``). Mirrors what the
    C++ ``derive()`` reads from ``cudnnGetVersion`` / ``cuda::sm_arch``.
    """
    import jax

    import transformer_engine_jax

    from transformer_engine.common.fused_attn_py.serialize import encode_cudnn_frontend_version

    if device is None:
        device = jax.local_devices()[0]
    sm_arch = _parse_sm_arch(device.compute_capability)
    cudnn_version = int(cudnn.backend_version())
    fe_python = encode_cudnn_frontend_version(getattr(cudnn, "__version__"))
    fe_cpp = int(transformer_engine_jax.get_cudnn_frontend_version())
    return RuntimeInfo(
        sm_arch=sm_arch,
        cudnn_version=cudnn_version,
        cudnn_frontend_version=fe_python,
        cudnn_build_version=fe_cpp,
    )


def config_from_jax_params(params: Any, runtime: RuntimeInfo, **overrides) -> FusedAttnConfig:
    """Adapt a JAX ``FusedAttnParams`` into a derived neutral ``FusedAttnConfig``.

    ``FusedAttnParams`` (jax/cpp_extensions/attention.py) already mirrors the C++
    ``FusedAttnConfig`` field order with ``NVTE_*`` enums, so the neutral config
    copies the same-named fields and normalizes the enums by name. ``overrides``
    forwards fields the JAX param object does not carry (e.g. paged-KV dims).
    """
    return FusedAttnConfig.from_params(params, **overrides).derive(runtime)


def plan_ffi_attrs(plan: Plan) -> Dict[str, Any]:
    """Convert a :class:`Plan` into the FFI attribute dict the C++ handler reads.

    Mirrors the attribute set passed by ``flex_attention.py``: the serialized
    blob, its two hash halves, the frontend version, and the input/output/scalar
    UID arrays plus the packed scalar bytes.
    """
    return {
        "serialized_graph": plan.serialized_graph,
        "graph_hash0": plan.graph_hash[0],
        "graph_hash1": plan.graph_hash[1],
        "cudnn_frontend_version": plan.cudnn_frontend_version,
        "input_uids": np.asarray(plan.input_uids, dtype=np.int64),
        "output_uids": np.asarray(plan.output_uids, dtype=np.int64),
        "scalar_uids": np.asarray(plan.scalar_uids, dtype=np.int64),
        "scalar_sizes": np.asarray(plan.scalar_sizes, dtype=np.int64),
        "scalar_values": np.frombuffer(plan.scalar_values, dtype=np.uint8),
    }


def fused_attn_blob_fwd(
    cfg: FusedAttnConfig,
    buffers: Dict[str, Any],
    out_shape: Any,
    stats_shape: Any,
    *,
    attn_scale: float,
    cudnn_version: int | None = None,
    cudnn_frontend_version: int | None = None,
) -> Tuple[Any, Any]:
    """Lower an F16/BF16 forward through the blob FFI. Returns ``(output, stats)``.

    ``cfg`` must be derived; ``buffers`` maps input role names (``"Q"``/``"K"``/
    ``"V"`` plus any bias/seq/offset/page-table/dropout roles the graph declares)
    to JAX arrays. ``out_shape`` / ``stats_shape`` are ``jax.ShapeDtypeStruct``s
    for the outputs. Building and serializing the graph happens here (once per
    trace); the blob is a compile-time constant.
    """
    import jax
    import jax.numpy as jnp
    from jax import ffi

    cfg.check_derived()
    cudnn = _import_cudnn()
    plan = build_plan(
        cudnn,
        cfg,
        Pass.Fwd,
        attn_scale=attn_scale,
        cudnn_version=cudnn_version,
        cudnn_frontend_version=cudnn_frontend_version,
    )
    operands = ordered_input_operands(plan, buffers)
    workspace = jax.ShapeDtypeStruct((plan.workspace_size,), jnp.uint8)
    output, stats, _ = ffi.ffi_call(_FWD_FFI, (out_shape, stats_shape, workspace))(
        *operands, **plan_ffi_attrs(plan)
    )
    return output, stats


def fused_attn_blob_bwd(
    cfg: FusedAttnConfig,
    buffers: Dict[str, Any],
    grad_shapes: Tuple[Any, Any, Any],
    *,
    attn_scale: float,
    cudnn_version: int | None = None,
    cudnn_frontend_version: int | None = None,
) -> Tuple[Any, Any, Any]:
    """Lower an F16/BF16 backward through the blob FFI. Returns ``(dQ, dK, dV)``.

    ``buffers`` must include the backward inputs (``"Q"``/``"K"``/``"V"``/``"O"``/
    ``"dO"``/``"stats"`` plus any bias/seq/offset/dropout roles). ``grad_shapes``
    are the ``jax.ShapeDtypeStruct``s for ``(dQ, dK, dV)``.
    """
    import jax
    import jax.numpy as jnp
    from jax import ffi

    cfg.check_derived()
    cudnn = _import_cudnn()
    plan = build_plan(
        cudnn,
        cfg,
        Pass.Bwd,
        attn_scale=attn_scale,
        cudnn_version=cudnn_version,
        cudnn_frontend_version=cudnn_frontend_version,
    )
    operands = ordered_input_operands(plan, buffers)
    workspace = jax.ShapeDtypeStruct((plan.workspace_size,), jnp.uint8)
    dq, dk, dv, _ = ffi.ffi_call(_BWD_FFI, (*grad_shapes, workspace))(
        *operands, **plan_ffi_attrs(plan)
    )
    return dq, dk, dv


__all__ = [
    "fused_attn_py_enabled",
    "make_runtime_info",
    "config_from_jax_params",
    "plan_ffi_attrs",
    "ordered_input_operands",
    "fused_attn_blob_fwd",
    "fused_attn_blob_bwd",
]
