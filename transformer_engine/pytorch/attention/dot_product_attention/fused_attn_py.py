# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch glue for the framework-neutral ``fused_attn_py`` core.

The decision core, graph builders, cache, and probe live in
``transformer_engine/common/fused_attn_py`` and know nothing about PyTorch. This
module supplies the framework-specific runtime the core needs:

* the ``NVTE_FUSED_ATTN_PY`` opt-in gate,
* a per-device cuDNN handle bound to PyTorch's current stream (the same pattern
  as ``flex_attention.py``),
* a process-wide graph cache keyed by ``FusedAttnConfig.make_cache_key``,
* the F16/BF16 forward execute path: allocate outputs, build/lookup the graph,
  bind the variant pack from the builder's role->tensor map, and run
  ``graph.execute``.

Scope: F16/BF16 forward + backward (exposed as a ``torch.autograd.Function`` via
:func:`fused_attn_py_f16`), including THD ragged offsets and paged-KV forward, plus
the FP8/MXFP8 forward + backward execute functions (:func:`fused_attn_fwd_fp8` /
:func:`fused_attn_bwd_fp8`), which bind the caller's quantized inputs / scale
tensors and allocate outputs. Dispatch wiring into ``FusedAttnFunc`` (and the FP8
quantizer plumbing) is intentionally left out here so it can be placed alongside
the ``FusedAttentionParams`` work without churn -- callers reach this path
explicitly via the ``fused_attn_fwd_*`` / ``fused_attn_bwd_*`` functions.

Everything cuDNN/torch-specific is imported lazily so this module can be
imported (and its gate queried) without a GPU or the cuDNN Python package.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, Optional, Tuple

from transformer_engine.common.fused_attn_py import GraphCache
from transformer_engine.common.fused_attn_py.builders.f16 import (
    build_f16_bwd_graph,
    build_f16_fwd_graph,
)
from transformer_engine.common.fused_attn_py.builders.fp8 import (
    build_fp8_bwd_graph,
    build_fp8_fwd_graph,
)
from transformer_engine.common.fused_attn_py.config import (
    FusedAttnConfig,
    Pass,
    RuntimeInfo,
    _canonical_dtype,
)
from transformer_engine.common.fused_attn_py.serialize import encode_cudnn_frontend_version
from transformer_engine.common.fused_attn_py.strides import qkvo_dims_strides

# Process-wide graph caches, shared with the support probe so a graph the probe
# built is reused here rather than rebuilt.
FWD_GRAPH_CACHE = GraphCache()
BWD_GRAPH_CACHE = GraphCache()
FP8_FWD_GRAPH_CACHE = GraphCache()
FP8_BWD_GRAPH_CACHE = GraphCache()

_CUDNN_HANDLES: Dict[Any, Any] = {}
_F16_AUTOGRAD_FN = None


def fused_attn_py_enabled() -> bool:
    """True when the Python fused-attention path is opted in via env var."""
    return os.getenv("NVTE_FUSED_ATTN_PY", "0") not in ("0", "", "false", "False")


def _import_cudnn():
    try:
        return importlib.import_module("cudnn")
    except ImportError as exc:  # pragma: no cover - requires the package
        raise ImportError(
            "The Python fused-attention path needs the cuDNN frontend package. "
            "Install it with: pip install nvidia-cudnn-frontend"
        ) from exc


def _encode_cudnn_backend_version(version: Tuple[int, int, int]) -> int:
    """Encode a ``(major, minor, patch)`` cuDNN backend version as ``M*10000+m*100+p``."""
    major, minor, patch = version
    return int(major) * 10000 + int(minor) * 100 + int(patch)


def make_runtime_info(*, device=None) -> RuntimeInfo:
    """Collect the device/library facts ``FusedAttnConfig.derive()`` needs on PyTorch.

    ``sm_arch`` comes from the target CUDA device's compute capability, the cuDNN
    *backend* version from ``transformer_engine_torch.get_cudnn_version()`` (used
    both as the runtime and the build version -- PyTorch links a single cuDNN, so
    they match), and the *frontend* version from the Python ``cudnn`` package
    (``cudnn.__version__``). Mirrors the JAX ``make_runtime_info`` and what the C++
    ``derive()`` reads from ``cudnnGetVersion`` / ``cuda::sm_arch``.
    """
    import torch

    from transformer_engine.pytorch.utils import get_cudnn_version

    cudnn = _import_cudnn()
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(device)
    backend = _encode_cudnn_backend_version(get_cudnn_version())
    fe_python = encode_cudnn_frontend_version(getattr(cudnn, "__version__"))
    return RuntimeInfo(
        sm_arch=int(major) * 10 + int(minor),
        cudnn_version=backend,
        cudnn_frontend_version=fe_python,
        cudnn_build_version=backend,
        device_id=device.index or 0,
    )


def config_from_fused_attn_params(
    params: Any, runtime: RuntimeInfo, **overrides
) -> FusedAttnConfig:
    """Adapt a PyTorch ``FusedAttentionParams`` into a derived neutral ``FusedAttnConfig``.

    ``FusedAttentionParams`` (dot_product_attention/utils.py) mirrors the C++
    ``FusedAttnConfig`` field names with ``tex.NVTE_*`` enums, so the neutral config
    copies the same-named fields and normalizes the enums/dtypes by name via
    :meth:`FusedAttnConfig.from_params`. ``overrides`` forwards any field the params
    object does not carry (e.g. ``check_for_forward_support``). This is the seam a
    future ``FusedAttnFunc`` dispatch uses to build the config from call-site args.
    """
    return FusedAttnConfig.from_params(params, **overrides).derive(runtime)


def _get_cudnn_handle(device) -> Any:
    """Return a per-device cuDNN handle bound to PyTorch's current stream."""
    import torch

    cudnn = _import_cudnn()
    if device.type != "cuda":
        raise ValueError(f"fused_attn_py only supports CUDA tensors, got device {device}.")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    handle = _CUDNN_HANDLES.get(device)
    with torch.cuda.device(device):
        if handle is None:
            handle = cudnn.create_handle()
            _CUDNN_HANDLES[device] = handle
        cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream(device).cuda_stream)
    return handle


def _get_fwd_graph(cudnn, handle, cfg: FusedAttnConfig):
    """Build or fetch the cached F16 forward graph for this config."""
    cfg.check_derived()
    key = cfg.make_cache_key(Pass.Fwd)
    return FWD_GRAPH_CACHE.get_or_build(key, lambda: build_f16_fwd_graph(cudnn, handle, cfg))


def _get_bwd_graph(cudnn, handle, cfg: FusedAttnConfig):
    """Build or fetch the cached F16 backward graph for this config."""
    cfg.check_derived()
    key = cfg.make_cache_key(Pass.Bwd)
    return BWD_GRAPH_CACHE.get_or_build(key, lambda: build_f16_bwd_graph(cudnn, handle, cfg))


def fused_attn_fwd_f16(
    cfg: FusedAttnConfig,
    q,
    k,
    v,
    *,
    attn_scale: float,
    bias=None,
    seq_len_q=None,
    seq_len_kv=None,
    dropout_seed=None,
    dropout_offset=None,
    ragged_offsets: Optional[Dict[str, Any]] = None,
    page_tables: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any]:
    """Run the F16/BF16 forward SDPA through the Python cuDNN graph path.

    ``cfg`` must be derived. ``q``/``k``/``v`` are the input tensors laid out per
    ``cfg.qkv_layout`` (their strides therefore match ``qkvo_dims_strides``).
    Returns ``(o, stats)`` where ``stats`` is the softmax LSE ``(B, H, S_q, 1)``.
    Optional tensors are required exactly when the matching ``cfg`` flag is set
    (``is_bias`` / ``is_padding`` / ``is_dropout``). ``ragged_offsets`` (THD) maps
    ``"offset_q"``/``"offset_k"``/``"offset_v"``/``"offset_o"``/``"offset_stats"``
    to their int offset tensors; ``page_tables`` (paged KV) maps
    ``"page_table_k"``/``"page_table_v"``.
    """
    import torch

    cudnn = _import_cudnn()
    handle = _get_cudnn_handle(q.device)
    entry = _get_fwd_graph(cudnn, handle, cfg)

    ds = qkvo_dims_strides(cfg)
    o_dim, o_stride = ds["O"]
    b, h, s_q = int(cfg.graph_batch_size_fwd), int(cfg.num_attn_heads), int(cfg.graph_max_seqlen_q)
    o = torch.empty_strided(o_dim, o_stride, dtype=q.dtype, device=q.device)
    stats = torch.empty((b, h, s_q, 1), dtype=torch.float32, device=q.device)

    t = entry.tensors
    # attn_scale is a pass-by-value scalar tensor in the graph; feed the value.
    scale = torch.full((1, 1, 1, 1), float(attn_scale), dtype=torch.float32, device=q.device)
    variant_pack: Dict[Any, Any] = {
        t["Q"]: q,
        t["K"]: k,
        t["V"]: v,
        t["attn_scale"]: scale,
        t["O"]: o,
        t["Stats"]: stats,
    }
    if cfg.is_bias:
        _require(bias, "bias")
        variant_pack[t["bias"]] = bias
    if cfg.is_padding:
        _require(seq_len_q, "seq_len_q")
        _require(seq_len_kv, "seq_len_kv")
        variant_pack[t["seq_q"]] = seq_len_q
        variant_pack[t["seq_kv"]] = seq_len_kv
    if cfg.is_dropout:
        _require(dropout_seed, "dropout_seed")
        _require(dropout_offset, "dropout_offset")
        variant_pack[t["dropout_seed"]] = dropout_seed
        variant_pack[t["dropout_offset"]] = dropout_offset
    _bind_extra(variant_pack, t, ragged_offsets, page_tables)

    workspace = torch.empty(entry.workspace_size, dtype=torch.uint8, device=q.device)
    entry.graph.execute(variant_pack, workspace, handle=handle)
    return o, stats


def fused_attn_bwd_f16(
    cfg: FusedAttnConfig,
    q,
    k,
    v,
    o,
    d_o,
    stats,
    *,
    attn_scale: float,
    bias=None,
    seq_len_q=None,
    seq_len_kv=None,
    dropout_seed=None,
    dropout_offset=None,
    ragged_offsets: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any, Any, Optional[Any]]:
    """Run the F16/BF16 backward SDPA. Returns ``(dQ, dK, dV, dBias)``.

    ``dBias`` is ``None`` unless the graph emits it (bias not fully broadcast).
    Gradient tensors are allocated ``empty_like`` their inputs, so they carry the
    same layout the graph's dQ/dK/dV expect. ``ragged_offsets`` (THD) supplies the
    int offset tensors keyed as in :func:`fused_attn_fwd_f16` (paged KV never runs
    backward, so there are no page tables here).
    """
    import torch

    cudnn = _import_cudnn()
    handle = _get_cudnn_handle(q.device)
    entry = _get_bwd_graph(cudnn, handle, cfg)

    d_q = torch.empty_like(q)
    d_k = torch.empty_like(k)
    d_v = torch.empty_like(v)
    scale = torch.full((1, 1, 1, 1), float(attn_scale), dtype=torch.float32, device=q.device)
    t = entry.tensors
    variant_pack: Dict[Any, Any] = {
        t["Q"]: q,
        t["K"]: k,
        t["V"]: v,
        t["O"]: o,
        t["dO"]: d_o,
        t["stats"]: stats,
        t["attn_scale"]: scale,
        t["dQ"]: d_q,
        t["dK"]: d_k,
        t["dV"]: d_v,
    }
    d_bias = None
    if cfg.is_bias:
        _require(bias, "bias")
        variant_pack[t["bias"]] = bias
        if "dBias" in t:
            d_bias = torch.empty_like(bias)
            variant_pack[t["dBias"]] = d_bias
    if cfg.is_padding:
        _require(seq_len_q, "seq_len_q")
        _require(seq_len_kv, "seq_len_kv")
        variant_pack[t["seq_q"]] = seq_len_q
        variant_pack[t["seq_kv"]] = seq_len_kv
    if cfg.is_dropout:
        _require(dropout_seed, "dropout_seed")
        _require(dropout_offset, "dropout_offset")
        variant_pack[t["dropout_seed"]] = dropout_seed
        variant_pack[t["dropout_offset"]] = dropout_offset
    _bind_extra(variant_pack, t, ragged_offsets, None)

    workspace = torch.empty(entry.workspace_size, dtype=torch.uint8, device=q.device)
    entry.graph.execute(variant_pack, workspace, handle=handle)
    return d_q, d_k, d_v, d_bias


def _f16_autograd_fn():
    """Lazily build and cache the F16 autograd.Function (needs torch at call time)."""
    global _F16_AUTOGRAD_FN
    if _F16_AUTOGRAD_FN is not None:
        return _F16_AUTOGRAD_FN

    import torch

    class _FusedAttnPyF16Func(torch.autograd.Function):
        """cuDNN Python-graph F16/BF16 fused attention (forward + backward)."""

        @staticmethod
        def forward(
            ctx, cfg, attn_scale, q, k, v, bias, seq_len_q, seq_len_kv, dropout_seed, dropout_offset
        ):
            # pylint: disable=missing-function-docstring
            o, stats = fused_attn_fwd_f16(
                cfg,
                q,
                k,
                v,
                attn_scale=attn_scale,
                bias=bias,
                seq_len_q=seq_len_q,
                seq_len_kv=seq_len_kv,
                dropout_seed=dropout_seed,
                dropout_offset=dropout_offset,
            )
            ctx.cfg = cfg
            ctx.attn_scale = attn_scale
            ctx.save_for_backward(
                q, k, v, o, stats, bias, seq_len_q, seq_len_kv, dropout_seed, dropout_offset
            )
            return o

        @staticmethod
        def backward(ctx, d_o):
            # pylint: disable=missing-function-docstring
            (q, k, v, o, stats, bias, seq_len_q, seq_len_kv, seed, offset) = ctx.saved_tensors
            d_q, d_k, d_v, d_bias = fused_attn_bwd_f16(
                ctx.cfg,
                q,
                k,
                v,
                o,
                d_o.contiguous(),
                stats,
                attn_scale=ctx.attn_scale,
                bias=bias,
                seq_len_q=seq_len_q,
                seq_len_kv=seq_len_kv,
                dropout_seed=seed,
                dropout_offset=offset,
            )
            # Grad order matches forward's inputs:
            # (cfg, attn_scale, q, k, v, bias, seq_len_q, seq_len_kv, seed, offset).
            return None, None, d_q, d_k, d_v, d_bias, None, None, None, None

    _F16_AUTOGRAD_FN = _FusedAttnPyF16Func
    return _F16_AUTOGRAD_FN


def fused_attn_py_f16(
    cfg: FusedAttnConfig,
    q,
    k,
    v,
    *,
    attn_scale: float,
    bias=None,
    seq_len_q=None,
    seq_len_kv=None,
    dropout_seed=None,
    dropout_offset=None,
):
    """Autograd-aware F16/BF16 fused attention through the Python cuDNN path.

    Returns the attention output ``o``; gradients flow to ``q``, ``k``, ``v`` (and
    ``bias`` when ``cfg.is_bias`` and a ``dBias`` is emitted).
    """
    return _f16_autograd_fn().apply(
        cfg, attn_scale, q, k, v, bias, seq_len_q, seq_len_kv, dropout_seed, dropout_offset
    )


def _torch_dtype(dtype):
    """Map a neutral canonical dtype (or NVTE enum/name) to a torch dtype."""
    import torch

    name = _canonical_dtype(dtype)
    mapping = {
        "FLOAT16": torch.float16,
        "BFLOAT16": torch.bfloat16,
        "FLOAT32": torch.float32,
        "FLOAT8E4M3": torch.float8_e4m3fn,
        "FLOAT8E5M2": torch.float8_e5m2,
    }
    if name not in mapping:
        raise ValueError(f"fused_attn_py: no torch dtype for canonical dtype {name!r}.")
    return mapping[name]


def _get_fp8_fwd_graph(cudnn, handle, cfg: FusedAttnConfig):
    """Build or fetch the cached FP8/MXFP8 forward graph for this config."""
    cfg.check_derived()
    key = cfg.make_cache_key(Pass.Fwd)
    return FP8_FWD_GRAPH_CACHE.get_or_build(key, lambda: build_fp8_fwd_graph(cudnn, handle, cfg))


def _get_fp8_bwd_graph(cudnn, handle, cfg: FusedAttnConfig):
    """Build or fetch the cached FP8/MXFP8 backward graph for this config."""
    cfg.check_derived()
    key = cfg.make_cache_key(Pass.Bwd)
    return FP8_BWD_GRAPH_CACHE.get_or_build(key, lambda: build_fp8_bwd_graph(cudnn, handle, cfg))


def _bind_fp8_inputs(variant_pack, t, fp8_tensors) -> None:
    """Bind the caller's FP8 device inputs (descales/scales + MXFP8 helpers) by role.

    ``fp8_tensors`` maps builder role names (``"DescaleQ"``, ``"ScaleS"``,
    ``"ScaleO"``, ..., and for MXFP8 ``"Qt"``/``"Kt"``/``"dOf16"``/``"dOt"`` and
    their block descales) to device tensors. Only roles the graph actually declares
    are bound (current-scaling ``Scale_*`` are baked-in 1.0 constants, so those
    roles are absent from ``t`` and correctly skipped).
    """
    if not fp8_tensors:
        return
    for role, tensor in fp8_tensors.items():
        if role in t:
            _require(tensor, role)
            variant_pack[t[role]] = tensor


def _alloc_amaxes(variant_pack, t, roles, device):
    """Allocate + bind the amax scalar outputs the graph declares. Returns a dict."""
    import torch

    amaxes: Dict[str, Any] = {}
    for role in roles:
        if role in t:
            a = torch.empty((1, 1, 1, 1), dtype=torch.float32, device=device)
            variant_pack[t[role]] = a
            amaxes[role] = a
    return amaxes


def fused_attn_fwd_fp8(
    cfg: FusedAttnConfig,
    q,
    k,
    v,
    *,
    attn_scale: float,
    fp8_tensors: Dict[str, Any],
    seq_len_q=None,
    seq_len_kv=None,
    dropout_seed=None,
    dropout_offset=None,
    ragged_offsets: Optional[Dict[str, Any]] = None,
    page_tables: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Run the FP8/MXFP8 forward SDPA through the Python cuDNN graph path.

    ``q``/``k``/``v`` are the FP8 inputs. ``fp8_tensors`` supplies every FP8 device
    input the graph declares (tensor scaling: ``Descale_q/k/v/s``, ``Scale_s`` and,
    for delayed scaling, ``Scale_o``; MXFP8: the ``FP8_E8M0`` block descales). ``O``
    is allocated with ``cfg.o_dtype``, ``Stats`` is float32; the amax outputs are
    allocated only when the graph surfaces them (tensor scaling only). Returns
    ``(o, stats, amaxes)`` where ``amaxes`` is a role-keyed dict (empty for MXFP8).
    """
    import torch

    cudnn = _import_cudnn()
    handle = _get_cudnn_handle(q.device)
    entry = _get_fp8_fwd_graph(cudnn, handle, cfg)

    ds = qkvo_dims_strides(cfg)
    o_dim, o_stride = ds["O"]
    b, h, s_q = int(cfg.graph_batch_size_fwd), int(cfg.num_attn_heads), int(cfg.graph_max_seqlen_q)
    o = torch.empty_strided(o_dim, o_stride, dtype=_torch_dtype(cfg.o_dtype), device=q.device)
    stats = torch.empty((b, h, s_q, 1), dtype=torch.float32, device=q.device)

    t = entry.tensors
    scale = torch.full((1, 1, 1, 1), float(attn_scale), dtype=torch.float32, device=q.device)
    variant_pack: Dict[Any, Any] = {
        t["Q"]: q,
        t["K"]: k,
        t["V"]: v,
        t["attn_scale"]: scale,
        t["O"]: o,
        t["Stats"]: stats,
    }
    _bind_fp8_inputs(variant_pack, t, fp8_tensors)
    amaxes = _alloc_amaxes(variant_pack, t, ("AmaxS", "AmaxO"), q.device)
    if cfg.is_padding:
        _require(seq_len_q, "seq_len_q")
        _require(seq_len_kv, "seq_len_kv")
        variant_pack[t["seq_q"]] = seq_len_q
        variant_pack[t["seq_kv"]] = seq_len_kv
    if cfg.is_dropout:
        _require(dropout_seed, "dropout_seed")
        _require(dropout_offset, "dropout_offset")
        variant_pack[t["dropout_seed"]] = dropout_seed
        variant_pack[t["dropout_offset"]] = dropout_offset
    _bind_extra(variant_pack, t, ragged_offsets, page_tables)

    workspace = torch.empty(entry.workspace_size, dtype=torch.uint8, device=q.device)
    entry.graph.execute(variant_pack, workspace, handle=handle)
    return o, stats, amaxes


def fused_attn_bwd_fp8(
    cfg: FusedAttnConfig,
    q,
    k,
    v,
    o,
    d_o,
    stats,
    *,
    attn_scale: float,
    fp8_tensors: Dict[str, Any],
    seq_len_q=None,
    seq_len_kv=None,
    dropout_seed=None,
    dropout_offset=None,
    ragged_offsets: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """Run the FP8/MXFP8 backward SDPA. Returns ``(dQ, dK, dV, amaxes)``.

    ``fp8_tensors`` supplies every FP8 device input the graph declares -- the
    descales/scales (tensor scaling) or block descales plus the transpose/f16
    helper tensors ``Qt``/``Kt``/``dOf16``/``dOt`` (MXFP8). Grads are allocated with
    ``cfg.dqkv_dtype``; the grad amaxes are allocated only when the graph surfaces
    them (tensor scaling only) and returned in the ``amaxes`` dict.
    """
    import torch

    cudnn = _import_cudnn()
    handle = _get_cudnn_handle(q.device)
    entry = _get_fp8_bwd_graph(cudnn, handle, cfg)

    ds = qkvo_dims_strides(cfg, batch_size=int(cfg.graph_batch_size_bwd))
    dqkv_dtype = _torch_dtype(cfg.dqkv_dtype)
    d_q = torch.empty_strided(ds["Q"][0], ds["Q"][1], dtype=dqkv_dtype, device=q.device)
    d_k = torch.empty_strided(ds["K"][0], ds["K"][1], dtype=dqkv_dtype, device=q.device)
    d_v = torch.empty_strided(ds["V"][0], ds["V"][1], dtype=dqkv_dtype, device=q.device)

    t = entry.tensors
    scale = torch.full((1, 1, 1, 1), float(attn_scale), dtype=torch.float32, device=q.device)
    variant_pack: Dict[Any, Any] = {
        t["Q"]: q,
        t["K"]: k,
        t["V"]: v,
        t["O"]: o,
        t["dO"]: d_o,
        t["stats"]: stats,
        t["attn_scale"]: scale,
        t["dQ"]: d_q,
        t["dK"]: d_k,
        t["dV"]: d_v,
    }
    _bind_fp8_inputs(variant_pack, t, fp8_tensors)
    amaxes = _alloc_amaxes(variant_pack, t, ("AmaxdQ", "AmaxdK", "AmaxdV", "AmaxdP"), q.device)
    if cfg.is_padding:
        _require(seq_len_q, "seq_len_q")
        _require(seq_len_kv, "seq_len_kv")
        variant_pack[t["seq_q"]] = seq_len_q
        variant_pack[t["seq_kv"]] = seq_len_kv
    if cfg.is_dropout:
        _require(dropout_seed, "dropout_seed")
        _require(dropout_offset, "dropout_offset")
        variant_pack[t["dropout_seed"]] = dropout_seed
        variant_pack[t["dropout_offset"]] = dropout_offset
    _bind_extra(variant_pack, t, ragged_offsets, None)

    workspace = torch.empty(entry.workspace_size, dtype=torch.uint8, device=q.device)
    entry.graph.execute(variant_pack, workspace, handle=handle)
    return d_q, d_k, d_v, amaxes


def _require(value: Optional[Any], name: str) -> None:
    if value is None:
        raise ValueError(f"fused_attn_fwd_f16: cfg requires '{name}' but it was not provided.")


def _bind_extra(variant_pack, t, ragged_offsets, page_tables) -> None:
    """Bind THD ragged-offset and paged-KV page-table tensors into the variant pack.

    Each role is bound only if the built graph declares it (in ``t``); a missing
    tensor for a declared role is a caller error, surfaced via :func:`_require`.
    """
    for group in (ragged_offsets, page_tables):
        if not group:
            continue
        for role, graph_tensor in t.items():
            if role in group:
                _require(group[role], role)
                variant_pack[graph_tensor] = group[role]


__all__ = [
    "fused_attn_py_enabled",
    "make_runtime_info",
    "config_from_fused_attn_params",
    "fused_attn_fwd_f16",
    "fused_attn_bwd_f16",
    "fused_attn_py_f16",
    "fused_attn_fwd_fp8",
    "fused_attn_bwd_fp8",
    "FWD_GRAPH_CACHE",
    "BWD_GRAPH_CACHE",
    "FP8_FWD_GRAPH_CACHE",
    "FP8_BWD_GRAPH_CACHE",
]
