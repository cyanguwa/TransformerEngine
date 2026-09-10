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

Scope: forward only (the F16 backward builder is Stage 3, at which point this
becomes a ``torch.autograd.Function`` / ``torch.library`` op). Dispatch wiring
into ``FusedAttnFunc`` is intentionally left out here so it can be placed
alongside the ``FusedAttentionParams`` work without churn -- callers reach this
path explicitly via :func:`fused_attn_fwd_f16`.

Everything cuDNN/torch-specific is imported lazily so this module can be
imported (and its gate queried) without a GPU or the cuDNN Python package.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, Optional, Tuple

from transformer_engine.common.fused_attn_py import GraphCache
from transformer_engine.common.fused_attn_py.builders.f16 import build_f16_fwd_graph
from transformer_engine.common.fused_attn_py.config import FusedAttnConfig, Pass
from transformer_engine.common.fused_attn_py.strides import qkvo_dims_strides

# Process-wide forward graph cache, shared with the support probe so a graph the
# probe built is reused here rather than rebuilt.
FWD_GRAPH_CACHE = GraphCache()

_CUDNN_HANDLES: Dict[Any, Any] = {}


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
    return FWD_GRAPH_CACHE.get_or_build(
        key, lambda: build_f16_fwd_graph(cudnn, handle, cfg)
    )


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
) -> Tuple[Any, Any]:
    """Run the F16/BF16 forward SDPA through the Python cuDNN graph path.

    ``cfg`` must be derived. ``q``/``k``/``v`` are the input tensors laid out per
    ``cfg.qkv_layout`` (their strides therefore match ``qkvo_dims_strides``).
    Returns ``(o, stats)`` where ``stats`` is the softmax LSE ``(B, H, S_q, 1)``.
    Optional tensors are required exactly when the matching ``cfg`` flag is set
    (``is_bias`` / ``is_padding`` / ``is_dropout``).
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

    workspace = torch.empty(entry.workspace_size, dtype=torch.uint8, device=q.device)
    entry.graph.execute(variant_pack, workspace, handle=handle)
    return o, stats


def _require(value: Optional[Any], name: str) -> None:
    if value is None:
        raise ValueError(f"fused_attn_fwd_f16: cfg requires '{name}' but it was not provided.")


__all__ = ["fused_attn_py_enabled", "fused_attn_fwd_f16", "FWD_GRAPH_CACHE"]
