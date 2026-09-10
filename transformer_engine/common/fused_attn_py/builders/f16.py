# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""F16/BF16 arbitrary-seqlen fused-attention graph builders (cuDNN Python API).

Port of ``create_graph_f16_fwd`` in
``common/fused_attn/fused_attn_f16_arbitrary_seqlen.cu`` onto the cuDNN Python
graph API (``graph.sdpa``). The C++ builder keys its tensors by
``shared_ptr``/UID; here we return the graph tensor objects in a role->tensor
map (:class:`~transformer_engine.common.fused_attn_py.cache.GraphEntry`), which
the framework glue binds into the variant pack -- the same pattern the in-repo
``flex_attention.py`` uses.

Scope of this stage (Stage 2): the dense path -- BSHD/SBHD/BHSD and packed
QKV layouts, with post-scale bias, ALiBi, causal / bottom-right / sliding-window
masking, materialized-``seq_len`` padding (the non-``cu_seqlens``-direct path),
dropout, and GQA. Deferred to later stages and guarded with a clear
``NotImplementedError``:

* paged KV (``cfg.is_paged_kv``)                       -- Stage 4
* THD ragged offsets / ``cu_seqlens``-direct            -- Stage 4
* sink token / learnable softmax (``cfg.is_softmax_offset``)
* ``return_max_logit`` (logit-max output)

The masking/window option planning is factored into
:func:`plan_f16_fwd_masking`, a pure function that needs no ``cudnn`` and is
unit-tested on CPU; only tensor creation and the ``graph.sdpa`` call touch the
cuDNN Python package.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..cache import GraphEntry
from ..config import FusedAttnConfig, _canonical_dtype
from ..strides import qkvo_dims_strides


def _io_data_type(cudnn, dtype) -> Any:
    """Map an F16/BF16 dtype (enum / name) to a ``cudnn.data_type``."""
    name = _canonical_dtype(dtype)
    if name == "FLOAT16":
        return cudnn.data_type.HALF
    if name == "BFLOAT16":
        return cudnn.data_type.BFLOAT16
    raise ValueError(f"F16 fused-attention builder requires FP16/BF16, got {name}.")


def plan_f16_fwd_masking(cfg: FusedAttnConfig, cudnn_version: int) -> Dict[str, Any]:
    """Pure port of the mask/window option block of ``create_graph_f16_fwd``.

    Returns the scalar SDPA options as a dict:
      * ``diagonal_alignment`` : "TOP_LEFT" | "BOTTOM_RIGHT"
      * ``diagonal_band_left_bound`` : Optional[int]
      * ``diagonal_band_right_bound`` : Optional[int]  (0 == causal)
      * ``use_alibi_mask`` : bool
    """
    alignment = "BOTTOM_RIGHT" if cfg.bottom_right_diagonal else "TOP_LEFT"
    left_bound: Optional[int] = None
    right_bound: Optional[int] = None
    if cudnn_version >= 90200 and cfg.window_size_left != -1:
        left_bound = cfg.window_size_left + 1
    if cudnn_version >= 90600 and cfg.window_size_right != -1:
        right_bound = cfg.window_size_right
    if cfg.is_causal or cfg.is_causal_bottom_right:
        right_bound = 0
    return {
        "diagonal_alignment": alignment,
        "diagonal_band_left_bound": left_bound,
        "diagonal_band_right_bound": right_bound,
        "use_alibi_mask": cfg.is_alibi,
    }


def _reject_unsupported(cfg: FusedAttnConfig) -> None:
    if cfg.is_paged_kv:
        raise NotImplementedError(
            "fused_attn_py F16 builder: paged KV is not supported yet (Stage 4)."
        )
    if cfg.is_ragged_q or cfg.is_ragged_kv or cfg.uses_cu_seqlens_directly:
        raise NotImplementedError(
            "fused_attn_py F16 builder: THD / cu_seqlens-direct is not supported yet (Stage 4)."
        )
    if cfg.is_softmax_offset:
        raise NotImplementedError(
            "fused_attn_py F16 builder: sink token / learnable softmax is not supported yet."
        )
    if cfg.return_max_logit:
        raise NotImplementedError(
            "fused_attn_py F16 builder: return_max_logit is not supported yet."
        )


def build_f16_fwd_graph(
    cudnn: Any,
    handle: Any,
    cfg: FusedAttnConfig,
    *,
    cudnn_version: Optional[int] = None,
) -> GraphEntry:
    """Build the F16/BF16 forward SDPA graph. Mirrors ``create_graph_f16_fwd``.

    ``cudnn`` is the imported frontend module, ``handle`` a cuDNN handle bound to
    the current stream. All tensor dims/strides are derived from ``cfg`` (via
    ``strides.qkvo_dims_strides``), so the graph is a pure function of the
    normalized config -- the same graph the support probe builds and the execute
    path reuses under one ``make_cache_key``. Returns a :class:`GraphEntry` whose
    ``tensors`` maps role names ("Q","K","V","attn_scale","O","Stats", and
    optionally "bias","seq_q","seq_kv","dropout_seed","dropout_offset") to graph
    tensor objects for the glue to bind into the variant pack.
    """
    cfg.check_derived()
    _reject_unsupported(cfg)
    if cudnn_version is None:
        cudnn_version = cudnn.backend_version()

    io_dtype = _io_data_type(cudnn, cfg.qkv_dtype)
    b = int(cfg.graph_batch_size_fwd)
    h = int(cfg.num_attn_heads)
    s_q = int(cfg.graph_max_seqlen_q)
    ds = qkvo_dims_strides(cfg)

    graph = cudnn.pygraph(
        io_data_type=io_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )

    q = graph.tensor(name="Q", dim=list(ds["Q"][0]), stride=list(ds["Q"][1]))
    k = graph.tensor(name="K", dim=list(ds["K"][0]), stride=list(ds["K"][1]))
    v = graph.tensor(name="V", dim=list(ds["V"][0]), stride=list(ds["V"][1]))
    # attn_scale is a pass-by-value scalar so one cached graph serves every scale
    # (make_cache_key() normalizes attn_scale to 1.0).
    attn_scale = graph.tensor(
        name="attn_scale",
        dim=[1, 1, 1, 1],
        stride=[1, 1, 1, 1],
        data_type=cudnn.data_type.FLOAT,
        is_pass_by_value=True,
    )

    tensors: Dict[str, Any] = {"Q": q, "K": k, "V": v, "attn_scale": attn_scale}

    masking = plan_f16_fwd_masking(cfg, cudnn_version)
    sdpa_kwargs: Dict[str, Any] = {
        "name": "flash_attention",
        "generate_stats": True,  # C++ always returns stats
        "attn_scale": attn_scale,
        "diagonal_alignment": getattr(cudnn.diagonal_alignment, masking["diagonal_alignment"]),
        "use_alibi_mask": masking["use_alibi_mask"],
    }
    if masking["diagonal_band_left_bound"] is not None:
        sdpa_kwargs["diagonal_band_left_bound"] = masking["diagonal_band_left_bound"]
    if masking["diagonal_band_right_bound"] is not None:
        sdpa_kwargs["diagonal_band_right_bound"] = masking["diagonal_band_right_bound"]

    if cfg.is_bias:
        bias_b = int(cfg.bias_batch_size)
        bias_h = int(cfg.bias_num_heads)
        bias_sq = int(cfg.bias_seqlen_q)
        bias_skv = int(cfg.bias_seqlen_kv)
        bias_dim = [bias_b, bias_h, bias_sq, bias_skv]
        bias_stride = [bias_h * bias_sq * bias_skv, bias_sq * bias_skv, bias_skv, 1]
        bias = graph.tensor(name="bias", dim=bias_dim, stride=bias_stride)
        sdpa_kwargs["bias"] = bias
        tensors["bias"] = bias

    if cfg.is_padding:
        # Non-cu_seqlens-direct path: materialized per-batch actual seqlens.
        seq_q = graph.tensor(
            name="seq_q", dim=[b, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32
        )
        seq_kv = graph.tensor(
            name="seq_kv", dim=[b, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32
        )
        sdpa_kwargs["use_padding_mask"] = True
        sdpa_kwargs["seq_len_q"] = seq_q
        sdpa_kwargs["seq_len_kv"] = seq_kv
        tensors["seq_q"] = seq_q
        tensors["seq_kv"] = seq_kv

    if cfg.is_dropout:
        seed = graph.tensor(
            name="Seed", dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64
        )
        offset = graph.tensor(
            name="Offset", dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64
        )
        sdpa_kwargs["dropout"] = (float(cfg.dropout), seed, offset)
        tensors["dropout_seed"] = seed
        tensors["dropout_offset"] = offset

    o, stats = graph.sdpa(q, k, v, **sdpa_kwargs)

    o.set_output(True).set_dim(list(ds["O"][0])).set_stride(list(ds["O"][1]))
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim([b, h, s_q, 1]).set_stride(
        [h * s_q, s_q, 1, 1]
    )
    tensors["O"] = o
    tensors["Stats"] = stats

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)
    workspace_size = max(graph.get_workspace_size(), 1)

    return GraphEntry(graph=graph, tensors=tensors, workspace_size=workspace_size)


__all__ = ["plan_f16_fwd_masking", "build_f16_fwd_graph"]
