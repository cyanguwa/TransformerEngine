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

Scope: BSHD/SBHD/BHSD and packed QKV layouts, with post-scale bias, ALiBi,
causal / bottom-right / sliding-window masking, padding (both materialized
``seq_len`` and ``cu_seqlens``-direct), dropout, GQA, THD ragged offsets, and
(forward only) paged KV. Still deferred and guarded with a clear
``NotImplementedError``:

* sink token / learnable softmax (``cfg.is_softmax_offset``)
* ``return_max_logit`` (logit-max output)
* paged KV in the backward pass (backward never runs paged; it is inference-only)

The masking/window option planning is factored into
:func:`plan_f16_fwd_masking`, a pure function that needs no ``cudnn`` and is
unit-tested on CPU; only tensor creation and the ``graph.sdpa`` call touch the
cuDNN Python package.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..cache import GraphEntry
from ..config import FusedAttnConfig, ScaleDType, _canonical_dtype
from ..strides import paged_kv_dims_strides, qkvo_dims_strides, ragged_offset_multipliers
from ..uids import FusedAttnUIDF16 as _U

# Role -> stable UID for the F16/BF16 graphs. Roles are the keys used in the
# ``GraphEntry.tensors`` map; forward uses "Stats", backward uses "stats" (both
# map to the same LSE UID -- they are separate graphs, so no clash). Assigning
# UIDs lets a serialized graph be executed from a plain {uid: ptr} variant pack.
_ROLE_UID = {
    "Q": _U.Q,
    "K": _U.K,
    "V": _U.V,
    "O": _U.O,
    "dO": _U.dO,
    "dQ": _U.dQ,
    "dK": _U.dK,
    "dV": _U.dV,
    "bias": _U.Bias,
    "dBias": _U.dBias,
    "Stats": _U.Stats,
    "stats": _U.Stats,
    "attn_scale": _U.AttnScale,
    "seq_q": _U.SeqQ,
    "seq_kv": _U.SeqKV,
    "page_table_k": _U.PageTableK,
    "page_table_v": _U.PageTableV,
    "offset_q": _U.OffsetQ,
    "offset_k": _U.OffsetK,
    "offset_v": _U.OffsetV,
    "offset_o": _U.OffsetO,
    "offset_stats": _U.OffsetStats,
    "dropout_seed": _U.DropoutSeed,
    "dropout_offset": _U.DropoutOffset,
}


def _assign_uids(tensors: Dict[str, Any]) -> Dict[str, int]:
    """Assign each graph tensor its stable UID and return the role->uid map.

    Called once, just before finalizing, so every declared tensor (inputs and
    outputs) carries a UID in the serialized graph.
    """
    uids: Dict[str, int] = {}
    for role, tensor in tensors.items():
        uid = int(_ROLE_UID[role])
        tensor.set_uid(uid)
        uids[role] = uid
    return uids


def _io_data_type(cudnn, dtype) -> Any:
    """Map an F16/BF16 dtype (enum / name) to a ``cudnn.data_type``."""
    name = _canonical_dtype(dtype)
    if name == "FLOAT16":
        return cudnn.data_type.HALF
    if name == "BFLOAT16":
        return cudnn.data_type.BFLOAT16
    raise ValueError(f"F16 fused-attention builder requires FP16/BF16, got {name}.")


def _ragged_dtype(cudnn, scale_dtype: ScaleDType) -> Any:
    """Map a ragged-offset ``ScaleDType`` (INT32/INT64) to a ``cudnn.data_type``."""
    return cudnn.data_type.INT64 if scale_dtype is ScaleDType.INT64 else cudnn.data_type.INT32


def _offset_tensor(graph, cudnn, name: str, b: int, dtype: Any):
    """A ragged-offset / cu_seqlen tensor of shape (b+1, 1, 1, 1)."""
    return graph.tensor(
        name=name, dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=dtype
    )


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


def _reject_unsupported(cfg: FusedAttnConfig, *, allow_paged: bool) -> None:
    if cfg.is_paged_kv and not allow_paged:
        raise NotImplementedError(
            "fused_attn_py F16 builder: paged KV is forward-only (inference); the backward "
            "graph does not support it."
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
    optionally "bias","seq_q","seq_kv","dropout_seed","dropout_offset",
    "page_table_k","page_table_v","offset_q","offset_k","offset_v","offset_o",
    "offset_stats") to graph tensor objects for the glue to bind.
    """
    cfg.check_derived()
    _reject_unsupported(cfg, allow_paged=True)
    if cudnn_version is None:
        cudnn_version = cudnn.backend_version()

    io_dtype = _io_data_type(cudnn, cfg.qkv_dtype)
    b = int(cfg.graph_batch_size_fwd)
    h = int(cfg.num_attn_heads)
    s_q = int(cfg.graph_max_seqlen_q)
    s_kv = int(cfg.graph_max_seqlen_kv)
    ds = qkvo_dims_strides(cfg)
    ragged_dtype = _ragged_dtype(cudnn, cfg.ragged_offset_type_fwd)
    mults = ragged_offset_multipliers(cfg)
    cu_direct = cfg.uses_cu_seqlens_directly

    graph = cudnn.pygraph(
        io_data_type=io_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )

    tensors: Dict[str, Any] = {}

    q = graph.tensor(name="Q", dim=list(ds["Q"][0]), stride=list(ds["Q"][1]))
    if cfg.is_ragged_q:
        offset_q = _offset_tensor(graph, cudnn, "offset_q", b, ragged_dtype)
        q.set_ragged_offset(offset_q)
        if cu_direct:
            q.set_ragged_offset_multiplier(mults.q)
        tensors["offset_q"] = offset_q

    if cfg.is_paged_kv:
        paged = paged_kv_dims_strides(cfg)
        k = graph.tensor(name="K", dim=list(paged["K"][0]), stride=list(paged["K"][1]))
        v = graph.tensor(name="V", dim=list(paged["V"][0]), stride=list(paged["V"][1]))
    else:
        k = graph.tensor(name="K", dim=list(ds["K"][0]), stride=list(ds["K"][1]))
        v = graph.tensor(name="V", dim=list(ds["V"][0]), stride=list(ds["V"][1]))
        if cfg.is_ragged_kv:
            offset_k = _offset_tensor(graph, cudnn, "offset_k", b, ragged_dtype)
            offset_v = _offset_tensor(graph, cudnn, "offset_v", b, ragged_dtype)
            k.set_ragged_offset(offset_k)
            v.set_ragged_offset(offset_v)
            if cu_direct:
                k.set_ragged_offset_multiplier(mults.k)
                v.set_ragged_offset_multiplier(mults.v)
            tensors["offset_k"] = offset_k
            tensors["offset_v"] = offset_v

    # attn_scale is a pass-by-value scalar so one cached graph serves every scale
    # (make_cache_key() normalizes attn_scale to 1.0).
    attn_scale = graph.tensor(
        name="attn_scale",
        dim=[1, 1, 1, 1],
        stride=[1, 1, 1, 1],
        data_type=cudnn.data_type.FLOAT,
        is_pass_by_value=True,
    )
    tensors.update({"Q": q, "K": k, "V": v, "attn_scale": attn_scale})

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
        # seq_q/seq_kv hold materialized actual seqlens (b), or (b+1) cu_seqlens on
        # the cu_seqlens-direct path (which pins the UNIFIED engine).
        seq_len = b + 1 if cu_direct else b
        seq_q = graph.tensor(
            name="seq_q", dim=[seq_len, 1, 1, 1], stride=[1, 1, 1, 1],
            data_type=cudnn.data_type.INT32,
        )
        seq_kv = graph.tensor(
            name="seq_kv", dim=[seq_len, 1, 1, 1], stride=[1, 1, 1, 1],
            data_type=cudnn.data_type.INT32,
        )
        sdpa_kwargs["use_padding_mask"] = True
        if cu_direct:
            sdpa_kwargs["cu_seq_len_q"] = seq_q
            sdpa_kwargs["cu_seq_len_kv"] = seq_kv
            sdpa_kwargs["implementation"] = cudnn.attention_implementation.UNIFIED
        else:
            sdpa_kwargs["seq_len_q"] = seq_q
            sdpa_kwargs["seq_len_kv"] = seq_kv
        tensors["seq_q"] = seq_q
        tensors["seq_kv"] = seq_kv

    if cfg.is_paged_kv:
        mppk = int(cfg.max_pages_per_seq_k)
        mppv = int(cfg.max_pages_per_seq_v)
        page_table_k = graph.tensor(
            name="page_table_k", dim=[b, 1, mppk, 1], stride=[mppk, mppv, 1, 1],
            data_type=cudnn.data_type.INT32,
        )
        page_table_v = graph.tensor(
            name="page_table_v", dim=[b, 1, mppv, 1], stride=[mppv, mppv, 1, 1],
            data_type=cudnn.data_type.INT32,
        )
        sdpa_kwargs["paged_attention_k_table"] = page_table_k
        sdpa_kwargs["paged_attention_v_table"] = page_table_v
        sdpa_kwargs["paged_attention_max_seq_len_kv"] = s_kv
        tensors["page_table_k"] = page_table_k
        tensors["page_table_v"] = page_table_v

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
    if cfg.is_ragged_q:
        offset_o = _offset_tensor(graph, cudnn, "offset_o", b, ragged_dtype)
        o.set_ragged_offset(offset_o)
        if cu_direct:
            o.set_ragged_offset_multiplier(mults.o)
        tensors["offset_o"] = offset_o

    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim([b, h, s_q, 1])
    if cfg.uses_ragged_stats:
        offset_stats = _offset_tensor(graph, cudnn, "offset_stats", b, ragged_dtype)
        stats.set_stride([h * s_q, 1, h, 1]).set_ragged_offset(offset_stats)
        if cu_direct:
            stats.set_ragged_offset_multiplier(mults.stats)
        tensors["offset_stats"] = offset_stats
    else:
        stats.set_stride([h * s_q, s_q, 1, 1])
    tensors["O"] = o
    tensors["Stats"] = stats

    uids = _assign_uids(tensors)
    _finalize(cudnn, graph)
    ws = max(graph.get_workspace_size(), 1)
    return GraphEntry(graph=graph, tensors=tensors, workspace_size=ws, uids=uids)


def _finalize(cudnn: Any, graph: Any) -> None:
    """Run the cuDNN build+support+plan sequence (shared by fwd and bwd)."""
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)


def build_f16_bwd_graph(
    cudnn: Any,
    handle: Any,
    cfg: FusedAttnConfig,
    *,
    cudnn_version: Optional[int] = None,
) -> GraphEntry:
    """Build the F16/BF16 backward SDPA graph. Mirrors ``create_graph_f16_bwd``.

    Inputs Q/K/V/O/dO/stats + pass-by-value attn_scale; outputs dQ/dK/dV (and
    dBias when bias is not fully broadcast). Dims/strides are derived from ``cfg``
    with the backward graph batch (``graph_batch_size_bwd``). Returns a
    :class:`GraphEntry` whose ``tensors`` maps role names ("Q","K","V","O","dO",
    "stats","attn_scale","dQ","dK","dV", plus optional "bias","dBias","seq_q",
    "seq_kv","dropout_seed","dropout_offset") to graph tensor objects.
    """
    cfg.check_derived()
    _reject_unsupported(cfg, allow_paged=False)
    if cudnn_version is None:
        cudnn_version = cudnn.backend_version()

    io_dtype = _io_data_type(cudnn, cfg.qkv_dtype)
    b = int(cfg.graph_batch_size_bwd)
    h = int(cfg.num_attn_heads)
    s_q = int(cfg.graph_max_seqlen_q)
    s_kv = int(cfg.graph_max_seqlen_kv)
    ds = qkvo_dims_strides(cfg, batch_size=b)
    # Backward always uses wide ragged offsets and no multiplier (multipliers are
    # a UNIFIED-forward-only feature).
    ragged_dtype = _ragged_dtype(cudnn, cfg.ragged_offset_type_bwd)

    graph = cudnn.pygraph(
        io_data_type=io_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )

    tensors: Dict[str, Any] = {}

    q = graph.tensor(name="Q", dim=list(ds["Q"][0]), stride=list(ds["Q"][1]))
    k = graph.tensor(name="K", dim=list(ds["K"][0]), stride=list(ds["K"][1]))
    v = graph.tensor(name="V", dim=list(ds["V"][0]), stride=list(ds["V"][1]))
    # O and dO share the O layout.
    o = graph.tensor(name="O", dim=list(ds["O"][0]), stride=list(ds["O"][1]))
    d_o = graph.tensor(name="dO", dim=list(ds["O"][0]), stride=list(ds["O"][1]))

    offset_q = offset_k = offset_v = None
    if cfg.is_ragged_q:
        offset_q = _offset_tensor(graph, cudnn, "offset_q", b, ragged_dtype)
        offset_o = _offset_tensor(graph, cudnn, "offset_o", b, ragged_dtype)
        q.set_ragged_offset(offset_q)
        o.set_ragged_offset(offset_o)
        d_o.set_ragged_offset(offset_o)
        tensors["offset_q"] = offset_q
        tensors["offset_o"] = offset_o
    if cfg.is_ragged_kv:
        offset_k = _offset_tensor(graph, cudnn, "offset_k", b, ragged_dtype)
        offset_v = _offset_tensor(graph, cudnn, "offset_v", b, ragged_dtype)
        k.set_ragged_offset(offset_k)
        v.set_ragged_offset(offset_v)
        tensors["offset_k"] = offset_k
        tensors["offset_v"] = offset_v

    stats = graph.tensor(name="stats", dim=[b, h, s_q, 1], data_type=cudnn.data_type.FLOAT)
    if cfg.uses_ragged_stats:
        offset_stats = _offset_tensor(graph, cudnn, "offset_stats", b, ragged_dtype)
        stats.set_stride([h * s_q, 1, h, 1]).set_ragged_offset(offset_stats)
        tensors["offset_stats"] = offset_stats
    else:
        stats.set_stride([h * s_q, s_q, 1, 1])

    attn_scale = graph.tensor(
        name="attn_scale",
        dim=[1, 1, 1, 1],
        stride=[1, 1, 1, 1],
        data_type=cudnn.data_type.FLOAT,
        is_pass_by_value=True,
    )

    tensors.update(
        {"Q": q, "K": k, "V": v, "O": o, "dO": d_o, "stats": stats, "attn_scale": attn_scale}
    )

    masking = plan_f16_fwd_masking(cfg, cudnn_version)
    bwd_kwargs: Dict[str, Any] = {
        "name": "flash_attention_backward",
        "attn_scale": attn_scale,
        "diagonal_alignment": getattr(cudnn.diagonal_alignment, masking["diagonal_alignment"]),
        "use_alibi_mask": masking["use_alibi_mask"],
    }
    if masking["diagonal_band_left_bound"] is not None:
        bwd_kwargs["diagonal_band_left_bound"] = masking["diagonal_band_left_bound"]
    if masking["diagonal_band_right_bound"] is not None:
        bwd_kwargs["diagonal_band_right_bound"] = masking["diagonal_band_right_bound"]
    if cudnn_version >= 90000:
        bwd_kwargs["use_deterministic_algorithm"] = bool(cfg.deterministic)
    # Ragged workspace bounds (mirrors set_max_total_seq_len_* in the C++ builder).
    if cfg.uses_ragged_stats:
        bwd_kwargs["max_total_seq_len_q"] = s_q
    if cfg.is_ragged_kv and cfg.uses_ragged_graph:
        bwd_kwargs["max_total_seq_len_kv"] = s_kv

    if cfg.is_bias:
        bias_b = int(cfg.bias_batch_size)
        bias_h = int(cfg.bias_num_heads)
        bias_sq = int(cfg.bias_seqlen_q)
        bias_skv = int(cfg.bias_seqlen_kv)
        bias_dim = [bias_b, bias_h, bias_sq, bias_skv]
        bias_stride = [bias_h * bias_sq * bias_skv, bias_sq * bias_skv, bias_skv, 1]
        bias = graph.tensor(name="bias", dim=bias_dim, stride=bias_stride)
        bwd_kwargs["bias"] = bias
        tensors["bias"] = bias
        # dBias is computable unless the bias is fully broadcast over (b, h, s_q)
        # (a [1, 1, 1, s_kv] bias has no dbias, as of cuDNN 9.18).
        if not (bias_b == 1 and bias_h == 1 and bias_sq == 1):
            d_bias = graph.tensor(name="dBias", dim=bias_dim, stride=bias_stride)
            bwd_kwargs["dBias"] = d_bias
            tensors["dBias"] = d_bias

    if cfg.is_padding:
        seq_q = graph.tensor(
            name="seq_q", dim=[b, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32
        )
        seq_kv = graph.tensor(
            name="seq_kv", dim=[b, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32
        )
        bwd_kwargs["use_padding_mask"] = True
        bwd_kwargs["seq_len_q"] = seq_q
        bwd_kwargs["seq_len_kv"] = seq_kv
        tensors["seq_q"] = seq_q
        tensors["seq_kv"] = seq_kv

    if cfg.is_dropout:
        seed = graph.tensor(
            name="Seed", dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64
        )
        offset = graph.tensor(
            name="Offset", dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64
        )
        bwd_kwargs["dropout"] = (float(cfg.dropout), seed, offset)
        tensors["dropout_seed"] = seed
        tensors["dropout_offset"] = offset

    d_q, d_k, d_v = graph.sdpa_backward(q, k, v, o, d_o, stats, **bwd_kwargs)
    d_q.set_output(True).set_dim(list(ds["Q"][0])).set_stride(list(ds["Q"][1]))
    d_k.set_output(True).set_dim(list(ds["K"][0])).set_stride(list(ds["K"][1]))
    d_v.set_output(True).set_dim(list(ds["V"][0])).set_stride(list(ds["V"][1]))
    if cfg.is_ragged_q:
        d_q.set_ragged_offset(offset_q)
    if cfg.is_ragged_kv:
        d_k.set_ragged_offset(offset_k)
        d_v.set_ragged_offset(offset_v)
    tensors["dQ"] = d_q
    tensors["dK"] = d_k
    tensors["dV"] = d_v

    uids = _assign_uids(tensors)
    _finalize(cudnn, graph)
    ws = max(graph.get_workspace_size(), 1)
    return GraphEntry(graph=graph, tensors=tensors, workspace_size=ws, uids=uids)


__all__ = ["plan_f16_fwd_masking", "build_f16_fwd_graph", "build_f16_bwd_graph"]
