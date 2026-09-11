# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FP8 fused-attention graph builders (cuDNN Python API).

Port of ``create_graph_fp8_fwd`` in ``common/fused_attn/fused_attn_fp8.cu`` onto
the cuDNN Python graph API (``graph.sdpa_fp8``). Like the F16 builders, tensors
are keyed by stable UID (:class:`FusedAttnUIDFP8`) so a serialized graph runs
from a plain ``{uid: ptr}`` variant pack.

Scope of this module (Stage 6): the **forward, tensor-scaling** path -- delayed
scaling (FP8 output, real ``Scale_o``) and current scaling (F16 output, constant
``Scale_o`` of 1.0), with the same masking / padding (materialized ``seq_len``
and ``cu_seqlens``-direct) / THD-ragged / dropout structure as F16. FP8 carries
the extra descale (Q/K/V/S) and scale (S/O) inputs and the amax_s / amax_o
outputs.

Deferred with a clear ``NotImplementedError`` (mirrors what the C++ builder
supports but is out of scope here):

* MXFP8 block scaling (``cfg.is_mxfp8``) -- needs the FP8_E8M0 block-scale
  tensors + ``generateMatrixStridesWithFormat`` / ``pad_s_d_for_mxfp8`` port
  (Stage 7).
* FP8 backward (``build_fp8_bwd_graph``) -- next step.
* post-scale bias / ALiBi (commented out in the C++ FP8 builder) and sink-token
  / learnable softmax (``cfg.is_softmax_offset``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..cache import GraphEntry
from ..config import FusedAttnConfig, _canonical_dtype
from ..strides import qkvo_dims_strides, ragged_offset_multipliers
from ..uids import FusedAttnUIDFP8 as _U
from .f16 import _finalize, _offset_tensor, _ragged_dtype, plan_f16_fwd_masking

# Role -> stable UID for the FP8 graphs (mirror of FusedAttnUIDFP8). Only the
# roles this module actually creates are listed; backward-only roles are added
# when build_fp8_bwd_graph lands.
_ROLE_UID = {
    "Q": _U.Q,
    "K": _U.K,
    "V": _U.V,
    "O": _U.O,
    "Stats": _U.Stats,
    "attn_scale": _U.AttnScale,
    "DescaleQ": _U.DescaleQ,
    "DescaleK": _U.DescaleK,
    "DescaleV": _U.DescaleV,
    "DescaleS": _U.DescaleS,
    "ScaleS": _U.ScaleS,
    "ScaleO": _U.ScaleO,
    "AmaxS": _U.AmaxS,
    "AmaxO": _U.AmaxO,
    "seq_q": _U.SeqQ,
    "seq_kv": _U.SeqKV,
    "offset_q": _U.OffsetQ,
    "offset_k": _U.OffsetK,
    "offset_v": _U.OffsetV,
    "offset_o": _U.OffsetO,
    "offset_stats": _U.OffsetStats,
    "dropout_seed": _U.DropoutSeed,
    "dropout_offset": _U.DropoutOffset,
}


def _assign_uids(tensors: Dict[str, Any]) -> Dict[str, int]:
    """Assign each graph tensor its stable FP8 UID; return the role->uid map."""
    uids: Dict[str, int] = {}
    for role, tensor in tensors.items():
        uid = int(_ROLE_UID[role])
        tensor.set_uid(uid)
        uids[role] = uid
    return uids


def _fp8_data_type(cudnn, dtype) -> Any:
    """Map an FP8 dtype (enum / name) to a ``cudnn.data_type``."""
    name = _canonical_dtype(dtype)
    if name == "FLOAT8E4M3":
        return cudnn.data_type.FP8_E4M3
    if name == "FLOAT8E5M2":
        return cudnn.data_type.FP8_E5M2
    raise ValueError(f"FP8 fused-attention builder requires FP8 Q/K/V, got {name}.")


def _o_data_type(cudnn, dtype) -> Any:
    """Output dtype: FP8 (delayed scaling) or F16/BF16 (current scaling)."""
    name = _canonical_dtype(dtype)
    if name == "FLOAT8E4M3":
        return cudnn.data_type.FP8_E4M3
    if name == "FLOAT8E5M2":
        return cudnn.data_type.FP8_E5M2
    if name == "FLOAT16":
        return cudnn.data_type.HALF
    if name == "BFLOAT16":
        return cudnn.data_type.BFLOAT16
    raise ValueError(f"FP8 fused-attention builder: unsupported output dtype {name}.")


def _scalar(graph, name: str, cudnn) -> Any:
    """A scalar (descale/scale/amax) tensor of shape (1,1,1,1), FLOAT."""
    return graph.tensor(
        name=name, dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT
    )


def _reject_unsupported_fp8(cfg: FusedAttnConfig) -> None:
    if cfg.is_mxfp8:
        raise NotImplementedError(
            "fused_attn_py FP8 builder: MXFP8 block scaling is deferred to Stage 7 "
            "(needs the FP8_E8M0 block-scale tensors + format-stride/padding port)."
        )
    if not cfg.is_tensor_scaling:
        raise NotImplementedError(
            "fused_attn_py FP8 builder: only delayed/current tensor scaling and MXFP8 are "
            "FP8 scaling modes; got a config that is neither."
        )
    if cfg.is_bias or cfg.is_alibi:
        raise NotImplementedError(
            "fused_attn_py FP8 builder: post-scale bias / ALiBi are not supported (they are "
            "commented out in the C++ FP8 builder)."
        )
    if cfg.is_softmax_offset:
        raise NotImplementedError(
            "fused_attn_py FP8 builder: sink token / learnable softmax is not supported yet."
        )
    if cfg.return_max_logit:
        raise NotImplementedError(
            "fused_attn_py FP8 builder: return_max_logit is not supported yet."
        )


def build_fp8_fwd_graph(
    cudnn: Any,
    handle: Any,
    cfg: FusedAttnConfig,
    *,
    cudnn_version: Optional[int] = None,
) -> GraphEntry:
    """Build the FP8 forward SDPA graph (tensor scaling). Mirrors ``create_graph_fp8_fwd``.

    Inputs: Q/K/V (FP8), Descale_q/k/v/s, Scale_s, Scale_o (delayed scaling only;
    current scaling uses a constant 1.0), pass-by-value attn_scale, plus optional
    padding seq lengths, THD ragged offsets, and dropout RNG state. Outputs:
    O (FP8 for delayed, F16 for current), Stats, Amax_s, Amax_o. Dims/strides come
    from ``cfg`` via ``strides.qkvo_dims_strides``. Returns a :class:`GraphEntry`.
    """
    cfg.check_derived()
    _reject_unsupported_fp8(cfg)
    if cudnn_version is None:
        cudnn_version = cudnn.backend_version()

    qkv_dtype = _fp8_data_type(cudnn, cfg.qkv_dtype)
    o_dtype = _o_data_type(cudnn, cfg.o_dtype)
    b = int(cfg.graph_batch_size_fwd)
    h = int(cfg.num_attn_heads)
    s_q = int(cfg.graph_max_seqlen_q)
    ds = qkvo_dims_strides(cfg)
    ragged_dtype = _ragged_dtype(cudnn, cfg.ragged_offset_type_fwd)
    mults = ragged_offset_multipliers(cfg)
    cu_direct = cfg.uses_cu_seqlens_directly
    is_delayed = cfg.is_delayed_scaling_fwd

    graph = cudnn.pygraph(
        io_data_type=qkv_dtype,
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

    attn_scale = graph.tensor(
        name="attn_scale",
        dim=[1, 1, 1, 1],
        stride=[1, 1, 1, 1],
        data_type=cudnn.data_type.FLOAT,
        is_pass_by_value=True,
    )

    # Descales for the inputs and the softmax numerator; scale for S. Scale_o is a
    # real input for delayed scaling (FP8 output) but a baked-in 1.0 constant for
    # current scaling (F16 output).
    descale_q = _scalar(graph, "Descale_q", cudnn)
    descale_k = _scalar(graph, "Descale_k", cudnn)
    descale_v = _scalar(graph, "Descale_v", cudnn)
    descale_s = _scalar(graph, "Descale_s", cudnn)
    scale_s = _scalar(graph, "Scale_s", cudnn)
    scale_o = _scalar(graph, "Scale_o", cudnn) if is_delayed else graph.tensor(1.0)

    tensors.update(
        {
            "Q": q,
            "K": k,
            "V": v,
            "attn_scale": attn_scale,
            "DescaleQ": descale_q,
            "DescaleK": descale_k,
            "DescaleV": descale_v,
            "DescaleS": descale_s,
            "ScaleS": scale_s,
        }
    )
    if is_delayed:
        tensors["ScaleO"] = scale_o

    masking = plan_f16_fwd_masking(cfg, cudnn_version)
    sdpa_kwargs: Dict[str, Any] = {
        "name": "sdpa_fp8",
        "generate_stats": True,
        "attn_scale": attn_scale,
        "diagonal_alignment": getattr(cudnn.diagonal_alignment, masking["diagonal_alignment"]),
    }
    if masking["diagonal_band_left_bound"] is not None:
        sdpa_kwargs["diagonal_band_left_bound"] = masking["diagonal_band_left_bound"]
    if masking["diagonal_band_right_bound"] is not None:
        sdpa_kwargs["diagonal_band_right_bound"] = masking["diagonal_band_right_bound"]

    if cfg.is_padding:
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

    o, stats, amax_s, amax_o = graph.sdpa_fp8(
        q, k, v, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o, **sdpa_kwargs
    )

    o.set_output(True).set_dim(list(ds["O"][0])).set_stride(list(ds["O"][1])).set_data_type(o_dtype)
    if cfg.is_ragged_q:
        offset_o = _offset_tensor(graph, cudnn, "offset_o", b, ragged_dtype)
        o.set_ragged_offset(offset_o)
        if cu_direct:
            o.set_ragged_offset_multiplier(mults.o)
        tensors["offset_o"] = offset_o

    amax_s.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(
        cudnn.data_type.FLOAT
    )
    amax_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(
        cudnn.data_type.FLOAT
    )

    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim([b, h, s_q, 1])
    if cfg.uses_ragged_stats:
        offset_stats = _offset_tensor(graph, cudnn, "offset_stats", b, ragged_dtype)
        stats.set_stride([h * s_q, 1, h, 1]).set_ragged_offset(offset_stats)
        if cu_direct:
            stats.set_ragged_offset_multiplier(mults.stats)
        tensors["offset_stats"] = offset_stats
    else:
        stats.set_stride([h * s_q, s_q, 1, 1])

    tensors.update({"O": o, "Stats": stats, "AmaxS": amax_s, "AmaxO": amax_o})

    uids = _assign_uids(tensors)
    _finalize(cudnn, graph)
    ws = max(graph.get_workspace_size(), 1)
    return GraphEntry(graph=graph, tensors=tensors, workspace_size=ws, uids=uids)


__all__ = ["build_fp8_fwd_graph"]
