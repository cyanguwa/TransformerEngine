# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FP8 fused-attention graph builders (cuDNN Python API).

Port of ``create_graph_fp8_fwd`` / ``create_graph_fp8_bwd`` in
``common/fused_attn/fused_attn_fp8.cu`` onto the cuDNN Python graph API
(``graph.sdpa_fp8`` / ``graph.sdpa_fp8_backward``). Like the F16 builders, tensors
are keyed by stable UID (:class:`FusedAttnUIDFP8`) so a serialized graph runs
from a plain ``{uid: ptr}`` variant pack.

Scope of this module -- both FP8 scaling families:

* **Tensor scaling** -- delayed scaling (FP8 grads/output, real scale tensors)
  and current scaling (F16 grads/output, constant ``1.0`` scales). Carries scalar
  descale/scale inputs and the amax outputs (forward: amax_s/amax_o; backward:
  amax_dQ/dK/dV + amax_dP).
* **MXFP8** block scaling (``cfg.is_mxfp8``) -- FP8_E8M0 block-scale descales laid
  out by QKV format via ``generateMatrixStridesWithFormat`` / ``pad_s_d_for_mxfp8``
  (see ``strides.py``); backward also carries the transpose / f16 helper tensors
  (Q_t, K_t, dO_t, dO_f16). The amaxes are computed but not surfaced (C++
  ``set_output(!is_mxfp8)``): forward emits O/Stats, backward emits dQ/dK/dV only.

All paths share the masking / padding / THD-ragged / dropout structure of F16.

Deferred with a clear ``NotImplementedError`` (mirrors what the C++ builder
supports but is out of scope here):

* post-scale bias / ALiBi (commented out in the C++ FP8 builder) and sink-token
  / learnable softmax (``cfg.is_softmax_offset``).

Note (executor wiring, a later stage): FP8 graphs emit amax outputs beyond the
2 (forward: O/stats) / 3 (backward: dQ/dK/dV) the F16 ``score_mod`` FFI handlers
return, so the FP8 execute path needs its own handlers with extra ``Ret``
buffers. This module only builds/serializes the graph; that is unaffected.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..cache import GraphEntry
from ..config import FusedAttnConfig, _canonical_dtype, _name
from ..strides import (
    generate_matrix_strides_with_format,
    pad_s_d_for_mxfp8,
    qkvo_dims_strides,
    ragged_offset_multipliers,
)
from ..uids import FusedAttnUIDFP8 as _U
from .f16 import _finalize, _offset_tensor, _ragged_dtype, plan_f16_fwd_masking

# Role -> stable UID for the FP8 graphs (mirror of FusedAttnUIDFP8). Forward uses
# "Stats"/"O" as outputs; backward reuses "O" as an *input* and "stats"
# (lowercase) for the input LSE, so the two never collide within one graph.
_ROLE_UID = {
    "Q": _U.Q,
    "K": _U.K,
    "V": _U.V,
    "O": _U.O,
    "dO": _U.dO,
    "dQ": _U.dQ,
    "dK": _U.dK,
    "dV": _U.dV,
    "Stats": _U.Stats,
    "stats": _U.Stats,
    "attn_scale": _U.AttnScale,
    # Descales / scales (forward)
    "DescaleQ": _U.DescaleQ,
    "DescaleK": _U.DescaleK,
    "DescaleV": _U.DescaleV,
    "DescaleS": _U.DescaleS,
    "ScaleS": _U.ScaleS,
    "ScaleO": _U.ScaleO,
    "AmaxS": _U.AmaxS,
    "AmaxO": _U.AmaxO,
    # Descales / scales (backward-only)
    "DescaleO": _U.DescaleO,
    "DescaledO": _U.DescaledO,
    "DescaledP": _U.DescaledP,
    "ScaledP": _U.ScaledP,
    "ScaledQ": _U.ScaledQ,
    "ScaledK": _U.ScaledK,
    "ScaledV": _U.ScaledV,
    "AmaxdP": _U.AmaxdP,
    "AmaxdQ": _U.AmaxdQ,
    "AmaxdK": _U.AmaxdK,
    "AmaxdV": _U.AmaxdV,
    # MXFP8 transpose / f16 helper tensors (backward-only)
    "Qt": _U.Qt,
    "Kt": _U.Kt,
    "dOf16": _U.dOf16,
    "dOt": _U.dOt,
    "DescaleQt": _U.DescaleQt,
    "DescaleKt": _U.DescaleKt,
    "DescaledOt": _U.DescaledOt,
    # Sequence lengths / ragged offsets / dropout (shared)
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


def _block_scale(graph, cudnn, name: str, dim, stride) -> Any:
    """An MXFP8 block-scale tensor: FP8_E8M0 with F8_128x4 reordering."""
    return graph.tensor(
        name=name,
        dim=list(dim),
        stride=list(stride),
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )


def _scale_inv_format(cfg: FusedAttnConfig, fallback):
    """qkv_scale_inv_format when set, else the given q/kv format (matches C++)."""
    if _name(cfg.qkv_scale_inv_format) in ("", "NVTE_QKV_Format_NOT_SET"):
        return fallback
    return cfg.qkv_scale_inv_format


def _reject_unsupported_fp8(cfg: FusedAttnConfig) -> None:
    if not (cfg.is_tensor_scaling or cfg.is_mxfp8):
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
    """Build the FP8 forward SDPA graph. Mirrors ``create_graph_fp8_fwd``.

    Two FP8 scaling modes:

    * **Tensor scaling** -- Q/K/V (FP8) + scalar Descale_q/k/v/s, Scale_s, Scale_o
      (a real input for delayed scaling / FP8 output; a baked-in 1.0 constant for
      current scaling / F16 output); outputs O, Stats, Amax_s, Amax_o.
    * **MXFP8** (block scaling) -- Q/K/V (FP8) + FP8_E8M0 block-scale Descale_q/k/v
      laid out by QKV format; outputs O and Stats only (the amax is computed but
      not surfaced, matching the C++ ``set_output(!is_mxfp8)``).

    Both share the pass-by-value attn_scale and the padding / THD-ragged / dropout
    machinery. Returns a :class:`GraphEntry`.
    """
    cfg.check_derived()
    _reject_unsupported_fp8(cfg)
    if cudnn_version is None:
        cudnn_version = cudnn.backend_version()

    is_mxfp8 = cfg.is_mxfp8
    qkv_dtype = _fp8_data_type(cudnn, cfg.qkv_dtype)
    o_dtype = _o_data_type(cudnn, cfg.o_dtype)
    b = int(cfg.graph_batch_size_fwd)
    h = int(cfg.num_attn_heads)
    hg = int(cfg.num_gqa_groups)
    s_q = int(cfg.graph_max_seqlen_q)
    s_kv = int(cfg.graph_max_seqlen_kv)
    d_qk = int(cfg.head_dim_qk)
    d_v = int(cfg.head_dim_v)
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

    tensors.update({"Q": q, "K": k, "V": v, "attn_scale": attn_scale})
    if is_mxfp8:
        # Block-scale descales (FP8_E8M0), laid out by QKV format and MXFP8 padding.
        pad = pad_s_d_for_mxfp8(s_q, s_kv, d_qk, d_v)
        q_fmt = _scale_inv_format(cfg, cfg.q_format)
        kv_fmt = _scale_inv_format(cfg, cfg.kv_format)
        descale_q = _block_scale(
            graph, cudnn, "Descale_q", [b, h, pad.s_q_padded, pad.d_qk_scale_padded],
            generate_matrix_strides_with_format(b, h, pad.s_q_padded, pad.d_qk_scale_padded, q_fmt),
        )
        descale_k = _block_scale(
            graph, cudnn, "Descale_k", [b, hg, pad.s_kv_padded, pad.d_qk_scale_padded],
            generate_matrix_strides_with_format(
                b, hg, pad.s_kv_padded, pad.d_qk_scale_padded, kv_fmt
            ),
        )
        descale_v = _block_scale(
            graph, cudnn, "Descale_v", [b, hg, pad.s_kv_scale_padded, pad.d_v_padded],
            generate_matrix_strides_with_format(
                b, hg, pad.s_kv_scale_padded, pad.d_v_padded, kv_fmt
            ),
        )
        tensors.update({"DescaleQ": descale_q, "DescaleK": descale_k, "DescaleV": descale_v})
    else:
        # Scalar descales for the inputs and softmax numerator; scale for S. Scale_o
        # is a real input for delayed scaling (FP8 output) but a 1.0 constant for
        # current scaling (F16 output).
        descale_q = _scalar(graph, "Descale_q", cudnn)
        descale_k = _scalar(graph, "Descale_k", cudnn)
        descale_v = _scalar(graph, "Descale_v", cudnn)
        descale_s = _scalar(graph, "Descale_s", cudnn)
        scale_s = _scalar(graph, "Scale_s", cudnn)
        scale_o = _scalar(graph, "Scale_o", cudnn) if is_delayed else graph.tensor(1.0)
        tensors.update(
            {
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

    if is_mxfp8:
        outputs = graph.sdpa_fp8(q, k, v, descale_q, descale_k, descale_v, **sdpa_kwargs)
        o, stats, amax_o = outputs[0], outputs[1], outputs[2]
        amax_s = None
    else:
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

    # amax_o is a real output only for tensor scaling (C++ set_output(!is_mxfp8));
    # amax_s exists for tensor scaling only.
    amax_o.set_output(not is_mxfp8).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(
        cudnn.data_type.FLOAT
    )
    if not is_mxfp8:
        amax_s.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(
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

    tensors.update({"O": o, "Stats": stats})
    if not is_mxfp8:
        tensors.update({"AmaxS": amax_s, "AmaxO": amax_o})

    uids = _assign_uids(tensors)
    _finalize(cudnn, graph)
    ws = max(graph.get_workspace_size(), 1)
    return GraphEntry(graph=graph, tensors=tensors, workspace_size=ws, uids=uids)


def build_fp8_bwd_graph(
    cudnn: Any,
    handle: Any,
    cfg: FusedAttnConfig,
    *,
    cudnn_version: Optional[int] = None,
) -> GraphEntry:
    """Build the FP8 backward SDPA graph. Mirrors ``create_graph_fp8_bwd``.

    Two FP8 scaling modes:

    * **Tensor scaling** -- Q/K/V (FP8), O, dO, Stats, attn_scale, scalar descales
      (Q/K/V/O/dO/S/dP) and scales (S). ``Descale_o`` is a 1.0 constant when
      current scaling keeps O in F16; the output scales (Scale_dQ/dK/dV/dP) are
      real inputs for delayed scaling but 1.0 constants for current scaling.
      Outputs dQ/dK/dV + amax_dQ/dK/dV + amax_dP.
    * **MXFP8** -- additionally carries the transpose / f16 helpers (Q_t, K_t,
      dO_t, dO_f16) and FP8_E8M0 block-scale descales (Q/Q_t/K/K_t/V/dO/dO_t) laid
      out by QKV format. Outputs dQ/dK/dV only (amaxes computed but not surfaced,
      mirroring the C++ ``set_output(!is_mxfp8)``); no amax_dP.

    Returns a :class:`GraphEntry` whose ``output_roles`` are declared explicitly
    (the shared "O"/"Stats" names are *inputs* here, not outputs).
    """
    cfg.check_derived()
    _reject_unsupported_fp8(cfg)
    if cudnn_version is None:
        cudnn_version = cudnn.backend_version()

    is_mxfp8 = cfg.is_mxfp8
    qkv_dtype = _fp8_data_type(cudnn, cfg.qkv_dtype)
    o_dtype = _o_data_type(cudnn, cfg.o_dtype)
    do_dtype = _o_data_type(cudnn, cfg.do_dtype)
    dqkv_dtype = _o_data_type(cudnn, cfg.dqkv_dtype)
    b = int(cfg.graph_batch_size_bwd)
    h = int(cfg.num_attn_heads)
    hg = int(cfg.num_gqa_groups)
    s_q = int(cfg.graph_max_seqlen_q)
    s_kv = int(cfg.graph_max_seqlen_kv)
    d_qk = int(cfg.head_dim_qk)
    d_v = int(cfg.head_dim_v)
    ds = qkvo_dims_strides(cfg, batch_size=b)
    ragged_dtype = _ragged_dtype(cudnn, cfg.ragged_offset_type_bwd)
    is_delayed = cfg.is_delayed_scaling_bwd
    is_current = cfg.is_current_scaling_bwd

    graph = cudnn.pygraph(
        io_data_type=qkv_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )

    tensors: Dict[str, Any] = {}

    q = graph.tensor(name="Q", dim=list(ds["Q"][0]), stride=list(ds["Q"][1]))
    k = graph.tensor(name="K", dim=list(ds["K"][0]), stride=list(ds["K"][1]))
    v = graph.tensor(name="V", dim=list(ds["V"][0]), stride=list(ds["V"][1]))
    o = graph.tensor(name="O", dim=list(ds["O"][0]), stride=list(ds["O"][1]), data_type=o_dtype)
    d_o = graph.tensor(name="dO", dim=list(ds["O"][0]), stride=list(ds["O"][1]), data_type=do_dtype)

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

    stats = graph.tensor(name="Stats", dim=[b, h, s_q, 1], data_type=cudnn.data_type.FLOAT)
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

    tensors.update({"Q": q, "K": k, "V": v, "O": o, "dO": d_o, "stats": stats})
    tensors["attn_scale"] = attn_scale

    if is_mxfp8:
        # Transpose / f16 helpers (laid out by QKV format) and FP8_E8M0 block
        # descales for Q/Q_t/K/K_t/V/dO/dO_t.
        pad = pad_s_d_for_mxfp8(s_q, s_kv, d_qk, d_v)
        q_fmt, kv_fmt = cfg.q_format, cfg.kv_format
        # dO shares O's format; fall back to the query format when the framework
        # left do_format unset (it normally populates it).
        do_fmt = cfg.do_format
        if _name(do_fmt) in ("", "NVTE_QKV_Format_NOT_SET"):
            do_fmt = q_fmt
        q_sfmt = _scale_inv_format(cfg, q_fmt)
        kv_sfmt = _scale_inv_format(cfg, kv_fmt)
        do_sfmt = cfg.do_scale_inv_format
        if _name(do_sfmt) in ("", "NVTE_QKV_Format_NOT_SET"):
            do_sfmt = do_fmt
        fmt = generate_matrix_strides_with_format
        q_t = graph.tensor(
            name="Q_t", dim=[b, h, s_q, d_qk], stride=list(fmt(b, h, s_q, d_qk, q_fmt)),
            data_type=qkv_dtype,
        )
        k_t = graph.tensor(
            name="K_t", dim=[b, hg, s_kv, d_qk], stride=list(fmt(b, hg, s_kv, d_qk, kv_fmt)),
            data_type=qkv_dtype,
        )
        d_o_t = graph.tensor(
            name="dO_t", dim=[b, h, s_q, d_v], stride=list(fmt(b, h, s_q, d_v, do_fmt)),
            data_type=do_dtype,
        )
        d_o_f16 = graph.tensor(
            name="dO_f16", dim=[b, h, s_q, d_v], stride=list(fmt(b, h, s_q, d_v, do_fmt)),
            data_type=o_dtype,
        )
        descale_q = _block_scale(
            graph, cudnn, "Descale_q", [b, h, pad.s_q_padded, pad.d_qk_scale_padded],
            fmt(b, h, pad.s_q_padded, pad.d_qk_scale_padded, q_sfmt),
        )
        descale_q_t = _block_scale(
            graph, cudnn, "Descale_q_t", [b, h, pad.s_q_scale_padded, pad.d_qk_padded],
            fmt(b, h, pad.s_q_scale_padded, pad.d_qk_padded, q_sfmt),
        )
        descale_k = _block_scale(
            graph, cudnn, "Descale_k", [b, hg, pad.s_kv_padded, pad.d_qk_scale_padded],
            fmt(b, hg, pad.s_kv_padded, pad.d_qk_scale_padded, kv_sfmt),
        )
        descale_k_t = _block_scale(
            graph, cudnn, "Descale_k_t", [b, hg, pad.s_kv_scale_padded, pad.d_qk_padded],
            fmt(b, hg, pad.s_kv_scale_padded, pad.d_qk_padded, kv_sfmt),
        )
        descale_v = _block_scale(
            graph, cudnn, "Descale_v", [b, hg, pad.s_kv_padded, pad.d_v_scale_padded],
            fmt(b, hg, pad.s_kv_padded, pad.d_v_scale_padded, kv_sfmt),
        )
        descale_do = _block_scale(
            graph, cudnn, "Descale_dO", [b, h, pad.s_q_padded, pad.d_v_scale_padded],
            fmt(b, h, pad.s_q_padded, pad.d_v_scale_padded, do_sfmt),
        )
        descale_do_t = _block_scale(
            graph, cudnn, "Descale_dO_t", [b, h, pad.s_q_scale_padded, pad.d_v_padded],
            fmt(b, h, pad.s_q_scale_padded, pad.d_v_padded, do_sfmt),
        )
        tensors.update(
            {
                "Qt": q_t,
                "Kt": k_t,
                "dOt": d_o_t,
                "dOf16": d_o_f16,
                "DescaleQ": descale_q,
                "DescaleQt": descale_q_t,
                "DescaleK": descale_k,
                "DescaleKt": descale_k_t,
                "DescaleV": descale_v,
                "DescaledO": descale_do,
                "DescaledOt": descale_do_t,
            }
        )
    else:
        # Scalar descales for inputs / softmax / dP; Descale_o is a 1.0 constant
        # when current scaling keeps O in F16. Output scales are real inputs for
        # delayed scaling but 1.0 constants for current scaling.
        descale_q = _scalar(graph, "Descale_q", cudnn)
        descale_k = _scalar(graph, "Descale_k", cudnn)
        descale_v = _scalar(graph, "Descale_v", cudnn)
        descale_s = _scalar(graph, "Descale_s", cudnn)
        descale_dp = _scalar(graph, "Descale_dP", cudnn)
        descale_do = _scalar(graph, "Descale_dO", cudnn)
        scale_s = _scalar(graph, "Scale_s", cudnn)
        scale_dp = _scalar(graph, "Scale_dP", cudnn)
        o_in_f16 = cfg.is_o_in_f16
        if is_current and o_in_f16:
            descale_o = graph.tensor(1.0)
        else:
            descale_o = _scalar(graph, "Descale_O", cudnn)
        if is_delayed:
            scale_dq = _scalar(graph, "Scale_dQ", cudnn)
            scale_dk = _scalar(graph, "Scale_dK", cudnn)
            scale_dv = _scalar(graph, "Scale_dV", cudnn)
        else:
            scale_dq = graph.tensor(1.0)
            scale_dk = graph.tensor(1.0)
            scale_dv = graph.tensor(1.0)
        tensors.update(
            {
                "DescaleQ": descale_q,
                "DescaleK": descale_k,
                "DescaleV": descale_v,
                "DescaleS": descale_s,
                "DescaledP": descale_dp,
                "DescaledO": descale_do,
                "ScaleS": scale_s,
                "ScaledP": scale_dp,
            }
        )
        if not (is_current and o_in_f16):
            tensors["DescaleO"] = descale_o
        if is_delayed:
            tensors.update({"ScaledQ": scale_dq, "ScaledK": scale_dk, "ScaledV": scale_dv})

    masking = plan_f16_fwd_masking(cfg, cudnn_version)
    bwd_kwargs: Dict[str, Any] = {
        "name": "sdpa_fp8_backward",
        "attn_scale": attn_scale,
        "diagonal_alignment": getattr(cudnn.diagonal_alignment, masking["diagonal_alignment"]),
    }
    if masking["diagonal_band_left_bound"] is not None:
        bwd_kwargs["diagonal_band_left_bound"] = masking["diagonal_band_left_bound"]
    if masking["diagonal_band_right_bound"] is not None:
        bwd_kwargs["diagonal_band_right_bound"] = masking["diagonal_band_right_bound"]
    if cudnn_version >= 91900:
        bwd_kwargs["use_deterministic_algorithm"] = bool(cfg.deterministic)

    if cfg.is_padding:
        # Backward always uses materialized actual seqlens (b), never cu_seqlens.
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

    if is_mxfp8:
        outputs = graph.sdpa_fp8_backward(
            q, q_t, k, k_t, v, o, d_o_f16, d_o, d_o_t, stats,
            descale_q, descale_q_t, descale_k, descale_k_t, descale_v, descale_do, descale_do_t,
            **bwd_kwargs,
        )
        d_q, d_k, d_v = outputs[0], outputs[1], outputs[2]
        amax_dq, amax_dk, amax_dv = outputs[3], outputs[4], outputs[5]
        amax_dp = None
    else:
        d_q, d_k, d_v, amax_dq, amax_dk, amax_dv, amax_dp = graph.sdpa_fp8_backward(
            q, k, v, o, d_o, stats,
            descale_q, descale_k, descale_v, descale_o, descale_do, descale_s, descale_dp,
            scale_s, scale_dq, scale_dk, scale_dv, scale_dp,
            **bwd_kwargs,
        )

    d_q.set_output(True).set_dim(list(ds["Q"][0])).set_stride(list(ds["Q"][1])).set_data_type(
        dqkv_dtype
    )
    d_k.set_output(True).set_dim(list(ds["K"][0])).set_stride(list(ds["K"][1])).set_data_type(
        dqkv_dtype
    )
    d_v.set_output(True).set_dim(list(ds["V"][0])).set_stride(list(ds["V"][1])).set_data_type(
        dqkv_dtype
    )
    if cfg.is_ragged_q:
        d_q.set_ragged_offset(offset_q)
    if cfg.is_ragged_kv:
        d_k.set_ragged_offset(offset_k)
        d_v.set_ragged_offset(offset_v)

    # amaxes are real outputs only for tensor scaling (C++ set_output(!is_mxfp8));
    # amax_dP exists for tensor scaling only.
    grad_amaxes = (amax_dq, amax_dk, amax_dv) if is_mxfp8 else (amax_dq, amax_dk, amax_dv, amax_dp)
    for amax in grad_amaxes:
        amax.set_output(not is_mxfp8).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(
            cudnn.data_type.FLOAT
        )

    tensors.update({"dQ": d_q, "dK": d_k, "dV": d_v})
    if is_mxfp8:
        # Grad amaxes are computed but not surfaced for MXFP8 -> not bound / no UID.
        output_roles = frozenset({"dQ", "dK", "dV"})
    else:
        tensors.update(
            {"AmaxdQ": amax_dq, "AmaxdK": amax_dk, "AmaxdV": amax_dv, "AmaxdP": amax_dp}
        )
        output_roles = frozenset({"dQ", "dK", "dV", "AmaxdQ", "AmaxdK", "AmaxdV", "AmaxdP"})

    uids = _assign_uids(tensors)
    _finalize(cudnn, graph)
    ws = max(graph.get_workspace_size(), 1)
    return GraphEntry(
        graph=graph, tensors=tensors, workspace_size=ws, uids=uids, output_roles=output_roles
    )


__all__ = ["build_fp8_fwd_graph", "build_fp8_bwd_graph"]
