# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""BHSD logical dim/stride generation, ported from ``generateMatrixStrides``.

cuDNN fused-attention tensors are always logically ``[B, H, S, D]``; the
physical layout (BSHD / SBHD / BHSD / packed QK V / THD) is expressed entirely
through the stride vector. This is a faithful Python port of
``generateMatrixStrides`` in ``common/fused_attn/utils.cu`` for the four data
matrices (Q, K, V, O); the FP8-only transpose matrices and the S matrix are not
needed by the Python builders and are omitted.

Making dims/strides a pure function of the normalized ``FusedAttnConfig`` (never
of a runtime tensor's ``.stride()``) is what lets the cache key
(``make_cache_key``) soundly identify a graph and lets the support probe build
the exact same graph the execute path will reuse.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

from .config import FusedAttnConfig, QKVLayoutGroup, _name, get_qkv_layout_group

# Dim indices in the BHSD logical layout.
_BATCH, _HEAD, _SEQ, _HID = 0, 1, 2, 3

# Layout groupings, matching the switch cases in generateMatrixStrides.
_SB3HD = {"NVTE_SB3HD"}
_SBH3D = {"NVTE_SBH3D"}
_SBHD_SB2HD = {"NVTE_SBHD_SB2HD"}
_SBHD_SBH2D = {"NVTE_SBHD_SBH2D"}
_SBHD_SEP = {"NVTE_SBHD_SBHD_SBHD", "NVTE_Paged_KV_SBHD_SBHD_SBHD"}
_BS3HD = {"NVTE_BS3HD", "NVTE_T3HD"}
_BSH3D = {"NVTE_BSH3D", "NVTE_TH3D"}
_BSHD_BS2HD = {"NVTE_BSHD_BS2HD", "NVTE_THD_T2HD"}
_BSHD_BSH2D = {"NVTE_BSHD_BSH2D", "NVTE_THD_TH2D"}
_BSHD_SEP = {
    "NVTE_BSHD_BSHD_BSHD",
    "NVTE_THD_THD_THD",
    "NVTE_THD_BSHD_BSHD",
    "NVTE_Paged_KV_BSHD_BSHD_BSHD",
    "NVTE_Paged_KV_THD_BSHD_BSHD",
}
_SBHD_BSHD_BSHD = {"NVTE_SBHD_BSHD_BSHD", "NVTE_Paged_KV_SBHD_BSHD_BSHD"}
_BSHD_SBHD_SBHD = {
    "NVTE_BSHD_SBHD_SBHD",
    "NVTE_THD_SBHD_SBHD",
    "NVTE_Paged_KV_BSHD_SBHD_SBHD",
    "NVTE_Paged_KV_THD_SBHD_SBHD",
}
_BHSD_SEP = {"NVTE_BHSD_BHSD_BHSD"}

_Stride = Tuple[int, int, int, int]


def generate_matrix_strides(
    b: int, h: int, s_q: int, s_kv: int, d: int, layout: object, matrix: str
) -> _Stride:
    """Return the BHSD stride for one data matrix. ``matrix`` in {Q, K, V, O}.

    ``h`` and ``d`` are the head count and head dim *of this matrix* (Q/O use the
    query head count and query/value head dim; K/V use the KV head count), exactly
    as the C++ callers pass them.
    """
    name = _name(layout)
    s = [0, 0, 0, 0]

    def sb_packed(mult: int, head_stride: int) -> _Stride:
        # S-major packed (SB3HD / SBH3D): batch fastest within a timestep.
        return (mult * h * d, head_stride, b * mult * h * d, 1)

    def sbhd_plain() -> _Stride:
        return (h * d, d, b * h * d, 1)

    def bs_packed(mult: int, head_stride: int, seq: int) -> _Stride:
        # B-major packed (BS3HD / BSH3D): sequence stride within a batch row.
        return (seq * mult * h * d, head_stride, mult * h * d, 1)

    def bshd_plain(seq: int) -> _Stride:
        return (seq * h * d, d, h * d, 1)

    is_q, is_k, is_v, is_o = (matrix == "Q", matrix == "K", matrix == "V", matrix == "O")

    if name in _SB3HD:
        return sb_packed(3, d) if (is_q or is_k or is_v) else sbhd_plain()
    if name in _SBH3D:
        return sb_packed(3, 3 * d) if (is_q or is_k or is_v) else sbhd_plain()
    if name in _SBHD_SB2HD:
        return sb_packed(2, d) if (is_k or is_v) else sbhd_plain()
    if name in _SBHD_SBH2D:
        return sb_packed(2, 2 * d) if (is_k or is_v) else sbhd_plain()
    if name in _SBHD_SEP:
        return sbhd_plain()
    if name in _BS3HD:
        return bs_packed(3, d, s_q) if (is_q or is_k or is_v) else bshd_plain(s_q)
    if name in _BSH3D:
        return bs_packed(3, 3 * d, s_q) if (is_q or is_k or is_v) else bshd_plain(s_q)
    if name in _BSHD_BS2HD:
        return bs_packed(2, d, s_kv) if (is_k or is_v) else bshd_plain(s_q)
    if name in _BSHD_BSH2D:
        return bs_packed(2, 2 * d, s_kv) if (is_k or is_v) else bshd_plain(s_q)
    if name in _BSHD_SEP:
        return bshd_plain(s_q) if (is_q or is_o) else bshd_plain(s_kv)
    if name in _SBHD_BSHD_BSHD:
        return bshd_plain(s_kv) if (is_k or is_v) else sbhd_plain()
    if name in _BSHD_SBHD_SBHD:
        return sbhd_plain() if (is_k or is_v) else bshd_plain(s_q)
    if name in _BHSD_SEP:
        if is_q or is_o:
            return (h * s_q * d, s_q * d, d, 1)
        return (h * s_kv * d, s_kv * d, d, 1)

    raise ValueError(f"generate_matrix_strides: unsupported qkv_layout {name!r}")


def qkvo_dims_strides(cfg: FusedAttnConfig, batch_size: Optional[int] = None):
    """Return ``{role: (dim, stride)}`` for Q, K, V, O from a derived config.

    Dense path only (no paged/ragged): matches the non-paged branch of
    ``create_graph_f16_fwd`` / ``create_graph_f16_bwd``. Q/O use ``num_attn_heads``
    and the query/value head dims; K/V use ``num_gqa_groups``. ``batch_size``
    defaults to the forward graph batch (``graph_batch_size_fwd``); the backward
    builder passes ``graph_batch_size_bwd``.
    """
    b = int(cfg.graph_batch_size_fwd if batch_size is None else batch_size)
    s_q = int(cfg.graph_max_seqlen_q)
    s_kv = int(cfg.graph_max_seqlen_kv)
    h = int(cfg.num_attn_heads)
    hg = int(cfg.num_gqa_groups)
    d_qk = int(cfg.head_dim_qk)
    d_v = int(cfg.head_dim_v)
    layout = cfg.qkv_layout
    return {
        "Q": ((b, h, s_q, d_qk), generate_matrix_strides(b, h, s_q, s_kv, d_qk, layout, "Q")),
        "K": ((b, hg, s_kv, d_qk), generate_matrix_strides(b, hg, s_q, s_kv, d_qk, layout, "K")),
        "V": ((b, hg, s_kv, d_v), generate_matrix_strides(b, hg, s_q, s_kv, d_v, layout, "V")),
        "O": ((b, h, s_q, d_v), generate_matrix_strides(b, h, s_q, s_kv, d_v, layout, "O")),
    }


def paged_kv_dims_strides(cfg: FusedAttnConfig):
    """Return ``{"K": (dim, stride), "V": (dim, stride)}`` for the paged KV containers.

    Mirrors the ``is_paged_kv`` branch of ``create_graph_f16_fwd``: K/V live in
    page containers dimensioned by ``num_pages`` / ``page_size`` (not batch /
    seqlen), and their strides come from ``generateMatrixStrides`` called with
    ``num_pages`` as the batch and the page sizes as the seq extents.
    """
    hg = int(cfg.num_gqa_groups)
    d_qk = int(cfg.head_dim_qk)
    d_v = int(cfg.head_dim_v)
    npk = int(cfg.num_pages_k)
    npv = int(cfg.num_pages_v)
    psk = int(cfg.page_size_k)
    psv = int(cfg.page_size_v)
    layout = cfg.qkv_layout
    return {
        "K": ((npk, hg, psk, d_qk), generate_matrix_strides(npk, hg, psk, psv, d_qk, layout, "K")),
        "V": ((npv, hg, psv, d_v), generate_matrix_strides(npv, hg, psk, psv, d_v, layout, "V")),
    }


class RaggedOffsetMultipliers(NamedTuple):
    """Per-tensor ragged-offset multipliers (port of the C++ struct).

    On the ``cu_seqlens``-direct path a token-unit ``cu_seqlens`` tensor is bound
    directly as the ragged offset; the multiplier recovers element offsets. For
    interleaved QKV (``3HD``/``H3D``) the K/V offsets scale the Q-side cu_seqlens
    (``kv_from_q``).
    """

    q: int
    k: int
    v: int
    o: int
    stats: int
    kv_from_q: bool


def ragged_offset_multipliers(cfg: FusedAttnConfig) -> RaggedOffsetMultipliers:
    """Port of ``RaggedOffsetMultipliers`` (utils.h)."""
    h = int(cfg.num_attn_heads)
    hg = int(cfg.num_gqa_groups)
    d_qk = int(cfg.head_dim_qk)
    d_v = int(cfg.head_dim_v)
    q, k, v, o, stats, kv_from_q = h * d_qk, hg * d_qk, hg * d_v, h * d_v, h, False
    group = get_qkv_layout_group(cfg.qkv_layout)
    if group in (QKVLayoutGroup.THREE_HD, QKVLayoutGroup.H3D):
        q = k = v = 3 * h * d_qk
        kv_from_q = True
    elif group in (QKVLayoutGroup.HD_2HD, QKVLayoutGroup.HD_H2D):
        k = v = 2 * hg * d_qk
    return RaggedOffsetMultipliers(q, k, v, o, stats, kv_from_q)


__all__ = [
    "generate_matrix_strides",
    "qkvo_dims_strides",
    "paged_kv_dims_strides",
    "ragged_offset_multipliers",
    "RaggedOffsetMultipliers",
]
