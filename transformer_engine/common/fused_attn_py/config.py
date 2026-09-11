# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Normalized fused-attention config: the Python port of the C++ decision core.

This module mirrors, one-to-one, the C++ ``FusedAttnConfig`` struct together with
its ``derive()`` and ``make_cache_key()`` methods
(``common/fused_attn/config_and_params.{h,cpp}``) and the pure layout/bucketing
helpers it depends on (``common/fused_attn/fused_attn.cpp`` and
``common/fused_attn/utils.cu``).

Design choices that keep the core framework-neutral and CPU-testable:

* **Enum inputs are normalized by name.** cuDNN/NVTE enums are only exposed
  through the per-framework C extensions (``transformer_engine_torch`` /
  ``transformer_engine_jax``) and their integer values are an implementation
  detail. We therefore accept the framework enum object, a bare string, or one
  of the small canonical enums defined here, and switch on the *name*. The
  dual-oracle test pins the names against ``tex`` so drift is caught.
* **Device / library facts are injected**, not queried, via ``RuntimeInfo``
  (``sm_arch``, ``cudnn_version``, ``cudnn_frontend_version``). ``derive()`` is
  thus a pure function of (config, runtime) and can be unit-tested with a fake
  runtime, exactly as the C++ ``derive()`` reads ``cudnnGetVersion()`` /
  ``cuda::sm_arch``.

Only the graph-building / backend-selection *decisions* are ported here. The
device pointer plumbing, ragged-offset multipliers, and the cuDNN support probe
itself live with the Python graph builders (stages 2+).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Canonical enum categories (values chosen here; only used internally + returned)
# ---------------------------------------------------------------------------
class QKVFormat(str, Enum):
    """Mirror of the ``NVTE_QKV_Format`` names this core reasons about."""

    SBHD = "NVTE_SBHD"
    BSHD = "NVTE_BSHD"
    THD = "NVTE_THD"
    BHSD = "NVTE_BHSD"
    SBHD_2BSHD = "NVTE_SBHD_2BSHD"
    BSHD_2SBHD = "NVTE_BSHD_2SBHD"
    THD_2BSHD = "NVTE_THD_2BSHD"
    THD_2SBHD = "NVTE_THD_2SBHD"
    NOT_SET = "NVTE_QKV_Format_NOT_SET"


class QKVLayoutGroup(str, Enum):
    """Mirror of ``NVTE_QKV_Layout_Group``."""

    HD_HD_HD = "NVTE_HD_HD_HD"
    H3D = "NVTE_H3D"
    HD_H2D = "NVTE_HD_H2D"
    HD_2HD = "NVTE_HD_2HD"
    THREE_HD = "NVTE_3HD"
    PAGED_KV_HD_HD_HD = "NVTE_Paged_KV_HD_HD_HD"
    SD_SD_SD = "NVTE_SD_SD_SD"


class FusedAttnBackend(str, Enum):
    """Mirror of ``NVTE_Fused_Attn_Backend`` (names match ``tex``)."""

    No_Backend = "NVTE_No_Backend"
    F16_max512_seqlen = "NVTE_F16_max512_seqlen"
    F16_arbitrary_seqlen = "NVTE_F16_arbitrary_seqlen"
    FP8 = "NVTE_FP8"


class ScaleDType(str, Enum):
    """Ragged-offset element type, mirror of ``DType::kInt32`` / ``kInt64``."""

    INT32 = "int32"
    INT64 = "int64"


class Pass(str, Enum):
    """Mirror of the C++ ``Pass`` enum."""

    Fwd = "fwd"
    Bwd = "bwd"


# Canonical dtype categories, keyed off the tail of the enum name so both
# NVTEDType (``kNVTEBFloat16``) and framework DType (``kBFloat16``) normalize the
# same way.
_FP8_DTYPES = frozenset({"FLOAT8E4M3", "FLOAT8E5M2"})
_F16_DTYPES = frozenset({"FLOAT16", "BFLOAT16"})

INT32_MAX = 2**31 - 1


# ---------------------------------------------------------------------------
# Name normalization helpers
# ---------------------------------------------------------------------------
def _name(value) -> str:
    """Return the enum member name for a framework enum, or the string as-is."""
    if value is None:
        return ""
    return getattr(value, "name", str(value))


def _canonical_dtype(value) -> str:
    """Normalize an NVTEDType/DType enum (or name) to e.g. ``BFLOAT16``."""
    name = _name(value).upper()
    # Strip the ``kNVTE`` / ``k`` prefixes: kNVTEBFloat16 / kBFloat16 -> BFLOAT16.
    if name.startswith("KNVTE"):
        name = name[len("KNVTE") :]
    elif name.startswith("K"):
        name = name[1:]
    return name


def _is_fp8_dtype(value) -> bool:
    return _canonical_dtype(value) in _FP8_DTYPES


def _is_f16_dtype(value) -> bool:
    return _canonical_dtype(value) in _F16_DTYPES


# ---------------------------------------------------------------------------
# Layout -> {format, q_format, kv_format, group} maps
# Ported verbatim from nvte_get_qkv_{format,layout_group} in fused_attn.cpp.
# Keyed by NVTE_QKV_Layout member name.
# ---------------------------------------------------------------------------
_LAYOUT_TO_GROUP = {
    # NVTE_3HD
    "NVTE_SB3HD": QKVLayoutGroup.THREE_HD,
    "NVTE_BS3HD": QKVLayoutGroup.THREE_HD,
    "NVTE_T3HD": QKVLayoutGroup.THREE_HD,
    # NVTE_H3D
    "NVTE_SBH3D": QKVLayoutGroup.H3D,
    "NVTE_BSH3D": QKVLayoutGroup.H3D,
    "NVTE_TH3D": QKVLayoutGroup.H3D,
    # NVTE_HD_2HD
    "NVTE_SBHD_SB2HD": QKVLayoutGroup.HD_2HD,
    "NVTE_BSHD_BS2HD": QKVLayoutGroup.HD_2HD,
    "NVTE_THD_T2HD": QKVLayoutGroup.HD_2HD,
    # NVTE_HD_H2D
    "NVTE_SBHD_SBH2D": QKVLayoutGroup.HD_H2D,
    "NVTE_BSHD_BSH2D": QKVLayoutGroup.HD_H2D,
    "NVTE_THD_TH2D": QKVLayoutGroup.HD_H2D,
    # NVTE_HD_HD_HD
    "NVTE_SBHD_SBHD_SBHD": QKVLayoutGroup.HD_HD_HD,
    "NVTE_BSHD_BSHD_BSHD": QKVLayoutGroup.HD_HD_HD,
    "NVTE_THD_THD_THD": QKVLayoutGroup.HD_HD_HD,
    "NVTE_SBHD_BSHD_BSHD": QKVLayoutGroup.HD_HD_HD,
    "NVTE_BSHD_SBHD_SBHD": QKVLayoutGroup.HD_HD_HD,
    "NVTE_THD_SBHD_SBHD": QKVLayoutGroup.HD_HD_HD,
    "NVTE_THD_BSHD_BSHD": QKVLayoutGroup.HD_HD_HD,
    # NVTE_Paged_KV_HD_HD_HD
    "NVTE_Paged_KV_BSHD_BSHD_BSHD": QKVLayoutGroup.PAGED_KV_HD_HD_HD,
    "NVTE_Paged_KV_SBHD_BSHD_BSHD": QKVLayoutGroup.PAGED_KV_HD_HD_HD,
    "NVTE_Paged_KV_THD_BSHD_BSHD": QKVLayoutGroup.PAGED_KV_HD_HD_HD,
    "NVTE_Paged_KV_BSHD_SBHD_SBHD": QKVLayoutGroup.PAGED_KV_HD_HD_HD,
    "NVTE_Paged_KV_SBHD_SBHD_SBHD": QKVLayoutGroup.PAGED_KV_HD_HD_HD,
    "NVTE_Paged_KV_THD_SBHD_SBHD": QKVLayoutGroup.PAGED_KV_HD_HD_HD,
    # NVTE_SD_SD_SD
    "NVTE_BHSD_BHSD_BHSD": QKVLayoutGroup.SD_SD_SD,
}

_LAYOUT_TO_FORMAT = {
    # NVTE_SBHD
    "NVTE_SB3HD": QKVFormat.SBHD,
    "NVTE_SBH3D": QKVFormat.SBHD,
    "NVTE_SBHD_SB2HD": QKVFormat.SBHD,
    "NVTE_SBHD_SBH2D": QKVFormat.SBHD,
    "NVTE_SBHD_SBHD_SBHD": QKVFormat.SBHD,
    "NVTE_Paged_KV_SBHD_SBHD_SBHD": QKVFormat.SBHD,
    # NVTE_BSHD
    "NVTE_BS3HD": QKVFormat.BSHD,
    "NVTE_BSH3D": QKVFormat.BSHD,
    "NVTE_BSHD_BS2HD": QKVFormat.BSHD,
    "NVTE_BSHD_BSH2D": QKVFormat.BSHD,
    "NVTE_BSHD_BSHD_BSHD": QKVFormat.BSHD,
    "NVTE_Paged_KV_BSHD_BSHD_BSHD": QKVFormat.BSHD,
    # NVTE_THD
    "NVTE_T3HD": QKVFormat.THD,
    "NVTE_TH3D": QKVFormat.THD,
    "NVTE_THD_T2HD": QKVFormat.THD,
    "NVTE_THD_TH2D": QKVFormat.THD,
    "NVTE_THD_THD_THD": QKVFormat.THD,
    # hybrid formats
    "NVTE_SBHD_BSHD_BSHD": QKVFormat.SBHD_2BSHD,
    "NVTE_Paged_KV_SBHD_BSHD_BSHD": QKVFormat.SBHD_2BSHD,
    "NVTE_BSHD_SBHD_SBHD": QKVFormat.BSHD_2SBHD,
    "NVTE_Paged_KV_BSHD_SBHD_SBHD": QKVFormat.BSHD_2SBHD,
    "NVTE_THD_BSHD_BSHD": QKVFormat.THD_2BSHD,
    "NVTE_Paged_KV_THD_BSHD_BSHD": QKVFormat.THD_2BSHD,
    "NVTE_THD_SBHD_SBHD": QKVFormat.THD_2SBHD,
    "NVTE_Paged_KV_THD_SBHD_SBHD": QKVFormat.THD_2SBHD,
    # NVTE_BHSD
    "NVTE_BHSD_BHSD_BHSD": QKVFormat.BHSD,
}

# Second-level maps for q_format / kv_format, keyed by the *format* produced above.
# Ported verbatim from nvte_get_q_format / nvte_get_kv_format.
_FORMAT_TO_Q_FORMAT = {
    QKVFormat.SBHD: QKVFormat.SBHD,
    QKVFormat.SBHD_2BSHD: QKVFormat.SBHD,
    QKVFormat.BSHD: QKVFormat.BSHD,
    QKVFormat.BSHD_2SBHD: QKVFormat.BSHD,
    QKVFormat.THD: QKVFormat.THD,
    QKVFormat.THD_2BSHD: QKVFormat.THD,
    QKVFormat.THD_2SBHD: QKVFormat.THD,
    QKVFormat.BHSD: QKVFormat.BHSD,
}

_FORMAT_TO_KV_FORMAT = {
    QKVFormat.SBHD: QKVFormat.SBHD,
    QKVFormat.BSHD_2SBHD: QKVFormat.SBHD,
    QKVFormat.THD_2SBHD: QKVFormat.SBHD,
    QKVFormat.BSHD: QKVFormat.BSHD,
    QKVFormat.SBHD_2BSHD: QKVFormat.BSHD,
    QKVFormat.THD_2BSHD: QKVFormat.BSHD,
    QKVFormat.THD: QKVFormat.THD,
    QKVFormat.BHSD: QKVFormat.BHSD,
}


def get_qkv_format(qkv_layout) -> QKVFormat:
    """Port of ``nvte_get_qkv_format``."""
    name = _name(qkv_layout)
    if name in ("", "NVTE_QKV_Layout_NOT_SET"):
        return QKVFormat.NOT_SET
    try:
        return _LAYOUT_TO_FORMAT[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported qkv_layout {name!r} in get_qkv_format.") from exc


def get_q_format(qkv_layout) -> QKVFormat:
    """Port of ``nvte_get_q_format``."""
    fmt = get_qkv_format(qkv_layout)
    if fmt is QKVFormat.NOT_SET:
        return QKVFormat.NOT_SET
    return _FORMAT_TO_Q_FORMAT[fmt]


def get_kv_format(qkv_layout) -> QKVFormat:
    """Port of ``nvte_get_kv_format``."""
    fmt = get_qkv_format(qkv_layout)
    if fmt is QKVFormat.NOT_SET:
        return QKVFormat.NOT_SET
    return _FORMAT_TO_KV_FORMAT[fmt]


def get_qkv_layout_group(qkv_layout) -> Optional[QKVLayoutGroup]:
    """Port of ``nvte_get_qkv_layout_group``."""
    name = _name(qkv_layout)
    if name in ("", "NVTE_QKV_Layout_NOT_SET"):
        return None
    try:
        return _LAYOUT_TO_GROUP[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported qkv_layout {name!r} in get_qkv_layout_group.") from exc


# ---------------------------------------------------------------------------
# Bucketing + ragged offset width. Ported verbatim from utils.cu.
# ---------------------------------------------------------------------------
def get_max_batch_size(batch_size: int) -> int:
    """Port of ``get_max_batch_size`` (utils.cu)."""
    if batch_size == 0:
        return 0
    log2_b = math.ceil(math.log2(batch_size))
    if log2_b <= 5:
        return 32
    if log2_b <= 9:
        return 2**log2_b
    return (batch_size + 511) // 512 * 512


def get_max_tokens(num_tokens: int) -> int:
    """Port of ``get_max_tokens`` (utils.cu)."""
    if num_tokens == 0:
        return 0
    log2_t = math.ceil(math.log2(num_tokens))
    if log2_t <= 10:
        return 1024
    if log2_t <= 15:
        return 2**log2_t
    return (num_tokens + 32767) // 32768 * 32768


def get_ragged_offset_dtype(
    layout_group: Optional[QKVLayoutGroup],
    num_attn_heads: int,
    num_gqa_groups: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    head_dim_qk: int,
    head_dim_v: int,
) -> ScaleDType:
    """Port of ``get_ragged_offset_dtype`` (utils.cu)."""
    offsets = [0, 0, 0, 0]
    if layout_group in (QKVLayoutGroup.HD_HD_HD, QKVLayoutGroup.PAGED_KV_HD_HD_HD):
        offsets[0] = num_attn_heads * head_dim_qk * max_seqlen_q
        offsets[1] = num_gqa_groups * head_dim_qk * max_seqlen_kv
        offsets[2] = num_gqa_groups * head_dim_v * max_seqlen_kv
    elif layout_group in (QKVLayoutGroup.THREE_HD, QKVLayoutGroup.H3D):
        offsets[0] = 3 * num_attn_heads * head_dim_qk * max_seqlen_q
        offsets[1] = offsets[0]
        offsets[2] = offsets[0]
    elif layout_group in (QKVLayoutGroup.HD_2HD, QKVLayoutGroup.HD_H2D):
        offsets[0] = num_attn_heads * head_dim_qk * max_seqlen_q
        offsets[1] = 2 * num_gqa_groups * head_dim_qk * max_seqlen_kv
        offsets[2] = offsets[1]
    offsets[3] = num_attn_heads * head_dim_qk * max_seqlen_q
    if max(offsets) > INT32_MAX:
        return ScaleDType.INT64
    return ScaleDType.INT32


# ---------------------------------------------------------------------------
# Injected device / library facts (mirror cudnnGetVersion / CUDNN_FRONTEND_VERSION
# / cuda::sm_arch, which derive() and the gating rules read directly in C++).
# ---------------------------------------------------------------------------
@dataclass
class RuntimeInfo:
    """Device and cuDNN facts the C++ path reads from the environment."""

    sm_arch: int
    cudnn_version: int  # runtime cudnnGetVersion(), e.g. 91500
    cudnn_frontend_version: int  # CUDNN_FRONTEND_VERSION, e.g. 12600
    cudnn_build_version: int  # compile-time CUDNN_VERSION, e.g. 91500
    device_id: int = 0


# Canonical enum-name constants used by derive() / gating.
_MASK_PADDING = {
    "NVTE_PADDING_MASK",
    "NVTE_PADDING_CAUSAL_MASK",
    "NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK",
}
_MASK_CAUSAL = {"NVTE_CAUSAL_MASK", "NVTE_PADDING_CAUSAL_MASK"}
_MASK_CAUSAL_BR = {
    "NVTE_CAUSAL_BOTTOM_RIGHT_MASK",
    "NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK",
}


@dataclass
class FusedAttnConfig:
    """Python mirror of the C++ ``FusedAttnConfig`` (input + derived fields).

    Field order and names follow ``config_and_params.h`` so the two stay easy to
    diff. Enum-typed fields accept the framework enum object, a string, or the
    canonical enum defined in this module; ``derive()`` reads them by name.
    """

    # --- Basic attention settings ---
    is_training: bool = True
    deterministic: bool = False
    cuda_graph: bool = False
    return_max_logit: bool = False
    attn_mask_type: object = "NVTE_NO_MASK"
    bias_type: object = "NVTE_NO_BIAS"
    window_size_left: int = -1
    window_size_right: int = -1
    bottom_right_diagonal: bool = True
    softmax_type: object = "NVTE_VANILLA_SOFTMAX"
    scaling_mode: object = "NVTE_DELAYED_TENSOR_SCALING"
    dropout: float = 0.0
    attn_scale: float = 1.0

    # --- Tensor types ---
    qkv_dtype: object = "kNVTEBFloat16"
    o_dtype: object = "kNVTEBFloat16"
    do_dtype: object = "kNVTEBFloat16"
    dqkv_dtype: object = "kNVTEBFloat16"

    # --- Tensor layouts ---
    qkv_layout: object = "NVTE_QKV_Layout_NOT_SET"
    o_format: object = "NVTE_QKV_Format_NOT_SET"
    do_format: object = "NVTE_QKV_Format_NOT_SET"
    dqkv_layout: object = "NVTE_QKV_Layout_NOT_SET"
    qkv_scale_inv_format: object = "NVTE_QKV_Format_NOT_SET"
    do_scale_inv_format: object = "NVTE_QKV_Format_NOT_SET"

    # --- Tensor dimensions ---
    batch_size: int = 0
    num_attn_heads: int = 0
    num_gqa_groups: int = 0
    head_dim_qk: int = 0
    head_dim_v: int = 0
    max_seqlen_q: int = 0
    max_seqlen_kv: int = 0
    num_tokens_q: int = 0
    num_tokens_kv: int = 0

    # --- Paged KV dimensions ---
    num_pages_k: int = 0
    num_pages_v: int = 0
    page_size_k: int = 0
    page_size_v: int = 0
    max_pages_per_seq_k: int = 0
    max_pages_per_seq_v: int = 0

    # --- Bias dimensions ---
    bias_batch_size: int = 0
    bias_num_heads: int = 0
    bias_seqlen_q: int = 0
    bias_seqlen_kv: int = 0

    # --- Support-query directions (mirror check_for_{forward,backward}_support) ---
    check_for_forward_support: bool = True
    check_for_backward_support: bool = True

    # --- Derived fields (filled by derive(); do not set by hand) ---
    is_derived: bool = field(default=False, init=False)
    device_id: int = field(default=-1, init=False)
    qkv_format: QKVFormat = field(default=QKVFormat.NOT_SET, init=False)
    q_format: QKVFormat = field(default=QKVFormat.NOT_SET, init=False)
    kv_format: QKVFormat = field(default=QKVFormat.NOT_SET, init=False)
    is_ragged_q: bool = field(default=False, init=False)
    is_ragged_kv: bool = field(default=False, init=False)
    is_paged_kv: bool = field(default=False, init=False)
    is_padding: bool = field(default=False, init=False)
    is_causal: bool = field(default=False, init=False)
    is_causal_bottom_right: bool = field(default=False, init=False)
    is_bias: bool = field(default=False, init=False)
    is_alibi: bool = field(default=False, init=False)
    is_softmax_offset: bool = field(default=False, init=False)
    is_dropout: bool = field(default=False, init=False)
    is_o_in_f16: bool = field(default=False, init=False)
    is_tensor_scaling: bool = field(default=False, init=False)
    is_mxfp8: bool = field(default=False, init=False)
    is_delayed_scaling_fwd: bool = field(default=False, init=False)
    is_delayed_scaling_bwd: bool = field(default=False, init=False)
    is_current_scaling_fwd: bool = field(default=False, init=False)
    is_current_scaling_bwd: bool = field(default=False, init=False)
    is_mxfp8_fwd: bool = field(default=False, init=False)
    is_mxfp8_bwd: bool = field(default=False, init=False)
    uses_cu_seqlens_directly: bool = field(default=False, init=False)
    bucketed_batch_size: int = field(default=0, init=False)
    bucketed_num_tokens_q: int = field(default=0, init=False)
    bucketed_num_tokens_kv: int = field(default=0, init=False)
    uses_ragged_graph: bool = field(default=False, init=False)
    uses_ragged_stats: bool = field(default=False, init=False)
    graph_batch_size_fwd: int = field(default=0, init=False)
    graph_batch_size_bwd: int = field(default=0, init=False)
    graph_max_seqlen_q: int = field(default=0, init=False)
    graph_max_seqlen_kv: int = field(default=0, init=False)
    needs_64bit_ragged_offset: bool = field(default=False, init=False)
    ragged_offset_type_fwd: ScaleDType = field(default=ScaleDType.INT32, init=False)
    ragged_offset_type_bwd: ScaleDType = field(default=ScaleDType.INT32, init=False)
    # bottom_right_diagonal is an input but derive() may flip it, so track it.

    # ------------------------------------------------------------------
    @classmethod
    def from_params(cls, params: object, **overrides) -> "FusedAttnConfig":
        """Build an (underived) config from any object exposing the same fields.

        Duck-typed and framework-neutral: for each *input* field of this config
        (the ones that mirror the C++ ``FusedAttnConfig`` struct), copy the
        same-named attribute off ``params`` if present, else keep the default.
        ``overrides`` win over ``params`` (e.g. paged-KV dims a framework's param
        object does not carry). Enum/dtype values are normalized by name at
        ``derive()`` time, so passing framework ``NVTE_*`` enums or bare strings
        both work. This is how the PyTorch ``FusedAttentionParams`` and the JAX
        ``FusedAttnParams`` (both C++-struct mirrors) feed the neutral core.
        """
        kwargs = {}
        for f in fields(cls):
            if not f.init:
                continue
            if f.name in overrides:
                kwargs[f.name] = overrides[f.name]
            elif hasattr(params, f.name):
                kwargs[f.name] = getattr(params, f.name)
        return cls(**kwargs)

    def derive(self, runtime: RuntimeInfo) -> "FusedAttnConfig":
        """Port of ``FusedAttnConfig::derive()`` (config_and_params.cpp)."""
        if self.is_derived:
            return self

        # Common attributes
        self.qkv_format = get_qkv_format(self.qkv_layout)
        self.q_format = get_q_format(self.qkv_layout)
        self.kv_format = get_kv_format(self.qkv_layout)
        layout_group = get_qkv_layout_group(self.qkv_layout)
        self.is_paged_kv = layout_group is QKVLayoutGroup.PAGED_KV_HD_HD_HD
        self.is_ragged_q = self.q_format is QKVFormat.THD
        self.is_ragged_kv = self.kv_format is QKVFormat.THD

        mask = _name(self.attn_mask_type)
        self.is_padding = mask in _MASK_PADDING
        self.is_causal = mask in _MASK_CAUSAL
        self.is_causal_bottom_right = mask in _MASK_CAUSAL_BR
        if self.is_causal_bottom_right and not self.bottom_right_diagonal:
            self.bottom_right_diagonal = True
        if self.is_causal and self.bottom_right_diagonal:
            self.bottom_right_diagonal = False
        has_window = self.window_size_left != -1 or self.window_size_right != -1
        if not self.is_causal and not self.is_causal_bottom_right and not has_window:
            self.bottom_right_diagonal = False

        bias = _name(self.bias_type)
        self.is_bias = bias == "NVTE_POST_SCALE_BIAS"
        self.is_alibi = bias == "NVTE_ALIBI"
        self.is_softmax_offset = _name(self.softmax_type) != "NVTE_VANILLA_SOFTMAX"
        self.is_dropout = self.is_training and self.dropout != 0.0

        # FP8 recipe
        is_o_in_fp8 = _is_fp8_dtype(self.o_dtype)
        is_dqkv_in_fp8 = _is_fp8_dtype(self.dqkv_dtype)
        self.is_o_in_f16 = _is_f16_dtype(self.o_dtype)
        is_dqkv_in_f16 = _is_f16_dtype(self.dqkv_dtype)
        self.is_tensor_scaling = _name(self.scaling_mode) == "NVTE_DELAYED_TENSOR_SCALING"
        self.is_mxfp8 = _name(self.scaling_mode) == "NVTE_MXFP8_1D_SCALING"
        self.is_delayed_scaling_fwd = self.is_tensor_scaling and is_o_in_fp8
        self.is_delayed_scaling_bwd = self.is_tensor_scaling and is_dqkv_in_fp8
        self.is_current_scaling_fwd = self.is_tensor_scaling and self.is_o_in_f16
        self.is_current_scaling_bwd = self.is_tensor_scaling and is_dqkv_in_f16
        self.is_mxfp8_fwd = self.is_mxfp8 and self.is_o_in_f16
        self.is_mxfp8_bwd = self.is_mxfp8 and is_dqkv_in_f16

        # cu_seqlens vs actual_seqlens
        is_fp8_dtype = _is_fp8_dtype(self.qkv_dtype)
        min_frontend_version = 12600 if is_fp8_dtype else 12500
        min_cudnn_version = 92500 if is_fp8_dtype else 92400
        self.uses_cu_seqlens_directly = (
            runtime.cudnn_frontend_version >= min_frontend_version
            and runtime.cudnn_build_version >= min_cudnn_version
            and runtime.cudnn_version >= min_cudnn_version
            and not self.is_dropout
        )

        # Bucket batch size and token counts
        self.bucketed_batch_size = (
            get_max_batch_size(self.batch_size) if (self.is_ragged_q or self.is_ragged_kv) else 0
        )
        self.bucketed_num_tokens_q = get_max_tokens(self.num_tokens_q) if self.is_ragged_q else 0
        self.bucketed_num_tokens_kv = get_max_tokens(self.num_tokens_kv) if self.is_ragged_kv else 0

        # Ragged (TH1) vs dense (BHS1) graphs and stats
        self.uses_ragged_graph = (
            runtime.cudnn_version >= 90600 and runtime.sm_arch >= 90 and runtime.sm_arch != 120
        )
        self.uses_ragged_stats = self.is_ragged_q and self.uses_ragged_graph
        buckets_the_batch = (self.is_ragged_q or self.is_ragged_kv) and self.uses_ragged_graph
        self.graph_batch_size_fwd = (
            self.bucketed_batch_size
            if (buckets_the_batch and not self.uses_cu_seqlens_directly)
            else self.batch_size
        )
        self.graph_batch_size_bwd = (
            self.bucketed_batch_size if buckets_the_batch else self.batch_size
        )
        self.graph_max_seqlen_q = (
            self.bucketed_num_tokens_q
            if (self.is_ragged_q and self.uses_ragged_graph)
            else self.max_seqlen_q
        )
        self.graph_max_seqlen_kv = (
            self.bucketed_num_tokens_kv
            if (self.is_ragged_kv and self.uses_ragged_graph)
            else self.max_seqlen_kv
        )

        # Ragged offset widths
        self.needs_64bit_ragged_offset = (self.is_ragged_q or self.is_ragged_kv) and (
            get_ragged_offset_dtype(
                layout_group,
                self.num_attn_heads,
                self.num_gqa_groups,
                self.max_seqlen_q,
                self.max_seqlen_kv,
                self.head_dim_qk,
                self.head_dim_v,
            )
            is ScaleDType.INT64
        )
        wide = ScaleDType.INT64 if runtime.cudnn_version >= 90500 else ScaleDType.INT32
        self.ragged_offset_type_fwd = ScaleDType.INT32 if self.uses_cu_seqlens_directly else wide
        self.ragged_offset_type_bwd = wide

        self.device_id = runtime.device_id
        self.is_derived = True
        return self

    # ------------------------------------------------------------------
    def check_derived(self) -> None:
        if not self.is_derived:
            raise RuntimeError(
                "FusedAttnConfig's derived fields are not set. Call derive() first."
            )

    # ------------------------------------------------------------------
    def make_cache_key(self, pass_: Pass) -> Tuple:
        """Port of ``FusedAttnConfig::make_cache_key()``.

        Returns a hashable tuple of exactly the fields the C++ ``operator<``
        keys on, after the same normalization the C++ version applies. Two
        configs that share a cached cuDNN graph produce an equal key.
        """
        self.check_derived()

        # Start from the keyed fields (same order as C++ operator<).
        max_seqlen_q = self.graph_max_seqlen_q
        max_seqlen_kv = self.graph_max_seqlen_kv
        num_tokens_q = 0
        num_tokens_kv = 0
        batch_size = self.batch_size
        if (self.is_ragged_q or self.is_ragged_kv) and self.uses_ragged_graph:
            batch_size = (
                self.graph_batch_size_fwd if pass_ is Pass.Fwd else self.graph_batch_size_bwd
            )
        attn_scale = 1.0
        cuda_graph = False
        deterministic = self.deterministic
        return_max_logit = self.return_max_logit
        do_dtype = _canonical_dtype(self.do_dtype)
        dqkv_dtype = _canonical_dtype(self.dqkv_dtype)
        do_format = _name(self.do_format)
        dqkv_layout = _name(self.dqkv_layout)
        do_scale_inv_format = _name(self.do_scale_inv_format)

        if pass_ is Pass.Fwd:
            do_dtype = "BFLOAT16"
            dqkv_dtype = "BFLOAT16"
            do_format = "NVTE_QKV_Format_NOT_SET"
            dqkv_layout = "NVTE_QKV_Layout_NOT_SET"
            do_scale_inv_format = "NVTE_QKV_Format_NOT_SET"
            deterministic = False
        else:
            return_max_logit = False

        return (
            self.is_training,
            deterministic,
            cuda_graph,
            return_max_logit,
            _name(self.attn_mask_type),
            _name(self.bias_type),
            self.window_size_left,
            self.window_size_right,
            self.bottom_right_diagonal,
            _name(self.softmax_type),
            _name(self.scaling_mode),
            self.dropout,
            attn_scale,
            _canonical_dtype(self.qkv_dtype),
            _canonical_dtype(self.o_dtype),
            do_dtype,
            dqkv_dtype,
            _name(self.qkv_layout),
            _name(self.o_format),
            do_format,
            dqkv_layout,
            _name(self.qkv_scale_inv_format),
            do_scale_inv_format,
            batch_size,
            self.num_attn_heads,
            self.num_gqa_groups,
            self.head_dim_qk,
            self.head_dim_v,
            max_seqlen_q,
            max_seqlen_kv,
            num_tokens_q,
            num_tokens_kv,
            self.num_pages_k,
            self.num_pages_v,
            self.page_size_k,
            self.page_size_v,
            self.max_pages_per_seq_k,
            self.max_pages_per_seq_v,
            self.bias_batch_size,
            self.bias_num_heads,
            self.bias_seqlen_q,
            self.bias_seqlen_kv,
            self.device_id,
        )
