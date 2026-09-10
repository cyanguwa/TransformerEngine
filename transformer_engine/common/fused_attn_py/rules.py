# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused-attention backend selection: the Python port of the C++ gating rules.

This mirrors ``nvte_get_fused_attn_backend_v2`` in
``common/fused_attn/fused_attn.cpp``. The structure is identical: derive the
config, then apply TransformerEngine's version/arch/config rejections, then run
cuDNN's own support check ("first rejection wins").

The cuDNN support check builds the real graph and therefore lives with the
Python graph builders. It is injected here as ``probe`` so the pure-Python
TE-gating logic can be exercised on CPU:

    probe(cfg, pass_) -> str   # "" if cuDNN supports it, else the reason

When ``probe`` is None the cuDNN step is skipped and the verdict reflects only
the TE gating (useful for unit tests and for callers that run the probe
themselves). The dual-oracle test compares this verdict against the C++
``tex.get_fused_attn_backend`` for a config sweep, so the two never diverge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .config import (
    FusedAttnConfig,
    FusedAttnBackend,
    Pass,
    QKVFormat,
    RuntimeInfo,
    _is_f16_dtype,
    _is_fp8_dtype,
    _name,
    get_qkv_format,
)

ProbeFn = Callable[[FusedAttnConfig, Pass], str]

_FP8_OK_FORMATS = (QKVFormat.BSHD, QKVFormat.SBHD, QKVFormat.BHSD, QKVFormat.THD)


@dataclass
class Verdict:
    """Result of backend selection: the chosen backend and, on rejection, why."""

    backend: FusedAttnBackend
    reason: str = ""

    @property
    def supported(self) -> bool:
        return self.backend is not FusedAttnBackend.No_Backend


def _reject(reason: str) -> Verdict:
    return Verdict(FusedAttnBackend.No_Backend, reason)


def _is_set(layout) -> bool:
    return _name(layout) not in ("", "NVTE_QKV_Layout_NOT_SET")


def select_fused_attn_backend(
    cfg: FusedAttnConfig,
    runtime: RuntimeInfo,
    probe: Optional[ProbeFn] = None,
) -> Verdict:
    """Port of ``nvte_get_fused_attn_backend_v2``. First rejection wins."""
    if not cfg.is_derived:
        cfg.derive(runtime)

    cudnn = runtime.cudnn_version
    sm = runtime.sm_arch

    # THD + 64-bit ragged offsets require cuDNN >= 9.5
    if cfg.needs_64bit_ragged_offset and cudnn < 90500:
        return _reject(
            "This config requires 64-bit ragged offsets, which is only supported by cuDNN >= 9.5."
        )

    # THD input requires a padding mask
    if (cfg.is_ragged_q or cfg.is_ragged_kv) and not cfg.is_padding:
        return _reject(
            "THD format requires PADDING / PADDING_CAUSAL / PADDING_CAUSAL_BOTTOM_RIGHT mask."
        )

    if (cfg.is_ragged_q and cfg.num_tokens_q == 0) or (
        cfg.is_ragged_kv and cfg.num_tokens_kv == 0
    ):
        return _reject(
            "THD format requires num_tokens_q / num_tokens_kv to be set for the ragged inputs."
        )

    # Paged KV requires a padding mask
    if cfg.is_paged_kv and not cfg.is_padding:
        return _reject(
            "Paged KV requires PADDING / PADDING_CAUSAL / PADDING_CAUSAL_BOTTOM_RIGHT mask."
        )

    # Paged KV requires cache dimensions to be set
    if cfg.is_paged_kv and (
        cfg.num_pages_k == 0
        or cfg.num_pages_v == 0
        or cfg.page_size_k == 0
        or cfg.page_size_v == 0
        or cfg.max_pages_per_seq_k == 0
        or cfg.max_pages_per_seq_v == 0
    ):
        return _reject(
            "Paged KV requires num_pages, page_size and max_pages_per_seq to be set for both K "
            "and V."
        )

    # Fused-attention does not support pre-scale bias
    if _name(cfg.bias_type) == "NVTE_PRE_SCALE_BIAS":
        return _reject("Fused attention does not support pre-scale bias.")

    is_fp8 = _is_fp8_dtype(cfg.qkv_dtype)
    is_f16_or_bf16 = _is_f16_dtype(cfg.qkv_dtype)

    def each_pass(verdict: ProbeFn) -> str:
        if cfg.check_for_forward_support:
            reason = verdict(cfg, Pass.Fwd)
            if reason:
                return reason
        if cfg.is_training and cfg.check_for_backward_support:
            reason = verdict(cfg, Pass.Bwd)
            if reason:
                return reason
        return ""

    # F16/BF16 support checks
    if is_f16_or_bf16:
        if cfg.is_ragged_q and cfg.is_ragged_kv and sm < 90:
            return _reject("F16/BF16 fused attention with THD format requires sm90 or later.")
        if (cfg.is_ragged_q or cfg.is_ragged_kv) and sm < 90 and cudnn < 90700:
            return _reject(
                "F16/BF16 fused attention with a ragged Q and non-ragged KV requires cuDNN "
                ">= 9.7 before sm90."
            )
        has_sliding_window = not (
            cfg.window_size_left == -1
            and (cfg.window_size_right == -1 or cfg.window_size_right == 0)
        )
        if (
            cfg.is_causal_bottom_right
            and has_sliding_window
            and cfg.max_seqlen_q != cfg.max_seqlen_kv
            and cudnn <= 90700
            and sm >= 100
        ):
            return _reject(
                "Known cuDNN <= 9.7.0 issue with bottom-right causal masking and a sliding "
                "window for cross-attention on sm100. Please upgrade cuDNN."
            )
        if (
            cudnn <= 91500
            and cfg.is_training
            and cfg.qkv_format in (QKVFormat.BSHD, QKVFormat.SBHD)
            and (cfg.max_seqlen_kv % 128 != 0)
            and cfg.cuda_graph
            and not cfg.is_padding
        ):
            return _reject("Known cuDNN <= 9.15 issue with CUDA graph. Please upgrade cuDNN.")
        if (
            cfg.is_training
            and cfg.check_for_backward_support
            and cfg.uses_ragged_stats
            and _name(cfg.softmax_type) == "NVTE_LEARNABLE_SOFTMAX"
            and cudnn < 92600
        ):
            return _reject(
                "Known cuDNN < 9.26.0 issue with THD learnable softmax backward. Please upgrade "
                "cuDNN."
            )

        # Run cuDNN support checks
        if probe is not None:
            cudnn_reason = each_pass(probe)
            if cudnn_reason:
                return _reject(cudnn_reason)
        return Verdict(FusedAttnBackend.F16_arbitrary_seqlen)

    # FP8 support checks
    if is_fp8:
        if cfg.return_max_logit:
            return _reject("FP8 fused attention does not support return_max_logit=True.")
        if cfg.qkv_format not in _FP8_OK_FORMATS:
            return _reject(
                "FP8 fused attention supports BSHD/SBHD/BHSD/THD formats, found "
                f"{cfg.qkv_format.value}."
            )
        if cfg.is_training and cfg.check_for_backward_support and _is_set(cfg.dqkv_layout):
            dqkv_format = get_qkv_format(cfg.dqkv_layout)
            if dqkv_format not in _FP8_OK_FORMATS:
                return _reject(
                    "FP8 fused attention supports BSHD/SBHD/BHSD/THD gradient formats, found "
                    f"{dqkv_format.value}."
                )
        if cfg.qkv_format is QKVFormat.THD:
            if cudnn < 92300:
                return _reject(
                    "FP8 fused attention with THD format requires cuDNN 9.23.0 or later!"
                )
            if cfg.is_training and cfg.check_for_backward_support and sm < 100:
                return _reject(
                    "FP8 fused attention with THD format supports backward on sm100+ only!"
                )
            if (
                cfg.is_training
                and cfg.check_for_backward_support
                and _name(cfg.softmax_type) != "NVTE_VANILLA_SOFTMAX"
                and cudnn < 92600
            ):
                return _reject(
                    "FP8 fused attention with THD format and a sink token requires cuDNN 9.26.0 "
                    "or later for backward!"
                )
            if sm >= 100 and (cfg.head_dim_qk > 128 or cfg.head_dim_v > 128):
                return _reject(
                    "FP8 fused attention with THD format supports head dimensions up to 128 on "
                    "sm100+ only!"
                )
        if cfg.is_bias:
            return _reject("FP8 fused attention does not support pre/post_scale_bias yet!")
        if cfg.is_alibi:
            return _reject("FP8 fused attention does not support ALiBi yet!")
        recipe_reason = (
            "FP8 fused attention only supports FP8DelayedScaling or FP8CurrentScaling or MXFP8 "
            "recipes!"
        )
        if cfg.check_for_forward_support and not (
            cfg.is_delayed_scaling_fwd or cfg.is_current_scaling_fwd or cfg.is_mxfp8_fwd
        ):
            return _reject(recipe_reason)
        if (
            cfg.is_training
            and cfg.check_for_backward_support
            and not (cfg.is_delayed_scaling_bwd or cfg.is_current_scaling_bwd or cfg.is_mxfp8_bwd)
        ):
            return _reject(recipe_reason)
        if cfg.is_mxfp8 and cudnn < 92100:
            return _reject("MXFP8 fused attention requires cuDNN 9.21.0 or later!")

        # Run cuDNN support checks
        if probe is not None:
            cudnn_reason = each_pass(probe)
            if cudnn_reason:
                return _reject(cudnn_reason)
        return Verdict(FusedAttnBackend.FP8)

    # Unsupported dtype
    return _reject(f"Unsupported QKV dtype qkv_dtype={_name(cfg.qkv_dtype)} .")


__all__ = ["Verdict", "ProbeFn", "select_fused_attn_backend"]
