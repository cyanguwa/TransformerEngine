# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Stable UIDs for every logical fused-attention tensor.

This is a 1:1 Python mirror of ``common/fused_attn/fused_attn_uids.h``. cuDNN
graphs identify their inputs/outputs by an integer UID assigned via
``set_uid``; keying variant packs by these UIDs lets the Python builder feed
``graph.execute`` a plain ``{uid: ptr}`` map without owning any cuDNN tensor
objects.

Rules (identical to the C++ header):
  * Values are explicit and MUST stay stable and in lockstep with the C++ enum;
    never renumber an existing tensor. Append new tensors with the next unused
    value.
  * UIDs only need to be unique within a single graph. Forward and backward are
    distinct graphs, so a UID may appear in only one pass.
  * F16 and FP8 use separate enums because their tensor sets diverge. MXFP8
    reuses the FP8 graph path and therefore the FP8 UIDs.
"""

from enum import IntEnum


class FusedAttnUIDF16(IntEnum):
    """UIDs for the F16/BF16 arbitrary-seqlen graphs (forward and backward)."""

    # Core data tensors
    Q = 1
    K = 2
    V = 3
    O = 4
    dO = 5
    dQ = 6
    dK = 7
    dV = 8
    # Bias
    Bias = 9
    dBias = 10
    # Softmax statistics
    Stats = 11  # softmax LSE (forward S1 / backward stats)
    MaxLogit = 12  # forward S2, only when return_max_logit
    # Host scalar (attention scale), passed through the variant pack
    AttnScale = 13
    # Sequence lengths (cu_seqlens or materialized actual_seqlens)
    SeqQ = 14
    SeqKV = 15
    # Paged-KV page tables
    PageTableK = 16
    PageTableV = 17
    # Ragged offsets
    OffsetQ = 18
    OffsetK = 19
    OffsetV = 20
    OffsetO = 21
    OffsetStats = 22
    # Dropout RNG state
    DropoutSeed = 23
    DropoutOffset = 24
    # Learnable softmax offset (sink attention)
    SoftmaxOffset = 25
    dSoftmaxOffset = 26


class FusedAttnUIDFP8(IntEnum):
    """UIDs for the FP8 graphs (forward and backward). Also used by MXFP8."""

    # Core data tensors
    Q = 1
    K = 2
    V = 3
    O = 4
    dO = 5
    dQ = 6
    dK = 7
    dV = 8
    Stats = 9
    # Bias
    Bias = 10
    dBias = 11
    # Descales for inputs
    DescaleQ = 12
    DescaleK = 13
    DescaleV = 14
    DescaleO = 15
    DescaledO = 16
    DescaleS = 17
    DescaledP = 18
    # Scales
    ScaleO = 19
    ScaleS = 20
    ScaledP = 21
    ScaledQ = 22
    ScaledK = 23
    ScaledV = 24
    # Amaxes
    AmaxO = 25
    AmaxS = 26
    AmaxdP = 27
    AmaxdQ = 28
    AmaxdK = 29
    AmaxdV = 30
    # Transposed / f16 helper tensors (backward)
    Qt = 31
    Kt = 32
    dOf16 = 33
    dOt = 34
    DescaleQt = 35
    DescaleKt = 36
    DescaledOt = 37
    # Sequence lengths
    SeqQ = 38
    SeqKV = 39
    # Ragged offsets
    OffsetQ = 40
    OffsetK = 41
    OffsetV = 42
    OffsetO = 43
    OffsetStats = 44
    # Dropout RNG state
    DropoutSeed = 45
    DropoutOffset = 46
    # Learnable softmax offset (sink attention)
    SoftmaxOffset = 47
    dSoftmaxOffset = 48
    # Host scalar (attention scale), passed through the variant pack. Unlike the
    # C++ path (which keys attn_scale by shared_ptr), the Python builder binds it
    # by UID, so the FP8 graph needs one too.
    AttnScale = 49


__all__ = ["FusedAttnUIDF16", "FusedAttnUIDFP8"]
