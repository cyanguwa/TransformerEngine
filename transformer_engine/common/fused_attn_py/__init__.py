# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Framework-agnostic Python fused-attention backend.

This package is the Python home of the fused-attention path that is being
migrated off the cuDNN-frontend C++ API (``transformer_engine/common/fused_attn``)
and onto the cuDNN Python graph API (``import cudnn``). It is intentionally
framework-neutral: both the PyTorch and JAX extensions consume it, mirroring how
``transformer_engine/common`` already hosts framework-agnostic helpers.

Module layout (populated incrementally, one migration stage per module):

* ``uids``     -- stable per-tensor UIDs, a 1:1 mirror of the C++
                  ``common/fused_attn/fused_attn_uids.h``. Both the C++ builders
                  and the Python builders key their cuDNN variant packs by these
                  values, so the two agree on which pointer feeds which slot.
* ``config``   -- (stage 1) the normalized ``FusedAttnConfig`` dataclass and its
                  ``derive()`` / cache-key logic, ported from
                  ``common/fused_attn/config_and_params.{h,cpp}``.
* ``rules``    -- (stage 1) backend selection, ported from
                  ``select_fused_attn_backend`` in ``common/fused_attn/fused_attn.cpp``.
* ``builders`` -- (stages 2+) cuDNN Python graph builders for F16, then FP8/MXFP8.

Correctness during the migration is enforced by a dual-oracle test: the Python
``rules`` verdict is compared against the C++ ``tex.get_fused_attn_backend`` for a
sweep of configs, so the two never silently diverge.
"""

from .uids import FusedAttnUIDF16, FusedAttnUIDFP8
from .config import (
    FusedAttnConfig,
    FusedAttnBackend,
    QKVFormat,
    QKVLayoutGroup,
    Pass,
    RuntimeInfo,
)
from .rules import Verdict, select_fused_attn_backend

__all__ = [
    "FusedAttnUIDF16",
    "FusedAttnUIDFP8",
    "FusedAttnConfig",
    "FusedAttnBackend",
    "QKVFormat",
    "QKVLayoutGroup",
    "Pass",
    "RuntimeInfo",
    "Verdict",
    "select_fused_attn_backend",
]
