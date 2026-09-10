# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuDNN support probe that plugs the real graph build into ``rules``.

``rules.select_fused_attn_backend`` applies TransformerEngine's gating and then,
"first rejection wins", asks cuDNN whether it can actually build the graph. That
last step *is* a graph build, so it lives here with the builders and is injected
into ``rules`` as a ``ProbeFn``.

``make_cudnn_probe`` returns a ``probe(cfg, pass_) -> reason`` closure over the
``cudnn`` module and a handle. A successful build optionally seeds a
:class:`~transformer_engine.common.fused_attn_py.cache.GraphCache`, so the
execute path reuses the very graph the probe built (they key on the same
``make_cache_key`` because the graph is a pure function of ``cfg``).

Coverage tracks the builders: F16/BF16 forward is probed for real; F16 backward
(Stage 3) and FP8/MXFP8 (Stages 6/7) are not yet built in Python, so their probe
returns ``""`` (treated as supported) and defers to the C++ path / later stages.
"""

from __future__ import annotations

from typing import Any, Optional

from .builders.f16 import build_f16_bwd_graph, build_f16_fwd_graph
from .cache import GraphCache, build_and_probe
from .config import FusedAttnConfig, Pass, _is_f16_dtype
from .rules import ProbeFn


def make_cudnn_probe(
    cudnn: Any,
    handle: Any,
    *,
    cudnn_version: Optional[int] = None,
    fwd_cache: Optional[GraphCache] = None,
    bwd_cache: Optional[GraphCache] = None,
) -> ProbeFn:
    """Return a ``ProbeFn`` that builds the real cuDNN graph to test support.

    ``fwd_cache`` / ``bwd_cache``, if given, are populated on a successful build
    for the matching pass so the execute path reuses the graph instead of
    rebuilding it.
    """
    not_supported = cudnn.cudnnGraphNotSupportedError
    builders = {Pass.Fwd: build_f16_fwd_graph, Pass.Bwd: build_f16_bwd_graph}
    caches = {Pass.Fwd: fwd_cache, Pass.Bwd: bwd_cache}

    def probe(cfg: FusedAttnConfig, pass_: Pass) -> str:
        if not _is_f16_dtype(cfg.qkv_dtype):
            return ""  # FP8/MXFP8 builders not ported yet (Stages 6/7).

        build_graph = builders[pass_]
        cache = caches[pass_]
        key = cfg.make_cache_key(pass_)
        if cache is not None and cache.get(key) is not None:
            return ""  # Already built and cached => supported.

        entry, reason = build_and_probe(
            lambda: build_graph(cudnn, handle, cfg, cudnn_version=cudnn_version),
            not_supported,
        )
        if entry is not None and cache is not None:
            cache.get_or_build(key, lambda: entry)
        return reason

    return probe


__all__ = ["make_cudnn_probe"]
