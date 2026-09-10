# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Serialize a built cuDNN graph into a framework-neutral execution ``Plan``.

The migration's builder->executor contract is *data*, not code: a builder
produces a :class:`Plan` -- the serialized cuDNN graph blob plus the workspace
size, a content hash, the frontend version it was serialized with, and the
ordered input/output UID lists -- and any executor (the JAX blob FFI handler,
which deserializes the blob and binds a ``{uid: ptr}`` variant pack, or a future
PyTorch ``torch.library`` op) consumes it. This mirrors the ``score_mod`` path
already wired through ``jax/cpp_extensions/flex_attention.py`` and the C++
``FusedAttnScoreMod*Handler`` deserialize/execute handlers.

Kept dependency-free (no ``numpy`` / framework imports) so it stays part of the
neutral core and is unit-testable on CPU with a fake ``cudnn``; the JAX lowering
converts the plain lists here into the FFI attribute arrays it needs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .cache import GraphEntry

# Roles that are graph *outputs*; every other declared role is an input. Forward
# emits O and "Stats"; backward's "stats" (lowercase) is an *input*, and it emits
# dQ/dK/dV plus dBias (when computable).
_OUTPUT_ROLES = frozenset({"O", "Stats", "dQ", "dK", "dV", "dBias"})


@dataclass(frozen=True)
class Plan:
    """A serialized cuDNN graph plus everything an executor needs to run it.

    ``input_uids`` / ``output_uids`` are ascending-by-UID lists so the ordering
    is deterministic across builds; ``role_uids`` keeps the human-readable role
    names for the framework glue to map its buffers onto UIDs.
    """

    serialized_graph: bytes
    graph_hash: Tuple[int, int]
    cudnn_frontend_version: int
    workspace_size: int
    input_uids: List[int]
    output_uids: List[int]
    role_uids: Dict[str, int]


def encode_cudnn_frontend_version(version: str) -> int:
    """Encode a ``major.minor.patch`` frontend version as ``M*10000+m*100+p``.

    Port of ``_encode_cudnn_frontend_version`` in ``flex_attention.py``; the C++
    executor compares this against its build-time ``CUDNN_FRONTEND_VERSION``.
    """
    public_version = version.split("+", 1)[0].split("-", 1)[0]
    parts = public_version.split(".")
    if len(parts) < 3:
        raise RuntimeError(f"Could not parse cuDNN frontend Python version: {version!r}.")
    major, minor, patch = (int(part) for part in parts[:3])
    return major * 10000 + minor * 100 + patch


def graph_hash(serialized_graph: bytes) -> Tuple[int, int]:
    """Two signed 64-bit halves of the blob's SHA-256 (matches flex_attention)."""
    digest = hashlib.sha256(serialized_graph).digest()
    return (
        int.from_bytes(digest[0:8], byteorder="little", signed=True),
        int.from_bytes(digest[8:16], byteorder="little", signed=True),
    )


def _frontend_version(cudnn: Any, override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    version_string = getattr(cudnn, "__version__", None)
    if version_string is None:
        raise RuntimeError("cuDNN frontend Python package does not expose __version__.")
    return encode_cudnn_frontend_version(version_string)


def serialize_entry(
    cudnn: Any,
    entry: GraphEntry,
    *,
    cudnn_frontend_version: Optional[int] = None,
) -> Plan:
    """Serialize a finalized :class:`GraphEntry` into a :class:`Plan`.

    ``entry`` must already be finalized (``build_plans`` done) and carry the
    ``uids`` map assigned by the builder. Roles are split into inputs/outputs by
    :data:`_OUTPUT_ROLES`. The graph blob is produced with ``graph.serialize()``
    -- the same call the flex ``score_mod`` path uses.
    """
    if not entry.uids:
        raise ValueError("serialize_entry: GraphEntry has no UIDs; build with UID keying first.")

    inputs: List[int] = []
    outputs: List[int] = []
    for role, uid in entry.uids.items():
        (outputs if role in _OUTPUT_ROLES else inputs).append(int(uid))
    inputs.sort()
    outputs.sort()

    serialized = bytes(entry.graph.serialize())
    return Plan(
        serialized_graph=serialized,
        graph_hash=graph_hash(serialized),
        cudnn_frontend_version=_frontend_version(cudnn, cudnn_frontend_version),
        workspace_size=int(entry.workspace_size),
        input_uids=inputs,
        output_uids=outputs,
        role_uids=dict(entry.uids),
    )


__all__ = ["Plan", "serialize_entry", "encode_cudnn_frontend_version", "graph_hash"]
