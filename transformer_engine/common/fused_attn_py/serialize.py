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
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .cache import GraphEntry
from .config import FusedAttnConfig, Pass

# Roles that are graph *outputs*; every other declared (non-scalar) role is an
# input. Forward emits O and "Stats"; backward's "stats" (lowercase) is an
# *input*, and it emits dQ/dK/dV plus dBias (when computable). FP8 forward also
# emits the amax outputs (Amax_s / Amax_o); FP8 backward emits dQ/dK/dV amaxes.
# The FP8 descale/scale tensors are ordinary device *inputs*, so they are not
# listed here.
_OUTPUT_ROLES = frozenset(
    {
        "O",
        "Stats",
        "dQ",
        "dK",
        "dV",
        "dBias",
        "AmaxS",
        "AmaxO",
        "AmaxdP",
        "AmaxdQ",
        "AmaxdK",
        "AmaxdV",
    }
)

# Pass-by-value scalar roles: these are bound to the graph as host scalars, not
# device buffers, so the executor carries them in scalar_uids/values (16-byte
# packed rows), never in input_uids. Mirrors the score_mod scalar mechanism.
_SCALAR_ROLES = frozenset({"attn_scale"})

# cuDNN caps a pass-by-value scalar at 16 bytes (one packed row).
_SCALAR_ROW_BYTES = 16


@dataclass(frozen=True)
class Plan:
    """A serialized cuDNN graph plus everything an executor needs to run it.

    ``input_uids`` / ``output_uids`` / ``scalar_uids`` are ascending-by-UID lists
    so the ordering is deterministic across builds; the framework glue must feed
    the FFI its device operands in ``input_uids`` order (the executor zips them
    positionally). ``scalar_sizes`` and ``scalar_values`` (a flat ``16 * N``-byte
    buffer of packed rows) accompany ``scalar_uids``. ``role_uids`` keeps the
    human-readable role names for the glue to map its buffers onto UIDs.
    """

    serialized_graph: bytes
    graph_hash: Tuple[int, int]
    cudnn_frontend_version: int
    workspace_size: int
    input_uids: List[int]
    output_uids: List[int]
    scalar_uids: List[int]
    scalar_sizes: List[int]
    scalar_values: bytes
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


def pack_scalar_f32(value: float) -> bytes:
    """Pack an FP32 pass-by-value scalar (e.g. attn_scale) as little-endian bytes."""
    return struct.pack("<f", float(value))


def _pack_scalar_rows(values: List[bytes]) -> Tuple[List[int], bytes]:
    """Return ``(sizes, flat 16*N-byte buffer)`` for the ordered scalar ``values``."""
    sizes: List[int] = []
    packed = bytearray(_SCALAR_ROW_BYTES * len(values))
    for i, raw in enumerate(values):
        if len(raw) > _SCALAR_ROW_BYTES:
            raise ValueError(f"pass-by-value scalar exceeds {_SCALAR_ROW_BYTES} bytes: {len(raw)}.")
        sizes.append(len(raw))
        packed[i * _SCALAR_ROW_BYTES : i * _SCALAR_ROW_BYTES + len(raw)] = raw
    return sizes, bytes(packed)


def serialize_entry(
    cudnn: Any,
    entry: GraphEntry,
    *,
    cudnn_frontend_version: Optional[int] = None,
    scalar_values: Optional[Dict[str, bytes]] = None,
) -> Plan:
    """Serialize a finalized :class:`GraphEntry` into a :class:`Plan`.

    ``entry`` must already be finalized (``build_plans`` done) and carry the
    ``uids`` map assigned by the builder. Roles are split three ways: pass-by-value
    scalars (:data:`_SCALAR_ROLES`), graph outputs (:data:`_OUTPUT_ROLES`), and
    everything else as device inputs. ``scalar_values`` maps each present scalar
    role to its packed host bytes (e.g. ``{"attn_scale": pack_scalar_f32(s)}``);
    it is required exactly for the scalar roles the graph declares. The blob is
    produced with ``graph.serialize()`` -- the same call the flex ``score_mod``
    path uses.
    """
    if not entry.uids:
        raise ValueError("serialize_entry: GraphEntry has no UIDs; build with UID keying first.")
    scalar_values = scalar_values or {}

    inputs: List[int] = []
    outputs: List[int] = []
    scalar_roles: List[Tuple[int, str]] = []
    for role, uid in entry.uids.items():
        if role in _SCALAR_ROLES:
            scalar_roles.append((int(uid), role))
        elif role in _OUTPUT_ROLES:
            outputs.append(int(uid))
        else:
            inputs.append(int(uid))
    inputs.sort()
    outputs.sort()
    scalar_roles.sort()  # ascending by UID

    scalar_uids: List[int] = []
    ordered_scalar_bytes: List[bytes] = []
    for uid, role in scalar_roles:
        if role not in scalar_values:
            raise ValueError(f"serialize_entry: missing scalar value for pass-by-value '{role}'.")
        scalar_uids.append(uid)
        ordered_scalar_bytes.append(scalar_values[role])
    scalar_sizes, packed_scalars = _pack_scalar_rows(ordered_scalar_bytes)

    serialized = bytes(entry.graph.serialize())
    return Plan(
        serialized_graph=serialized,
        graph_hash=graph_hash(serialized),
        cudnn_frontend_version=_frontend_version(cudnn, cudnn_frontend_version),
        workspace_size=int(entry.workspace_size),
        input_uids=inputs,
        output_uids=outputs,
        scalar_uids=scalar_uids,
        scalar_sizes=scalar_sizes,
        scalar_values=packed_scalars,
        role_uids=dict(entry.uids),
    )


def ordered_input_operands(plan: "Plan", buffers: Dict[str, Any]) -> List[Any]:
    """Order ``buffers`` (role -> value) to match ``plan.input_uids`` positionally.

    The executor zips the FFI operands against ``input_uids`` in order, so device
    operands must be supplied in exactly that order. Pass-by-value scalars travel
    in the attributes, not as operands, so they are absent from ``input_uids``
    and must not appear in ``buffers`` here.
    """
    uid_to_role = {uid: role for role, uid in plan.role_uids.items()}
    operands: List[Any] = []
    for uid in plan.input_uids:
        role = uid_to_role[uid]
        if role not in buffers:
            raise ValueError(f"ordered_input_operands: missing input buffer for role '{role}'.")
        operands.append(buffers[role])
    return operands


def build_plan(
    cudnn: Any,
    cfg: FusedAttnConfig,
    pass_: Pass,
    *,
    attn_scale: float,
    handle: Any = None,
    cudnn_version: Optional[int] = None,
    cudnn_frontend_version: Optional[int] = None,
) -> Plan:
    """Build the graph for ``pass_`` and serialize it into a :class:`Plan`.

    The shared build+serialize step both framework bridges use at lowering time.
    Dispatches to the F16 or FP8 builder based on the derived config's scaling
    mode. ``handle`` may be ``None`` -- a graph can be built and serialized
    without a device handle (the executor deserializes against its own handle),
    which is what lets JAX produce the blob as a compile-time constant.
    ``attn_scale`` is the compile-time softmax scale packed as the graph's
    pass-by-value scalar. ``cudnn_version`` is the cuDNN *backend* version the
    builder gates masking features on (defaults to ``cudnn.backend_version()``);
    ``cudnn_frontend_version`` is the *frontend* version stamped into the Plan for
    the deserialize check.
    """
    # Imported here (not at module load) to keep the cudnn-free import surface of
    # serialize.py minimal; builders only pull in the neutral config/strides/uids.
    if cfg.is_tensor_scaling or cfg.is_mxfp8:
        from .builders.fp8 import build_fp8_fwd_graph

        if pass_ is not Pass.Fwd:
            raise NotImplementedError(
                "fused_attn_py: FP8 backward Plan is not wired yet (Stage 6 is FP8 forward only)."
            )
        entry = build_fp8_fwd_graph(cudnn, handle, cfg, cudnn_version=cudnn_version)
    else:
        from .builders.f16 import build_f16_bwd_graph, build_f16_fwd_graph

        builder = build_f16_fwd_graph if pass_ is Pass.Fwd else build_f16_bwd_graph
        entry = builder(cudnn, handle, cfg, cudnn_version=cudnn_version)
    return serialize_entry(
        cudnn,
        entry,
        cudnn_frontend_version=cudnn_frontend_version,
        scalar_values={"attn_scale": pack_scalar_f32(attn_scale)},
    )


__all__ = [
    "Plan",
    "serialize_entry",
    "build_plan",
    "ordered_input_operands",
    "encode_cudnn_frontend_version",
    "graph_hash",
    "pack_scalar_f32",
]
