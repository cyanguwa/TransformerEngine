# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Process-wide cuDNN graph cache and support probe.

Python counterpart of ``common/fused_attn/graph_cache.h``. The C++ cache maps a
normalized ``FusedAttnConfig`` to a built cuDNN graph guarded by a mutex; this
mirrors that with a dict keyed by ``FusedAttnConfig.make_cache_key(pass)`` under
a lock.

Two responsibilities, kept separate so the pure logic stays testable:

* ``GraphCache`` -- thread-safe ``key -> entry`` store (build-once).
* ``build_and_probe`` -- run a builder, translating cuDNN's "not supported"
  exception into a reason string. This is the ``probe`` that
  ``rules.select_fused_attn_backend`` injects: it is exactly the step that
  "builds the real cuDNN graph", which is why backend selection cannot be a
  pure lookup table.

The ``cudnn`` module and the builder are injected rather than imported at module
load, so this file has no hard dependency on a GPU or the cuDNN Python package
and can be unit-tested with fakes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Hashable, Optional, Tuple


@dataclass
class GraphEntry:
    """A built cuDNN graph plus the handles/metadata needed to execute it.

    ``graph`` is the ``cudnn.pygraph``; ``tensors`` maps a stable role name (e.g.
    ``"Q"``, ``"O"``, ``"Stats"``) to the graph tensor object returned at build
    time; ``workspace_size`` is the byte size ``graph.get_workspace_size()``
    reported (floored at 1, as cuDNN requires a non-empty workspace). ``uids``
    maps the same role names to the stable integer UID (``set_uid``) each tensor
    carries in the graph, so a serialized graph can be executed by binding a
    plain ``{uid: ptr}`` variant pack (the JAX blob / ``torch.library`` paths).

    ``output_roles``, when non-empty, is the exact set of role names the builder
    marked as graph *outputs* (``set_output(True)``). Serialization uses it to
    split inputs from outputs instead of guessing from role names -- necessary
    because the same role (e.g. ``"O"``/``"Stats"``) is an output in the forward
    graph but an input in the backward graph. Builders that leave it empty fall
    back to the name-based heuristic in ``serialize``.
    """

    graph: Any
    tensors: Dict[str, Any] = field(default_factory=dict)
    workspace_size: int = 1
    uids: Dict[str, int] = field(default_factory=dict)
    output_roles: FrozenSet[str] = field(default_factory=frozenset)


class GraphCache:
    """Thread-safe build-once cache, mirroring ``GraphCache`` in graph_cache.h."""

    def __init__(self) -> None:
        self._entries: Dict[Hashable, GraphEntry] = {}
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Optional[GraphEntry]:
        with self._lock:
            return self._entries.get(key)

    def get_or_build(self, key: Hashable, build_fn: Callable[[], GraphEntry]) -> GraphEntry:
        """Return the cached entry for ``key`` or build, insert, and return it.

        Matches the C++ cache semantics: the lookup/insert is locked, the build
        is not, so concurrent builders for the same key are allowed but only the
        first insert wins.
        """
        cached = self.get(key)
        if cached is not None:
            return cached
        entry = build_fn()
        with self._lock:
            # Another thread may have inserted while we built; that entry wins.
            return self._entries.setdefault(key, entry)


def build_and_probe(
    build_fn: Callable[[], GraphEntry],
    not_supported_exc: type,
) -> Tuple[Optional[GraphEntry], str]:
    """Build a graph, translating a cuDNN "not supported" rejection into a reason.

    Returns ``(entry, "")`` on success or ``(None, reason)`` if cuDNN declined
    the graph. Any other exception propagates -- a violated invariant or a real
    failure must surface as an error, not be reported as an unsupported config
    (same contract as ``UnsupportedByCudnn`` in graph_cache.h).
    """
    try:
        return build_fn(), ""
    except not_supported_exc as exc:  # cuDNN declined this config.
        reason = str(exc) or "cuDNN reported the fused-attention graph as unsupported."
        return None, reason


__all__ = ["GraphEntry", "GraphCache", "build_and_probe"]
