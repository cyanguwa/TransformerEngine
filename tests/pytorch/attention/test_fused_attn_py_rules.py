# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the Python fused-attention decision core (``fused_attn_py``).

Two tiers:

* **CPU unit tests** exercise the pure ports (bucketing, layout maps, ``derive``,
  ``make_cache_key``) with an injected fake ``RuntimeInfo``. They need neither a
  GPU nor the built C extension.
* **Dual-oracle tests** (guarded by the presence of ``transformer_engine_torch``
  and a CUDA device) pin the Python gating against the C++
  ``tex.get_fused_attn_backend`` for a config sweep. Because the Python verdict
  (run without the cuDNN probe) is an upper bound on support, we assert:

      1. Python rejects  => C++ rejects        (TE gating is identical & first)
      2. C++ accepts (X) => Python accepts (X)  (C++ never accepts what TE gating
                                                 in Python rejects)

  These two implications fully pin the TE-gating layer without needing the cuDNN
  probe, which is added with the Python graph builders in later stages.
"""

import importlib

import pytest

fa = importlib.import_module("transformer_engine.common.fused_attn_py")
config = importlib.import_module("transformer_engine.common.fused_attn_py.config")
rules = importlib.import_module("transformer_engine.common.fused_attn_py.rules")


# ---------------------------------------------------------------------------
# CPU unit tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "b,expected",
    [(1, 32), (32, 32), (33, 64), (64, 64), (512, 512), (513, 1024), (1025, 1536), (0, 0)],
)
def test_get_max_batch_size(b, expected):
    assert config.get_max_batch_size(b) == expected


@pytest.mark.parametrize(
    "t,expected",
    [(1, 1024), (1024, 1024), (1025, 2048), (32768, 32768), (32769, 65536), (0, 0)],
)
def test_get_max_tokens(t, expected):
    assert config.get_max_tokens(t) == expected


def test_layout_map_coverage():
    # The format and group maps must cover exactly the same set of layouts.
    assert set(config._LAYOUT_TO_FORMAT) == set(config._LAYOUT_TO_GROUP)


def _fake_runtime(sm_arch=90, cudnn=91500):
    return config.RuntimeInfo(
        sm_arch=sm_arch,
        cudnn_version=cudnn,
        cudnn_frontend_version=12600,
        cudnn_build_version=cudnn,
    )


def test_derive_thd_padding_causal():
    cfg = config.FusedAttnConfig(
        qkv_layout="NVTE_THD_THD_THD",
        attn_mask_type="NVTE_PADDING_CAUSAL_MASK",
        batch_size=4,
        num_attn_heads=8,
        num_gqa_groups=8,
        head_dim_qk=64,
        head_dim_v=64,
        max_seqlen_q=512,
        max_seqlen_kv=512,
        num_tokens_q=2048,
        num_tokens_kv=2048,
    )
    cfg.derive(_fake_runtime())
    assert cfg.is_ragged_q and cfg.is_ragged_kv
    assert cfg.is_padding and cfg.is_causal
    assert cfg.bottom_right_diagonal is False  # causal + bottom_right -> false
    assert cfg.uses_ragged_graph and cfg.uses_ragged_stats
    assert cfg.graph_max_seqlen_q == config.get_max_tokens(2048)
    assert cfg.bucketed_batch_size == config.get_max_batch_size(4)


def test_derive_causal_bottom_right_flips_diagonal():
    cfg = config.FusedAttnConfig(
        qkv_layout="NVTE_BSHD_BSHD_BSHD",
        attn_mask_type="NVTE_CAUSAL_BOTTOM_RIGHT_MASK",
        bottom_right_diagonal=False,
        batch_size=2,
        num_attn_heads=4,
        num_gqa_groups=4,
        head_dim_qk=64,
        head_dim_v=64,
        max_seqlen_q=128,
        max_seqlen_kv=256,
    )
    cfg.derive(_fake_runtime())
    assert cfg.is_causal_bottom_right
    assert cfg.bottom_right_diagonal is True  # forced on for bottom-right


def test_make_cache_key_forward_normalization():
    cfg = config.FusedAttnConfig(
        qkv_layout="NVTE_BSHD_BSHD_BSHD",
        attn_mask_type="NVTE_NO_MASK",
        batch_size=8,
        num_attn_heads=4,
        num_gqa_groups=4,
        head_dim_qk=64,
        head_dim_v=64,
        max_seqlen_q=128,
        max_seqlen_kv=128,
        attn_scale=0.125,
        do_dtype="kNVTEFloat16",
        deterministic=True,
    )
    cfg.derive(_fake_runtime())
    key = cfg.make_cache_key(config.Pass.Fwd)
    assert len(key) == 43  # exactly the fields C++ operator< keys on
    # Forward normalization: attn_scale collapses, do_dtype -> BF16, deterministic dropped.
    key_scaled = config.FusedAttnConfig(
        qkv_layout="NVTE_BSHD_BSHD_BSHD",
        attn_mask_type="NVTE_NO_MASK",
        batch_size=8,
        num_attn_heads=4,
        num_gqa_groups=4,
        head_dim_qk=64,
        head_dim_v=64,
        max_seqlen_q=128,
        max_seqlen_kv=128,
        attn_scale=1.0,
        do_dtype="kNVTEBFloat16",
        deterministic=False,
    )
    key_scaled.derive(_fake_runtime())
    assert key == key_scaled.make_cache_key(config.Pass.Fwd)


def test_select_backend_pure_gating():
    rt = _fake_runtime(sm_arch=90, cudnn=91500)
    # BF16 dense should pass TE gating (no probe).
    cfg = config.FusedAttnConfig(
        qkv_layout="NVTE_BSHD_BSHD_BSHD",
        attn_mask_type="NVTE_CAUSAL_MASK",
        batch_size=2,
        num_attn_heads=8,
        num_gqa_groups=8,
        head_dim_qk=64,
        head_dim_v=64,
        max_seqlen_q=512,
        max_seqlen_kv=512,
    )
    v = rules.select_fused_attn_backend(cfg, rt)
    assert v.backend is config.FusedAttnBackend.F16_arbitrary_seqlen

    # Pre-scale bias is always rejected.
    cfg2 = config.FusedAttnConfig(
        qkv_layout="NVTE_BSHD_BSHD_BSHD",
        bias_type="NVTE_PRE_SCALE_BIAS",
        batch_size=1,
        num_attn_heads=8,
        num_gqa_groups=8,
        head_dim_qk=64,
        head_dim_v=64,
        max_seqlen_q=128,
        max_seqlen_kv=128,
    )
    v2 = rules.select_fused_attn_backend(cfg2, rt)
    assert not v2.supported and "pre-scale bias" in v2.reason


f16_builder = importlib.import_module("transformer_engine.common.fused_attn_py.builders.f16")


def test_plan_f16_fwd_masking():
    """Pure mask/window planning must mirror create_graph_f16_fwd (CPU-only)."""

    def cfg(**kw):
        c = config.FusedAttnConfig()
        for key, val in kw.items():
            setattr(c, key, val)
        return c

    # Top-left causal: right bound 0, no left bound.
    p = f16_builder.plan_f16_fwd_masking(cfg(is_causal=True, bottom_right_diagonal=False), 90600)
    assert p == {
        "diagonal_alignment": "TOP_LEFT",
        "diagonal_band_left_bound": None,
        "diagonal_band_right_bound": 0,
        "use_alibi_mask": False,
    }

    # Bottom-right causal.
    p = f16_builder.plan_f16_fwd_masking(
        cfg(is_causal_bottom_right=True, bottom_right_diagonal=True), 90600
    )
    assert p["diagonal_alignment"] == "BOTTOM_RIGHT" and p["diagonal_band_right_bound"] == 0

    # Sliding window: left = window_left + 1, right = window_right (cuDNN 9.6+).
    p = f16_builder.plan_f16_fwd_masking(
        cfg(window_size_left=128, window_size_right=64, bottom_right_diagonal=True), 90600
    )
    assert p["diagonal_band_left_bound"] == 129 and p["diagonal_band_right_bound"] == 64

    # Right bound needs cuDNN 9.6+; at 9.2 it is dropped while left is kept.
    p = f16_builder.plan_f16_fwd_masking(
        cfg(window_size_left=128, window_size_right=64, bottom_right_diagonal=True), 90200
    )
    assert p["diagonal_band_left_bound"] == 129 and p["diagonal_band_right_bound"] is None

    # ALiBi rides along with causal (right bound 0).
    p = f16_builder.plan_f16_fwd_masking(cfg(is_alibi=True, is_causal=True), 90600)
    assert p["use_alibi_mask"] is True and p["diagonal_band_right_bound"] == 0


def test_f16_builder_rejects_deferred_features():
    """Deferred features must fail loud, not silently build a wrong graph."""
    import pytest as _pytest

    for field in ("is_paged_kv", "is_ragged_q", "is_softmax_offset", "return_max_logit"):
        cfg = config.FusedAttnConfig()
        setattr(cfg, field, True)
        with _pytest.raises(NotImplementedError):
            f16_builder._reject_unsupported(cfg)


strides = importlib.import_module("transformer_engine.common.fused_attn_py.strides")


def _contig(dim):
    st = [1] * len(dim)
    for i in range(len(dim) - 2, -1, -1):
        st[i] = st[i + 1] * dim[i + 1]
    return st


def test_generate_matrix_strides_vs_flex_and_hand():
    """strides.py must match generateMatrixStrides for separate + packed layouts."""
    b, h, hg, s_q, s_kv, dqk, dv = 2, 4, 2, 8, 8, 16, 16
    gen = strides.generate_matrix_strides

    # Separate SBHD/BSHD checked against flex_attention._bhsd_dim_stride semantics.
    def sbhd_gt(bb, hh, ss, dd):
        st = _contig([ss, bb, hh, dd])
        return (st[1], st[2], st[0], st[3])

    def bshd_gt(bb, hh, ss, dd):
        st = _contig([bb, ss, hh, dd])
        return (st[0], st[2], st[1], st[3])

    assert gen(b, h, s_q, s_kv, dqk, "NVTE_SBHD_SBHD_SBHD", "Q") == sbhd_gt(b, h, s_q, dqk)
    assert gen(b, hg, s_q, s_kv, dqk, "NVTE_SBHD_SBHD_SBHD", "K") == sbhd_gt(b, hg, s_kv, dqk)
    assert gen(b, h, s_q, s_kv, dv, "NVTE_BSHD_BSHD_BSHD", "Q") == bshd_gt(b, h, s_q, dv)
    assert gen(b, hg, s_q, s_kv, dv, "NVTE_BSHD_BSHD_BSHD", "V") == bshd_gt(b, hg, s_kv, dv)

    # Packed and BHSD, hand-computed against the C++ switch.
    assert gen(b, h, s_q, s_kv, dqk, "NVTE_BS3HD", "Q") == (s_q * 3 * h * dqk, dqk, 3 * h * dqk, 1)
    assert gen(b, h, s_q, s_kv, dqk, "NVTE_SB3HD", "Q") == (3 * h * dqk, dqk, b * 3 * h * dqk, 1)
    assert gen(b, h, s_q, s_kv, dv, "NVTE_SB3HD", "O") == (h * dv, dv, b * h * dv, 1)
    bhsd_q = gen(b, h, s_q, s_kv, dqk, "NVTE_BHSD_BHSD_BHSD", "Q")
    assert bhsd_q == (h * s_q * dqk, s_q * dqk, dqk, 1)
    # Hybrid: Q is SBHD, K is BSHD.
    assert gen(b, h, s_q, s_kv, dqk, "NVTE_SBHD_BSHD_BSHD", "Q") == (h * dqk, dqk, b * h * dqk, 1)
    hybrid_k = gen(b, hg, s_q, s_kv, dqk, "NVTE_SBHD_BSHD_BSHD", "K")
    assert hybrid_k == (s_kv * hg * dqk, dqk, hg * dqk, 1)


class _FakeCudnnTensor:
    def __init__(self, name=None, **kw):
        self.name = name

    def set_output(self, v):
        return self

    def set_dim(self, d):
        return self

    def set_stride(self, s):
        return self

    def set_data_type(self, dt):
        return self


class _FakeCudnnGraph:
    def __init__(self, **kw):
        self.calls = []
        self.sdpa_kwargs = None

    def tensor(self, **kw):
        return _FakeCudnnTensor(**kw)

    def sdpa(self, q, k, v, **kw):
        self.sdpa_kwargs = kw
        return _FakeCudnnTensor(name="O"), _FakeCudnnTensor(name="Stats")

    def validate(self):
        self.calls.append("validate")

    def build_operation_graph(self):
        self.calls.append("bog")

    def create_execution_plans(self, modes):
        self.calls.append("cep")

    def check_support(self):
        self.calls.append("cs")

    def build_plans(self, policy):
        self.calls.append("bp")

    def get_workspace_size(self):
        return 4096


class _FakeCudnn:
    class data_type:  # noqa: N801
        HALF = "HALF"
        BFLOAT16 = "BF16"
        FLOAT = "FLOAT"
        INT32 = "INT32"
        INT64 = "INT64"

    class diagonal_alignment:  # noqa: N801
        TOP_LEFT = "TL"
        BOTTOM_RIGHT = "BR"

    class heur_mode:  # noqa: N801
        A = "A"
        FALLBACK = "FB"

    class build_plan_policy:  # noqa: N801
        HEURISTICS_CHOICE = "HC"

    def __init__(self):
        self.g = None

    def pygraph(self, **kw):
        self.g = _FakeCudnnGraph(**kw)
        return self.g

    def backend_version(self):
        return 91500


def _fake_runtime():
    return config.RuntimeInfo(
        sm_arch=90, cudnn_version=91500, cudnn_frontend_version=10700, cudnn_build_version=91500
    )


def test_build_f16_fwd_graph_control_flow():
    """Exercise the full builder path (tensor/sdpa/finalize) with a mock cudnn."""
    cudnn = _FakeCudnn()
    rt = _fake_runtime()
    base = dict(
        qkv_layout="NVTE_BSHD_BSHD_BSHD",
        batch_size=2,
        num_attn_heads=8,
        num_gqa_groups=8,
        head_dim_qk=64,
        head_dim_v=64,
        max_seqlen_q=128,
        max_seqlen_kv=128,
        qkv_dtype="kNVTEBFloat16",
        o_dtype="kNVTEBFloat16",
    )

    def build(**overrides):
        cfg = config.FusedAttnConfig(**{**base, **overrides}).derive(rt)
        return f16_builder.build_f16_fwd_graph(cudnn, handle=1234, cfg=cfg)

    e = build()
    assert set(e.tensors) == {"Q", "K", "V", "attn_scale", "O", "Stats"}
    assert e.workspace_size == 4096
    assert cudnn.g.calls == ["validate", "bog", "cep", "cs", "bp"]
    assert cudnn.g.sdpa_kwargs["generate_stats"] is True

    assert build(attn_mask_type="NVTE_CAUSAL_MASK") and cudnn.g.sdpa_kwargs[
        "diagonal_band_right_bound"
    ] == 0

    e = build(
        bias_type="NVTE_POST_SCALE_BIAS",
        bias_batch_size=2,
        bias_num_heads=8,
        bias_seqlen_q=128,
        bias_seqlen_kv=128,
    )
    assert "bias" in e.tensors and "bias" in cudnn.g.sdpa_kwargs

    e = build(attn_mask_type="NVTE_PADDING_MASK")
    assert cudnn.g.sdpa_kwargs.get("use_padding_mask") is True
    assert "seq_q" in e.tensors and "seq_kv" in e.tensors

    e = build(dropout=0.1)
    assert "dropout" in cudnn.g.sdpa_kwargs and e.tensors.get("dropout_seed") is not None

    # GQA + SBHD + head_dim_v != head_dim_qk must still build.
    assert build(qkv_layout="NVTE_SBHD_SBHD_SBHD", num_gqa_groups=2, head_dim_v=128)


# ---------------------------------------------------------------------------
# In-container tests: enum-name parity + dual oracle vs the C++ backend query
# ---------------------------------------------------------------------------
tex = pytest.importorskip(
    "transformer_engine_torch", reason="C++ extension required for oracle tests"
)


def test_enum_name_parity():
    """The names the core switches on must exist in the matching tex enums."""

    def names(enum_cls):
        return {m.name for m in enum_cls}

    assert set(config._LAYOUT_TO_FORMAT).issubset(names(tex.NVTE_QKV_Layout))
    assert {f.value for f in config.QKVFormat if f is not config.QKVFormat.NOT_SET}.issubset(
        names(tex.NVTE_QKV_Format)
    )
    mask_names = (
        config._MASK_PADDING | config._MASK_CAUSAL | config._MASK_CAUSAL_BR | {"NVTE_NO_MASK"}
    )
    assert mask_names.issubset(names(tex.NVTE_Mask_Type))
    assert {
        "NVTE_NO_BIAS",
        "NVTE_PRE_SCALE_BIAS",
        "NVTE_POST_SCALE_BIAS",
        "NVTE_ALIBI",
    }.issubset(names(tex.NVTE_Bias_Type))
    assert {"NVTE_VANILLA_SOFTMAX", "NVTE_LEARNABLE_SOFTMAX"}.issubset(
        names(tex.NVTE_Softmax_Type)
    )
    assert {"NVTE_DELAYED_TENSOR_SCALING", "NVTE_MXFP8_1D_SCALING"}.issubset(
        names(tex.NVTEScalingMode)
    )
    assert {b.value for b in config.FusedAttnBackend}.issubset(
        names(tex.NVTE_Fused_Attn_Backend)
    )


torch = pytest.importorskip("torch", reason="torch required for dual-oracle test")


def _init_kwargs(cls, spec):
    """Filter a spec dict down to the constructor-settable fields of a dataclass."""
    import dataclasses

    allowed = {f.name for f in dataclasses.fields(cls) if f.init}
    return {k: v for k, v in spec.items() if k in allowed}


def _real_runtime():
    major, minor = torch.cuda.get_device_capability()
    cudnn = tex.get_cudnn_version()
    return config.RuntimeInfo(
        sm_arch=major * 10 + minor,
        cudnn_version=cudnn,
        cudnn_frontend_version=cudnn,  # does not affect the backend verdict
        cudnn_build_version=cudnn,
    )


def _dual_oracle_specs():
    """A small representative sweep of (name, spec) pairs shared by both paths."""
    from transformer_engine.pytorch.constants import TE_DType as _  # noqa: F401

    Mask = tex.NVTE_Mask_Type
    Bias = tex.NVTE_Bias_Type
    Layout = tex.NVTE_QKV_Layout
    dims = dict(
        batch_size=2,
        num_attn_heads=8,
        num_gqa_groups=8,
        head_dim_qk=64,
        head_dim_v=64,
        max_seqlen_q=512,
        max_seqlen_kv=512,
    )
    specs = []
    for mask in (Mask.NVTE_NO_MASK, Mask.NVTE_CAUSAL_MASK, Mask.NVTE_PADDING_CAUSAL_MASK):
        for layout in (Layout.NVTE_BSHD_BSHD_BSHD, Layout.NVTE_SBHD_SBHD_SBHD):
            for training in (True, False):
                spec = dict(qkv_layout=layout, attn_mask_type=mask, is_training=training, **dims)
                if mask.name.startswith("NVTE_PADDING"):
                    spec.update(num_tokens_q=1024, num_tokens_kv=1024)
                specs.append((f"{mask.name}-{layout.name}-train{training}", spec))
    # A rejected-by-TE case: pre-scale bias.
    specs.append(
        ("pre_scale_bias", dict(qkv_layout=Layout.NVTE_BSHD_BSHD_BSHD,
                                bias_type=Bias.NVTE_PRE_SCALE_BIAS, is_training=True, **dims))
    )
    return specs


_SPECS = _dual_oracle_specs()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("name,spec", _SPECS, ids=[s[0] for s in _SPECS])
def test_dual_oracle_backend_selection(name, spec):
    """Python TE-gating verdict is an upper bound on the C++ verdict."""
    from transformer_engine.pytorch.attention.dot_product_attention.utils import (
        FusedAttentionParams,
    )

    runtime = _real_runtime()

    cfg = config.FusedAttnConfig(**_init_kwargs(config.FusedAttnConfig, spec))
    py = rules.select_fused_attn_backend(cfg, runtime)  # no cuDNN probe

    params = FusedAttentionParams(**_init_kwargs(FusedAttentionParams, spec))
    c_backend = tex.get_fused_attn_backend(params)
    c_supported = c_backend != tex.NVTE_Fused_Attn_Backend.NVTE_No_Backend

    # (1) Python rejects => C++ rejects.
    if not py.supported:
        assert not c_supported, (
            f"[{name}] Python rejected ({py.reason}) but C++ chose {c_backend.name}"
        )
    # (2) C++ accepts X => Python accepts the same X.
    if c_supported:
        assert py.backend.value == c_backend.name, (
            f"[{name}] C++ chose {c_backend.name} but Python chose {py.backend.value} "
            f"({py.reason})"
        )

