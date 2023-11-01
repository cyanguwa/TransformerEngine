# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from importlib.metadata import version
import os
import math
from typing import Any, Dict, List, Tuple, Union

from pkg_resources import packaging
import pytest
import torch

from transformer_engine.common import recipe
from transformer_engine.pytorch import TransformerLayer, fp8_autocast
from transformer_engine.pytorch.attention import (
    DotProductAttention,
    RotaryPositionEmbedding,
)
from transformer_engine.pytorch.constants import TE_DType
import transformer_engine.pytorch.cpp_extensions as ext
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    AttnBiasType,
    AttnMaskType,
    FusedAttnBackend,
    QKVLayout,
    fused_attn_bwd,
    fused_attn_fwd,
)
import transformer_engine.pytorch.fp8 as fp8
from transformer_engine.pytorch.module.base import (
    TransformerEngineBaseModule,
    _prepare_backward,
)
from transformer_engine.pytorch.utils import (
    get_device_compute_capability,
    init_method_normal,
    scaled_init_method_normal,
)
from transformer_engine.pytorch.distributed import _set_cuda_rng_state, CudaRNGStatesTracker
import transformer_engine_extensions as tex

# Only run FP8 tests on H100.
fp8_available, reason_for_no_fp8 = fp8.FP8GlobalStateManager.is_fp8_available()
_flash_attn_version = packaging.version.Version(version("flash-attn"))
_flash_attn_2_available = _flash_attn_version >= packaging.version.Version("2")

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# Record initial RNG state from script run.
_cpu_rng_state = torch.get_rng_state()
_cuda_rng_state = torch.cuda.get_rng_state()

def _get_cudnn_version():
    cudnn_version_encoded = ext.get_cudnn_version()
    cudnn_major = cudnn_version_encoded // 1000
    cudnn_minor = (cudnn_version_encoded - cudnn_major * 1000) // 100
    cudnn_patch = cudnn_version_encoded - 1000 * cudnn_major - 100 * cudnn_minor
    return [cudnn_major, cudnn_minor, cudnn_patch]

def reset_rng_states() -> None:
    """revert back to initial RNG state."""
    torch.set_rng_state(_cpu_rng_state)
    _set_cuda_rng_state(_cuda_rng_state)

_cudnn_version = _get_cudnn_version()

class ModelConfig:
    def __init__(
        self,
        num_layers: int,
        num_attention_heads: int,
        num_gqa_groups: int,
        head_dim: int,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        dropout_p: float,
        attn_mask_type: str,
        attn_bias_type: str,
    ):
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_gqa_groups = num_gqa_groups
        self.head_dim = head_dim
        self.hidden_size = num_attention_heads * head_dim 
        self.hidden_size_kv = num_gqa_groups * head_dim 
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv
        self.dropout_p = dropout_p
        self.attn_mask_type  = attn_mask_type
        self.attn_bias_type  = attn_bias_type
        self.attn_type  = "self" if (max_seqlen_q == max_seqlen_kv) else "cross" 

def _is_fused_attention_supported(
    config: ModelConfig,
    dtype: torch.dtype,
    qkv_layout: str = "sbh3d",
) -> bool:
    if config.attn_mask_type == 'padding,causal':
        attn_mask_type = 'padding_causal'
    else:
        attn_mask_type = config.attn_mask_type
    backend = tex.get_fused_attn_backend(
        TE_DType[dtype],
        TE_DType[dtype],
        QKVLayout[qkv_layout],
        AttnBiasType[config.attn_bias_type],
        AttnMaskType[attn_mask_type],
        config.dropout_p,
        config.max_seqlen_q,
        config.max_seqlen_kv,
        config.head_dim,
        config.num_attention_heads,
        config.num_gqa_groups,
    )
    return backend != FusedAttnBackend["No_Backend"]

def _is_flash_attention_supported(config: ModelConfig) -> bool:
    if get_device_compute_capability() < (8, 0):
        return False
    if config.attn_bias_type != "no_bias":
        return False
    return True

def _is_unfused_attention_supported(config: ModelConfig) -> bool:
    if ("padding" in config.attn_mask_type or config.attn_bias_type == "alibi"):
        return False
    if ("causal" in config.attn_mask_type and config.attn_type == 'cross'):
        return False
    return True

################ common model configs and other params ################ 
model_configs = {
    #     test:    num_layers,  h, hg,   d,   sq,  skv,   p,      mask,      bias 
    "base_1_0": ModelConfig(1, 16, 16,  64,  128,  128, 0.0, "no_mask", "no_bias"), # self
    "base_1_1": ModelConfig(1, 16, 16,  64,  128,  256, 0.0, "no_mask", "no_bias"), # cross
    "base_2_0": ModelConfig(1, 16, 16,  64, 2048, 2048, 0.0, "no_mask", "no_bias"), # self
    "base_2_1": ModelConfig(1, 16, 16,  64, 2048, 4096, 0.0, "no_mask", "no_bias"), # cross
    "base_3_0": ModelConfig(1, 16, 16, 128,  128,  128, 0.0, "no_mask", "no_bias"), # self
    "base_3_1": ModelConfig(1, 16, 16, 128,  128,  256, 0.0, "no_mask", "no_bias"), # cross
    "base_4_0": ModelConfig(1, 24, 24, 128, 2048, 2048, 0.0, "no_mask", "no_bias"), # self
    "base_4_1": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0, "no_mask", "no_bias"), # cross
}

param_types = [torch.float16]
if torch.cuda.is_bf16_supported():
    param_types.append(torch.bfloat16)
batch_sizes = [1, 32]

param_types_lean = [torch.float16]
batch_sizes_lean = [2]

################ test basic DPA cases ################ 
# To run a test in model_configs:
#   pytest -s -v tests/pytorch/test_fused_attn.py::test_dot_product_attention[True-base_3_0-2-dtype0]
# all passed 

@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes_lean)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("ckpt_attn", [True, False])
def test_dot_product_attention(dtype, bs, model, ckpt_attn):
    """Test DotProductAttention module with different backends"""

    # Get configs
    config = model_configs[model]
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)

    if config.attn_type == 'self':
        qkv_layout = 'sbh3d'
    else:
        qkv_layout = 'sbhd_sbh2d'
    workspace_opt = True

    # Skip if only unfused backend is supported
    fused_attn_supported = _is_fused_attention_supported(
        config, dtype, qkv_layout=qkv_layout,
    )
    flash_attn_supported = _is_flash_attention_supported(config)
    if not (fused_attn_supported or flash_attn_supported):
        pytest.skip(
            "Neither FusedAttention nor FlashAttention support this model config"
        )

    unfused_attn_supported = _is_unfused_attention_supported(config)
    # UnfusedDotProductAttention backend
    if unfused_attn_supported:
        unfused_attn_fwd, unfused_attn_bwd = _run_dot_product_attention(
            dtype, bs, config, "UnfusedDotProductAttention", ckpt_attn, qkv_layout, workspace_opt,
        )

    # FusedAttention backend
    if fused_attn_supported:
        #if config.max_seqlen_q <= 512 and config.max_seqlen_kv <= 512:
        #    os.environ["NVTE_FUSED_ATTN_BACKEND"] = "0"
        fused_attn_fwd, fused_attn_bwd = _run_dot_product_attention(
            dtype, bs, config, "FusedAttention", ckpt_attn, qkv_layout, workspace_opt,
        )

    # FlashAttention backend
    if flash_attn_supported:
        flash_attn_fwd, flash_attn_bwd = _run_dot_product_attention(
            dtype, bs, config, "FlashAttention", ckpt_attn, qkv_layout, workspace_opt,
        )

    if unfused_attn_supported and fused_attn_supported:
        torch.testing.assert_close(fused_attn_fwd, unfused_attn_fwd, **tols)
        torch.testing.assert_close(fused_attn_bwd, unfused_attn_bwd, **tols)
        print("test_dot_product_attention: fused_attn matches unfused_attn")
    if unfused_attn_supported and flash_attn_supported:
        torch.testing.assert_close(flash_attn_fwd, unfused_attn_fwd, **tols)
        torch.testing.assert_close(flash_attn_bwd, unfused_attn_bwd, **tols)
        print("test_dot_product_attention: flash_attn matches unfused_attn")
    if fused_attn_supported and flash_attn_supported:
        torch.testing.assert_close(flash_attn_fwd, fused_attn_fwd, **tols)
        torch.testing.assert_close(flash_attn_bwd, fused_attn_bwd, **tols)
        print("test_dot_product_attention: flash_attn matches fused_attn")

################ test DPA with dropout ################ 
# To run a test in model_configs_dropout:
#   rm *.pt
#   NVTE_FUSED_ATTN_OLD=1 pytest -s -v tests/pytorch/test_fused_attn.py::test_dpa_dropout[True-dropout_1_2-2-dtype0]
#   NVTE_FUSED_ATTN_OLD=0 pytest -s -v tests/pytorch/test_fused_attn.py::test_dpa_dropout[True-dropout_1_2-2-dtype0]
# failing: all except dropout_1_1 fwd

model_configs_dropout = {
    #        test:    num_layers,  h, hg,   d,   sq,  skv,   p,      mask,      bias 
    "dropout_1_0": ModelConfig(1, 16, 16,  64,  128,  128, 0.1, "no_mask", "no_bias"), # self
    "dropout_1_1": ModelConfig(1, 16, 16,  64,  128,  128, 0.1,  "causal", "no_bias"), # self
    "dropout_1_2": ModelConfig(1, 16, 16,  64,  128,  512, 0.1, "no_mask", "no_bias"), # cross
}

@pytest.mark.skipif(
    _cudnn_version < [8,9,5], reason="cuDNN 8.9.5+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("bs", batch_sizes_lean)
@pytest.mark.parametrize("model", model_configs_dropout.keys())
@pytest.mark.parametrize("workspace_opt", [True, False])
def test_dpa_dropout(dtype, bs, model, workspace_opt):
    """Test DotProductAttention module with dropout"""

    # Get configs
    config = model_configs_dropout[model]
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)

    ckpt_attn = False
    if config.attn_type == 'self':
        qkv_layout = 'sbh3d'
    else:
        qkv_layout = 'sbhd_sbh2d'

    # Skip if fused backend is not supported
    fused_attn_supported = _is_fused_attention_supported(
        config, dtype, qkv_layout=qkv_layout,
    )
    if not fused_attn_supported:
        pytest.skip(
            "FusedAttention does not support this model config"
        )
    else:
        # run old implementation with v0.9.2
        if os.environ["NVTE_FUSED_ATTN_OLD"] == "1":
            fused_attn_fwd_old, fused_attn_bwd_old = _run_dot_product_attention(
                dtype, bs, config, "FusedAttention", ckpt_attn, qkv_layout, workspace_opt,
            )
            torch.save(fused_attn_fwd_old, 'fused_attn_fwd_old.pt')
            for i in range(len(fused_attn_bwd_old)):
                torch.save(fused_attn_bwd_old[i], 'fused_attn_bwd_old_'+str(i)+'.pt')

        # run new implementation with v1/prerelease 4
        if os.environ["NVTE_FUSED_ATTN_OLD"] == "0":
            fused_attn_fwd, fused_attn_bwd = _run_dot_product_attention(
                dtype, bs, config, "FusedAttention", ckpt_attn, qkv_layout, workspace_opt,
            )
            #torch.save(fused_attn_fwd, 'fused_attn_fwd.pt')
            #for i in range(len(fused_attn_bwd)):
            #    torch.save(fused_attn_bwd[i], 'fused_attn_bwd_'+str(i)+'.pt')

            print('new fwd:')
            print(fused_attn_fwd.min().item(), fused_attn_fwd.max().item())
            fused_attn_fwd_old = torch.load('fused_attn_fwd_old.pt')
            print('old fwd:')
            print(fused_attn_fwd_old.min().item(), fused_attn_fwd_old.max().item())
            torch.testing.assert_close(fused_attn_fwd, fused_attn_fwd_old, **tols)

            for i in range(len(fused_attn_bwd)):
                print('new bwd ',i,':')
                print(fused_attn_bwd[i].min().item(), fused_attn_bwd[i].max().item())
                fused_attn_bwd_old_i = torch.load('fused_attn_bwd_old_'+str(i)+'.pt')
                print('old bwd ',i,':')
                print(fused_attn_bwd_old_i.min().item(), fused_attn_bwd_old_i.max().item())
                torch.testing.assert_close(fused_attn_bwd[i], fused_attn_bwd_old_i, **tols)

################ test DPA with different masks ################ 
# To run a test in model_configs_mask:
#   pytest -s -v tests/pytorch/test_fused_attn.py::test_dpa_mask[True-mask_1_0-2-dtype0]
# failing: cross-attn + causal, padding + causal
# [True-mask_1_1-2-dtype0] - AssertionError: Tensor-likes are not close!
# [True-mask_2_1-2-dtype0] - AssertionError: Tensor-likes are not close!
# [True-mask_4_0-2-dtype0] - AssertionError: Tensor-likes are not close!
# [True-mask_4_1-2-dtype0] - AssertionError: Tensor-likes are not close!

model_configs_mask = {
    # testname:    num_layers,  h, hg,   d,   sq,  skv,   p,             mask,      bias 
    "mask_1_0": ModelConfig(1, 16, 16,  64,  128,  128, 0.0,         "causal", "no_bias"), # self
    "mask_1_1": ModelConfig(1, 16, 16,  64,  128,  256, 0.0,         "causal", "no_bias"), # cross
    "mask_2_0": ModelConfig(1, 24, 24, 128, 2048, 2048, 0.0,         "causal", "no_bias"), # self
    "mask_2_1": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0,         "causal", "no_bias"), # cross
    "mask_3_0": ModelConfig(1, 16, 16,  64,  128,  128, 0.0,        "padding", "no_bias"), # self
    "mask_3_1": ModelConfig(1, 16, 16,  64,  128,  256, 0.0,        "padding", "no_bias"), # cross
    "mask_4_0": ModelConfig(1, 16, 16,  64,  128,  128, 0.0, "padding,causal", "no_bias"), # self
    "mask_4_1": ModelConfig(1, 16, 16,  64,  128,  256, 0.0, "padding,causal", "no_bias"), # cross
}

@pytest.mark.skipif(
    _cudnn_version < [8,9,5], reason="cuDNN 8.9.5+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("bs", batch_sizes_lean)
@pytest.mark.parametrize("model", model_configs_mask.keys())
@pytest.mark.parametrize("workspace_opt", [True])#, False])
def test_dpa_mask(dtype, bs, model, workspace_opt):
    """Test DotProductAttention module with different QKV layouts"""

    # Get configs
    config = model_configs_mask[model]
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)

    ckpt_attn = False
    if config.attn_type == 'self':
        qkv_layout = 'sb3hd'
    else:
        qkv_layout = 'sbhd_sbh2d'

    # Skip if only unfused backend is supported
    fused_attn_supported = _is_fused_attention_supported(
        config, dtype, qkv_layout=qkv_layout,
    )
    flash_attn_supported = _is_flash_attention_supported(config)
    unfused_attn_supported = _is_unfused_attention_supported(config)
    if not (fused_attn_supported or flash_attn_supported):
        pytest.skip(
            "Neither FusedAttention nor FlashAttention support this model config"
        )

    #if config.attn_mask_type in ['padding']: #, 'padding_causal']:
    #    os.environ["NVTE_FUSED_ATTN_BACKEND"] = "0"
    #    fused_attn_fwd, fused_attn_bwd = _run_dot_product_attention(
    #        dtype, bs, config, "FusedAttention", ckpt_attn, qkv_layout, workspace_opt,
    #    )
    #    os.environ["NVTE_FUSED_ATTN_BACKEND"] = "1"
    #    fused_attn_fwd_arbi, fused_attn_bwd_arbi = _run_dot_product_attention(
    #        dtype, bs, config, "FusedAttention", ckpt_attn, qkv_layout, workspace_opt,
    #    )
    #    print('max512 fwd:')
    #    print(fused_attn_fwd.min().item(), fused_attn_fwd.max().item())
    #    #torch.save(fused_attn_fwd, 'fused_attn_fwd.pt')
    #    print('arbi fwd:')
    #    print(fused_attn_fwd_arbi.min().item(), fused_attn_fwd_arbi.max().item())
    #    #torch.save(fused_attn_fwd_arbi, 'fused_attn_fwd_arbi.pt')
    #    torch.testing.assert_close(fused_attn_fwd, fused_attn_fwd_arbi, **tols)
    #    for i in range(len(fused_attn_bwd)):
    #        print('max512 bwd ',i,':')
    #        print(fused_attn_bwd[i].min().item(), fused_attn_bwd[i].max().item())
    #        #torch.save(fused_attn_bwd[i], 'fused_attn_bwd_'+str(i)+'.pt')
    #        print('arbi bwd ',i,':')
    #        print(fused_attn_bwd_arbi[i].min().item(), fused_attn_bwd_arbi[i].max().item())
    #        #torch.save(fused_attn_bwd_arbi[i], 'fused_attn_bwd_arbi_'+str(i)+'.pt')
    #        torch.testing.assert_close(fused_attn_bwd[i], fused_attn_bwd_arbi[i], **tols)
    #    print("test_dpa_mask: fused_attn max512 matches fused_attn arbi")
    #else:
    if True:
        if unfused_attn_supported:
            unfused_attn_fwd, unfused_attn_bwd = _run_dot_product_attention(
                dtype, bs, config, "UnfusedDotProductAttention", ckpt_attn, qkv_layout, workspace_opt)
        if fused_attn_supported:
            #if config.max_seqlen_q <= 512 and config.max_seqlen_kv <= 512:
            #    os.environ["NVTE_FUSED_ATTN_BACKEND"] = "0"
            fused_attn_fwd, fused_attn_bwd = _run_dot_product_attention(
                dtype, bs, config, "FusedAttention", ckpt_attn, qkv_layout, workspace_opt)
        if flash_attn_supported:
            flash_attn_fwd, flash_attn_bwd = _run_dot_product_attention(
                dtype, bs, config, "FlashAttention", ckpt_attn, qkv_layout, workspace_opt)
        if unfused_attn_supported and fused_attn_supported:
            torch.testing.assert_close(fused_attn_fwd, unfused_attn_fwd, **tols)
            torch.testing.assert_close(fused_attn_bwd, unfused_attn_bwd, **tols)
            print("test_dpa_mask: fused_attn matches unfused_attn")
        if unfused_attn_supported and flash_attn_supported:
            torch.testing.assert_close(flash_attn_fwd, unfused_attn_fwd, **tols)
            torch.testing.assert_close(flash_attn_bwd, unfused_attn_bwd, **tols)
            print("test_dpa_mask: flash_attn matches unfused_attn")
        if fused_attn_supported and flash_attn_supported:
            torch.testing.assert_close(flash_attn_fwd, fused_attn_fwd, **tols)
            torch.testing.assert_close(flash_attn_bwd, fused_attn_bwd, **tols)
            print("test_dpa_mask: flash_attn matches fused_attn")

################ test DPA with different bias types ################ 
# To run a test in model_configs_bias:
# pytest -s -v tests/pytorch/test_fused_attn.py::test_dpa_bias[True-bias_1_0-2-dtype0]
# failing: all 5
# need to fix alibi for m512 with cross attn, bias_3_1

model_configs_bias = {
    #     test:    num_layers,  h, hg,   d,   sq,  skv,   p,      mask,             bias 
    "bias_1_0": ModelConfig(1, 16, 16,  64,  128,  128, 0.0, "no_mask", "post_scale_bias"), # self
    "bias_1_1": ModelConfig(1, 16, 16,  64,  128,  256, 0.0, "no_mask", "post_scale_bias"), # cross
    "bias_2_0": ModelConfig(1, 24, 24, 128, 2048, 2048, 0.0, "no_mask", "post_scale_bias"), # self
    "bias_2_1": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0, "no_mask", "post_scale_bias"), # cross
    "bias_3_0": ModelConfig(1, 16, 16,  64,  128,  128, 0.0, "no_mask",           "alibi"), # self
#    "bias_3_1": ModelConfig(1, 16, 16,  64,  128,  256, 0.0, "no_mask",           "alibi"), # cross
}

@pytest.mark.skipif(
    _cudnn_version < [8,9,5], reason="cuDNN 8.9.5+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("bs", batch_sizes_lean)
@pytest.mark.parametrize("model", model_configs_bias.keys())
@pytest.mark.parametrize("workspace_opt", [True])#, False])
def test_dpa_bias(dtype, bs, model, workspace_opt):
    """Test DotProductAttention module with different QKV layouts"""

    # Get configs
    config = model_configs_bias[model]
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)

    ckpt_attn = False
    if config.attn_type == 'self':
        qkv_layout = 'sb3hd'
    else:
        qkv_layout = 'sbhd_sbh2d'

    # Skip if only unfused backend is supported
    fused_attn_supported = _is_fused_attention_supported(
        config, dtype, qkv_layout=qkv_layout,
    )
    flash_attn_supported = _is_flash_attention_supported(config)
    unfused_attn_supported = _is_unfused_attention_supported(config)
    if not (fused_attn_supported or flash_attn_supported):
        pytest.skip(
            "Neither FusedAttention nor FlashAttention support this model config"
        )

    if config.attn_bias_type in ['post_scale_bias', 'alibi']:
        os.environ["NVTE_FUSED_ATTN_BACKEND"] = "0"
        fused_attn_fwd, fused_attn_bwd = _run_dot_product_attention(
            dtype, bs, config, "FusedAttention", ckpt_attn, qkv_layout, workspace_opt,
        )
        os.environ["NVTE_FUSED_ATTN_BACKEND"] = "1"
        fused_attn_fwd_arbi, fused_attn_bwd_arbi = _run_dot_product_attention(
            dtype, bs, config, "FusedAttention", ckpt_attn, qkv_layout, workspace_opt,
        )
        print('max512 fwd:')
        print(fused_attn_fwd.min().item(), fused_attn_fwd.max().item())
        #torch.save(fused_attn_fwd, 'fused_attn_fwd.pt')
        print('arbi fwd:')
        print(fused_attn_fwd_arbi.min().item(), fused_attn_fwd_arbi.max().item())
        #torch.save(fused_attn_fwd_arbi, 'fused_attn_fwd_arbi.pt')
        torch.testing.assert_close(fused_attn_fwd, fused_attn_fwd_arbi, **tols)
        for i in range(len(fused_attn_bwd)):
            print('max512 bwd ',i,':')
            print(fused_attn_bwd[i].min().item(), fused_attn_bwd[i].max().item())
            #torch.save(fused_attn_bwd[i], 'fused_attn_bwd_'+str(i)+'.pt')
            print('arbi bwd ',i,':')
            print(fused_attn_bwd_arbi[i].min().item(), fused_attn_bwd_arbi[i].max().item())
            #torch.save(fused_attn_bwd_arbi[i], 'fused_attn_bwd_arbi_'+str(i)+'.pt')
            torch.testing.assert_close(fused_attn_bwd[i], fused_attn_bwd_arbi[i], **tols)
        print("test_dpa_bias: fused_attn max512 matches fused_attn arbi")

        if unfused_attn_supported:
            unfused_attn_fwd, unfused_attn_bwd = _run_dot_product_attention(
                dtype, bs, config, "UnfusedDotProductAttention", ckpt_attn, qkv_layout, workspace_opt)
        if unfused_attn_supported and fused_attn_supported:
            torch.testing.assert_close(fused_attn_fwd, unfused_attn_fwd, **tols)
            torch.testing.assert_close(fused_attn_bwd, unfused_attn_bwd, **tols)
            print("test_dpa_bias: fused_attn max512 matches unfused_attn")
    #else:
    #    if unfused_attn_supported:
    #        unfused_attn_fwd, unfused_attn_bwd = _run_dot_product_attention(
    #            dtype, bs, config, "UnfusedDotProductAttention", ckpt_attn, qkv_layout, workspace_opt)
    #    if fused_attn_supported:
    #        fused_attn_fwd, fused_attn_bwd = _run_dot_product_attention(
    #            dtype, bs, config, "FusedAttention", ckpt_attn, qkv_layout, workspace_opt)
    #    if flash_attn_supported:
    #        flash_attn_fwd, flash_attn_bwd = _run_dot_product_attention(
    #            dtype, bs, config, "FlashAttention", ckpt_attn, qkv_layout, workspace_opt)
    #    if unfused_attn_supported and fused_attn_supported:
    #        torch.testing.assert_close(fused_attn_fwd, unfused_attn_fwd, **tols)
    #        torch.testing.assert_close(fused_attn_bwd, unfused_attn_bwd, **tols)
    #        print("test_dpa_bias: fused_attn matches unfused_attn")
    #    if unfused_attn_supported and flash_attn_supported:
    #        torch.testing.assert_close(flash_attn_fwd, unfused_attn_fwd, **tols)
    #        torch.testing.assert_close(flash_attn_bwd, unfused_attn_bwd, **tols)
    #        print("test_dpa_bias: flash_attn matches unfused_attn")
    #    if fused_attn_supported and flash_attn_supported:
    #        torch.testing.assert_close(flash_attn_fwd, fused_attn_fwd, **tols)
    #        torch.testing.assert_close(flash_attn_bwd, fused_attn_bwd, **tols)
    #        print("test_dpa_bias: flash_attn matches fused_attn")

################ test basic DPA with different qkv layouts ################ 
# To run a test in model_configs_layout:
# pytest -s -v tests/pytorch/test_fused_attn.py::test_dpa_layout[sbh3d-True-layout_2_1-2-dtype0]
# failing: all layout_1_ and layout_2_ due to bias

qkv_layouts = [
    'sb3hd', 'sbh3d', 'sbhd_sb2hd', 'sbhd_sbh2d', 'sbhd_sbhd_sbhd',
    'bs3hd', 'bsh3d', 'bshd_bs2hd', 'bshd_bsh2d', 'bshd_bshd_bshd',
    # will add tests for thd layouts later when the support is available in fused attention
    #'t3hd', 'th3d', 'thd_t2hd', 'thd_th2d', 'thd_thd_thd',
    ]

model_configs_layout = {
    # testname:    num_layers,  h, hg,   d,   sq,  skv,   p,      mask,             bias 
    "layout_0_0": ModelConfig(1, 16, 16,  64,  128,  128, 0.0,  "causal",         "no_bias"), # self
    "layout_0_1": ModelConfig(1, 16, 16,  64,  128,  256, 0.0, "no_mask",         "no_bias"), # cross 
    "layout_0_2": ModelConfig(1, 16, 16,  64,  128,  256, 0.0, "padding",         "no_bias"), # cross 
    "layout_x_0": ModelConfig(1, 16, 16,  64,  128,  128, 0.0,  "causal",         "no_bias"), # self
    "layout_x_1": ModelConfig(1, 16, 16,  64,  128,  256, 0.0, "no_mask",         "no_bias"), # cross 
    "layout_x_2": ModelConfig(1, 16, 16,  64,  128,  256, 0.0, "padding",         "no_bias"), # cross 
#    "layout_1_0": ModelConfig(1, 16, 16,  64,  128,  128, 0.0,  "causal", "post_scale_bias"), # self
#    "layout_1_1": ModelConfig(1, 16, 16,  64,  128,  256, 0.0, "no_mask",           "alibi"), # cross
#    "layout_1_2": ModelConfig(1, 16, 16,  64,  128,  256, 0.0, "padding", "post_scale_bias"), # cross
#    "layout_2_0": ModelConfig(1, 24, 24, 128, 2048, 2048, 0.0,  "causal", "post_scale_bias"), # self
#    "layout_2_1": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0, "no_mask", "post_scale_bias"), # cross
#    "layout_2_2": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0, "padding", "post_scale_bias"), # cross
}

@pytest.mark.skipif(
    _cudnn_version < [8,9,5], reason="cuDNN 8.9.5+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("bs", batch_sizes_lean)
@pytest.mark.parametrize("model", model_configs_layout.keys())
@pytest.mark.parametrize("workspace_opt", [True])#, False])
@pytest.mark.parametrize("qkv_layout", qkv_layouts)
def test_dpa_qkv_layout(dtype, bs, model, workspace_opt, qkv_layout):
    """Test DotProductAttention module with different QKV layouts"""

    # Get configs
    config = model_configs_layout[model]
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)

    ckpt_attn = False
    if '3' in qkv_layout and config.attn_type == 'cross':
        pytest.skip(
            "No need to test this layout for cross attention"
        )

    # Skip if fused backend is not supported
    fused_attn_supported = _is_fused_attention_supported(
        config, dtype, qkv_layout=qkv_layout,
    )
    unfused_attn_supported = _is_unfused_attention_supported(config)
    if not (fused_attn_supported or flash_attn_supported):
        pytest.skip(
            "Neither FusedAttention nor FlashAttention support this model config"
        )

    # UnfusedDotProductAttention backend
    if unfused_attn_supported:
        unfused_attn_fwd, unfused_attn_bwd = _run_dot_product_attention(
            dtype, bs, config, "UnfusedDotProductAttention", ckpt_attn, qkv_layout, workspace_opt)
    # FusedAttention backend
    if fused_attn_supported:
        fused_attn_fwd, fused_attn_bwd = _run_dot_product_attention(
            dtype, bs, config, "FusedAttention", ckpt_attn, qkv_layout, workspace_opt)

    if unfused_attn_supported and fused_attn_supported:
        torch.testing.assert_close(fused_attn_fwd, unfused_attn_fwd, **tols)
        torch.testing.assert_close(fused_attn_bwd, unfused_attn_bwd, **tols)

        print('fused fwd:')
        print(fused_attn_fwd.min().item(), fused_attn_fwd.max().item())
        print('unfused fwd:')
        print(unfused_attn_fwd.min().item(), unfused_attn_fwd.max().item())
        #torch.save(fused_attn_fwd, 'fused_attn_fwd.pt')
        #torch.save(unfused_attn_fwd, 'unfused_attn_fwd.pt')
        torch.testing.assert_close(fused_attn_fwd, unfused_attn_fwd, **tols)
        for i in range(len(unfused_attn_bwd)):
            print('fused bwd:',i)
            print(fused_attn_bwd[i].min().item(), fused_attn_bwd[i].max().item())
            print('unfused bwd:',i)
            print(unfused_attn_bwd[i].min().item(), unfused_attn_bwd[i].max().item())
            #torch.save(fused_attn_bwd[i], 'fused_attn_bwd_'+str(i)+'.pt')
            #torch.save(unfused_attn_bwd[i], 'unfused_attn_bwd_'+str(i)+'.pt')
            torch.testing.assert_close(fused_attn_bwd[i], unfused_attn_bwd[i], **tols)

################ core DPA function ################ 
def _run_dot_product_attention(dtype, bs, config, backend, ckpt_attn, qkv_layout, workspace_opt):

    # set rng and env vars
    reset_rng_states()
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"
        os.environ["NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT"] = "1" if workspace_opt else "0"

    # generate seqlens
    qkv_format = ''.join([i for i in qkv_layout.split('_')[0] if i.isalpha()])
    if "padding" in config.attn_mask_type or qkv_format == 'thd':
        if config.attn_type == 'self':
            seqlens_q = torch.randint(1, config.max_seqlen_q, [bs], dtype = torch.int32).cuda()
            seqlens_kv = seqlens_q
        if config.attn_type == 'cross':
            seqlens_q = torch.randint(1, config.max_seqlen_q, [bs], dtype = torch.int32).cuda()
            seqlens_kv = torch.randint(1, config.max_seqlen_kv, [bs], dtype = torch.int32).cuda()
    else:
        seqlens_q = torch.empty(bs, dtype = torch.int32).cuda()
        seqlens_q.fill_(config.max_seqlen_q)
        seqlens_kv = torch.empty(bs, dtype = torch.int32).cuda()
        seqlens_kv.fill_(config.max_seqlen_kv)
    #print('seqlens q:',seqlens_q)
    #print('seqlens kv:',seqlens_kv)
    cu_seqlens_q = torch.zeros(bs + 1, dtype = torch.int32).cuda()
    cu_seqlens_kv = torch.zeros(bs + 1, dtype = torch.int32).cuda()
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim = 0)
    cu_seqlens_kv[1:] = torch.cumsum(seqlens_kv, dim = 0)
    attention_mask = None
    if "padding" in config.attn_mask_type:
        if config.attn_type == 'self':
            attention_mask_q = torch.Tensor([])
            for i in range(bs):
                attention_mask_q = torch.cat([attention_mask_q,
                    torch.Tensor([True]*seqlens_q[i] + [False]*(config.max_seqlen_q-seqlens_q[i]))
                    .to(torch.bool).unsqueeze(0).unsqueeze(0).unsqueeze(0)], dim=0) 
            attention_mask = attention_mask_q.cuda()
        if config.attn_type == 'cross':
            attention_mask_q = torch.Tensor([])
            attention_mask_kv = torch.Tensor([])
            for i in range(bs):
                attention_mask_q = torch.cat([attention_mask_q,
                    torch.Tensor([True]*seqlens_q[i] + [False]*(config.max_seqlen_q-seqlens_q[i]))
                    .to(torch.bool).unsqueeze(0).unsqueeze(0).unsqueeze(0)], dim=0) 
                attention_mask_kv = torch.cat([attention_mask_kv,
                    torch.Tensor([True]*seqlens_kv[i] + [False]*(config.max_seqlen_kv-seqlens_kv[i]))
                    .to(torch.bool).unsqueeze(0).unsqueeze(0).unsqueeze(0)], dim=0) 
            attention_mask = (attention_mask_q.cuda(), attention_mask_kv.cuda())

    # generate input tensor
    dim_to_num = {'b': bs,
        'sq': config.max_seqlen_q,
        'skv': config.max_seqlen_kv,
        'h': config.num_attention_heads,
        'hg': config.num_gqa_groups,
        'd': config.head_dim,
        't': cu_seqlens_q[-1],
        'tg': cu_seqlens_kv[-1],
        '3': 3,
        '2': 2}
    inp = []
    for i,layout in enumerate(qkv_layout.split('_')):
        layout = '_'.join(layout)
        if i == 0:
            layout = layout.replace('s', 'sq')
        else:
            layout = layout.replace('s', 'skv')
            layout = layout.replace('h', 'hg')
            layout = layout.replace('t', 'tg')
        tensor_shape = [dim_to_num[j] for j in layout.split('_')]
        tensor = 0.1 * torch.randn(tensor_shape, dtype = dtype).cuda()
        tensor_count = 1
        split_dim = 0
        for dim,l in enumerate(layout.split('_')):
            if l.isdigit():
                tensor_count = int(l)
                split_dim = dim
                break
        tensors = torch.split(tensor, 1, dim = split_dim) if split_dim != 0 else [tensor]
        for j in range(tensor_count):
            if split_dim != 0:
                inp.append(tensors[j].squeeze(split_dim))
            else:
                inp.append(tensors[j])
    for i in range(3):
        inp[i].requires_grad=True

    # generate output gradient
    qkv_format_kv = '_'.join(qkv_format)
    qkv_format_kv = qkv_format_kv.replace('s', 'sq')
    op_grad_shape = [dim_to_num[i] for i in qkv_format_kv.split('_')]
    op_grad_shape_new = [*op_grad_shape[:-2], op_grad_shape[-2] * op_grad_shape[-1]]
    op_grad = 0.001 * torch.randint(0, 200, op_grad_shape_new, dtype = dtype).cuda()

    # generate bias
    if config.attn_bias_type == 'no_bias':
        bias = None
    if config.attn_bias_type == 'post_scale_bias':
        bias = torch.randn(1, config.num_attention_heads, config.max_seqlen_q, config.max_seqlen_kv, dtype = dtype).cuda()
    elif config.attn_bias_type == 'alibi':
        if os.environ['NVTE_FUSED_ATTN_BACKEND'] == '0':
            config.attn_bias_type = 'post_scale_bias'
            n = 2 ** math.floor(math.log2(config.num_attention_heads))
            m_0 = 2.0 ** (-8.0 / n)
            m = torch.pow(m_0, torch.arange(1, 1 + n))

            a = torch.ones(config.max_seqlen_q, config.max_seqlen_kv)
            b = torch.triu(a,diagonal=1)
            c = b.cumsum(dim=-1)
            d = c - torch.transpose(c, 0, 1)
            bias = d.expand(1, config.num_attention_heads, config.max_seqlen_q, config.max_seqlen_kv)
            for i in range(config.num_attention_heads):
                bias[0,i,:,:] = m[i] *  bias[0,i,:,:]
            bias = bias.to(dtype = dtype).cuda()
        else:
            bias = None

    # generate rng
    _DUMMY_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()
    _DUMMY_CUDA_RNG_STATE_TRACKER.add("model-parallel-rng", seed)

    def get_dummy_cuda_rng_tracker():
        """Get cuda rng tracker."""
        return _DUMMY_CUDA_RNG_STATE_TRACKER

    # set up model 
    block = (
         DotProductAttention(
                config.num_attention_heads,
                config.head_dim,
                num_gqa_groups = config.num_gqa_groups,
                attention_dropout = config.dropout_p,
                qkv_format = qkv_format,
                attn_mask_type = config.attn_mask_type,
                sequence_parallel = False,
                tp_size = 1,
                get_rng_state_tracker = get_dummy_cuda_rng_tracker,
                tp_group = None,
                layer_number = 1,
                attention_type = config.attn_type,
        ).to(dtype = dtype).cuda()
    )

    op = block(inp[0], inp[1], inp[2],
            attention_mask = attention_mask,
            qkv_format = qkv_format,
            cu_seqlens_q = cu_seqlens_q,
            cu_seqlens_kv = cu_seqlens_kv,
            attn_mask_type = config.attn_mask_type,
            checkpoint_core_attention = ckpt_attn,
            core_attention_bias_type = config.attn_bias_type,
            core_attention_bias = bias,
            fast_zero_fill = True)

    op.backward(op_grad)

    return op, (inp[0].grad, inp[1].grad, inp[2].grad)

################ test basic TE layer cases ################ 
# To run a test in model_configs_lean:
#   pytest -s -v tests/pytorch/test_fused_attn.py::test_transformer_layer[False-True-lean_1_0-2-dtype0] 
# failing: lean_1_1, lean_2_1 due to bias

model_configs_lean = {
    #     test:    num_layers,  h, hg,   d,   sq,  skv,   p,      mask,             bias 
    "lean_1_0": ModelConfig(1, 16, 16,  64,  128,  128, 0.0,  "causal",         "no_bias"),
    "lean_1_1": ModelConfig(1, 16, 16,  64,  128,  128, 0.0, "no_mask", "post_scale_bias"),
#    "lean_1_2": ModelConfig(1, 16, 16,  64,  128,  128, 0.0, "padding", "post_scale_bias"),
    "lean_2_0": ModelConfig(1, 24, 24, 128, 2048, 2048, 0.0,  "causal",         "no_bias"),
    "lean_2_1": ModelConfig(1, 24, 24, 128, 2048, 2048, 0.0, "no_mask", "post_scale_bias"),
#    "lean_2_2": ModelConfig(1, 24, 24, 128, 2048, 2048, 0.0, "padding", "post_scale_bias"),
}

@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("bs", batch_sizes_lean)
@pytest.mark.parametrize("model", model_configs_lean.keys())
@pytest.mark.parametrize("fused_qkv_params", [True, False])
@pytest.mark.parametrize("RoPE", [True, False])
def test_transformer_layer(dtype, bs, model, fused_qkv_params, RoPE):
    """Test TransformerLayer module when its DotProductAttention is enabled with
    FlashAttention, FusedAttention, or UnfusedDotProductAttention backend"""

    # Get configs
    config = model_configs_lean[model]
    tols = dict(atol=5e-1, rtol=5e-2)

    ckpt_attn = False
    qkv_format = 'sbhd'
    workspace_opt = True

    # Skip if only unfused backend is supported
    fused_attn_supported = _is_fused_attention_supported(
        config,
        dtype,
        qkv_layout="sbh3d" if fused_qkv_params else "sb3hd",
    )
    flash_attn_supported = _is_flash_attention_supported(config)
    unfused_attn_supported = _is_unfused_attention_supported(config)
    if not (fused_attn_supported or flash_attn_supported):
        pytest.skip(
            "Neither FusedAttention nor FlashAttention support this model config"
        )

    # UnfusedDotProductAttention backend
    if unfused_attn_supported:
        unfused_attn_fwd, unfused_attn_bwd = _run_transformer_layer(
            dtype,
            bs,
            config,
            "UnfusedDotProductAttention",
            ckpt_attn,
            qkv_format,
            workspace_opt,
            fused_qkv_params,
            RoPE,
        )

    # FusedAttention backend
    if fused_attn_supported and unfused_attn_supported:
        #if config.max_seqlen_q <= 512 and config.max_seqlen_kv <= 512:
        #    os.environ["NVTE_FUSED_ATTN_BACKEND"] = "0"
        fused_attn_fwd, fused_attn_bwd = _run_transformer_layer(
            dtype,
            bs,
            config,
            "FusedAttention",
            ckpt_attn,
            qkv_format,
            workspace_opt,
            fused_qkv_params,
            RoPE,
        )
        print(fused_attn_fwd.min().item(), fused_attn_fwd.max().item())
        print(unfused_attn_fwd.min().item(), unfused_attn_fwd.max().item())
        print(fused_attn_bwd.min().item(), fused_attn_bwd.max().item())
        print(unfused_attn_bwd.min().item(), unfused_attn_bwd.max().item())
        torch.testing.assert_close(fused_attn_fwd, unfused_attn_fwd, **tols)
        torch.testing.assert_close(fused_attn_bwd, unfused_attn_bwd, **tols)
        print("test_transformer_layer: fused_attn matches unfused_attn")

    # FlashAttention backend
    if flash_attn_supported and unfused_attn_supported:
        flash_attn_fwd, flash_attn_bwd = _run_transformer_layer(
            dtype,
            bs,
            config,
            "FlashAttention",
            ckpt_attn,
            qkv_format,
            workspace_opt,
            fused_qkv_params,
            RoPE,
        )
        torch.testing.assert_close(flash_attn_fwd, unfused_attn_fwd, **tols)
        torch.testing.assert_close(flash_attn_bwd, unfused_attn_bwd, **tols)
        print("test_transformer_layer: flash_attn matches unfused_attn")

################ test TE layer with MQA/GQA ################ 
# To run a test in model_configs_lean:
#   pytest -s -v tests/pytorch/test_fused_attn.py::test_transformer_layer_gqa[lean_1_0-2-dtype0] 
# all passed or skipped; fused attn MQA is still waiting dK/dV to reduce to b_hg_s_d

@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("bs", batch_sizes_lean)
@pytest.mark.parametrize("model", model_configs_lean.keys())
def test_transformer_layer_gqa(dtype, bs, model):
    """Test TransformerLayer module with MQA/GQA"""

    ckpt_attn = False
    qkv_format = 'sbhd'
    workspace_opt = True
    fused_qkv_params = True
    RoPE = False 

    config = model_configs_lean[model]
    def find_factors(x):
       f = []
       for i in range(2, x + 1):
           if x % i == 0:
               f.append(i)
       return f

    # Skip if only unfused backend is supported
    if not (_flash_attn_2_available and _is_flash_attention_supported(config)):
        pytest.skip("FlashAttention does not support this model config")

    num_querys_per_gqa_group = find_factors(config.num_attention_heads)

    for num_q_per_gqa_group in num_querys_per_gqa_group:
        config.num_gqa_groups=config.num_attention_heads // num_q_per_gqa_group
        flash_attn_fwd, flash_attn_bwd = _run_transformer_layer(
            dtype,
            bs,
            config,
            "FlashAttention",
            ckpt_attn,
            qkv_format,
            workspace_opt,
            fused_qkv_params,
            RoPE,
        )
        unfused_attn_fwd, unfused_attn_bwd = _run_transformer_layer(
            dtype,
            bs,
            config,
            "UnfusedDotProductAttention",
            ckpt_attn,
            qkv_format,
            workspace_opt,
            fused_qkv_params,
            RoPE,
        )

        atol, rtol = 5e-1, 5e-2
        torch.testing.assert_close(flash_attn_fwd, unfused_attn_fwd, atol=atol, rtol=rtol)
        torch.testing.assert_close(flash_attn_bwd, unfused_attn_bwd, atol=atol, rtol=rtol)
        print("test_transformer_layer_gqa: flash_attn matches unfused_attn")

        if config.num_gqa_groups == 1:
            fused_attn_fwd, fused_attn_bwd = _run_transformer_layer(
                dtype,
                bs,
                config,
                "FusedAttention",
                ckpt_attn,
                qkv_format,
                workspace_opt,
                fused_qkv_params,
                RoPE,
            )
            torch.testing.assert_close(fused_attn_fwd, unfused_attn_fwd, atol=atol, rtol=rtol)
            torch.testing.assert_close(fused_attn_bwd, unfused_attn_bwd, atol=atol, rtol=rtol)
            print("test_transformer_layer_gqa: fused_attn matches unfused_attn")

################ core TE layer function ################ 
def _run_transformer_layer(dtype, bs, config, backend, ckpt_attn, qkv_layout, workspace_opt, fused_qkv_params, RoPE):

    reset_rng_states()
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"

    inp = torch.randn(
            config.max_seqlen_q, bs, config.hidden_size,
            dtype=dtype).cuda()
    inp.requires_grad=True

    if "padding" in config.attn_mask_type:
        seqlens = torch.randint(1, config.max_seqlen_q, [bs], dtype = torch.int32).cuda()
    else:
        seqlens = torch.empty(bs, dtype=torch.int32).cuda()
        seqlens.fill_(config.max_seqlen_q)
    cu_seqlens = torch.zeros(bs + 1, device=inp.device, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)

    attention_mask = None
    if "padding" in config.attn_mask_type:
        attention_mask_q = torch.Tensor([])
        for i in range(bs):
            attention_mask_q = torch.cat([attention_mask_q,
                torch.Tensor([True]*seqlens_q[i] + [False]*(config.max_seqlen_q-seqlens_q[i]))
                .to(torch.bool).unsqueeze(0).unsqueeze(0).unsqueeze(0)])
        attention_mask = attention_mask_q

    sigma = 0.02
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    layer_number = 1
    drop_path_rate = 0.0
    drop_path_rates = [
            rate.item() for rate in torch.linspace(0, drop_path_rate, config.num_layers)]

    if config.attn_bias_type == 'no_bias':
        bias = None
    if config.attn_bias_type == 'post_scale_bias':
        bias = torch.randn(1, config.num_attention_heads, config.max_seqlen_q, config.max_seqlen_kv, dtype = dtype).cuda()
    elif config.attn_bias_type == 'alibi':
        if os.environ['NVTE_FUSED_ATTN_BACKEND'] == '0':
            config.attn_bias_type = 'post_scale_bias'
            n = 2 ** math.floor(math.log2(config.num_attention_heads))
            m_0 = 2.0 ** (-8.0 / n)
            m = torch.pow(m_0, torch.arange(1, 1 + n))

            a = torch.ones(config.max_seqlen_q, config.max_seqlen_kv)
            b = torch.triu(a,diagonal=1)
            c = b.cumsum(dim=-1)
            d = c - torch.transpose(c, 0, 1)
            bias = d.expand(1, config.num_attention_heads, config.max_seqlen_q, config.max_seqlen_kv)
            for i in range(config.num_attention_heads):
                bias[0,i,:,:] = m[i] *  bias[0,i,:,:]
            bias = bias.to(dtype = dtype).cuda()
        else:
            bias = None

    rotary_pos_emb = None
    if RoPE:
        PE = RotaryPositionEmbedding(dim=config.head_dim)
        rotary_pos_emb = PE(config.max_seqlen_q).cuda().to(dtype=dtype)

    block = (
        TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_attention_heads,
            num_gqa_groups=config.num_gqa_groups,
            layernorm_epsilon=1e-5,
            hidden_dropout=0.0,
            attention_dropout=config.dropout_p,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
            kv_channels=config.head_dim,
            self_attn_mask_type=config.attn_mask_type,
            tp_group=None,
            tp_size=1,
            params_dtype=dtype,
            get_rng_state_tracker=None,
            fuse_wgrad_accumulation=False,
            seq_length=config.max_seqlen_q,
            micro_batch_size=bs,
            sequence_parallel=False,
            apply_residual_connection_post_layernorm=False,
            output_layernorm=False,
            layer_type="encoder",
            drop_path_rate=drop_path_rates[layer_number - 1],
            set_parallel_mode=True,
            fuse_qkv_params=fused_qkv_params,
            zero_centered_gamma=False,
            qkv_weight_interleaved=False,
            ub_tp_comm_overlap=False,
            bias=True,
        )
        .to(dtype=dtype)
        .cuda()
    )

    op = block(inp,
        attention_mask=attention_mask,
        self_attn_mask_type=config.attn_mask_type,
        checkpoint_core_attention=False,
        rotary_pos_emb=rotary_pos_emb,
        core_attention_bias_type=config.attn_bias_type,
        core_attention_bias=bias)
    loss = op.sum()
    loss.backward()

    return op, inp.grad


model_configs_fp8 = {
    "test1": ModelConfig(1, 16, 16, 64, 512, 512, 0.0, "no_mask", "no_bias"),
}
batch_sizes_fp8 = [1, 4]
param_types_fp8 = [torch.float16]

@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("dtype", param_types_fp8)
@pytest.mark.parametrize("bs", batch_sizes_fp8)
@pytest.mark.parametrize("model", model_configs_fp8.keys())
def test_dpa_fp8(dtype, bs, model):
    """Test FP8 dot-product attention with different backends

    FusedAttention uses fused_attn_fwd/bwd_qkvpacked from
    cpp_extensions. UnfusedDotProductAttention uses plain PyTorch
    operations.

    """

    config = model_configs_fp8[model]

    # Skip if not supported
    if not _is_fused_attention_supported(config, dtype):
        pytest.skip("FusedAttention does not support this model config")

    # Run dot-product attention with different backends
    fused_attn_fwd, fused_attn_bwd = _run_dpa_fp8(
        dtype,
        bs,
        config,
        "FusedAttention"
    )
    unfused_attn_fwd, unfused_attn_bwd = _run_dpa_fp8_ref(
        dtype,
        bs,
        config,
        "UnfusedDotProductAttention",
    )

    # Check that results match
    tols = dict(atol=2.5e-2, rtol=2.5e-2)
    torch.testing.assert_close(fused_attn_fwd, unfused_attn_fwd, **tols)
    torch.testing.assert_close(fused_attn_bwd, unfused_attn_bwd, **tols)

def _run_dpa_fp8(dtype, bs, config, backend):

    reset_rng_states()
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"

    inp = 0.01 * torch.randn(
            bs * config.max_seqlen_q, config.num_attention_heads * config.head_dim,
            dtype=dtype).cuda()
    inp.requires_grad=True
    seqlens = torch.empty(bs, dtype=torch.int32).cuda()
    seqlens.fill_(config.max_seqlen_q)
    cu_seqlens = torch.zeros(bs + 1, device=inp.device, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    op_grad = 0.01 * torch.randn(
        bs * config.max_seqlen_q, config.num_attention_heads * config.head_dim,
        dtype=dtype).cuda()
    torch.save(op_grad, 'op_grad.pt')

    fp8_recipe = recipe.DelayedScaling(
        margin=0,
        interval=1,
        fp8_format=recipe.Format.HYBRID,
        amax_history_len=1,
        amax_compute_algo="most_recent",
    )

    dpa = DPA_FP8(config).to(dtype=torch.float16).cuda()
    with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        op = dpa(inp, cu_seqlens, config.max_seqlen_q)
        op.backward(op_grad)

    context = torch.load("ctx.pt")
    dqkv = torch.load('dqkv.pt')
    return (context.view(bs, config.max_seqlen_q, -1).transpose(0,1),
        dqkv.view(bs, config.max_seqlen_q, 3, config.num_attention_heads, config.head_dim).transpose(0,1).contiguous())

def _run_dpa_fp8_ref(dtype, bs, config, backend):

    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"

    inp = torch.load('qkv.pt').cuda()
    inp.requires_grad=True
    seqlens = torch.empty(bs, dtype=torch.int32).cuda()
    seqlens.fill_(config.max_seqlen_q)
    cu_seqlens = torch.zeros(bs + 1, device=inp.device, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    op_grad = torch.load('op_grad.pt').cuda().view(bs, config.max_seqlen_q, -1).transpose(0,1)

    _DUMMY_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()
    _DUMMY_CUDA_RNG_STATE_TRACKER.add("model-parallel-rng", seed)

    def get_dummy_cuda_rng_tracker():
        """Get cuda rng tracker."""
        return _DUMMY_CUDA_RNG_STATE_TRACKER

    block = (
         DotProductAttention(
                config.num_attention_heads,
                config.head_dim,
                attention_dropout=config.dropout_p,
                sequence_parallel=False,
                tp_size=1,
                get_rng_state_tracker=get_dummy_cuda_rng_tracker,
                tp_group=None,
                layer_number=1,
                attention_type="self"
        ).to(dtype=dtype).cuda()
    )

    q = inp[:, :,0,:,:]
    k = inp[:, :,1,:,:]
    v = inp[:, :,2,:,:]
    op = block(q, k, v, attn_mask_type=config.attn_mask_type)
    op.backward(op_grad)

    return op, inp.grad

_CUBLASLT_WORKSPACE_SIZE_BYTES = 33_554_432  # 32MiB
_2X_ACC_FPROP = False
_2X_ACC_DGRAD = False
_2X_ACC_WGRAD = False

META_QKV  = tex.FP8FwdTensors.GEMM1_OUTPUT
META_O    = tex.FP8FwdTensors.GEMM2_INPUT
META_DO   = tex.FP8BwdTensors.GRAD_INPUT2
META_DQKV = tex.FP8BwdTensors.GRAD_OUTPUT1

META_S    = tex.FP8FwdTensors.GEMM3_WEIGHT
META_DS   = tex.FP8BwdTensors.GRAD_INPUT3

class _dpa_fp8(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        qkv_weight: torch.Tensor,
        qkv_bias: torch.Tensor,
        cu_seqlens: torch.Tensor,
        num_attention_heads: int,
        p_dropout: float,
        max_s: int,
        fast_zero_fill: bool,
        fp8_meta: Dict[str, Any],
        workspace: torch.Tensor,
        is_training: bool,
    ) -> torch.Tensor:

        assert inp.dim() == 2
        in_features = qkv_weight.shape[-1]
        h = num_attention_heads
        d = in_features // h
        b = cu_seqlens.numel() - 1
        is_nl = False
        if b < 4 and b > 1:
            max_s = 512
            is_nl = True

        fp8_dtype_forward = fp8.get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)

        inputmat, inputmat_t = ext.fp8_cast_transpose_fused(
            inp,
            fp8_meta["scaling_fwd"],
            tex.FP8FwdTensors.GEMM1_INPUT,
            fp8_dtype_forward,
        )

        qkv_weight_fp8, qkv_weight_t_fp8 = ext.fp8_cast_transpose_fused(
            qkv_weight,
            fp8_meta["scaling_fwd"],
            tex.FP8FwdTensors.GEMM1_WEIGHT,
            fp8_dtype_forward,
        )

        M = None
        ZInv = None
        philox_unpacked = None

        qkv_out, _ = ext.fp8_gemm(
            qkv_weight_fp8,
            fp8_meta["scaling_fwd"].scale_inv,
            tex.FP8FwdTensors.GEMM1_WEIGHT,
            fp8_dtype_forward,
            inputmat,
            fp8_meta["scaling_fwd"].scale_inv,
            tex.FP8FwdTensors.GEMM1_INPUT,
            fp8_dtype_forward,
            torch.uint8,
            workspace,
            bias=qkv_bias,
            use_bias=True,
            out_index=META_QKV,
            fp8_meta_tensor=fp8_meta["scaling_fwd"],
            use_split_accumulator=_2X_ACC_FPROP,
            D_dtype=fp8_dtype_forward,
        )
        qkv_out = qkv_out.view(-1, 3, h, d)
        qkv_out_fp16 = ext.cast_from_fp8(qkv_out, fp8_meta["scaling_fwd"],
                META_QKV, fp8_dtype_forward,
                tex.DType.kFloat16).view(b, max_s, 3, h, d).transpose(0,1).contiguous()
        torch.save(qkv_out_fp16, 'qkv.pt')

        # FMHA
        context_, aux_ctx_tensors, *rest = fused_attn_fwd(
                is_training,
                max_s,
                max_s,
                cu_seqlens,
                cu_seqlens,
                qkv_out[:,0,:,:],
                qkv_out[:,1,:,:],
                qkv_out[:,2,:,:],
                fp8_dtype_forward,
                FusedAttnBackend["FP8"],
                None,
                fp8_meta["scaling_fwd"].scale_inv[META_QKV],
                fp8_meta["scaling_fwd"].scale[META_S],
                fp8_meta["scaling_fwd"].scale[META_O],
                fp8_meta["scaling_fwd"].amax_history[0][META_S],
                fp8_meta["scaling_fwd"].amax_history[0][META_O],
                attn_scale=None,
                dropout=p_dropout,
                fast_zero_fill=fast_zero_fill,
                qkv_layout="t3hd",
                attn_bias_type="no_bias",
                attn_mask_type="padding",
                rng_gen=None,
                )
        M, ZInv, philox_unpacked = aux_ctx_tensors

        context = context_.view(-1, in_features)
        context_t = tex.fp8_transpose(context, fp8_dtype_forward)

        ctx.save_for_backward(
            inputmat_t, qkv_weight_t_fp8, workspace,
            qkv_out,
            context_, context_t,
            fp8_meta["scaling_fwd"].scale,
            fp8_meta["scaling_fwd"].scale_inv,
        )
        ctx.aux_ctx_tensors = aux_ctx_tensors
        ctx.fp8_meta = fp8_meta
        ctx.cu_seqlens = cu_seqlens
        ctx.p_dropout = p_dropout
        ctx.max_s = max_s
        ctx.fast_zero_fill = fast_zero_fill
        ctx.is_nl = is_nl
        ctx.hidden_size = in_features
        ctx.num_attention_heads = num_attention_heads

        context_fp16 = ext.cast_from_fp8(context, fp8_meta["scaling_fwd"],
                META_O, fp8_dtype_forward, tex.DType.kFloat16)
        torch.save(context_fp16, 'ctx.pt')
        return context_fp16


    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:

        with _prepare_backward(True, ctx.fp8_meta, None, 1, name="_DPA"):
            (
                inputmat_t,
                qkv_weight_t_fp8,
                workspace,
                qkv_out,
                context, context_t,
                fwd_scales,
                fwd_scale_inverses,
            ) = ctx.saved_tensors
            fp8_dtype_forward = fp8.get_fp8_te_dtype(
                ctx.fp8_meta["recipe"], fprop_tensor=True
            )
            fp8_dtype_backward = fp8.get_fp8_te_dtype(
                ctx.fp8_meta["recipe"], fprop_tensor=False
            )

            proj_dgrad = ext.cast_to_fp8(
                grad_output, ctx.fp8_meta["scaling_bwd"], META_DO, fp8_dtype_backward
            )

            dq, dk, dv, *rest = fused_attn_bwd(
                    ctx.max_s,
                    ctx.max_s,
                    ctx.cu_seqlens,
                    ctx.cu_seqlens,
                    qkv_out[:,0,:,:],
                    qkv_out[:,1,:,:],
                    qkv_out[:,2,:,:],
                    context,
                    proj_dgrad.view_as(context),
                    fp8_dtype_forward,
                    ctx.aux_ctx_tensors,
                    FusedAttnBackend["FP8"],
                    fwd_scale_inverses[META_QKV], # d_scale_qkv,
                    fwd_scale_inverses[META_S], # d_scale_s,
                    fwd_scale_inverses[META_O], # d_scale_o,
                    ctx.fp8_meta['scaling_bwd'].scale_inv[META_DO], # d_scale_do
                    fwd_scales[META_S], # q_scale_s
                    ctx.fp8_meta['scaling_bwd'].scale[META_DS], # q_scale_ds
                    ctx.fp8_meta['scaling_bwd'].scale[META_DQKV], # q_scale_dqkv
                    ctx.fp8_meta['scaling_bwd'].amax_history[0][META_DS], # amax_ds
                    ctx.fp8_meta['scaling_bwd'].amax_history[0][META_DQKV], # amax_dqkv
                    None,
                    ctx.p_dropout,
                    ctx.fast_zero_fill,
                    "t3hd",
                    "no_bias",
                    "padding",
                    )
            dqkv = torch.cat([dq.unsqueeze(1), dk.unsqueeze(1), dv.unsqueeze(1)], dim=1)

            dqkv_grad_output_c = dqkv.view(-1, 3*ctx.hidden_size)
            dqkv_grad_output_c_fp16 = ext.cast_from_fp8(dqkv_grad_output_c,
                ctx.fp8_meta["scaling_bwd"], META_DQKV,
                fp8_dtype_backward, tex.DType.kFloat16)
            torch.save(dqkv_grad_output_c_fp16, 'dqkv.pt')

            qkv_bgrad, dqkv_grad_output_t = ext.fp8_transpose_bgrad_fused(
                dqkv_grad_output_c,
                ctx.fp8_meta["scaling_bwd"],
                META_DQKV,
                fp8_dtype_backward,
                torch.float16,
            )

            # QKV DGRAD
            qkv_dgrad, _ = ext.fp8_gemm(
                qkv_weight_t_fp8,
                fwd_scale_inverses,
                tex.FP8FwdTensors.GEMM1_WEIGHT,
                fp8_dtype_forward,
                dqkv_grad_output_c,
                ctx.fp8_meta["scaling_bwd"].scale_inv,
                META_DQKV,
                fp8_dtype_backward,
                torch.float16,
                workspace,
                use_split_accumulator=_2X_ACC_DGRAD,
            )
            # QKV WGRAD
            qkv_wgrad, _ = ext.fp8_gemm(
                inputmat_t,
                fwd_scale_inverses,
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                dqkv_grad_output_t,
                ctx.fp8_meta["scaling_bwd"].scale_inv,
                META_DQKV,
                fp8_dtype_backward,
                torch.float16,
                workspace,
                use_split_accumulator=_2X_ACC_WGRAD,
            )

        return (qkv_dgrad,
            qkv_wgrad,
            qkv_bgrad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None)

class DPA_FP8(TransformerEngineBaseModule):
    def __init__(
        self,
        config,
        params_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.p_dropout = config.dropout_p
        self.h = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.fast_zero_fill = True

        self.qkv_weight = torch.nn.Parameter(
            torch.empty(
                self.hidden_size * 3,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        self.fp8_weight_shapes.append(self.qkv_weight.shape)
        self.qkv_bias = torch.nn.Parameter(
            torch.empty(
                self.hidden_size * 3,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        with torch.no_grad():
            self.qkv_bias.zero_()
            self.qkv_weight.fill_(1.0)
        self.workspace = torch.empty(
            _CUBLASLT_WORKSPACE_SIZE_BYTES, dtype=torch.int8, device="cuda"
        )

    def forward(
        self, inp: torch.Tensor,
        cu_seqlens, max_s,
    ) -> torch.Tensor:
        with self.prepare_forward(inp, None, num_gemms=3) as inp:
            out = _dpa_fp8.apply(
                inp,
                self.qkv_weight,
                self.qkv_bias,
                cu_seqlens,
                self.h,
                self.p_dropout,
                max_s,
                self.fast_zero_fill,
                self.fp8_meta,
                self.workspace,
                self.training)
        return out

    def get_fp8_weights_scratchpad(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[torch.Tensor]:
        """Needs override."""
