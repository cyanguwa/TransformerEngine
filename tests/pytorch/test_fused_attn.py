# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import pytest

from transformer_engine.pytorch.utils import (
    init_method_normal,
    scaled_init_method_normal,
)
from transformer_engine.pytorch import TransformerLayer
from transformer_engine.pytorch.attention import DotProductAttention 
import os

class ModelConfig:
    def __init__(
        self, num_layers, hidden_size, num_attention_heads, head_dim, seq_len,
        dropout_p, attn_mask_type, 
    ):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim 
        assert (hidden_size == num_attention_heads * head_dim
                ), """hidden_size must be = num_heads x head_dim."""
        self.seq_len = seq_len
        self.dropout_p = dropout_p
        self.attn_mask_type  = attn_mask_type

model_configs = {
    "test1": ModelConfig(1, 1024, 16, 64, 128, 0.0, "causal"),
    "test2": ModelConfig(1, 1024, 16, 64, 512, 0.0, "causal"),
    "test3": ModelConfig(1, 1024, 16, 64, 2048, 0.0, "causal"),
    #"test4": ModelConfig(1, 1024, 16, 64, 2048, 0.1, "causal"),
}

param_types = [torch.float16]
if torch.cuda.is_bf16_supported():
    param_types.append(torch.bfloat16)

batch_sizes = [2, 48]

@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", model_configs.keys())
def test_dot_product_attention(dtype, bs, model):
    """Test DotProductAttention module with three backends,
    FlashAttention, FusedAttention and UnfusedDotProductAttention"""

    config = model_configs[model]

    flash_attn_fwd, flash_attn_bwd = _run_dot_product_attention(
            dtype, bs, config, "FlashAttention")
    fused_attn_fwd, fused_attn_bwd = _run_dot_product_attention(
            dtype, bs, config, "FusedAttention")
    unfused_attn_fwd, unfused_attn_bwd = _run_dot_product_attention(
            dtype, bs, config, "UnfusedDotProductAttention")

    atol, rtol = (2.5e-2, 2.5e-2) if dtype == torch.bfloat16 else (2.5e-3, 2.5e-3)
    #print('fused_attn_fwd  : min',fused_attn_fwd.min().item(),'max',fused_attn_fwd.max().item())
    #print('flash_attn_fwd  : min',flash_attn_fwd.min().item(),'max',flash_attn_fwd.max().item())
    #print('unfused_attn_fwd: min',unfused_attn_fwd.min().item(),'max',unfused_attn_fwd.max().item())
    #print('fused_attn_bwd  : min',fused_attn_bwd.min().item(),'max',fused_attn_bwd.max().item())
    #print('flash_attn_bwd  : min',flash_attn_bwd.min().item(),'max',flash_attn_bwd.max().item())
    #print('unfused_attn_bwd: min',unfused_attn_bwd.min().item(),'max',unfused_attn_bwd.max().item())
    assert torch.allclose(fused_attn_fwd, flash_attn_fwd, atol = atol, rtol = rtol)
    assert torch.allclose(fused_attn_bwd, flash_attn_bwd, atol = atol, rtol = rtol)
    assert torch.allclose(fused_attn_fwd, unfused_attn_fwd, atol = atol, rtol = rtol)
    assert torch.allclose(fused_attn_bwd, unfused_attn_bwd, atol = atol, rtol = rtol)
    
def _run_dot_product_attention(dtype, bs, config, backend):

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"
        #os.environ["NVTE_FUSED_ATTN_BACKEND"] = "1"

    inp = 0.1 * torch.randn(
            config.seq_len, bs, 3, config.num_attention_heads, config.head_dim,
            dtype = dtype).cuda()
    inp.requires_grad=True
    seqlens = torch.empty(bs, dtype = torch.int32).cuda()
    seqlens.fill_(config.seq_len)
    cu_seqlens = torch.zeros(bs + 1, device = inp.device, dtype = torch.int32)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim = 0)
    op_grad = 0.001 * torch.randint(0, 200, (
        config.seq_len, bs, config.num_attention_heads * config.head_dim
        ), dtype = dtype).cuda()

    block = (
         DotProductAttention(
                config.num_attention_heads,
                config.head_dim,
                attention_dropout = config.dropout_p,
                attn_mask_type = config.attn_mask_type,
                sequence_parallel = False,
                tp_size = 1,
                get_rng_state_tracker = None,
                tp_group = None,
                layer_number = 1,
                attention_type = "self"
        ).to(dtype = dtype).cuda()
    )

    q = inp[:, :,0,:,:]
    k = inp[:, :,1,:,:]
    v = inp[:, :,2,:,:]
    op = block(q, k, v)
    op.backward(op_grad)

    return op, inp.grad

#@pytest.mark.parametrize("dtype", param_types)
#@pytest.mark.parametrize("bs", batch_sizes[:1])
#@pytest.mark.parametrize("model", list(model_configs.keys())[:2])
#def test_transformer_layer(dtype, bs, model):
#    """Test TransformerLayer module when its DotProductAttention is enabled with
#    FlashAttention, FusedAttention, or UnfusedDotProductAttention backend"""
#
#    config = model_configs[model]
#
#    flash_attn_fwd, flash_attn_bwd = _run_transformer_layer(
#            dtype, bs, config, "FlashAttention")
#    fused_attn_fwd, fused_attn_bwd = _run_transformer_layer(
#            dtype, bs, config, "FusedAttention")
#    unfused_attn_fwd, unfused_attn_bwd = _run_transformer_layer(
#            dtype, bs, config, "UnfusedDotProductAttention")
#
#    atol, rtol = (5e-2, 5e-2) if dtype == torch.bfloat16 else (4e-3, 4e-3)
#    print('fused_attn_fwd  : min',fused_attn_fwd.min().item(),'max',fused_attn_fwd.max().item())
#    print('flash_attn_fwd  : min',flash_attn_fwd.min().item(),'max',flash_attn_fwd.max().item())
#    print('unfused_attn_fwd: min',unfused_attn_fwd.min().item(),'max',unfused_attn_fwd.max().item())
#    print('fused_attn_bwd  : min',fused_attn_bwd.min().item(),'max',fused_attn_bwd.max().item())
#    print('flash_attn_bwd  : min',flash_attn_bwd.min().item(),'max',flash_attn_bwd.max().item())
#    print('unfused_attn_bwd: min',unfused_attn_bwd.min().item(),'max',unfused_attn_bwd.max().item())
#    assert torch.allclose(fused_attn_fwd, flash_attn_fwd, atol = atol, rtol = rtol)
#    assert torch.allclose(fused_attn_bwd, flash_attn_bwd, atol = atol, rtol = rtol)
#    assert torch.allclose(fused_attn_fwd, unfused_attn_fwd, atol = atol, rtol = rtol)
#    assert torch.allclose(fused_attn_bwd, unfused_attn_bwd, atol = atol, rtol = rtol)
#    
#def _run_transformer_layer(dtype, bs, config, backend):
#
#    torch.manual_seed(1234)
#    torch.cuda.manual_seed(1234)
#    os.environ["NVTE_FLASH_ATTN"] = "0"
#    os.environ["NVTE_FUSED_ATTN"] = "0"
#    if backend == "FlashAttention":
#        os.environ["NVTE_FLASH_ATTN"] = "1"
#    if backend == "FusedAttention":
#        os.environ["NVTE_FUSED_ATTN"] = "1"
#        #os.environ["NVTE_FUSED_ATTN_BACKEND"] = "1"
#
#    inp = 0.1 * torch.randn(
#            config.seq_len, bs, config.num_attention_heads * config.head_dim,
#            dtype = dtype).cuda()
#    inp.requires_grad=True
#    seqlens = torch.empty(bs, dtype = torch.int32).cuda()
#    seqlens.fill_(config.seq_len)
#    cu_seqlens = torch.zeros(bs + 1, device = inp.device, dtype = torch.int32)
#    cu_seqlens[1:] = torch.cumsum(seqlens, dim = 0)
#    op_grad = 0.001 * torch.randint(0, 200, (
#        config.seq_len, bs, config.num_attention_heads * config.head_dim
#        ), dtype = dtype).cuda()
#
#    sigma = 0.02
#    init_method = init_method_normal(sigma)
#    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)
#
#    layer_number = 1
#    drop_path_rate = 0.0
#    drop_path_rates = [
#            rate.item() for rate in torch.linspace(0, drop_path_rate, config.num_layers)]
#
#    block = (
#        TransformerLayer(
#            config.hidden_size,
#            4 * config.hidden_size,
#            config.num_attention_heads,
#            layernorm_epsilon = 1e-5,
#            hidden_dropout = 0.0,
#            attention_dropout = config.dropout_p,
#            init_method = init_method,
#            output_layer_init_method = output_layer_init_method,
#            layer_number = layer_number,
#            kv_channels = config.head_dim,
#            self_attn_mask_type = config.attn_mask_type,
#            tp_group = None,
#            tp_size =  1,
#            params_dtype = dtype,
#            get_rng_state_tracker = None,
#            fuse_wgrad_accumulation = False,
#            seq_length = config.seq_len,
#            micro_batch_size = bs,
#            sequence_parallel = False,
#            apply_residual_connection_post_layernorm = False,
#            output_layernorm = False,
#            layer_type = "encoder",
#            drop_path_rate = drop_path_rates[layer_number - 1],
#            set_parallel_mode = True,
#            fuse_qkv_params = True,
#            zero_centered_gamma = False,
#            qkv_weight_interleaved = False,
#            ub_tp_comm_overlap = False,
#            bias = True,
#        )
#        .to(dtype = dtype)
#        .cuda()
#    )
#
#    op = block(inp)
#    op.backward(op_grad)
#
#    return op, inp.grad
