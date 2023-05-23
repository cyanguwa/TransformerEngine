# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import pytest

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
        self.seq_len = seq_len
        self.dropout_p = dropout_p
        self.attn_mask_type  = attn_mask_type

model_configs = {
    "test1": ModelConfig(1, 1024, 16, 64, 512, 0.0, "causal"),
    "test2": ModelConfig(1, 1024, 16, 64, 2048, 0.0, "causal"),
    #"test2": ModelConfig(1, 1024, 16, 64, 2048, 0.1, "causal"),
}

param_types = [torch.float16]
if torch.cuda.is_bf16_supported():
    param_types.append(torch.bfloat16)

batch_sizes = [2, 48]

@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", model_configs.keys())
def test_fused_against_flash_attn(dtype, bs, model):

    config = model_configs[model]
    flash_attn_fwd, flash_attn_bwd = _run_dpa(dtype, bs, config, "FlashAttention")
    fused_attn_fwd, fused_attn_bwd = _run_dpa(dtype, bs, config, "FusedAttention")
    unfused_attn_fwd, unfused_attn_bwd = _run_dpa(dtype, bs, config, "UnfusedDotProductAttention")

    assert torch.allclose(fused_attn_fwd, flash_attn_fwd, atol=1e-2, rtol=1e-3)
    assert torch.allclose(fused_attn_bwd, flash_attn_bwd, atol=1e-2, rtol=1e-3)
    assert torch.allclose(fused_attn_fwd, unfused_attn_fwd, atol=1e-2, rtol=1e-3)
    assert torch.allclose(fused_attn_bwd, unfused_attn_bwd, atol=1e-2, rtol=1e-3)
    
def _dot_product_attention(dtype, bs, config):

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

    return block

def _run_dpa(dtype, bs, config, backend):

    seq_len = config.seq_len
    h = config.num_attention_heads
    d = config.head_dim
    dropout_p = config.dropout_p

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"
        #os.environ["NVTE_FUSED_ATTN_BACKEND"] = "2"
    print('>>> backend: ', backend)

    inp_mha = 0.1*torch.randn(seq_len, bs, 3, h, d, dtype=dtype).cuda()
    inp_mha.requires_grad=True
    seqlens = torch.empty(bs, dtype = torch.int32).cuda()

    seqlens.fill_(seq_len)
    cu_seqlens = torch.zeros(bs+1, device = inp_mha.device, dtype = torch.int32)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim = 0)
    
    op_grad = 0.001*torch.randint(0,200,(seq_len, bs, h*d), dtype = dtype).cuda()

    block = _dot_product_attention(dtype, bs, config)
    q = inp_mha[:, :,0,:,:]#.contiguous()
    k = inp_mha[:, :,1,:,:]#.contiguous()
    v = inp_mha[:, :,2,:,:]#.contiguous()
    op_old = block(q, k, v)
    op_old.backward(op_grad)
    return op_old, inp_mha.grad
