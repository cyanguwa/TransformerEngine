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
    #"test1": ModelConfig(1, 1024, 16, 64, 512, 0.1, "causal"),
    "test2": ModelConfig(1, 1024, 16, 64, 2048, 0.1, "causal"),
}

#param_types = [torch.float16]
param_types = [torch.bfloat16]
#if torch.cuda.is_bf16_supported():
#    param_types.append(torch.bfloat16)

batch_sizes = [2] #[2, 48]

def _test_sanity_e2e(block, bs, dtype, config, backend):

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
        #os.environ["NVTE_FUSED_ATTN_BACKEND"] = "3"
    #print('flash',os.environ["NVTE_FLASH_ATTN"],'fused',os.environ["NVTE_FUSED_ATTN"])
    print('>>> backend: ', backend)

    inp_mha = 0.1*torch.randn(seq_len, bs, h*d, dtype=dtype).cuda()
    seqlens = torch.empty(bs, dtype = torch.int32).cuda()

    seqlens.fill_(seq_len)
    cu_seqlens = torch.zeros(bs+1, device = inp_mha.device, dtype = torch.int32)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim = 0)
    
    op_grad = 0.001*torch.randint(0,200,(seq_len, bs, h*d), dtype = dtype).cuda()
    
    #mha = block() #bs, seq_len, h, d, dropout_p, "causal", True, data_type = dtype)
    #old_times = []
    #for i in range(num_iters):
    #    #torch.cuda.synchronize()
    #    #start_old = time.time()
    #    op_old = block(inp_mha)
    #    op_old.backward(op_grad)
    #    #torch.cuda.synchronize()
    #    #end_old = time.time()
    #    #old_times.append(end_old - start_old)
    ##old_time = sum(old_times[10:])/num_iters
    op_old = block(inp_mha)
    op_old.backward(op_grad)
    name = 'flash_attn.pt' if int(os.getenv("NVTE_FLASH_ATTN")) == 1 else 'fused_attn.pt'
    torch.save(op_old,name)
    #torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", model_configs.keys())
def test_sanity_gpt(dtype, bs, model):

    config = model_configs[model]

    sigma = 0.02
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    layer_number = 1
    drop_path_rate = 0.0
    drop_path_rates = [rate.item() for rate in torch.linspace(0, drop_path_rate, config.num_layers)]

    block = (
        TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_attention_heads,
            layernorm_epsilon = 1e-5,
            hidden_dropout = 0.0,
            attention_dropout = config.dropout_p,
            init_method = init_method,
            output_layer_init_method = output_layer_init_method,
            layer_number = layer_number,
            kv_channels = config.head_dim,
            self_attn_mask_type = config.attn_mask_type,
            tp_group = None,
            tp_size =  1,
            params_dtype = dtype,
            get_rng_state_tracker = None,
            fuse_wgrad_accumulation = False,
            seq_length = config.seq_len,
            micro_batch_size = bs,
            sequence_parallel = False,
            apply_residual_connection_post_layernorm = False,
            output_layernorm = False,
            layer_type = "encoder",
            drop_path_rate = drop_path_rates[layer_number - 1],
            set_parallel_mode = True,
            fuse_qkv_params = True,
            zero_centered_gamma = False,
            qkv_weight_interleaved = False,
            ub_tp_comm_overlap = False,
            bias = True,
        )
        .to(dtype = dtype)
        .cuda()
    )

    _test_sanity_e2e(block, bs, dtype, config, "FlashAttention")
    _test_sanity_e2e(block, bs, dtype, config, "FusedAttention")
    flash_attn_pt = torch.load('flash_attn.pt')
    fused_attn_pt = torch.load('fused_attn.pt')
    print('flash_attn_pt: min',flash_attn_pt.min().item(),'max',flash_attn_pt.max().item())
    print('fused_attn_pt: min',fused_attn_pt.min().item(),'max',fused_attn_pt.max().item())
    print('flash_attn_pt == fused_attn_pt (atol=0.02): ', torch.allclose(flash_attn_pt, fused_attn_pt, atol=0.02))
    print('flash_attn_pt == fused_attn_pt (atol=0.04): ', torch.allclose(flash_attn_pt, fused_attn_pt, atol=0.04))

