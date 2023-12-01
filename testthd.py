import torch
from tests.pytorch.test_fused_attn import ModelConfig, _run_dot_product_attention

dtype = torch.float16
ckpt_attn = False
workspace_opt = True

for i in range(5):
    #if i == 2:
    #    torch.cuda.cudart().cudaProfilerStart()

    config = ModelConfig(2, 16, 16,  64, 2048, 2048, 0.0, "no_mask", "no_bias")
    qkv_layout = 'sb3hd'

    #print('begin ',qkv_layout)
    #flash_attn_fwd, flash_attn_bwd = _run_dot_product_attention(
    #    dtype, config, "FusedAttention", ckpt_attn, qkv_layout, workspace_opt,
    #)
    
    print('begin ',qkv_layout)
    flash_attn_fwd, flash_attn_bwd = _run_dot_product_attention(
        dtype, config, "FlashAttention", ckpt_attn, qkv_layout, workspace_opt,
    )
    
    config = ModelConfig(2, 16, 16,  64, 2048, 2048, 0.0, "no_mask", "no_bias")
    qkv_layout = 't3hd'
    
    print('begin ',qkv_layout)
    flash_attn_fwd, flash_attn_bwd = _run_dot_product_attention(
        dtype, config, "FlashAttention", ckpt_attn, qkv_layout, workspace_opt,
    )
    #if i == 4:
    #    torch.cuda.cudart().cudaProfilerStop()
print('finished')
