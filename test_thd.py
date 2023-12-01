import torch
from tests.pytorch.test_fused_attn import ModelConfig, _run_dot_product_attention

dtype = torch.float16
ckpt_attn = False
workspace_opt = True

num_iter = 23
for i in range(num_iter):
    if i == 3:
        torch.cuda.cudart().cudaProfilerStart()

    config = ModelConfig(2, 16, 16,  64, 2048, 2048, 0.0, "no_mask", "no_bias")
    qkv_layout = 'sbhd_sbhd_sbhd' #'sb3hd'

    #print('begin ',qkv_layout)
    #flash_attn_fwd, flash_attn_bwd = _run_dot_product_attention(
    #    dtype, config, "FusedAttention", ckpt_attn, qkv_layout, workspace_opt,
    #)
    
    print('begin ',qkv_layout)
    flash_attn_fwd, flash_attn_bwd = _run_dot_product_attention(
        dtype, config, "FlashAttention", ckpt_attn, qkv_layout, workspace_opt,
    )
    
    config = ModelConfig(2, 16, 16,  64, 2048, 2048, 0.0, "no_mask", "no_bias")
    qkv_layout = 'thd_thd_thd' #'t3hd'
    
    print('begin ',qkv_layout)
    flash_attn_fwd, flash_attn_bwd = _run_dot_product_attention(
        dtype, config, "FlashAttention", ckpt_attn, qkv_layout, workspace_opt,
    )
    if i == num_iter - 1:
        torch.cuda.cudart().cudaProfilerStop()
print('finished')
