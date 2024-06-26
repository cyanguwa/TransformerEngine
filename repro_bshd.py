import enum
import math
from itertools import product
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

import torch
import transformer_engine
import transformer_engine.pytorch.cpp_extensions
import transformer_engine_torch as tex

from triton.testing import do_bench


def repo():
    #batch = [4] #1, 2, 4, 8]
    #seqlen = [128, 512] #256, 512, 1024]
    #nheads = [32] #16, 32]
    #headdim = [128]#, 128]
    batch = [1, 2, 4, 8]
    seqlen = [128, 256, 512, 1024]
    nheads = [16, 32]
    headdim = [128, 128]

    for _ in range(2):
        for b, s in product(batch, seqlen):
            for h, d in zip(nheads, headdim):
                print()
                print()
                print(f"Benchmarking {b=}, {s=}, {h=}, {d=}")
                
                torch.cuda.empty_cache()
                dropout = 0.0
                softmax_scale = 1.0 / math.sqrt(d)
                is_causal = True
                is_training = False

                q, k, v = [
                    torch.randn(
                        b, s,
                        h,
                        d,
                        dtype=torch.bfloat16,
                        device="cuda",
                        requires_grad=False,
                    ).contiguous()
                    for _ in range(3)
                ]
                cu_seqlens = (
                    torch.arange(start=0, end=(b + 1) * s, step=s, dtype=torch.int32)
                    .cuda()
                    .contiguous()
                )

                cu_seqlens_q = cu_seqlens
                cu_seqlens_kv = cu_seqlens
                max_seqlen_q = s
                max_seqlen_kv = s
                
                fn = lambda: transformer_engine.pytorch.cpp_extensions.fused_attn.fused_attn_fwd(
                    q=q,
                    k=k,
                    v=v,
                    qkv_dtype=tex.DType.kBFloat16,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    cu_seqlens_q_padded=cu_seqlens_q,
                    cu_seqlens_kv_padded=cu_seqlens_kv,
                    #seq_offsets_q=(cu_seqlens_q * h * d),
                    #seq_offsets_k=(cu_seqlens_kv * h * d),
                    #seq_offsets_v=(cu_seqlens_kv * h * d),
                    #seq_offsets_o=(cu_seqlens_q * h * d),
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    dropout=dropout,
                    attn_scale=softmax_scale,
                    attn_mask_type="padding_causal" if is_causal else "padding",
                    is_training=is_training,
                    fused_attention_backend=tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen,
                    #qkv_layout="thd_thd_thd",
                    qkv_layout="bshd_bshd_bshd",
                )

                fn()
                print(f"Time: {do_bench(fn)} s")



if __name__ == "__main__":
    print(f"{torch.version.cuda=}")
    print(f"{torch.backends.cudnn.version()=}")
    repo()
