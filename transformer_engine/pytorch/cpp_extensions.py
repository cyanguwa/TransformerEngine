# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""TE FP8 extensions and GEMMs"""
import math
from typing import Optional, Tuple, List, Union
import torch
import transformer_engine_extensions as tex
from .constants import TE_DType

TORCH_DType = {
    tex.DType.kFloat8E4M3: torch.uint8,
    tex.DType.kFloat8E5M2: torch.uint8,
    tex.DType.kFloat16: torch.half,
    tex.DType.kBFloat16: torch.bfloat16,
    tex.DType.kFloat32: torch.float32,
    tex.DType.kInt32: torch.int32,
}

def check_tensor(x: torch.Tensor):
    """Check tensor properties."""
    assert (
            x.is_cuda and x.is_contiguous()
            ), "Tensor should be on CUDA and contiguous."

def check_qkv(qkv: torch.Tensor, dtype: torch.dtype):
    """Check tensor properties."""
    check_tensor(qkv)
    assert (
            qkv.dtype is dtype 
            and qkv.dim() == 4
            and qkv.shape[1] == 3
            ), f"QKV should be in [total_seqs, 3, num_heads, head_dim] shape"
    "and {dtype} dtype."

def check_o(o: torch.Tensor, dtype: torch.dtype):
    """Check tensor properties."""
    check_tensor(o)
    assert (
            o.dtype is dtype 
            and o.dim() == 3
            ), f"O and dO should be a 3D tensor in {dtype}."

def check_stats(stats: torch.Tensor, b: int, h: int, s: int):
    """Check tensor properties."""
    check_tensor(stats)
    assert (
            stats.dtype is torch.float32
            and stats.dim() == 4
            and stats.shape == torch.Size([b, h, s, 1])
            ), "M and ZInv should be in [batch_size, num_heads, max_seq_len, 1] and float32."

def check_cu_seqlens(cu_seqlens: torch.Tensor):
    """Check tensor properties."""
    check_tensor(cu_seqlens)
    assert (
            cu_seqlens.dtype is torch.int32
            and cu_seqlens.dim() == 1
            ), "cu_seqlens should be in [batch_size +1] and int32."

def check_scalar(scalar: torch.Tensor):
    """Check tensor properties."""
    check_tensor(scalar)
    assert (
            scalar.dtype is torch.float32
            and scalar.dim() <= 1
            and scalar.numel() == 1
            ), "amax/scale/descale tensor should be a float32 scalar."

def check_rng_state(rng_state: torch.Tensor):
    """Check tensor properties."""
    check_tensor(rng_state)
    assert (
            rng_state.dtype is torch.int64
            and rng_state.numel() == 2
            ), "rng_state should be [seed, offset] and in int64."

def get_mha_layout(qkv_layout: str):
    """Get MHA Layout in integers."""
    qkv_layout = qkv_layout.lower()
    qkv_layout_dict = {
        "not_interleaved": 0,
        "qkv_interleaved": 1,
        "kv_interleaved": 2,
    }
    return qkv_layout_dict[qkv_layout]

def fused_attn_fwd(
    is_training: bool,
    max_seq_len: int,
    cu_seqlens: torch.Tensor,
    QKV: torch.Tensor,
    QKV_dtype: tex.DType,
    Bias: torch.Tensor = None,
    d_scale_QKV: torch.Tensor = None,
    q_scale_S: torch.Tensor = None,
    q_scale_O: torch.Tensor = None,
    amax_S: torch.Tensor = None,
    amax_O: torch.Tensor = None,
    attn_scale: float = None,
    p_dropout: float = 0.0,
    set_zero: bool = True,
    QKV_layout: str = "qkv_interleaved",
    Bias_type: str = "no_bias",
    masking: str = "padding",
    rng_gen: torch.Generator = None,
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Fused Attention FWD.

    Parameters
    ----------
    is_training: bool
                if True, produce auxilary tensors such as M and ZInv for the backward;
                otherwise, not (inference doesn't need those tensors)
    max_seq_len: int
                the max seq len used for compute; it may not always be equal to max(cu_seqlens)
    cu_seqlens: torch.Tensor
                accumulative seqlens, [batch_size + 1]
    QKV: torch.Tensor
                input tensor, [total_seqs, 3, num_heads, head_dim]
    QKV_dtype: tex.DType
                QKV's data type in tex.DType, not torch.dtype
    Bias: torch.Tensor, default = None
                input tensor, [total_seqs, num_heads, head_dim]
    d_scale_QKV: torch.Tensor, default = None
                input tensor, for the dequantization of QKV in FP8 calculations 
    q_scale_S: torch.Tensor, default = None
                input tensor, for the quantization of S in FP8 calculations
    q_scale_O: torch.Tensor, default = None
                input tensor, for the quantization of O in FP8 calculations
    amax_S: torch.Tensor, default = None
                output tensor for the next iteration, amax of S in FP8 calculations
    amax_O: torch.Tensor, default = None
                output tensor for the next iteration, amax of O in FP8 calculations
    attn_scale: float, default = None
                if set, use attn_scale; otherwise, use 1.0/sqrt(head_dim)
    p_dropout: float, default = 0.1
                dropout probability
    set_zero: bool, default = True
                whether to initialize tensor O to zero, initialization uses mha_fill method 
    QKV_layout: str, default = `qkv_interleaved`
                matrix layout of QKV, {`qkv_interleaved`, `kv_interleaved`, `not_interleaved`}
    Bias_type: str, default = `no_bias`
                bias type, {`no_bias`, ...}
    masking: str, default = `padding`
                masking type, {`padding`, `causal`, `none`}
    rng_gen: torch.Generator, default = None
                random number generator; if not set, use default CUDA generator

    Returns 
    ----------
    O: torch.Tensor
                output tensor of the fused attention, same dtype as QKV
    aux_fwd_tensors: List[torch.Tensor]
                auxilary output tensors if is_training is True, e.g. [M, ZInv]
    rng_state: torch.Tensor
                random number generator state, [seed, offset] in uint64
    """

    check_cu_seqlens(cu_seqlens)
    ## sequence lengths, non accumulative
    #seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    # batch size
    b = cu_seqlens.numel() - 1
    # torch.dtype
    QKV_type = TORCH_DType[QKV_dtype]
    # check properties
    check_qkv(QKV, QKV_type)

    assert b <= QKV.size(0), f"b must be <= QKV.size(0)."
    total_seqs = QKV.size(0)
    h = QKV.size(2)
    d = QKV.size(3)

    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(d)
    #qkv_layout = get_mha_layout(qkv_layout)

    ############### FP8 fused attention API ################
    if QKV_type is torch.uint8 and max_seq_len <=512 and d == 64:
        assert (d_scale_QKV is not None), "d_scale_QKV is required for the FP8 API."
        assert (q_scale_S is not None), "q_scale_S is required for the FP8 API."
        assert (q_scale_O is not None), "q_scale_O is required for the FP8 API."
        assert (amax_S is not None), "amax_S is required for the FP8 API."
        assert (amax_O is not None), "amax_O is required for the FP8 API."
        assert (masking == "padding"), "Currently the FP8 API only supports padding masking."
        check_scalar(d_scale_QKV)
        check_scalar(q_scale_S)
        check_scalar(q_scale_O)
        check_scalar(amax_S)
        check_scalar(amax_O)

        #qkv_ragged_offset = cu_seqlens * 3 * h * d
        #o_ragged_offset = cu_seqlens * h * d
        #seqlen_list = [qkv_ragged_offset, o_ragged_offset, seqlens]

        output_tensors = tex.fused_attn_fwd(
                b, max_seq_len, total_seqs, h, d,
                is_training, attn_scale, p_dropout, set_zero, QKV_layout,
                cu_seqlens,
                QKV,
                QKV_dtype,
                d_scale_QKV,
                q_scale_S,
                q_scale_O,
                amax_S,
                amax_O,
                Bias, # None
                Bias_type, # not used
                rng_gen,
        )

        return output_tensors

    ############### BF16/FP16 fused attention API from fmha_v2 ################
    elif QKV_type is torch.bfloat16 or QKV_type is torch.float16:
        #TODO add BF/FP16 support for >512 sequence length
        if Bias_type is not "no_bias":
            assert (Bias is not None), "Bias is required if bias_type is `no_bias`."
        output_tensors = tex.fused_attn_fwd(
                b, max_seq_len, total_seqs, h, d,
                is_training, attn_scale, p_dropout, set_zero, qkv_layout,
                cu_seqlens,
                QKV,
                QKV_dtype,
                None,
                None,
                None,
                None,
                None,
                Bias,
                Bias_type, 
                rng_gen,
        )
        return output_tensors

    ############### BF16/FP16 fused attention API from fmha_v1 apex ################
    elif (QKV_type is torch.bfloat16 or QKV_type is torch.float16) and (max_seq_len <=512):
        #TODO add BF/FP16 support for <=512 sequence length
        pass

    else:
        assert False, "No support for this dtype and max_seq_len combination."
        return

def fused_attn_bwd(
    max_seq_len: int,
    cu_seqlens: torch.Tensor,
    QKV: torch.Tensor,
    O: torch.Tensor,
    dO: torch.Tensor,
    QKV_dtype: tex.DType,
    rng_state: torch.Tensor,
    aux_fwd_tensors: List[torch.Tensor] = None,
    Bias: torch.Tensor = None,
    d_scale_QKV: torch.Tensor = None,
    d_scale_S: torch.Tensor = None,
    d_scale_O: torch.Tensor = None,
    d_scale_dO: torch.Tensor = None,
    q_scale_S: torch.Tensor = None,
    q_scale_dS: torch.Tensor = None,
    q_scale_dQKV: torch.Tensor = None,
    amax_dS: torch.Tensor = None,
    amax_dQKV: torch.Tensor = None,
    attn_scale: float = None,
    p_dropout: float = 0.0,
    set_zero: bool = True,
    QKV_layout: str = "qkv_interleaved",
    Bias_type: str = "no_bias",
    masking: str = "padding",
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Fused Attention BWD.

    Parameters
    ----------
    max_seq_len: int
                the max seq len used for compute; it may not always be equal to max(cu_seqlens)
    cu_seqlens: torch.Tensor
                accumulative seqlens, [batch_size + 1]
    QKV: torch.Tensor
                input tensor, [total_seqs, 3, num_heads, head_dim]
    O: torch.Tensor
                input tensor, [total_seqs, num_heads, head_dim]
    dO: torch.Tensor
                input tensor, [total_seqs, num_heads, head_dim]
    QKV_dtype: tex.DType
                QKV's data type in tex.DType, not torch.dtype
    rng_state: torch.Tensor
                random number generator state, in [seed, offset] format and uint64
    aux_fwd_tensors: List[torch.Tensor], default = None
                auxilary output tensors of fwd, e.g. [M, ZInv]
    Bias: torch.Tensor, default = None
                input tensor, [total_seqs, num_heads, head_dim]
    d_scale_QKV: torch.Tensor, default = None
                input tensor, for the dequantization of QKV in FP8 calculations 
    d_scale_S: torch.Tensor, default = None
                input tensor, for the dequantization of S in FP8 calculations 
    d_scale_O: torch.Tensor, default = None
                input tensor, for the dequantization of O in FP8 calculations 
    d_scale_dO: torch.Tensor, default = None
                input tensor, for the dequantization of dO in FP8 calculations 
    q_scale_S: torch.Tensor, default = None
                input tensor, for the quantization of S in FP8 calculations
    q_scale_dS: torch.Tensor, default = None
                input tensor, for the quantization of dS in FP8 calculations
    q_scale_dQKV: torch.Tensor, default = None
                input tensor, for the quantization of dQKV in FP8 calculations
    amax_dS: torch.Tensor, default = None
                output tensor for the next iteration, amax of dS in FP8 calculations
    amax_dQKV: torch.Tensor, default = None
                output tensor for the next iteration, amax of dQKV in FP8 calculations
    attn_scale: float, default = None
                if set, use attn_scale; otherwise, use 1.0/sqrt(head_dim)
    p_dropout: float, default = 0.1
                dropout probability
    set_zero: bool, default = True
                whether to initialize tensor O to zero, initialization uses mha_fill method 
    QKV_layout: str, default = `qkv_interleaved`
                matrix layout of QKV, {`qkv_interleaved`, `kv_interleaved`, `not_interleaved`}
    Bias_type: str, default = `no_bias`
                bias type, {`no_bias`, ...}
    masking: str, default = `padding`
                masking type, {`padding`, `causal`, `none`}

    Returns 
    ----------
    dQKV: torch.Tensor
                the gradient tensor of QKV, same dtype as QKV
    """

    check_cu_seqlens(cu_seqlens)
    ## sequence lengths, non accumulative
    #seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    # batch size
    b = cu_seqlens.numel() - 1
    # torch.dtype
    QKV_type = TORCH_DType[QKV_dtype]
    # check properties
    check_qkv(QKV, QKV_type)
    check_o(O, QKV_type)
    check_o(dO, QKV_type)

    assert b <= QKV.size(0), f"b must be <= QKV.size(0)."
    total_seqs = QKV.size(0)
    h = QKV.size(2)
    d = QKV.size(3)

    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(d)
    #qkv_ragged_offset = cu_seqlens * 3 * h * d
    #o_ragged_offset = cu_seqlens * h * d

    check_rng_state(rng_state)

    if QKV_type is torch.uint8 and max_seq_len <=512 and d == 64:
        assert (d_scale_QKV is not None), "d_scale_QKV is required for the FP8 API."
        assert (d_scale_S is not None), "d_scale_S is required for the FP8 API."
        assert (d_scale_O is not None), "d_scale_O is required for the FP8 API."
        assert (d_scale_dO is not None), "d_scale_dO is required for the FP8 API."
        assert (q_scale_S is not None), "q_scale_S is required for the FP8 API."
        assert (q_scale_dS is not None), "q_scale_dS is required for the FP8 API."
        assert (q_scale_dQKV is not None), "q_scale_dQKV is required for the FP8 API."
        assert (amax_dS is not None), "amax_dS is required for the FP8 API."
        assert (amax_dQKV is not None), "amax_dQKV is required for the FP8 API."
        assert (len(aux_fwd_tensors) == 2
                ), "aux_fwd_tensors is required for the FP8 API, e.g. [M, ZInv]"
        assert (masking == "padding"), "Currently the FP8 API only supports padding masking."
        check_scalar(d_scale_QKV)
        check_scalar(d_scale_S)
        check_scalar(d_scale_O)
        check_scalar(d_scale_dO)
        check_scalar(q_scale_S)
        check_scalar(q_scale_dS)
        check_scalar(q_scale_dQKV)
        check_scalar(amax_dS)
        check_scalar(amax_dQKV)
        M, ZInv = aux_fwd_tensors
        check_stats(M, b, h, max_seq_len)
        check_stats(ZInv, b, h, max_seq_len)

        #qkv_layout = get_mha_layout(qkv_layout)

        dQKV = tex.fused_attn_bwd(
                b, max_seq_len, total_seqs, h, d,
                attn_scale, p_dropout, set_zero, QKV_layout,
                cu_seqlens,
                QKV, O, dO,
                QKV_dtype,
                M, ZInv,
                d_scale_QKV, d_scale_S, d_scale_O, d_scale_dO,
                q_scale_S, q_scale_dS, q_scale_dQKV,
                amax_dS, amax_dQKV,
                Bias, # None
                Bias_type, # not used
                rng_state,
        )

        return dQKV

    ############### BF16/FP16 fused attention API from fmha_v2 ################
    elif QKV_type is torch.bfloat16 or QKV_type is torch.float16:
        #TODO add BF/FP16 support for >512 sequence length
        if Bias_type is not "no_bias":
            assert (Bias is not None), "Bias is required if bias_type is `no_bias`."
        dQKV = tex.fused_attn_bwd(
                b, max_seq_len, total_seqs, h, d,
                attn_scale, p_dropout, set_zero, QKV_layout,
                cu_seqlens,
                QKV, O, dO,
                QKV_dtype,
                M, ZInv, # does BF16 use M/ZInv or S?
                None, None, None, None,
                None, None, None, 
                None, None, 
                Bias,
                Bias_type,
                rng_state,
        )
        return dQKV 

    ############### BF16/FP16 fused attention API from fmha_v1 apex ################
    elif (QKV_type is torch.bfloat16 or QKV_type is torch.float16) and (max_seq_len <=512):
        #TODO add BF/FP16 support for <=512 sequence length
        pass

    else:
        assert False, "No support for this dtype and max_seq_len combination."
        return

def fp8_gemm(
    A: torch.Tensor,
    A_scale_inv: torch.Tensor,
    A_fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    A_dtype: tex.DType,
    B: torch.Tensor,
    B_scale_inv: torch.Tensor,
    B_fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    B_dtype: tex.DType,
    out_dtype: torch.dtype,
    workspace: torch.Tensor,
    gelu: bool = False,
    accumulate: bool = False,
    out: Optional[torch.Tensor] = None,
    out_index = None,
    fp8_meta_tensor: tex.FP8TensorMeta = None,
    bias: Optional[torch.Tensor] = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,
    D_dtype: Optional[tex.DType] = None,
) -> torch.Tensor:
    """TN layout GEMM with fp8 inputs."""

    empty_tensor = torch.Tensor()
    if D_dtype is not None and D_dtype in [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2]:
        assert fp8_meta_tensor is not None and out_index is not None

    return_output = False
    if out is None:
        out = torch.empty(
            B.shape[0],
            A.shape[0],
            dtype=out_dtype,
            device="cuda",
        )
        return_output = True
    # Use bfloat16 as default bias_dtype
    bias_dtype = torch.bfloat16 if bias is None else bias.dtype
    if gelu:
        gelu_input = torch.empty_like(out, dtype=bias_dtype)
    else:
        gelu_input = empty_tensor
    bias_dtype = TE_DType[bias_dtype]

    out_dtype = TE_DType[out.dtype] if D_dtype is None else D_dtype

    _ = torch.ops.tex_ts.te_gemm_ts(
        A,
        A_scale_inv,
        A_fp8_tensor,
        A_dtype,
        True,  # transa
        B,
        B_scale_inv,
        B_fp8_tensor,
        B_dtype,
        False,  # transb
        out,
        empty_tensor if out_index is None else fp8_meta_tensor.scale[out_index],
        out_dtype,
        empty_tensor if out_index is None else fp8_meta_tensor.amax_history[0][out_index],
        bias if use_bias else empty_tensor,
        bias_dtype,
        gelu_input,  # this is pre_gelu_out
        False,  # grad
        workspace,
        workspace.shape[0],
        accumulate,
        use_split_accumulator,
    )

    if return_output:
        if gelu:
            return out, gelu_input
        return out
    if gelu:
        return gelu_input
    return None


def gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    dtype: torch.dtype,
    workspace: torch.Tensor,
    gelu: bool = False,
    gelu_input: Optional[torch.Tensor] = None,
    grad: bool = False,
    accumulate: bool = False,
    layout: str = "TN",
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_bias: bool = False,
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Non FP8 GEMM."""

    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    empty_tensor = torch.Tensor()
    fp8_index = -1 # dummy index

    return_output = False
    if out is None:
        out = torch.empty(
            B.shape[1] if transb else B.shape[0],
            A.shape[0] if transa else A.shape[1],
            dtype=dtype,
            device="cuda",
        )
        return_output = True

    if gelu and not grad:
        gelu_input = torch.empty_like(out, dtype=dtype)
    elif not gelu:
        gelu_input = empty_tensor

    if grad and use_bias:
        grad_bias = torch.empty(B.shape[1], dtype=out.dtype, device="cuda")
    else:
        grad_bias = empty_tensor

    bias = bias if use_bias else empty_tensor

    assert A.dtype == dtype and B.dtype == dtype, \
        f'Expected dtype={dtype}, but found A.dtype={A.dtype} and B.dtype={B.dtype}'
    input_dtype = TE_DType[dtype]
    output_dtype = TE_DType[out.dtype]
    if use_bias:
        bias_dtype = TE_DType[grad_bias.dtype] if grad else TE_DType[bias.dtype]
    else:
        bias_dtype = output_dtype

    _ = torch.ops.tex_ts.te_gemm_ts(
        A,
        empty_tensor,
        fp8_index,
        input_dtype,
        transa,
        B,
        empty_tensor,
        fp8_index,
        input_dtype,
        transb,
        out,
        empty_tensor, # out_scale
        output_dtype,
        empty_tensor, # out_amax
        grad_bias if grad else bias,
        bias_dtype,
        gelu_input,
        grad,
        workspace,
        workspace.shape[0],
        accumulate,
        False,  # use_split_accumulator
    )

    if return_output:
        return out, grad_bias, gelu_input
    return None, grad_bias, gelu_input


def fp8_cast_transpose_fused(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    cast_out: Optional[torch.Tensor] = None,
    transpose_out: Optional[torch.Tensor] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor], None]:
    """Cast + Transpose with FP8 output"""

    return_outputs = False
    if cast_out is None or transpose_out is None:
        cast_out = torch.empty_like(inp, dtype=torch.uint8)
        transpose_out = torch.empty(
            inp.shape[1], inp.shape[0], device="cuda", dtype=torch.uint8
        )
        return_outputs = True

    tex.fused_cast_transpose(
        inp,
        fp8_meta_tensor.scale[fp8_tensor],
        fp8_meta_tensor.amax_history[0][fp8_tensor],
        fp8_meta_tensor.scale_inv[fp8_tensor],
        cast_out,
        transpose_out,
        otype,
    )

    if return_outputs:
        return cast_out, transpose_out
    return None


def fp8_cast_transpose_bgrad_fused(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cast + Transpose + BGRAD with FP8 output"""
    return tex.fused_cast_transpose_bgrad(
        inp,
        fp8_meta_tensor.scale[fp8_tensor],
        fp8_meta_tensor.amax_history[0][fp8_tensor],
        fp8_meta_tensor.scale_inv[fp8_tensor],
        otype,
    )


def fp8_transpose_bgrad_fused(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    grad_bias_type: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transpose + BGRAD with FP8 output"""
    return tex.fused_fp8_transpose_bgrad(
        inp,
        fp8_meta_tensor.scale[fp8_tensor],
        fp8_meta_tensor.amax_history[0][fp8_tensor],
        fp8_meta_tensor.scale_inv[fp8_tensor],
        otype,
        TE_DType[grad_bias_type],
    )


def fp8_cast_transpose_bgrad_dgelu_fused(
    grad_output: torch.Tensor,
    gelu_input: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cast + Transpose + BGRAD + DGELU with FP8 output"""
    return tex.fused_cast_transpose_bgrad_dgelu(
        grad_output,
        gelu_input,
        fp8_meta_tensor.scale[fp8_tensor],
        fp8_meta_tensor.amax_history[0][fp8_tensor],
        fp8_meta_tensor.scale_inv[fp8_tensor],
        otype,
    )


def fp8_gelu(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
) -> torch.Tensor:
    """GeLU with FP8 output"""
    return torch.ops.tex_ts.fp8_gelu_ts(
        inp,
        fp8_meta_tensor.scale,
        fp8_meta_tensor.amax_history,
        fp8_meta_tensor.scale_inv,
        fp8_tensor,
        otype,
    )


def layernorm_fwd_fp8(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    sm_margin: int,
    zero_centered_gamma: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LayerNorm with FP8 output"""
    return tex.layernorm_fwd_fp8(
        inp,
        weight,
        bias,
        eps,
        fp8_meta_tensor.scale[fp8_tensor],
        fp8_meta_tensor.amax_history[0][fp8_tensor],
        fp8_meta_tensor.scale_inv[fp8_tensor],
        otype,
        sm_margin,
        zero_centered_gamma
    )


def layernorm_fwd_fp8_inf(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    zero_centered_gamma,
) -> torch.Tensor:
    """LayerNorm with FP8 output.

    This version of layernorm_fwd_fp8 is specialized for inference, and returns
    only the normalized output.
    """
    ret = torch.ops.tex_ts.layernorm_fwd_fp8_inf_ts(
        inp,
        weight,
        bias,
        eps,
        fp8_meta_tensor.scale,
        fp8_meta_tensor.amax_history,
        fp8_meta_tensor.scale_inv,
        fp8_tensor,
        otype,
        zero_centered_gamma)
    return ret


def layernorm_fwd_inf(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    zero_centered_gamma: bool,
) -> torch.Tensor:
    """LayerNorm with FP8 output"""
    return torch.ops.tex_ts.layernorm_fwd_inf_ts(
        inp,
        weight,
        bias,
        eps,
        zero_centered_gamma,
    )


def cast_to_fp8(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
) -> torch.Tensor:
    """Cast input to FP8"""
    return torch.ops.tex_ts.cast_to_fp8_ts(
        inp,
        fp8_meta_tensor.scale,
        fp8_meta_tensor.amax_history,
        fp8_meta_tensor.scale_inv,
        fp8_tensor,
        otype,
    )


def cast_from_fp8(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    itype: tex.DType,
    otype: tex.DType,
) -> torch.Tensor:
    """Cast input from FP8"""
    return torch.ops.tex_ts.cast_from_fp8_ts(
        inp,
        fp8_meta_tensor.scale_inv,
        fp8_tensor,
        itype,
        otype,
    )
