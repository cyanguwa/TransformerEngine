/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file fused_attn_arbitrary_seqlen.h
 *  \brief Functions for fused attention with seqlen > 512
 */

#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_ARBITRARY_SEQLEN_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_ARBITRARY_SEQLEN_H_

#include "transformer_engine/fused_attn.h"

#include <cudnn.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend_utils.h>

#include "common/common.h"

namespace transformer_engine {
#if (CUDNN_VERSION >= 8900)
namespace fused_attn {
void fused_attn_arbitrary_seqlen_fwd_impl(
                int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
                bool is_training, float scaling_factor, float dropout_probability,
                NVTE_QKV_Layout layout,
                NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                void *devPtrQ, void *devPtrK, void *devPtrV,
                void *devPtrSoftmaxStats, void *devPtrO,
                void* devPtrDropoutSeed, void* devPtrDropoutOffset,
                void* devPtrCuSeqlensQ, void* devPtrCuSeqlensKV,
                cudnn_frontend::DataType_t tensorType,
                void *workspace, size_t *workspace_size,
                cudaStream_t stream, cudnnHandle_t handle, bool* check_support);

void fused_attn_arbitrary_seqlen_bwd_impl(
                int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
                float scaling_factor, float dropout_probability, NVTE_QKV_Layout layout,
                NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                void* devPtrQ, void* devPtrKTranspose, void* devPtrVTranspose,
                void* devPtrO, void* devPtrSoftmaxStats,
                void* devPtrdQ, void* devPtrdK, void* devPtrdV, void* devPtrdO,
                void* devPtrDropoutSeed, void* devPtrDropoutOffset,
                void* devPtrCuSeqlensQ, void* devPtrCuSeqlensKV,
                cudnn_frontend::DataType_t tensorType, void *workspace, size_t *workspace_size,
                cudaStream_t stream, cudnnHandle_t handle, bool* check_support);
}

void fused_attn_arbitrary_seqlen_fwd_qkvpacked(size_t batch, size_t max_seqlen, size_t num_head,
                                      size_t head_size, bool is_training, float attn_scale,
                                      float p_dropout, NVTE_QKV_Layout qkv_layout,
                                      NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                                      const Tensor *input_QKV, const Tensor *input_Bias,
                                      Tensor *output_O, NVTETensorPack *Aux_CTX_Tensors,
                                      const Tensor *cu_seqlens, const Tensor *rng_state,
                                      Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

void fused_attn_arbitrary_seqlen_bwd_qkvpacked(size_t batch, size_t max_seqlen, size_t num_head,
                                      size_t head_dim, float attn_scale, float p_dropout,
                                      NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                                      NVTE_Mask_Type mask_type, const Tensor *input_QKV,
                                      const Tensor *input_O,
                                      const Tensor *input_dO, Tensor *output_S,
                                      Tensor *output_dQKV, Tensor *output_dBias,
                                      const Tensor *cu_seqlens, const Tensor *rng_state,
                                      Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

void fused_attn_arbitrary_seqlen_fwd(size_t batch, size_t max_seqlen_q, size_t max_seqlen_kv,
                                      size_t num_head, size_t head_size, bool is_training,
                                      float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout,
                                      NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                                      const Tensor *input_Q, const Tensor *input_K,
                                      const Tensor *input_V, const Tensor *input_Bias,
                                      Tensor *output_O, NVTETensorPack *Aux_CTX_Tensors,
                                      const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
                                      const Tensor *rng_state,
                                      Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

void fused_attn_arbitrary_seqlen_bwd(size_t batch, size_t max_seqlen_q, size_t max_seqlen_kv,
                                      size_t num_head, size_t head_dim, float attn_scale,
                                      float p_dropout, NVTE_QKV_Layout qkv_layout,
                                      NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                                      const Tensor *input_Q, const Tensor *input_K,
                                      const Tensor *input_V, const Tensor *input_O,
                                      const Tensor *input_dO, Tensor *output_S,
                                      Tensor *output_dQ, Tensor *output_dK,
                                      Tensor *output_dV, Tensor *output_dBias,
                                      const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
                                      const Tensor *rng_state,
                                      Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

#endif  // CUDNN_VERSION >= 8900
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_ARBITRARY_SEQLEN_H_
