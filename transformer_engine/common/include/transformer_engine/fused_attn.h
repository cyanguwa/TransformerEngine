/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_FP8_H_
#define TRANSFORMER_ENGINE_FUSED_ATTN_FP8_H_

#include "transformer_engine.h"
//#include <string>

#ifdef __cplusplus
extern "C" {
#endif

enum NVTE_QKV_Layout {
    NOT_INTERLEAVED = 0,
    QKV_INTERLEAVED = 1,
    KV_INTERLEAVED = 2
};

enum NVTE_Bias_Type {
    NO_BIAS = 0,
    PRE_SCALE_BIAS = 1,
    POST_SCALE_BIAS = 2
};

enum NVTE_Mask_Type {
    PADDING = 0,
    CAUSAL = 1,
    NO_MASK = 2
};

//NVTE_QKV_Layout get_nvte_qkv_layout(const std::string qkv_layout);
//
//NVTE_Bias_Type get_nvte_bias_type(const std::string bias_type);
//
//NVTE_Mask_Type get_nvte_mask_type(const std::string mask_type);

struct NVTETensorPack {
  static const int MAX_SIZE = 10;
  NVTETensor tensors[MAX_SIZE];
  size_t size = 0;
};

void nvte_tensor_pack_create(NVTETensorPack* pack);

void nvte_tensor_pack_destroy(NVTETensorPack* pack);

void nvte_fused_attn_fwd_qkvpacked(
            size_t max_seqlen,
            bool is_training, float attn_scale, float p_dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            const NVTETensor cu_seqlens,
            const NVTETensor rng_state,
            const NVTETensor QKV,
            const NVTETensor Bias,
            NVTETensor S,
            NVTETensor O,
            NVTETensorPack* Aux_Output_Tensors,
            NVTETensor workspace,
            cudaStream_t stream);

void nvte_fused_attn_bwd_qkvpacked(
            size_t max_seqlen,
            float attn_scale, float p_dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            const NVTETensor cu_seqlens,
            const NVTETensor QKV,
            const NVTETensor Bias,
            const NVTETensor O,
            const NVTETensor dO,
            const NVTETensor S,
            NVTETensor dS,
            const NVTETensorPack* Aux_CTX_Tensors,
            NVTETensor dQKV,
            NVTETensor workspace,
            cudaStream_t stream);

void nvte_fused_attn_fwd_kvpacked(
            size_t max_seqlen_q, size_t max_seqlen_kv,
            bool is_training, float attn_scale, float p_dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            const NVTETensor cu_seqlens_q,
            const NVTETensor cu_seqlens_kv,
            const NVTETensor rng_state,
            const NVTETensor Q,
            const NVTETensor KV,
            const NVTETensor Bias,
            NVTETensor S,
            NVTETensor O,
            NVTETensorPack* Aux_Output_Tensors,
            NVTETensor workspace,
            cudaStream_t stream);

void nvte_fused_attn_bwd_kvpacked(
            size_t max_seqlen_q, size_t max_seqlen_kv,
            float attn_scale, float p_dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            const NVTETensor cu_seqlens_q,
            const NVTETensor cu_seqlens_kv,
            const NVTETensor Q,
            const NVTETensor KV,
            const NVTETensor Bias,
            const NVTETensor O,
            const NVTETensor dO,
            const NVTETensor S,
            NVTETensor dS,
            const NVTETensorPack* Aux_CTX_Tensors,
            NVTETensor dQ,
            NVTETensor dKV,
            NVTETensor workspace,
            cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
