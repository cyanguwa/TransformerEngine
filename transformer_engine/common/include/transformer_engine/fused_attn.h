/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_FP8_H_
#define TRANSFORMER_ENGINE_FUSED_ATTN_FP8_H_

#include "transformer_engine.h"

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

struct NVTETensorPack {
  static const int MAX_SIZE = 10;
  NVTETensor tensors[MAX_SIZE];
  size_t size = 0;
};

/*! \brief Create NVTETensors in NVTETensorPack.
 */
void nvte_tensor_pack_create(NVTETensorPack* pack);

/*! \brief Destroy NVTETensors in NVTETensorPack.
 */
void nvte_tensor_pack_destroy(NVTETensorPack* pack);

/*! \brief Compute dot product attention with packed QKV input.
 *
 * Computes:
 *  - `P = Q * K.T + B`
 *  - `S = Softmax(P)`
 *  - `O = AttentionMask(Dropout(S))`
 *
 *  \param[in]     QKV                   The QKV tensor in packed format.
 *  \param[in]     Bias                  The B tensor.
 *  \param[in,out] S                     The S tensor.
 *  \param[out]    O                     The output tensor O.
 *  \param[out]    Aux_Output_tensors    Auxiliary output tensors when training.
 *  \param[in]     cu_seqlens            Accumulative sequence lengths in a batch.
 *  \param[in]     rng_state             Seed and offset of the random number generator.
 *  \param[in]     max_seqlen            Max sequence length of this batch.  
 *  \param[in]     is_training           Whether this is in training mode or inference.
 *  \param[in]     attn_scale            Scaling factor for Q * K.T.
 *  \param[in]     p_dropout             Dropout probability.
 *  \param[in]     qkv_layout            Layout of QKV.
 *  \param[in]     bias_type             Type of the bias.
 *  \param[in]     attn_mask_type        Type of the attention mask.
 *  \param[in]     workspace             The workspace tensor.
 *  \param[in]     stream                CUDA stream used for this function.
 */
void nvte_fused_attn_fwd_qkvpacked(
            const NVTETensor QKV,
            const NVTETensor Bias,
            NVTETensor S,
            NVTETensor O,
            NVTETensorPack* Aux_Output_Tensors,
            const NVTETensor cu_seqlens,
            const NVTETensor rng_state,
            size_t max_seqlen,
            bool is_training, float attn_scale, float p_dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream);

/*! \brief Perform the backprop of dot product attention with packed QKV input.
 *
 *  \param[in]     QKV                   The QKV tensor in packed format.
 *  \param[in]     dBias                 The gradient of the B tensor.
 *  \param[in]     O                     The O tensor from the forward pass.
 *  \param[in]     dO                    The gradient of the O tensor.
 *  \param[in]     S                     The S tensor, S = Softmax(Q * K.T).
 *  \param[in,out] dP                    The gradient of the P tensor, P = Q * K.T.
 *  \param[in]     Aux_CTX_tensors       Auxiliary tensors from the forward pass when training.
 *  \param[out]    dQKV                  The gradient of the QKV tensor.
 *  \param[in]     cu_seqlens            Accumulative sequence lengths in a batch.
 *  \param[in]     rng_state             Seed and offset of the random number generator.
 *  \param[in]     max_seqlen            Max sequence length of this batch.  
 *  \param[in]     attn_scale            Scaling factor for Q * K.T.
 *  \param[in]     p_dropout             Dropout probability.
 *  \param[in]     qkv_layout            Layout of QKV.
 *  \param[in]     bias_type             Type of the bias.
 *  \param[in]     attn_mask_type        Type of the attention mask.
 *  \param[in]     workspace             The workspace tensor.
 *  \param[in]     stream                CUDA stream used for this function.
 */
void nvte_fused_attn_bwd_qkvpacked(
            const NVTETensor QKV,
            const NVTETensor dBias,
            const NVTETensor O,
            const NVTETensor dO,
            const NVTETensor S,
            NVTETensor dP,
            const NVTETensorPack* Aux_CTX_Tensors,
            NVTETensor dQKV,
            const NVTETensor cu_seqlens,
            size_t max_seqlen,
            float attn_scale, float p_dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream);

/*! \brief Compute dot product attention with packed KV input.
 *
 * Computes:
 *  - `P = Q * K.T + B`
 *  - `S = Softmax(P)`
 *  - `O = AttentionMask(Dropout(S))`
 *
 *  \param[in]     Q                     The Q tensor.
 *  \param[in]     KV                    The KV tensor.
 *  \param[in]     Bias                  The B tensor.
 *  \param[in,out] S                     The S tensor.
 *  \param[out]    O                     The output tensor O.
 *  \param[out]    Aux_Output_tensors    Auxiliary output tensors when training.
 *  \param[in]     cu_seqlens_q          Accumulative sequence lengths in a batch for Q.
 *  \param[in]     cu_seqlens_kv         Accumulative sequence lengths in a batch for KV.
 *  \param[in]     rng_state             Seed and offset of the random number generator.
 *  \param[in]     max_seqlen_q          Max sequence length of this batch for Q.  
 *  \param[in]     max_seqlen_kv         Max sequence length of this batch for KV.  
 *  \param[in]     is_training           Whether this is in training mode or inference.
 *  \param[in]     attn_scale            Scaling factor for Q * K.T.
 *  \param[in]     p_dropout             Dropout probability.
 *  \param[in]     qkv_layout            Layout of QKV.
 *  \param[in]     bias_type             Type of the bias.
 *  \param[in]     attn_mask_type        Type of the attention mask.
 *  \param[in]     workspace             The workspace tensor.
 *  \param[in]     stream                CUDA stream used for this function.
 */
void nvte_fused_attn_fwd_kvpacked(
            const NVTETensor Q,
            const NVTETensor KV,
            const NVTETensor Bias,
            NVTETensor S,
            NVTETensor O,
            NVTETensorPack* Aux_Output_Tensors,
            const NVTETensor cu_seqlens_q,
            const NVTETensor cu_seqlens_kv,
            const NVTETensor rng_state,
            size_t max_seqlen_q, size_t max_seqlen_kv,
            bool is_training, float attn_scale, float p_dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream);

/*! \brief Perform the backprop of dot product attention with packed KV input.
 *
 *  \param[in]     Q                     The Q tensor.
 *  \param[in]     KV                    The KV tensor.
 *  \param[in]     dBias                 The gradient of the B tensor.
 *  \param[in]     O                     The O tensor from the forward pass.
 *  \param[in]     dO                    The gradient of the O tensor.
 *  \param[in]     S                     The S tensor, S = Softmax(Q * K.T).
 *  \param[in,out] dP                    The gradient of the P tensor, P = Q * K.T.
 *  \param[in]     Aux_CTX_tensors       Auxiliary tensors from the forward pass when training.
 *  \param[out]    dQ                    The gradient of the Q tensor.
 *  \param[out]    dKV                   The gradient of the KV tensor.
 *  \param[in]     cu_seqlens_q          Accumulative sequence lengths in a batch for Q.
 *  \param[in]     cu_seqlens_kv         Accumulative sequence lengths in a batch for KV.
 *  \param[in]     rng_state             Seed and offset of the random number generator.
 *  \param[in]     max_seqlen_q          Max sequence length of this batch for Q.  
 *  \param[in]     max_seqlen_kv         Max sequence length of this batch for KV.  
 *  \param[in]     attn_scale            Scaling factor for Q * K.T.
 *  \param[in]     p_dropout             Dropout probability.
 *  \param[in]     qkv_layout            Layout of QKV.
 *  \param[in]     bias_type             Type of the bias.
 *  \param[in]     attn_mask_type        Type of the attention mask.
 *  \param[in]     workspace             The workspace tensor.
 *  \param[in]     stream                CUDA stream used for this function.
 */
void nvte_fused_attn_bwd_kvpacked(
            const NVTETensor Q,
            const NVTETensor KV,
            const NVTETensor dBias,
            const NVTETensor O,
            const NVTETensor dO,
            const NVTETensor S,
            NVTETensor dP,
            const NVTETensorPack* Aux_CTX_Tensors,
            NVTETensor dQ,
            NVTETensor dKV,
            const NVTETensor cu_seqlens_q,
            const NVTETensor cu_seqlens_kv,
            size_t max_seqlen_q, size_t max_seqlen_kv,
            float attn_scale, float p_dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
