/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/fused_attn.h"
#include "../common.h"
#include "utils.h"
#include "fused_attn_f16_max512_seqlen.h"
#include "fused_attn_f16_arbitrary_seqlen.h"
#include "fused_attn_fp8.h"

// NVTE fused attention FWD FP8 with packed QKV
void nvte_fused_attn_fwd_qkvpacked(
            const NVTETensor QKV,
            const NVTETensor Bias,
            NVTETensor S,
            NVTETensor O,
            NVTETensorPack* Aux_CTX_Tensors,
            const NVTETensor cu_seqlens,
            const NVTETensor rng_state,
            size_t max_seqlen,
            bool is_training, float attn_scale, float dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream,
            bool return_softmax,
            int num_split,
            NVTE_Fused_Attn_Backend fused_attention_backend) {
  NVTE_API_CALL(nvte_flash_attn_fwd_qkvpacked);
  using namespace transformer_engine;

  const Tensor *input_cu_seqlens = reinterpret_cast<const Tensor*>(cu_seqlens);
  const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(rng_state);
  const Tensor *input_QKV = reinterpret_cast<const Tensor*>(QKV);
  const Tensor *input_Bias = reinterpret_cast<const Tensor*>(Bias);
  // TODO should S be included in aux_output_tensors (no? amax/scale tensors need to be provided)
  // should S be used as Softmax, S_dmask, softmax_stats, or softmax_lse
  Tensor *input_output_S = reinterpret_cast<Tensor*>(S);
  Tensor *output_O = reinterpret_cast<Tensor*>(O);
  Tensor *wkspace = reinterpret_cast<Tensor*>(workspace);

  // QKV shape is [total_seqs, 3, h, d]
  auto ndim = input_QKV->data.shape.size();
  size_t b = input_cu_seqlens->data.shape[0] - 1;
  size_t h = input_QKV->data.shape[ndim - 2];
  size_t d = input_QKV->data.shape[ndim - 1];

  auto handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();
  const DType QKV_type = input_QKV->data.dtype;

  // TODO use enums instead
  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_FlashAttn) {
    // return_softmax and num_split are used in backend 1
    const char *err_msg =
    "Fused attention backend 1 is currently a placeholder. "
    "Please use one of the other backends instead. \n";
    NVTE_ERROR(err_msg);
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
#if (CUDNN_VERSION >= 8901)
      fused_attn_max_512_fwd_qkvpacked(
          b, max_seqlen, h, d,
          is_training, attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
          input_QKV, input_Bias, output_O,
          Aux_CTX_Tensors,
          input_cu_seqlens,
          input_rng_state,
          wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.1 is required for BF16/FP16 fused attention with max_seqlen<=512. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
#if (CUDNN_VERSION >= 8900)
      fused_attn_arbitrary_seqlen_fwd_qkvpacked(
          b, max_seqlen, h, d,
          is_training, attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
          input_QKV, input_Bias, output_O,
          Aux_CTX_Tensors,
          input_cu_seqlens,
          input_rng_state,
          wkspace, stream, handle);
#else
    NVTE_ERROR(
      "cuDNN 8.9.0 is required for BF16/FP16 fused attention with arbitrary sequence length. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
#if (CUDNN_VERSION >= 8900)
    // FP8 API doesn't use input_Bias, bias_type or attn_mask_type
    fused_attn_fp8_fwd_qkvpacked(
            b, max_seqlen, h, d,
            is_training, attn_scale, dropout, qkv_layout,
            input_QKV, input_output_S, output_O,
            Aux_CTX_Tensors,
            input_cu_seqlens,
            input_rng_state,
            wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.0 is required for FP8 fused attention. \n");
#endif
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }
}
// NVTE fused attention BWD FP8 with packed QKV
void nvte_fused_attn_bwd_qkvpacked(
            const NVTETensor QKV,
            const NVTETensor O,
            const NVTETensor dO,
            const NVTETensor S,
            NVTETensor dP,
            const NVTETensorPack* Aux_CTX_Tensors,
            NVTETensor dQKV,
            NVTETensor dBias,
            const NVTETensor cu_seqlens,
            size_t max_seqlen,
            float attn_scale, float dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream,
            int num_split,
            NVTE_Fused_Attn_Backend fused_attention_backend) {
  NVTE_API_CALL(nvte_flash_attn_bwd_qkvpacked);
  using namespace transformer_engine;

  const Tensor *input_cu_seqlens = reinterpret_cast<const Tensor*>(cu_seqlens);
  const Tensor *input_QKV = reinterpret_cast<const Tensor*>(QKV);
  const Tensor *input_O = reinterpret_cast<const Tensor*>(O);
  const Tensor *input_dO = reinterpret_cast<const Tensor*>(dO);
  const Tensor *input_S = reinterpret_cast<const Tensor*>(S);
  Tensor *input_output_dP = reinterpret_cast<Tensor*>(dP);
  Tensor *output_dQKV = reinterpret_cast<Tensor*>(dQKV);
  Tensor *output_dBias = reinterpret_cast<Tensor*>(dBias);
  Tensor *wkspace = reinterpret_cast<Tensor*>(workspace);

  // QKV shape is [total_seqs, 3, h, d]
  auto ndim = input_QKV->data.shape.size();
  size_t b = input_cu_seqlens->data.shape[0] - 1;
  size_t h = input_QKV->data.shape[ndim - 2];
  size_t d = input_QKV->data.shape[ndim - 1];

  auto handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();
  const DType QKV_type = input_QKV->data.dtype;

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_FlashAttn) {
    // return_softmax and num_split are used in backend 1
    const char *err_msg =
    "Fused attention backend 1 is currently a placeholder. "
    "Please use one of the other backends instead. \n";
    NVTE_ERROR(err_msg);
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
#if (CUDNN_VERSION >= 8901)
      // TODO input_S or output_S?
      // TODO naming of functions
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      fused_attn_max_512_bwd_qkvpacked(
          b, max_seqlen, h, d,
          attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
          input_QKV, input_dO,
//          Aux_CTX_Tensors,
          output_S,
          output_dQKV, output_dBias,
          input_cu_seqlens,
          wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.1 is required for BF16/FP16 fused attention with max_seqlen<=512. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
#if (CUDNN_VERSION >= 8900)
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[1]);
      fused_attn_arbitrary_seqlen_bwd_qkvpacked(
          b, max_seqlen, h, d,
          attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
          input_QKV, input_O, input_dO,
//          Aux_CTX_Tensors,
          output_S,
          output_dQKV, output_dBias,
          input_cu_seqlens, input_rng_state,
          wkspace, stream, handle);
#else
    const char *err_msg =
    "cuDNN 8.9.0 is required for BF16/FP16 fused attention "
    "with arbitrary sequence length. \n";
    NVTE_ERROR(err_msg);
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
#if (CUDNN_VERSION >= 8900)
    // Aux_CTX_Tensors contain [M, ZInv, rng_state] generated by the forward pass
    const Tensor *input_M = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[0]);
    const Tensor *input_ZInv = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[1]);
    const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[2]);

    // FP8 API doesn't use input_dBias, bias_type or attn_mask_type
    fused_attn_fp8_bwd_qkvpacked(
                    b, max_seqlen, h, d,
                    attn_scale, dropout, qkv_layout,
                    input_QKV, input_O, input_dO,
                    input_M, input_ZInv,
                    input_S, input_output_dP,
                    output_dQKV,
                    input_cu_seqlens,
                    input_rng_state,
                    wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.0 is required for FP8 fused attention. \n");
#endif
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }
}
// NVTE fused attention FWD FP8 with packed KV
void nvte_fused_attn_fwd_kvpacked(
            const NVTETensor Q,
            const NVTETensor KV,
            const NVTETensor Bias,
            NVTETensor S,
            NVTETensor O,
            NVTETensorPack* Aux_CTX_Tensors,
            const NVTETensor cu_seqlens_q,
            const NVTETensor cu_seqlens_kv,
            const NVTETensor rng_state,
            size_t max_seqlen_q, size_t max_seqlen_kv,
            bool is_training, float attn_scale, float dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream,
            bool return_softmax,
            int num_split,
            NVTE_Fused_Attn_Backend fused_attention_backend) {
  NVTE_API_CALL(nvte_flash_attn_fwd_kvpacked);
  using namespace transformer_engine;
  const Tensor *input_cu_seqlens_q = reinterpret_cast<const Tensor*>(cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = reinterpret_cast<const Tensor*>(cu_seqlens_kv);
  const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(rng_state);
  const Tensor *input_Q = reinterpret_cast<const Tensor*>(Q);
  const Tensor *input_KV = reinterpret_cast<const Tensor*>(KV);
  const Tensor *input_Bias = reinterpret_cast<const Tensor*>(Bias);
  Tensor *input_output_S = reinterpret_cast<Tensor*>(S);
  Tensor *output_O = reinterpret_cast<Tensor*>(O);
  Tensor *wkspace = reinterpret_cast<Tensor*>(workspace);

  // Q shape is [total_seqs, h, d]
  // KV shape is [total_seqs, h, d]
  auto ndim = input_Q->data.shape.size();
  size_t b = input_cu_seqlens_q->data.shape[0] - 1;
  size_t h = input_Q->data.shape[ndim - 2];
  size_t d = input_Q->data.shape[ndim - 1];

  auto handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();
  const DType QKV_type = input_Q->data.dtype;

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_FlashAttn) {
    // return_softmax and num_split are used in backend 1
    const char *err_msg =
    "Fused attention backend 1 is currently a placeholder. "
    "Please use one of the other backends instead. \n";
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
#if (CUDNN_VERSION >= 8901)
      fused_attn_max_512_fwd_kvpacked(
          b, max_seqlen_q, max_seqlen_kv, h, d,
          is_training, attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
          input_Q, input_KV, input_Bias, output_O,
          Aux_CTX_Tensors,
          input_cu_seqlens_q, input_cu_seqlens_kv,
          input_rng_state,
          wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.1 is required for BF16/FP16 fused attention with max_seqlen<=512. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
    const char* err_msg =
    "The FP16/BF16 fused attention (arbitrary seqlen) currently "
    "only supports packed QKV input.\n";
    NVTE_ERROR(err_msg);
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
    NVTE_ERROR("The FP8 fused attention API only supports packed QKV input. \n");
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }
}
// NVTE fused attention BWD FP8 with packed KV
void nvte_fused_attn_bwd_kvpacked(
            const NVTETensor Q,
            const NVTETensor KV,
            const NVTETensor O,
            const NVTETensor dO,
            const NVTETensor S,
            NVTETensor dP,
            const NVTETensorPack* Aux_CTX_Tensors,
            NVTETensor dQ,
            NVTETensor dKV,
            NVTETensor dBias,
            const NVTETensor cu_seqlens_q,
            const NVTETensor cu_seqlens_kv,
            size_t max_seqlen_q, size_t max_seqlen_kv,
            float attn_scale, float dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream,
            int num_split,
            NVTE_Fused_Attn_Backend fused_attention_backend) {
  NVTE_API_CALL(nvte_flash_attn_bwd_kvpacked);
  using namespace transformer_engine;
  const Tensor *input_cu_seqlens_q = reinterpret_cast<const Tensor*>(cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = reinterpret_cast<const Tensor*>(cu_seqlens_kv);
  const Tensor *input_Q = reinterpret_cast<const Tensor*>(Q);
  const Tensor *input_KV = reinterpret_cast<const Tensor*>(KV);
  const Tensor *input_O = reinterpret_cast<const Tensor*>(O);
  const Tensor *input_dO = reinterpret_cast<const Tensor*>(dO);
  const Tensor *input_S = reinterpret_cast<const Tensor*>(S);
  Tensor *input_output_dP = reinterpret_cast<Tensor*>(dP);
  Tensor *output_dQ = reinterpret_cast<Tensor*>(dQ);
  Tensor *output_dKV = reinterpret_cast<Tensor*>(dKV);
  Tensor *output_dBias = reinterpret_cast<Tensor*>(dBias);
  Tensor *wkspace = reinterpret_cast<Tensor*>(workspace);

  // Q shape is [total_seqs, h, d]
  // KV shape is [total_seqs, h, d]
  auto ndim = input_Q->data.shape.size();
  size_t b = input_cu_seqlens_q->data.shape[0] - 1;
  size_t h = input_Q->data.shape[ndim - 2];
  size_t d = input_Q->data.shape[ndim - 1];

  auto handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();
  const DType QKV_type = input_Q->data.dtype;

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_FlashAttn) {
    // return_softmax and num_split are used in backend 1
    const char *err_msg =
    "Fused attention backend 1 is currently a placeholder. "
    "Please use one of the other backends instead. \n";
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
#if (CUDNN_VERSION >= 8901)
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      fused_attn_max_512_bwd_kvpacked(
          b, max_seqlen_q, max_seqlen_kv, h, d,
          attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
          input_Q, input_KV, input_dO,
//          Aux_CTX_Tensors,
          output_S,
          output_dQ, output_dKV, output_dBias,
          input_cu_seqlens_q, input_cu_seqlens_kv,
          wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.1 is required for BF16/FP16 fused attention with max_seqlen<=512. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
    const char* err_msg =
    "The FP16/BF16 fused attention (arbitrary seqlen) currently "
    "only supports packed QKV input.\n";
    NVTE_ERROR(err_msg);
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
    NVTE_ERROR("The FP8 fused attention API only supports packed QKV input. \n");
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }
}
