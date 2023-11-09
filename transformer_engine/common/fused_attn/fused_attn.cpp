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
#include "../util/cuda_runtime.h"
#include "../util/system.h"

// map NVTE_QKV_Layout to NVTE_QKV_Layout_Group
NVTE_QKV_Layout_Group nvte_get_qkv_layout_group(NVTE_QKV_Layout qkv_layout) {
    switch (qkv_layout) {
        case NVTE_QKV_Layout::NVTE_SB3HD:
        case NVTE_QKV_Layout::NVTE_BS3HD:
        case NVTE_QKV_Layout::NVTE_T3HD:
            return NVTE_QKV_Layout_Group::NVTE_3HD;
        case NVTE_QKV_Layout::NVTE_SBH3D:
        case NVTE_QKV_Layout::NVTE_BSH3D:
        case NVTE_QKV_Layout::NVTE_TH3D:
            return NVTE_QKV_Layout_Group::NVTE_H3D;
        case NVTE_QKV_Layout::NVTE_SBHD_SB2HD:
        case NVTE_QKV_Layout::NVTE_BSHD_BS2HD:
        case NVTE_QKV_Layout::NVTE_THD_T2HD:
            return NVTE_QKV_Layout_Group::NVTE_HD_2HD;
        case NVTE_QKV_Layout::NVTE_SBHD_SBH2D:
        case NVTE_QKV_Layout::NVTE_BSHD_BSH2D:
        case NVTE_QKV_Layout::NVTE_THD_TH2D:
            return NVTE_QKV_Layout_Group::NVTE_HD_H2D;
        case NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD:
        case NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD:
        case NVTE_QKV_Layout::NVTE_THD_THD_THD:
            return NVTE_QKV_Layout_Group::NVTE_HD_HD_HD;
        default:
            NVTE_ERROR("qkv_layout not supported!");
    }
}

// map NVTE_QKV_Layout to NVTE_QKV_Format
NVTE_QKV_Format nvte_get_qkv_format(NVTE_QKV_Layout qkv_layout) {
    switch (qkv_layout) {
        case NVTE_QKV_Layout::NVTE_SB3HD:
        case NVTE_QKV_Layout::NVTE_SBH3D:
        case NVTE_QKV_Layout::NVTE_SBHD_SB2HD:
        case NVTE_QKV_Layout::NVTE_SBHD_SBH2D:
        case NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD:
            return NVTE_QKV_Format::NVTE_SBHD;
        case NVTE_QKV_Layout::NVTE_BS3HD:
        case NVTE_QKV_Layout::NVTE_BSH3D:
        case NVTE_QKV_Layout::NVTE_BSHD_BS2HD:
        case NVTE_QKV_Layout::NVTE_BSHD_BSH2D:
        case NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD:
            return NVTE_QKV_Format::NVTE_BSHD;
        case NVTE_QKV_Layout::NVTE_T3HD:
        case NVTE_QKV_Layout::NVTE_TH3D:
        case NVTE_QKV_Layout::NVTE_THD_T2HD:
        case NVTE_QKV_Layout::NVTE_THD_TH2D:
        case NVTE_QKV_Layout::NVTE_THD_THD_THD:
            return NVTE_QKV_Format::NVTE_THD;
        default:
            NVTE_ERROR("qkv_layout not supported!");
    }
}

// select a backend for fused attention
NVTE_Fused_Attn_Backend nvte_get_fused_attn_backend(
        NVTEDType q_dtype,
        NVTEDType kv_dtype,
        NVTE_QKV_Layout qkv_layout,
        NVTE_Bias_Type bias_type,
        NVTE_Mask_Type attn_mask_type,
        float dropout, size_t max_seqlen_q,
        size_t max_seqlen_kv, size_t head_dim,
        size_t num_attn_heads, size_t num_gqa_groups) {
  using namespace transformer_engine;
  NVTE_Fused_Attn_Backend backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
  const int device_id = cuda::current_device();
  const int sm_arch_ = cuda::sm_arch(device_id);
  NVTE_CHECK(q_dtype == kv_dtype, "Q and KV must have the same data type.");
  NVTE_QKV_Format qkv_format = nvte_get_qkv_format(qkv_layout);
  if ((q_dtype == NVTEDType::kNVTEFloat8E4M3) || (q_dtype == NVTEDType::kNVTEFloat8E5M2)
          && (sm_arch_ >= 90)
          && (max_seqlen_q == max_seqlen_kv)
          && (num_attn_heads == num_gqa_groups)
          && (max_seqlen_q <= 512)
          && (head_dim == 64)
          && (bias_type == NVTE_Bias_Type::NVTE_NO_BIAS)
          && (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK)
          && (qkv_layout == NVTE_QKV_Layout::NVTE_T3HD)) {
#if (CUDNN_VERSION >= 8900)
    backend = NVTE_Fused_Attn_Backend::NVTE_FP8;
#else
    backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
    std::cout << "Warning: FP8 fused attention is supported by cuDNN 8.9.0+."
           " Please upgrade your cuDNN version if possible." << std::endl;
#endif
  } else if ((q_dtype == NVTEDType::kNVTEFloat16) || (q_dtype == NVTEDType::kNVTEBFloat16)) {
    bool flag_m512 = false;
    bool flag_arb = false;
    if ((sm_arch_ == 80 || sm_arch_ == 90)
            && (head_dim == 64)
            && (num_attn_heads == num_gqa_groups)
            && ((bias_type == NVTE_Bias_Type::NVTE_NO_BIAS)
                || (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS))
            && ((attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK)
                || (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK)
                || (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK)  // TODO
                || (attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK))
            && ((qkv_layout == NVTE_QKV_Layout::NVTE_SB3HD)
                || (qkv_layout == NVTE_QKV_Layout::NVTE_SBHD_SB2HD)
                || (qkv_layout == NVTE_QKV_Layout::NVTE_BS3HD)
                || (qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BS2HD)
                || (qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD))) {
      flag_m512 = true;
    }
    if (
#if (CUDNN_VERSION >= 8903)
        (sm_arch_ >= 80)
#else
        (sm_arch_ == 80 || sm_arch_ == 90)
#endif
            && (max_seqlen_q % 64 == 0)
            && (max_seqlen_kv % 64 == 0)
            && (num_attn_heads == num_gqa_groups)
            //&& ((num_attn_heads == num_gqa_groups)
            //    || ((num_attn_heads != num_gqa_groups) && (num_gqa_groups == 1)))  // TODO
            && ((head_dim <= 128) && (head_dim % 8 == 0))
            && (bias_type == NVTE_Bias_Type::NVTE_NO_BIAS)
            //&& ((bias_type == NVTE_Bias_Type::NVTE_NO_BIAS)
            //    || (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS)
            //    || (bias_type == NVTE_Bias_Type::NVTE_ALIBI))
            && ((attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK)
                || (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK)
            //    || (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK)
                || (attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK))
            && ((qkv_format == NVTE_QKV_Format::NVTE_SBHD)
                || (qkv_format == NVTE_QKV_Format::NVTE_BSHD))) {
      flag_arb = true;
    }
//    std::cout << " testing arbi... " << flag_arb << std::endl;
//    int64_t batch = 2;
//    int64_t num_head = 16;
//    head_dim = 15;
//    bool is_training = true;
//    //cudnn_frontend::DataType_t tensorType,
//    //cudnnHandle_t handle,
//
//    auto handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();
//    //flag_arb = true;
//    bool check_support = true;
//    fused_attn::fused_attn_arbitrary_seqlen_fwd_impl(
//                 batch, num_head, max_seqlen_q, max_seqlen_kv, head_dim,
//                 is_training, 0.1f, dropout, qkv_layout,
//                 bias_type, attn_mask_type,
//                 nullptr, nullptr, nullptr, nullptr, nullptr,
//                 nullptr, nullptr, nullptr, nullptr,
//                 get_cudnn_fe_dtype(static_cast<DType>(q_dtype)),
//                 nullptr, nullptr, nullptr, handle, &check_support);
//    flag_arb = check_support;
//    std::cout << " after testing arbi fwd... " << flag_arb << std::endl;
//    check_support = true;
//    fused_attn::fused_attn_arbitrary_seqlen_bwd_impl(
//                 batch, num_head, max_seqlen_q, max_seqlen_kv, head_dim,
//                 0.1f, dropout, qkv_layout,
//                 bias_type, attn_mask_type,
//                 nullptr, nullptr, nullptr, nullptr, nullptr,
//                 nullptr, nullptr, nullptr, nullptr,
//                 nullptr, nullptr, nullptr, nullptr,
//                 get_cudnn_fe_dtype(static_cast<DType>(q_dtype)),
//                 nullptr, nullptr, nullptr, handle, &check_support);
//    flag_arb = check_support;
//    std::cout << " after testing arbi bwd... " << flag_arb << std::endl;

    if (((max_seqlen_q > 512) || (max_seqlen_kv > 512))
            && (flag_arb == true)) {
      backend = NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen;
    }
    if ((max_seqlen_q <= 512) && (max_seqlen_kv <= 512)) {
      if (flag_arb == true) {
        backend = NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen;
      } else if ((flag_arb == false) && (flag_m512 == true)) {
        backend = NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen;
      }
      int env_backend = static_cast<int>(backend);
      env_backend = transformer_engine::getenv<int>("NVTE_FUSED_ATTN_BACKEND", env_backend);
      if (((env_backend == static_cast<int>(NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen))
      && flag_m512)
          || ((env_backend == static_cast<int>(NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen))
      && flag_arb)) {
          backend = static_cast<NVTE_Fused_Attn_Backend>(env_backend);
      }
    }
    std::cout << "[FusedAttn]: selected backend " << static_cast<int>(backend) << "." << std::endl;
#if (CUDNN_VERSION < 8901)
    if (backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
      backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
      std::cout << "Warning: FP16/BF16 fused attention is supported by cuDNN 8.9.1+."
           " Please upgrade your cuDNN version if possible." << std::endl;
    }
#endif
#if (CUDNN_VERSION < 8900)
    if (backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
      backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
      std::cout << "Warning: FP16/BF16 fused attention is supported by cuDNN 8.9.0+."
           " Please upgrade your cuDNN version if possible." << std::endl;
    }
#endif
  } else {
    backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
  }
  return backend;
}

// NVTE fused attention FWD with packed QKV
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
            cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_fwd_qkvpacked);
  using namespace transformer_engine;

  const Tensor *input_cu_seqlens = reinterpret_cast<const Tensor*>(cu_seqlens);
  const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(rng_state);
  const Tensor *input_QKV = reinterpret_cast<const Tensor*>(QKV);
  const Tensor *input_Bias = reinterpret_cast<const Tensor*>(Bias);
  Tensor *input_output_S = reinterpret_cast<Tensor*>(S);
  Tensor *output_O = reinterpret_cast<Tensor*>(O);
  Tensor *wkspace = reinterpret_cast<Tensor*>(workspace);

  // QKV shape is [total_seqs, 3, h, d]
  auto ndim = input_QKV->data.shape.size();
  size_t b = input_cu_seqlens->data.shape[0] - 1;
  size_t h = input_QKV->data.shape[ndim - 2];
  size_t d = input_QKV->data.shape[ndim - 1];

  auto handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();
  const NVTEDType QKV_type = static_cast<NVTEDType>(input_QKV->data.dtype);

  NVTE_Fused_Attn_Backend fused_attention_backend =
              nvte_get_fused_attn_backend(
                          QKV_type, QKV_type,
                          qkv_layout, bias_type, attn_mask_type,
                          dropout, max_seqlen, max_seqlen, d, h, h);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
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
// NVTE fused attention BWD with packed QKV
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
            cudaStream_t stream) {
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
  const NVTEDType QKV_type = static_cast<NVTEDType>(input_QKV->data.dtype);

  NVTE_Fused_Attn_Backend fused_attention_backend =
              nvte_get_fused_attn_backend(
                          QKV_type, QKV_type,
                          qkv_layout, bias_type, attn_mask_type,
                          dropout, max_seqlen, max_seqlen, d, h, h);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
#if (CUDNN_VERSION >= 8901)
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      fused_attn_max_512_bwd_qkvpacked(
          b, max_seqlen, h, d,
          attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
          input_QKV, input_dO,
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
    const Tensor *input_M = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[0]);
    const Tensor *input_ZInv = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[1]);
    const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[2]);
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
// NVTE fused attention FWD with packed KV
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
            cudaStream_t stream) {
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
  size_t hg = input_KV->data.shape[ndim - 2];
  size_t d = input_Q->data.shape[ndim - 1];

  auto handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();
  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_KV->data.dtype);

  NVTE_Fused_Attn_Backend fused_attention_backend =
              nvte_get_fused_attn_backend(
                          Q_type, KV_type,
                          qkv_layout, bias_type, attn_mask_type,
                          dropout, max_seqlen_q, max_seqlen_kv, d, h, hg);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
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
// NVTE fused attention BWD with packed KV
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
            cudaStream_t stream) {
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
  size_t hg = input_KV->data.shape[ndim - 2];
  size_t d = input_Q->data.shape[ndim - 1];

  auto handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();
  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_KV->data.dtype);

  NVTE_Fused_Attn_Backend fused_attention_backend =
              nvte_get_fused_attn_backend(
                          Q_type, KV_type,
                          qkv_layout, bias_type, attn_mask_type,
                          dropout, max_seqlen_q, max_seqlen_kv, d, h, hg);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
#if (CUDNN_VERSION >= 8901)
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      fused_attn_max_512_bwd_kvpacked(
          b, max_seqlen_q, max_seqlen_kv, h, d,
          attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
          input_Q, input_KV, input_dO,
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
// NVTE fused attention FWD with separate Q, K and V
void nvte_fused_attn_fwd(
            const NVTETensor Q,
            const NVTETensor K,
            const NVTETensor V,
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
            cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_fwd);
  using namespace transformer_engine;
  const Tensor *input_cu_seqlens_q = reinterpret_cast<const Tensor*>(cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = reinterpret_cast<const Tensor*>(cu_seqlens_kv);
  const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(rng_state);
  const Tensor *input_Q = reinterpret_cast<const Tensor*>(Q);
  const Tensor *input_K = reinterpret_cast<const Tensor*>(K);
  const Tensor *input_V = reinterpret_cast<const Tensor*>(V);
  const Tensor *input_Bias = reinterpret_cast<const Tensor*>(Bias);
  Tensor *input_output_S = reinterpret_cast<Tensor*>(S);
  Tensor *output_O = reinterpret_cast<Tensor*>(O);
  Tensor *wkspace = reinterpret_cast<Tensor*>(workspace);

  auto ndim = input_Q->data.shape.size();
  size_t b = input_cu_seqlens_q->data.shape[0] - 1;
  size_t h = input_Q->data.shape[ndim - 2];
  size_t hg = input_K->data.shape[ndim - 2];
  size_t d = input_Q->data.shape[ndim - 1];

  auto handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();
  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_K->data.dtype);

  NVTE_Fused_Attn_Backend fused_attention_backend =
              nvte_get_fused_attn_backend(
                          Q_type, KV_type,
                          qkv_layout, bias_type, attn_mask_type,
                          dropout, max_seqlen_q, max_seqlen_kv, d, h, hg);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
#if (CUDNN_VERSION >= 8901)
      fused_attn_max_512_fwd(
          b, max_seqlen_q, max_seqlen_kv, h, d,
          is_training, attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
          input_Q, input_K, input_V, input_Bias, output_O,
          Aux_CTX_Tensors,
          input_cu_seqlens_q, input_cu_seqlens_kv,
          input_rng_state,
          wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.1 is required for BF16/FP16 fused attention with max_seqlen<=512. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
#if (CUDNN_VERSION >= 8900)
      fused_attn_arbitrary_seqlen_fwd(
          b, max_seqlen_q, max_seqlen_kv, h, hg, d,
          is_training, attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
          input_Q, input_K, input_V, input_Bias, output_O,
          Aux_CTX_Tensors,
          input_cu_seqlens_q, input_cu_seqlens_kv,
          input_rng_state,
          wkspace, stream, handle);
#else
    NVTE_ERROR(
      "cuDNN 8.9.0 is required for BF16/FP16 fused attention with arbitrary sequence length. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
#if (CUDNN_VERSION >= 8900)
    fused_attn_fp8_fwd(
            b, max_seqlen_q, max_seqlen_kv, h, d,
            is_training, attn_scale, dropout, qkv_layout,
            input_Q, input_K, input_V, input_output_S, output_O,
            Aux_CTX_Tensors,
            input_cu_seqlens_q, input_cu_seqlens_kv,
            input_rng_state,
            wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.0 is required for FP8 fused attention. \n");
#endif
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }
}
// NVTE fused attention BWD with separate Q, K and V
void nvte_fused_attn_bwd(
            const NVTETensor Q,
            const NVTETensor K,
            const NVTETensor V,
            const NVTETensor O,
            const NVTETensor dO,
            const NVTETensor S,
            NVTETensor dP,
            const NVTETensorPack* Aux_CTX_Tensors,
            NVTETensor dQ,
            NVTETensor dK,
            NVTETensor dV,
            NVTETensor dBias,
            const NVTETensor cu_seqlens_q,
            const NVTETensor cu_seqlens_kv,
            size_t max_seqlen_q, size_t max_seqlen_kv,
            float attn_scale, float dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_bwd);
  using namespace transformer_engine;
  const Tensor *input_cu_seqlens_q = reinterpret_cast<const Tensor*>(cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = reinterpret_cast<const Tensor*>(cu_seqlens_kv);
  const Tensor *input_Q = reinterpret_cast<const Tensor*>(Q);
  const Tensor *input_K = reinterpret_cast<const Tensor*>(K);
  const Tensor *input_V = reinterpret_cast<const Tensor*>(V);
  const Tensor *input_O = reinterpret_cast<const Tensor*>(O);
  const Tensor *input_dO = reinterpret_cast<const Tensor*>(dO);
  const Tensor *input_S = reinterpret_cast<const Tensor*>(S);
  Tensor *input_output_dP = reinterpret_cast<Tensor*>(dP);
  Tensor *output_dQ = reinterpret_cast<Tensor*>(dQ);
  Tensor *output_dK = reinterpret_cast<Tensor*>(dK);
  Tensor *output_dV = reinterpret_cast<Tensor*>(dV);
  Tensor *output_dBias = reinterpret_cast<Tensor*>(dBias);
  Tensor *wkspace = reinterpret_cast<Tensor*>(workspace);

  auto ndim = input_Q->data.shape.size();
  size_t b = input_cu_seqlens_q->data.shape[0] - 1;
  size_t h = input_Q->data.shape[ndim - 2];
  size_t hg = input_K->data.shape[ndim - 2];
  size_t d = input_Q->data.shape[ndim - 1];

  auto handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();
  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_K->data.dtype);

  NVTE_Fused_Attn_Backend fused_attention_backend =
              nvte_get_fused_attn_backend(
                          Q_type, KV_type,
                          qkv_layout, bias_type, attn_mask_type,
                          dropout, max_seqlen_q, max_seqlen_kv, d, h, hg);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
#if (CUDNN_VERSION >= 8901)
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      fused_attn_max_512_bwd(
          b, max_seqlen_q, max_seqlen_kv, h, d,
          attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
          input_Q, input_K, input_V, input_dO,
          output_S,
          output_dQ, output_dK, output_dV, output_dBias,
          input_cu_seqlens_q, input_cu_seqlens_kv,
          wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.1 is required for BF16/FP16 fused attention with max_seqlen<=512. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
#if (CUDNN_VERSION >= 8900)
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[1]);
      fused_attn_arbitrary_seqlen_bwd(
          b, max_seqlen_q, max_seqlen_kv, h, hg, d,
          attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
          input_Q, input_K, input_V, input_O, input_dO,
          output_S,
          output_dQ, output_dK, output_dV, output_dBias,
          input_cu_seqlens_q, input_cu_seqlens_kv,
          input_rng_state, wkspace, stream, handle);
#else
    const char *err_msg =
    "cuDNN 8.9.0 is required for BF16/FP16 fused attention "
    "with arbitrary sequence length. \n";
    NVTE_ERROR(err_msg);
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
#if (CUDNN_VERSION >= 8900)
    const Tensor *input_M = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[0]);
    const Tensor *input_ZInv = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[1]);
    const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[2]);
    fused_attn_fp8_bwd(
                    b, max_seqlen_q, max_seqlen_kv, h, d,
                    attn_scale, dropout, qkv_layout,
                    input_Q, input_K, input_V, input_O, input_dO,
                    input_M, input_ZInv,
                    input_S, input_output_dP,
                    output_dQ, output_dK, output_dV,
                    input_cu_seqlens_q, input_cu_seqlens_kv,
                    input_rng_state,
                    wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.0 is required for FP8 fused attention. \n");
#endif
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }
}
