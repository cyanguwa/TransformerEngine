/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/fused_attn.h"
#include "../common.h"
#include "utils.h"

// get QKV layout in enums
NVTE_QKV_Layout get_nvte_qkv_layout(const std::string qkv_layout) {
  if (qkv_layout == "not_interleaved") {
      return NVTE_QKV_Layout::NOT_INTERLEAVED;
  } else if (qkv_layout == "qkv_interleaved") {
      return NVTE_QKV_Layout::QKV_INTERLEAVED;
  } else if (qkv_layout == "kv_interleaved") {
      return NVTE_QKV_Layout::KV_INTERLEAVED;
  } else {
      NVTE_ERROR("Invalid QKV layout. \n");
  }
}

// get bias type in enums
NVTE_Bias_Type get_nvte_bias_type(const std::string bias_type) {
  if (bias_type == "no_bias") {
      return NVTE_Bias_Type::NO_BIAS;
  } else if (bias_type == "pre_scale_bias") {
      return NVTE_Bias_Type::PRE_SCALE_BIAS;
  } else if (bias_type == "post_scale_bias") {
      return NVTE_Bias_Type::POST_SCALE_BIAS;
  } else {
      NVTE_ERROR("Invalid bias type. \n");
  }
}

// get attn mask type in enums
NVTE_Mask_Type get_nvte_mask_type(const std::string mask_type) {
  if (mask_type == "padding") {
      return NVTE_Mask_Type::PADDING;
  } else if (mask_type == "causal") {
      return NVTE_Mask_Type::CAUSAL;
  } else if (mask_type == "no_mask") {
      return NVTE_Mask_Type::NO_MASK;
  } else {
      NVTE_ERROR("Invalid attention mask type. \n");
  }
}

// create NVTETensorPack
void nvte_tensor_pack_create(NVTETensorPack* pack) {
  for (int i = 0; i < pack->MAX_SIZE; i++) {
     pack->tensors[i] = reinterpret_cast<NVTETensor>(new transformer_engine::Tensor);
  }
}

// destroy NVTETensorPack
void nvte_tensor_pack_destroy(NVTETensorPack* pack) {
  for (int i = 0; i < pack->MAX_SIZE; i++) {
     auto *t = reinterpret_cast<transformer_engine::Tensor*>(pack->tensors[i]);
     delete t;
  }
}

// get cuDNN data type
cudnnDataType_t get_cudnn_dtype(const transformer_engine::DType t) {
  using namespace transformer_engine;
  switch (t) {
    case DType::kFloat16:
      return CUDNN_DATA_HALF;
    case DType::kFloat32:
      return CUDNN_DATA_FLOAT;
    case DType::kBFloat16:
      return CUDNN_DATA_BFLOAT16;
    case DType::kFloat8E4M3:
      return CUDNN_DATA_FP8_E4M3;
    case DType::kFloat8E5M2:
      return CUDNN_DATA_FP8_E5M2;
    default:
      NVTE_ERROR("Invalid cuDNN data type. \n");
  }
}

// convert cu_seqlens_q to qkv/o_ragged_offset and actual_seqlens_q
__global__ void cu_seqlens_to_offsets(size_t b, size_t h, size_t d,
                int32_t *cu_seqlens_q, int32_t *actual_seqlens_q,
                int32_t *qkv_ragged_offset, int32_t *o_ragged_offset) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < b) {
    actual_seqlens_q[tid] = cu_seqlens_q[tid + 1] - cu_seqlens_q[tid];
  }
  if (tid < b + 1) {
    qkv_ragged_offset[tid] = cu_seqlens_q[tid] * 3 * h * d;
    o_ragged_offset[tid] = cu_seqlens_q[tid] * h * d;
  }
}


namespace transformer_engine {
namespace fused_attn {

using namespace transformer_engine;

// get matrix strides based on matrix type
void generateMHAStrides(
            int64_t b, int64_t h,
            int64_t s_q, int64_t s_kv,
            int64_t d, int64_t* strideA,
            NVTE_QKV_Layout layout, MHA_Matrix matrix) {
    constexpr int batch_dim_idx   = 0;
    constexpr int head_dim_idx    = 1;
    constexpr int seqlen_dim_idx  = 2;
    constexpr int hidden_dim_idx  = 3;

    constexpr int seqlen_transpose_dim_idx = 3;
    constexpr int hidden_transpose_dim_idx = 2;

    constexpr int seqlen_q_dim_idx = 2;
    constexpr int seqlen_kv_dim_idx = 3;

    switch (matrix) {
        case MHA_Matrix::Q_Matrix:
            if (layout == NVTE_QKV_Layout::QKV_INTERLEAVED) {
                strideA[hidden_dim_idx] = 1;
                strideA[seqlen_dim_idx] = 3 * h * d;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_q * 3 * h * d;
            } else {
                strideA[hidden_dim_idx] = 1;
                strideA[seqlen_dim_idx] = h * d;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_q * h * d;
            }
            break;
        case MHA_Matrix::K_Matrix:
            if (layout == NVTE_QKV_Layout::QKV_INTERLEAVED) {
                strideA[seqlen_dim_idx] = 3 * h * d;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 3 * h * d;
            } else if (layout == NVTE_QKV_Layout::KV_INTERLEAVED) {
                strideA[seqlen_transpose_dim_idx] = 2 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 2 * h * d;
            } else {
                strideA[seqlen_transpose_dim_idx] = h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * h * d;
            }
            break;
        case MHA_Matrix::K_Matrix_Transpose:
            if (layout == NVTE_QKV_Layout::QKV_INTERLEAVED) {
                strideA[seqlen_transpose_dim_idx] = 3 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 3 * h * d;
            } else if (layout == NVTE_QKV_Layout::KV_INTERLEAVED) {
                strideA[seqlen_transpose_dim_idx] = 2 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 2 * h * d;
            } else {
                strideA[seqlen_transpose_dim_idx] = h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * h * d;
            }
            break;
        case MHA_Matrix::V_Matrix:
            if (layout == NVTE_QKV_Layout::QKV_INTERLEAVED) {
                strideA[hidden_dim_idx] = 1;
                strideA[seqlen_dim_idx] = 3 * h * d;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 3 * h * d;
            } else if (layout == NVTE_QKV_Layout::KV_INTERLEAVED) {
                strideA[hidden_dim_idx] = 1;
                strideA[seqlen_dim_idx] = 2* h * d;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 2 * h * d;
            } else {
                strideA[hidden_dim_idx] = 1;
                strideA[seqlen_dim_idx] = h * d;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * h * d;
            }
            break;
        case MHA_Matrix::V_Matrix_Transpose:
            if (layout == NVTE_QKV_Layout::QKV_INTERLEAVED) {
                    strideA[hidden_transpose_dim_idx] = 1;
                    strideA[seqlen_transpose_dim_idx] = 3 * h * d;
                    strideA[head_dim_idx] = d;
                    strideA[batch_dim_idx] = s_kv * 3 * h * d;
                } else if (layout == NVTE_QKV_Layout::KV_INTERLEAVED) {
                    strideA[hidden_transpose_dim_idx] = 1;
                    strideA[seqlen_transpose_dim_idx] = 2* h * d;
                    strideA[head_dim_idx] = d;
                    strideA[batch_dim_idx] = s_kv * 2 * h * d;
                } else {
                    strideA[hidden_transpose_dim_idx] = 1;
                    strideA[seqlen_transpose_dim_idx] = h * d;
                    strideA[head_dim_idx] = d;
                    strideA[batch_dim_idx] = s_kv * h * d;
                }
            break;
        case MHA_Matrix::S_Matrix:
            strideA[seqlen_kv_dim_idx] = 1;
            strideA[seqlen_q_dim_idx] = s_kv;
            strideA[head_dim_idx] = s_q * s_kv;
            strideA[batch_dim_idx] = h * s_q * s_kv;
            break;
        case MHA_Matrix::O_Matrix:
            strideA[seqlen_kv_dim_idx] = 1;
            strideA[seqlen_q_dim_idx] = h * d;
            strideA[head_dim_idx] = d;
            strideA[batch_dim_idx] = s_q * h * d;
            break;
    }
}
}  // namespace fused_attn
}  // namespace transformer_engine
