/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_UTILS_H_
#define TRANSFORMER_ENGINE_FUSED_ATTN_UTILS_H_

#include "transformer_engine/transformer_engine.h"
#include <cudnn_frontend.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

enum MHA_Matrix {
    Q_Matrix            = 0,  // queries
    K_Matrix            = 1,  // keys
    K_Matrix_Transpose  = 2,  // keys transposed
    V_Matrix            = 3,  // values
    V_Matrix_Transpose  = 4,  // value matrix transposed
    S_Matrix            = 5,  // output of GEMM1
    O_Matrix            = 6,  // final output
};

class cudnnExecutionPlanManager {
 public:
    static cudnnExecutionPlanManager &Instance() {
        static thread_local cudnnExecutionPlanManager instance;
        return instance;
    }

    cudnnHandle_t GetCudnnHandle() {
        static thread_local std::once_flag flag;
        std::call_once(flag, [&] { cudnnCreate(&handle_); });
        return handle_;
    }

    ~cudnnExecutionPlanManager() {
        static thread_local std::once_flag flag;
        std::call_once(flag, [&] { cudnnDestroy(handle_); });
    }

 private:
    cudnnHandle_t handle_;
};

cudnnDataType_t get_cudnn_dtype(const transformer_engine::DType t);

__global__ void cu_seqlens_to_offsets(size_t b, size_t h, size_t d,
                int32_t *cu_seqlens_q, int32_t *actual_seqlens_q,
                int32_t *qkv_ragged_offset, int32_t *o_ragged_offset);

namespace transformer_engine {
namespace fused_attn {

using namespace transformer_engine;

void generateMHAStrides(
            int64_t b, int64_t h,
            int64_t s_q, int64_t s_kv,
            int64_t d, int64_t* strideA,
            MHA_Layout layout, MHA_Matrix matrix);

struct FADescriptor {
  std::int64_t b;
  std::int64_t h;
  std::int64_t s_q;
  std::int64_t s_kv;
  std::int64_t d;
  float attnScale;
  bool isTraining;
  float dropoutProbability;
  MHA_Layout layout;
  cudnnDataType_t tensor_type;

  bool operator<(const FADescriptor &rhs) const {
    return std::tie(b, h, s_q, s_kv, d,
                    attnScale, isTraining, dropoutProbability,
                    layout, tensor_type) < std::tie(
                            rhs.b, rhs.h, rhs.s_q, rhs.s_kv, rhs.d,
                            rhs.attnScale, rhs.isTraining,
                            rhs.dropoutProbability, rhs.layout, rhs.tensor_type);
  }
};
}  // namespace fused_attn
}  // namespace transformer_engine

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
