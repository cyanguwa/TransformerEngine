/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_FP8_H_
#define TRANSFORMER_ENGINE_FUSED_ATTN_FP8_H_

#include "transformer_engine.h"
#include <cudnn_frontend.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define CUDNN_FRONTEND_UNUSED(X) ((void)X)

enum class MHA_Layout {
    NOT_INTERLEAVED = 0,
    QKV_INTERLEAVED = 1,
    KV_INTERLEAVED = 2
};

enum class MHA_Matrix {
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

//struct NVTETensorPack {
//  static const int MAX_SIZE = 10;
//  NVTETensor tensors[MAX_SIZE];
//  bool set_zero[MAX_SIZE];
//  size_t size = MAX_SIZE;
//  bool is_allocated = false;
//};
struct OutputTensorPack {
  //static const int MAX_SIZE = 10;
  //std::vector<transformer_engine::TensorWrapper> tensors;  // [MAX_SIZE];
  std::vector<NVTETensor> tensors;  // [MAX_SIZE];
  //bool set_zero[MAX_SIZE];
  std::vector<bool> set_zero;
  //size_t size = MAX_SIZE;
  size_t size = 0; // = MAX_SIZE;
  //bool is_allocated = false;
};

void nvte_fused_attn_fwd(
            int64_t b, int64_t max_seq_len, int64_t total_seqs, int64_t h, int64_t d,
            bool is_training, float attn_scale,
            float p_dropout, int qkv_layout,
            size_t num_input_tensors,
            const NVTETensor* input_list,
            transformer_engine::DType qkv_tex_dtype,
            std::vector<float*> fp8_amax_dptr_list,
            std::vector<float*> fp8_scale_dptr_list,
            std::vector<float*> fp8_scale_inv_dptr_list,
            OutputTensorPack* output_pack,
            std::vector<int32_t*> seqlen_list,
            uint64_t *rng_state,
            NVTETensor workspace,
            cudaStream_t stream);

//void nvte_fused_attn_fwd(
//                int64_t b, int64_t max_seq_len,
//                int64_t h, int64_t d,
//                float attn_scale, float p_dropout,
//                int qkv_layout, bool is_training,
//                const NVTETensor QKV,
//                NVTETensor M,
//                NVTETensor ZInv,
//                NVTETensor S,
//                NVTETensor O,
//                int32_t *QKVRaggedOffset,
//                int32_t *ORaggedOffset,
//                int32_t *Seqlens,
//                uint64_t *RngState,
//                NVTETensor workspace,
//                cudaStream_t stream);

void nvte_fused_attn_bwd(
                int64_t b, int64_t max_seq_len,
                int64_t h, int64_t d,
                float attn_scale, float p_dropout, int qkv_layout,
                const NVTETensor QKV,
                NVTETensor dQKV,
                const NVTETensor M,
                const NVTETensor ZInv,
                const NVTETensor S,
                NVTETensor dS,
                const NVTETensor O,
                const NVTETensor dO,
                int32_t *QKVRaggedOffset,
                int32_t *ORaggedOffset,
                int32_t *Seqlens,
                uint64_t *RngState,
                NVTETensor workspace,
                cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
