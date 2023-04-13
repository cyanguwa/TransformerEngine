/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"
#include <vector>
#include <ATen/Dispatch.h>
#include <string>
#include <ATen/native/DispatchStub.h>
#include <c10/macros/Macros.h>
#include "../common.h"

constexpr int block_size = 512;
constexpr int ctas_per_sm = 4;

template <typename scalar_t>
__global__ void __launch_bounds__(block_size) mha_fill_kernel(scalar_t* out_tensor,
                const int32_t* const start_row,
                const size_t num_rows) {
  size_t row_stride = gridDim.y * blockDim.x;
  size_t row_index = blockIdx.x + static_cast<size_t>(start_row[0]);
  size_t col_index = blockIdx.y * blockDim.x + threadIdx.x;
  while (row_index < num_rows) {
    out_tensor[row_index*row_stride + col_index] = 0;
    row_index += gridDim.x;
  }
}

void mha_fill(const at::Tensor &self, const at::Tensor &start_index) {
  auto max_tokens = self.size(0);
  auto self_2d = self.view({max_tokens, -1});
  auto fcd_size = self_2d.size(1);
  TORCH_CHECK(self.is_contiguous(), "input not contiguous");
  TORCH_CHECK(fcd_size % block_size == 0, "input size not aligned to block size");
  const int num_mp = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  uint64_t num_blk_y = (uint64_t)(fcd_size / block_size);
  uint64_t num_blk_x = (uint64_t)std::ceil(num_mp * ctas_per_sm / num_blk_y);
  dim3 dim_grid(num_blk_x, num_blk_y);
  dim3 dim_block(block_size);
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
                  at::ScalarType::Half,
                  at::ScalarType::BFloat16,
                  self_2d.scalar_type(),
                  "mha_fill", [&]() {
                  mha_fill_kernel<<<dim_grid, dim_block, 0,
                                  at::cuda::getCurrentCUDAStream()>>>(
                                  self_2d.data_ptr<scalar_t>(),
                                  static_cast<int32_t*>(start_index.data_ptr()),
                                  max_tokens);
                  C10_CUDA_KERNEL_LAUNCH_CHECK();
                  });
}

__global__ void unpack(at::PhiloxCudaState arg, int64_t* rng_state_ptr) {
  if (arg.captured_) {
    rng_state_ptr[0] = static_cast<int64_t>(*arg.seed_.ptr);
    rng_state_ptr[1] = static_cast<int64_t>(
                    *(arg.offset_.ptr) + static_cast<int64_t>(arg.offset_intragraph_));
  } else {
    rng_state_ptr[0] = static_cast<int64_t>(arg.seed_.val);
    rng_state_ptr[1] = static_cast<int64_t>(arg.offset_.val);
  }
}

at::PhiloxCudaState init_philox_state(
                at::CUDAGeneratorImpl* gen,
                size_t max_seq_len,
                size_t threads_per_cta) {
  at::PhiloxCudaState philox_args;
  size_t elts_per_thread = (max_seq_len * max_seq_len + threads_per_cta - 1)/threads_per_cta;
  std::lock_guard<std::mutex> lock(gen->mutex_);
  philox_args = gen->philox_cuda_state(elts_per_thread);
  return philox_args;
}

std::vector<at::Tensor> fused_attn_fwd_qkvpacked(
                size_t b, size_t max_seqlen, size_t total_seqs,
		size_t h, size_t d,
                bool is_training, float attn_scale, float p_dropout,
                bool set_zero, std::string qkv_layout, std::string attn_mask_type,
                const at::Tensor cu_seqlens,
                const at::Tensor QKV,
                const transformer_engine::DType qkv_type,
                const c10::optional<at::Tensor> descale_QKV,
                const c10::optional<at::Tensor> scale_S,
                const c10::optional<at::Tensor> scale_O,
                c10::optional<at::Tensor> amax_S,
                c10::optional<at::Tensor> amax_O,
                const c10::optional<at::Tensor> Bias,
                const c10::optional<std::string> bias_type,
                const c10::optional<at::Generator> rng_gen) {
  using namespace transformer_engine;

  // create output tensor O
  printf("----- qkv_type %d\n", (int)qkv_type);
  auto options = torch::TensorOptions().dtype(GetATenDType(qkv_type)).device(torch::kCUDA);
  auto O = torch::empty({static_cast<int64_t>(total_seqs),
		  static_cast<int64_t>(h), static_cast<int64_t>(d)}, options);
  if (set_zero) {
    mha_fill(O, cu_seqlens.index({torch::indexing::Slice(-1, torch::indexing::None)}));
  }

  printf("----- enter nvte fwd\n");
  // construct NVTE tensors
  TensorWrapper te_QKV, te_S, te_O, te_cu_seqlens;

  te_cu_seqlens = makeTransformerEngineTensor(cu_seqlens.data_ptr(), {b+1},
		    DType::kInt32, nullptr, nullptr, nullptr); 
  if (qkv_type == DType::kFloat8E4M3 || qkv_type == DType::kFloat8E5M2) {
    if ((!descale_QKV.has_value()) || (!scale_S.has_value()) || (!scale_O.has_value())
                    || (!amax_S.has_value()) || (!amax_O.has_value())) {
      std::string err_tensors = "descale_QKV, scale_S, scale_O, amax_S and amax_O";
      NVTE_ERROR(err_tensors + std::string("are required for FP8 operation. \n"));
    }
    te_QKV = makeTransformerEngineTensor(QKV.data_ptr(), {total_seqs, 3, h, d},
                    qkv_type, nullptr, nullptr, descale_QKV.value().data_ptr());
    at::Tensor descale_S = torch::empty_like(scale_S.value());  // ghost tensor for now
    te_S = makeTransformerEngineTensor(nullptr, {0},
		    DType::kFloat32, amax_S.value().data_ptr(),
		    scale_S.value().data_ptr(), descale_S.data_ptr());
    te_O = makeTransformerEngineTensor(O.data_ptr(), {total_seqs, h, d},
		    qkv_type, amax_O.value().data_ptr(), scale_O.value().data_ptr(), nullptr);
  } else if (qkv_type == DType::kBFloat16 || qkv_type == DType::kFloat16) {
    te_QKV = makeTransformerEngineTensor(QKV.data_ptr(), {total_seqs, 3, h, d},
                    qkv_type, nullptr, nullptr, nullptr);
    te_S = makeTransformerEngineTensor(nullptr, {0},
		    DType::kFloat32, nullptr, nullptr, nullptr);
    te_O = makeTransformerEngineTensor(O.data_ptr(), {total_seqs, h, d},
                    qkv_type, nullptr, nullptr, nullptr);
  } else {
    NVTE_ERROR("Fused attention only supports FP8 and BF16/FP16 data types. \n");
  }

  printf("----- after constructing te's \n");
  TensorWrapper te_Bias;
  if (Bias.has_value()) {
    auto bias_shape = Bias.value().sizes().vec();
    std::vector<size_t> shape{bias_shape.begin(), bias_shape.end()};
    te_Bias = makeTransformerEngineTensor(Bias.value().data_ptr(), shape,
                    DType::kFloat32, nullptr, nullptr, nullptr);
  }
//  if (bias_type.has_value()) {
//    std::string bias_type = bias_type.value().data();
//  }

  // convert strings to enums
  MHA_Layout qkv_layout_enum = get_mha_layout(qkv_layout);
  MHA_Bias_Type bias_type_enum = get_mha_bias_type(bias_type.value().data());
  Attn_Mask_Type attn_mask_type_enum = get_attn_mask_type(attn_mask_type);

  printf("----- after enums %d, %d, %d \n", (int)qkv_layout_enum, (int)bias_type_enum, (int)attn_mask_type_enum);
  // extract random number generator seed and offset
//  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
                  rng_gen, at::cuda::detail::getDefaultCUDAGenerator());
  size_t threads_per_cta = 128;
  at::PhiloxCudaState philox_args = init_philox_state(gen, max_seqlen, threads_per_cta);
  auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
  unpack<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
                  philox_args, static_cast<int64_t*>(rng_state.data_ptr()));
  auto te_rng_state = makeTransformerEngineTensor(rng_state);

  printf("----- after rng\n");
  // create auxilary output tensor pack
  // if training, this tensor pack includes [M, ZInv] but not rng_state
  NVTETensorPack nvte_aux_tensor_pack;
  nvte_tensor_pack_create(&nvte_aux_tensor_pack);

  // create workspace
  TensorWrapper workspace;

  printf("----- first call\n");
  // populate tensors with appropriate shapes and dtypes
  nvte_fused_attn_fwd_qkvpacked(
                  max_seqlen,
                  is_training, attn_scale, p_dropout,
		  qkv_layout_enum, bias_type_enum, attn_mask_type_enum,
		  te_cu_seqlens.data(),
		  te_rng_state.data(),
                  te_QKV.data(),
                  te_Bias.data(),
                  te_S.data(),
		  te_O.data(),
                  &nvte_aux_tensor_pack,
		  //reinterpret_cast<int32_t*>(cu_seqlens.data_ptr()),
                  //reinterpret_cast<uint64_t*>(rng_state.data_ptr()),
                  workspace.data(),
                  at::cuda::getCurrentCUDAStream());

  // allocate memory for workspace and output pack tensors
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace = makeTransformerEngineTensor(
                  workspace_data.data_ptr(),
                  workspace.shape(), workspace.dtype());

  printf("----- second call %ld \n", workspace.shape().data[0]);
  // aux_ctx_tensors includes nvte_aux_tensor_pack and rng_state
  // allocate memory for nvte_aux_tensor_pack first
  std::vector<at::Tensor> output_tensors;//(nvte_aux_tensor_pack.size + 2);
  output_tensors.push_back(O);
  for (size_t i = 0; i < nvte_aux_tensor_pack.size; ++i) {
    auto tensor = reinterpret_cast<Tensor*>(nvte_aux_tensor_pack.tensors[i]);
    //output_tensors[i+1] 
    auto output_tensor = allocateSpace(tensor->data.shape, tensor->data.dtype, false);
    output_tensors.push_back(output_tensor);
    //tensor->data.dptr = output_tensors[i+1].data_ptr();
    tensor->data.dptr = output_tensor.data_ptr();
  }

  // second call to actually execute the kernel
  nvte_fused_attn_fwd_qkvpacked(
                  max_seqlen,
                  is_training, attn_scale, p_dropout,
		  qkv_layout_enum, bias_type_enum, attn_mask_type_enum,
		  te_cu_seqlens.data(),
		  te_rng_state.data(),
                  te_QKV.data(),
                  te_Bias.data(),
                  te_S.data(),
		  te_O.data(),
                  &nvte_aux_tensor_pack,
		  //reinterpret_cast<int32_t*>(cu_seqlens.data_ptr()),
                  //reinterpret_cast<uint64_t*>(rng_state.data_ptr()),
                  workspace.data(),
                  at::cuda::getCurrentCUDAStream());

  // destroy tensor wrappers, but not the allocated memory
  nvte_tensor_pack_destroy(&nvte_aux_tensor_pack);

  // append rng_state to aux_ctx_tensors before return
  output_tensors.push_back(rng_state);

  printf("----- end of call\n");
  // [O, M, ZInv, rng_state] if training; [O, rng_state] if inference
  return output_tensors;
}

std::vector<at::Tensor> fused_attn_bwd_qkvpacked(
                size_t b, size_t max_seqlen, size_t total_seqs,
		size_t h, size_t d,
                float attn_scale, float p_dropout,
                bool set_zero, std::string qkv_layout, std::string attn_mask_type,
                const at::Tensor cu_seqlens,
                const at::Tensor QKV,
                const at::Tensor O,
                const at::Tensor dO,
                const transformer_engine::DType qkv_type,
                const std::vector<at::Tensor> Aux_CTX_Tensors,
                const c10::optional<at::Tensor> descale_QKV,
                const c10::optional<at::Tensor> descale_S,
                const c10::optional<at::Tensor> descale_O,
                const c10::optional<at::Tensor> descale_dO,
                const c10::optional<at::Tensor> scale_S,
                const c10::optional<at::Tensor> scale_dS,
                const c10::optional<at::Tensor> scale_dQKV,
                c10::optional<at::Tensor> amax_dS,
                c10::optional<at::Tensor> amax_dQKV,
                const c10::optional<at::Tensor> Bias,
                const c10::optional<std::string> bias_type) {
  using namespace transformer_engine;

  // create output tensor dQKV
  at::Tensor dQKV = torch::empty_like(QKV);
  if (set_zero) {
    mha_fill(dQKV, cu_seqlens.index({torch::indexing::Slice(-1, torch::indexing::None)}));
  }

  // construct NVTE tensors
  TensorWrapper te_QKV, te_O, te_dO, te_S, te_dS, te_dQKV;

  if (qkv_type == DType::kFloat8E4M3 || qkv_type == DType::kFloat8E5M2) {
    if ((!descale_QKV.has_value()) || (!descale_S.has_value())
                    || (!descale_O.has_value()) || (!descale_dO.has_value())
                    || (!scale_S.has_value()) || (!scale_dS.has_value())
                    || (!scale_dQKV.has_value())
                    || (!amax_dS.has_value()) || (!amax_dQKV.has_value())) {
      std::string err_tensors = "descale_QKV, descale_S, descale_O, scale_S, scale_dS, ";
      err_tensors = err_tensors + std::string("scale_dQKV, amax_dS and amax_dQKV");
      NVTE_ERROR(err_tensors + std::string("are required for FP8 operation. \n"));
    }
    te_QKV = makeTransformerEngineTensor(QKV.data_ptr(), {total_seqs, 3, h, d},
                    qkv_type, nullptr, nullptr, descale_QKV.value().data_ptr());
    te_O = makeTransformerEngineTensor(O.data_ptr(), {total_seqs, h, d},
                    qkv_type, nullptr, nullptr, descale_O.value().data_ptr());
    te_dO = makeTransformerEngineTensor(dO.data_ptr(), {total_seqs, h, d},
                    qkv_type, nullptr, nullptr, descale_dO.value().data_ptr());
    te_S = makeTransformerEngineTensor(nullptr, {0},
                    DType::kFloat32, nullptr, scale_S.value().data_ptr(), descale_S.value().data_ptr());
    at::Tensor descale_dS = torch::empty_like(scale_dS.value());  // ghost tensor for now
    te_dS = makeTransformerEngineTensor(nullptr, {0},
                    DType::kFloat32, amax_dS.value().data_ptr(), scale_dS.value().data_ptr(),
                    descale_dS.data_ptr());
    te_dQKV = makeTransformerEngineTensor(dQKV.data_ptr(), {total_seqs, 3, h, d},
                    qkv_type,
		    amax_dQKV.value().data_ptr(), scale_dQKV.value().data_ptr(), nullptr);
  } else if (qkv_type == DType::kBFloat16 || qkv_type == DType::kFloat16) {
    te_QKV = makeTransformerEngineTensor(QKV.data_ptr(), {total_seqs, 3, h, d},
                    qkv_type, nullptr, nullptr, nullptr);
    // do we need any softmax tensor for BF16 API?
    te_O = makeTransformerEngineTensor(O.data_ptr(), {total_seqs, h, d},
                    qkv_type, nullptr, nullptr, nullptr);
    te_dO = makeTransformerEngineTensor(dO.data_ptr(), {total_seqs, h, d},
                    qkv_type, nullptr, nullptr, nullptr);
    te_S = makeTransformerEngineTensor(nullptr, {0},
                    DType::kFloat32, nullptr, nullptr, nullptr);
    te_dS = makeTransformerEngineTensor(nullptr, {0},
                    DType::kFloat32, nullptr, nullptr, nullptr);
    te_dQKV = makeTransformerEngineTensor(dQKV.data_ptr(), {total_seqs, 3, h, d},
                    qkv_type, nullptr, nullptr, nullptr);
  } else {
    NVTE_ERROR("Fused attention only supports FP8 and BF16/FP16 data types. \n");
  }

  TensorWrapper te_Bias;
  if (Bias.has_value()) {
    auto bias_shape = Bias.value().sizes().vec();
    std::vector<size_t> shape{bias_shape.begin(), bias_shape.end()};
    te_Bias = makeTransformerEngineTensor(
                    Bias.value().data_ptr(), shape, DType::kFloat32,
                    nullptr, nullptr, nullptr);
  }
//  if (bias_type.has_value()) {
//    std::string bias_type = bias_type.value().data();
//  }
  // convert strings to enums
  MHA_Layout qkv_layout_enum = get_mha_layout(qkv_layout);
  MHA_Bias_Type bias_type_enum = get_mha_bias_type(bias_type.value().data());
  Attn_Mask_Type attn_mask_type_enum = get_attn_mask_type(attn_mask_type);

  // convert auxilary tensor pack from fwd/ctx to NVTETensorPack
  // if training, tensor pack contains [M, ZInv, rng_state]
  NVTETensorPack nvte_aux_tensor_pack;
  nvte_tensor_pack_create(&nvte_aux_tensor_pack);
  nvte_aux_tensor_pack.size = Aux_CTX_Tensors.size();
  for (size_t i = 0; i < nvte_aux_tensor_pack.size; ++i) {
    auto tensor = reinterpret_cast<Tensor*>(nvte_aux_tensor_pack.tensors[i]);
    tensor->data.dptr = Aux_CTX_Tensors[i].data_ptr();
    std::vector<int64_t> tmp(Aux_CTX_Tensors[i].sizes().vec());
    tensor->data.shape = std::vector<size_t>(tmp.begin(), tmp.end());
    tensor->data.dtype = GetTransformerEngineDType(Aux_CTX_Tensors[i].scalar_type());
  }

  // create tensors for cu_seqlens
  TensorWrapper te_cu_seqlens;
  te_cu_seqlens = makeTransformerEngineTensor(cu_seqlens.data_ptr(), {b+1},
		    DType::kInt32, nullptr, nullptr, nullptr); 

  // create workspace
  TensorWrapper workspace;

  // populate tensors with appropriate shapes and dtypes
  nvte_fused_attn_bwd_qkvpacked(
                  max_seqlen,
                  attn_scale, p_dropout,
		  qkv_layout_enum, bias_type_enum, attn_mask_type_enum,
		  te_cu_seqlens.data(),
                  te_QKV.data(),
                  te_Bias.data(),
                  te_O.data(),
                  te_dO.data(),
                  te_S.data(),
                  te_dS.data(),
		  &nvte_aux_tensor_pack,
		  te_dQKV.data(),
                  workspace.data(),
                  at::cuda::getCurrentCUDAStream());

  // allocate memory for workspace
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace = makeTransformerEngineTensor(
                  workspace_data.data_ptr(),
                  workspace.shape(), workspace.dtype());

  // second call to execute the kernel
  nvte_fused_attn_bwd_qkvpacked(
                  max_seqlen,
                  attn_scale, p_dropout,
		  qkv_layout_enum, bias_type_enum, attn_mask_type_enum,
		  te_cu_seqlens.data(),
                  te_QKV.data(),
                  te_Bias.data(),
                  te_O.data(),
                  te_dO.data(),
                  te_S.data(),
                  te_dS.data(),
		  &nvte_aux_tensor_pack,
		  te_dQKV.data(),
                  workspace.data(),
                  at::cuda::getCurrentCUDAStream());

  // destroy the tensor wrappers
  nvte_tensor_pack_destroy(&nvte_aux_tensor_pack);

  return {dQKV};
}
//std::vector<at::Tensor> fused_attn_fwd_kvpacked(
//                size_t b, size_t max_seqlen_q, size_t max_seqlen_kv,
//                size_t total_seqs_q, size_t total_seqs_kv,
//		size_t h, size_t d,
//                bool is_training, float attn_scale, float p_dropout,
//                bool set_zero, std::string qkv_layout, std::string attn_mask_type,
//                const at::Tensor cu_seqlens_q,
//		const at::Tensor cu_seqlens_kv,
//                const at::Tensor Q,
//		const at::Tensor KV,
//                const transformer_engine::DType qkv_type,
//                const c10::optional<at::Tensor> descale_QKV,
//                const c10::optional<at::Tensor> scale_S,
//                const c10::optional<at::Tensor> scale_O,
//                c10::optional<at::Tensor> amax_S,
//                c10::optional<at::Tensor> amax_O,
//                const c10::optional<at::Tensor> Bias,
//                const c10::optional<std::string> bias_type,
//                const c10::optional<at::Generator> rng_gen) {
//  using namespace transformer_engine;
//
//  // create output tensor O
//  auto options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
//  auto O = torch::empty({total_seqs_q, h, d}, options);
//  if (set_zero) {
//    mha_fill(O, cu_seqlens_q.index({torch::indexing::Slice(-1, torch::indexing::None)}));
//  }
//
//  // construct NVTE tensors
//  TensorWrapper te_Q, te_KV, te_S, te_O, te_cu_seqlens_q, te_cu_seqlens_kv;
//
//  te_cu_seqlens_q = makeTransformerEngineTensor(cu_seqlens_q, {b+1},
//		    DType::kInt32, nullptr, nullptr, nullptr); 
//  te_cu_seqlens_kv = makeTransformerEngineTensor(cu_seqlens_kv, {b+1},
//		    DType::kInt32, nullptr, nullptr, nullptr); 
//  if (qkv_type == DType::kFloat8E4M3 || qkv_type == DType::kFloat8E5M2) {
//    if ((!descale_QKV.has_value()) || (!scale_S.has_value()) || (!scale_O.has_value())
//                    || (!amax_S.has_value()) || (!amax_O.has_value())) {
//      std::string err_tensors = "descale_QKV, scale_S, scale_O, amax_S and amax_O";
//      NVTE_ERROR(err_tensors + std::string("are required for FP8 operation. \n"));
//    }
//    te_Q = makeTransformerEngineTensor(Q.data_ptr(), {total_seqs_q, h, d},
//                    qkv_type, nullptr, nullptr, descale_QKV.value().data_ptr());
//    te_KV = makeTransformerEngineTensor(KV.data_ptr(), {total_seqs_kv, 2, h, d},
//                    qkv_type, nullptr, nullptr, descale_QKV.value().data_ptr());
//    at::Tensor descale_S = torch::empty_like(scale_S.value());  // ghost tensor for now
//    te_S = makeTransformerEngineTensor(nullptr, {0},
//		    DType::kFloat32, amax_S.value().data_ptr(),
//		    scale_S.value().data_ptr(), descale_S.data_ptr());
//    te_O = makeTransformerEngineTensor(O.data_ptr(), {total_seqs_q, h, d},
//		    qkv_type, amax_O.value().data_ptr(), scale_O.value().data_ptr(), nullptr);
//  } else if (qkv_type == DType::kBFloat16 || qkv_type == DType::kFloat16) {
//    te_Q = makeTransformerEngineTensor(Q.data_ptr(), {total_seqs_q, h, d},
//                    qkv_type, nullptr, nullptr, nullptr);
//    te_KV = makeTransformerEngineTensor(KV.data_ptr(), {total_seqs_kv, 2, h, d},
//                    qkv_type, nullptr, nullptr, nullptr);
//    te_S = makeTransformerEngineTensor(nullptr, {0},
//		    DType::kFloat32, nullptr, nullptr, nullptr);
//    te_O = makeTransformerEngineTensor(O.data_ptr(), {total_seqs_q, h, d},
//                    qkv_type, nullptr, nullptr, nullptr);
//  } else {
//    NVTE_ERROR("Fused attention only supports FP8 and BF16/FP16 data types. \n");
//  }
//
//  TensorWrapper te_Bias;
//  if (Bias.has_value()) {
//    auto bias_shape = Bias.value().sizes().vec();
//    std::vector<size_t> shape{bias_shape.begin(), bias_shape.end()};
//    te_Bias = makeTransformerEngineTensor(Bias.value().data_ptr(), shape,
//                    DType::kFloat32, nullptr, nullptr, nullptr);
//  }
//// convert to enum bias type
////  if (bias_type.has_value()) {
////    std::string bias_type = bias_type.value().data();
////  }
//// convert to enum mha layout type
//// attn_mask_type enum?
//
//  //// if training, create auxilary output tensor pack [M, ZInv]
//  //// if inference, no auxilary output pack, i.e. None
//  //if (is_training) {
//  NVTETensorPack nvte_aux_tensor_pack;
//  nvte_tensor_pack_create(&nvte_aux_tensor_pack);
//  //} else {
//  //  auto nvte_output_pack = torch::indexing::None;
//  //}
////  // set properties for O since O is always an output
////  // and some of its properties are known at this stage
////  nvte_output_pack.size = 1;
////  Tensor *output_O = reinterpret_cast<Tensor*>(nvte_output_pack.tensors[0]);
////  output_O->data.dptr = nullptr;
////  output_O->data.shape = {total_seqs_q, h, d};
////  output_O->data.dtype = qkv_type;
////  output_O->amax.dptr = nullptr;
////  output_O->scale.dptr = nullptr;
////  output_O->scale_inv.dptr = nullptr;
////  if (qkv_type == DType::kFloat8E4M3 || qkv_type == DType::kFloat8E5M2) {
////    output_O->amax.dptr = amax_O.value().data_ptr();
////    output_O->scale.dptr = scale_O.value().data_ptr();
////  }
//
//  // extract rng seed and offset
//  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
//  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
//                  rng_gen, at::cuda::detail::getDefaultCUDAGenerator());
//  size_t threads_per_cta = 128;
//  at::PhiloxCudaState philox_args = init_philox_state(gen, max_seq_len, threads_per_cta);
//  auto rng_state = torch::empty({2}, options);
//  unpack<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
//                  philox_args, static_cast<int64_t*>(rng_state.data_ptr()));
//  auto te_rng_state = makeTransformerEngineTensor(rng_state);
//
//  // create workspace
//  TensorWrapper workspace;
//
//  // populate tensors with appropriate shapes and dtypes
//  nvte_fused_attn_fwd_kvpacked(
//                  max_seqlen_q, max_seqlen_kv,
//		  //total_seqs_q, total_seqs_kv,
//                  is_training, attn_scale, p_dropout,
//		  qkv_layout, bias_type, attn_mask_type,
//		  te_cu_seqlens_q.data(),
//		  te_cu_seqlens_kv.data(),
//		  te_rng_state.data(),
//                  te_Q.data(),
//                  te_KV.data(),
//                  te_Bias.data(),
//                  te_S.data(),
//		  te_O.data(),
//                  &nvte_aux_tensor_pack,
//                  //reinterpret_cast<int32_t*>(cu_seqlens_q.data_ptr()),
//                  //reinterpret_cast<int32_t*>(cu_seqlens_kv.data_ptr()),
//                  //reinterpret_cast<uint64_t*>(rng_state.data_ptr()),
//                  workspace.data(),
//                  at::cuda::getCurrentCUDAStream());
//
//  // allocate memory for workspace and output_pack tensors
//  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
//  workspace = makeTransformerEngineTensor(
//                  workspace_data.data_ptr(),
//                  workspace.shape(), workspace.dtype());
//
//  std::vector<at::Tensor> aux_ctx_tensors(nvte_aux_tensor_pack.size);
//  for (size_t i = 0; i < nvte_aux_tensor_pack.size; ++i) {
//    auto tensor = reinterpret_cast<Tensor*>(nvte_aux_tensor_pack.tensors[i]);
//    aux_ctx_tensors[i] = allocateSpace(tensor->data.shape, tensor->data.dtype, false);
//    tensor->data.dptr = aux_ctx_tensors[i].data_ptr();
//  }
//
//  // actual call to kernel
//  nvte_fused_attn_fwd_kvpacked(
//                  max_seqlen_q, max_seqlen_kv,
//		  //total_seqs_q, total_seqs_kv,
//                  is_training, attn_scale, p_dropout,
//		  qkv_layout, bias_type, attn_mask_type,
//		  te_cu_seqlens_q.data(),
//		  te_cu_seqlens_kv.data(),
//		  te_rng_state.data(),
//                  te_Q.data(),
//                  te_KV.data(),
//                  te_Bias.data(),
//                  te_S.data(),
//		  te_O.data(),
//                  &nvte_aux_tensor_pack,
//                  //reinterpret_cast<int32_t*>(cu_seqlens_q.data_ptr()),
//                  //reinterpret_cast<int32_t*>(cu_seqlens_kv.data_ptr()),
//                  //reinterpret_cast<uint64_t*>(rng_state.data_ptr()),
//                  workspace.data(),
//                  at::cuda::getCurrentCUDAStream());
//
//  nvte_tensor_pack_destroy(&nvte_aux_tensor_pack);
//
//  // add rng_state to aux_ctx_tensors
//  aux_ctx_tensors.push_back(rng_state);
//
//  return aux_ctx_tensors;
//}
//
//std::vector<at::Tensor> fused_attn_bwd_kvpacked(
//                size_t b, size_t max_seqlen_q, size_t max_seqlen_kv,
//                size_t total_seqs_q, size_t total_seqs_kv,
//		size_t h, size_t d,
//                float attn_scale, float p_dropout,
//                bool set_zero, std::string qkv_layout, std::string attn_mask_type,
//                const at::Tensor cu_seqlens_q,
//                const at::Tensor cu_seqlens_kv,
//                const at::Tensor Q,
//                const at::Tensor KV,
//                const at::Tensor O,
//                const at::Tensor dO,
//                const transformer_engine::DType qkv_type,
//                const std::vector<at::Tensor> Aux_CTX_Tensors,
//                const c10::optional<at::Tensor> descale_QKV,
//                const c10::optional<at::Tensor> descale_S,
//                const c10::optional<at::Tensor> descale_O,
//                const c10::optional<at::Tensor> descale_dO,
//                const c10::optional<at::Tensor> scale_S,
//                const c10::optional<at::Tensor> scale_dS,
//                const c10::optional<at::Tensor> scale_dQKV,
//                c10::optional<at::Tensor> amax_dS,
//                c10::optional<at::Tensor> amax_dQKV,
//                const c10::optional<at::Tensor> Bias,
//                const c10::optional<std::string> bias_type) {
//  using namespace transformer_engine;
//
//  // create output tensor dQ, dKV
//  at::Tensor dQ = torch::empty_like(Q);
//  at::Tensor dKV = torch::empty_like(KV);
//  if (set_zero) {
//    mha_fill(dQ, cu_seqlens_q.index({torch::indexing::Slice(-1, torch::indexing::None)}));
//    mha_fill(dKV, cu_seqlens_kv.index({torch::indexing::Slice(-1, torch::indexing::None)}));
//  }
//
//  // construct NVTE tensors
//  TensorWrapper te_Q, te_KV, te_O, te_dO, te_S, te_dS, te_dQ, te_dKV;//, te_M, te_ZInv;
//
//  if (qkv_type == DType::kFloat8E4M3 || qkv_type == DType::kFloat8E5M2) {
//    if ((!descale_QKV.has_value()) || (!descale_S.has_value())
//                    || (!descale_O.has_value()) || (!descale_dO.has_value())
//                    || (!scale_S.has_value()) || (!scale_dS.has_value())
//                    || (!scale_dQKV.has_value())
//                    || (!amax_dS.has_value()) || (!amax_dQKV.has_value())) {
//      std::string err_tensors = "descale_QKV, descale_S, descale_O, scale_S, scale_dS, ";
//      err_tensors = err_tensors + std::string("scale_dQKV, amax_dS and amax_dQKV");
//      NVTE_ERROR(err_tensors + std::string("are required for FP8 operation. \n"));
//    }
//    te_Q = makeTransformerEngineTensor(Q.data_ptr(), {total_seqs_q, h, d},
//                    qkv_type, nullptr, nullptr, descale_QKV.value().data_ptr());
//    te_KV = makeTransformerEngineTensor(KV.data_ptr(), {total_seqs_kv, 2, h, d},
//                    qkv_type, nullptr, nullptr, descale_QKV.value().data_ptr());
//
//    // use output pack instead?
//    //if (!M.has_value() || !ZInv.has_value()) {
//    //  NVTE_ERROR("M and ZInv tensors are required for FP8 fused attention backward. \n");
//    //}
//    //te_M = makeTransformerEngineTensor(M.value());
//    //te_ZInv = makeTransformerEngineTensor(ZInv.value());
//    auto te_M = makeTransformerEngineTensor(Aux_CTX_Tensors[0].data_ptr());
//    auto te_ZInv = makeTransformerEngineTensor(Aux_CTX_Tensors[1].data_ptr());
//    te_O = makeTransformerEngineTensor(O.data_ptr(), {total_seqs, h, d},
//                    qkv_type, nullptr, nullptr, descale_O.value().data_ptr());
//    te_dO = makeTransformerEngineTensor(dO.data_ptr(), {total_seqs, h, d},
//                    qkv_type, nullptr, nullptr, descale_dO.value().data_ptr());
//    te_S = makeTransformerEngineTensor(nullptr, {0},
//                    qkv_type, nullptr, scale_S.value().data_ptr(), descale_S.value().data_ptr());
//    at::Tensor descale_dS = torch::empty_like(scale_dS.value());  // ghost tensor for now
//    te_dS = makeTransformerEngineTensor(nullptr, {0},
//                    qkv_type, amax_dS.value().data_ptr(), scale_dS.value().data_ptr(),
//                    descale_dS.data_ptr());
//    te_dQ = makeTransformerEngineTensor(dQ.data_ptr(), {total_seqs_q, h, d},
//                    qkv_type,
//		    amax_dQKV.value().data_ptr(), scale_dQKV.value().data_ptr(), nullptr);
//    te_dKV = makeTransformerEngineTensor(dKV.data_ptr(), {total_seqs_kv, 2, h, d},
//                    qkv_type,
//		    amax_dQKV.value().data_ptr(), scale_dQKV.value().data_ptr(), nullptr);
//  } else if (qkv_type == DType::kBFloat16 || qkv_type == DType::kFloat16) {
//    te_Q = makeTransformerEngineTensor(Q.data_ptr(), {total_seqs_q, h, d},
//                    qkv_type, nullptr, nullptr, nullptr);
//    te_KV = makeTransformerEngineTensor(KV.data_ptr(), {total_seqs_kv, 2, h, d},
//                    qkv_type, nullptr, nullptr, nullptr);
//    // do we need any softmax tensor for BF16 API?
//    te_O = makeTransformerEngineTensor(O.data_ptr(), {total_seqs, h, d},
//                    qkv_type, nullptr, nullptr, nullptr);
//    te_dO = makeTransformerEngineTensor(dO.data_ptr(), {total_seqs, h, d},
//                    qkv_type, nullptr, nullptr, nullptr);
//    te_S = makeTransformerEngineTensor(nullptr, {0},
//                    qkv_type, nullptr, nullptr, nullptr);
//    te_dS = makeTransformerEngineTensor(nullptr, {0},
//                    qkv_type, nullptr, nullptr, nullptr);
//    te_dQ = makeTransformerEngineTensor(dQ.data_ptr(), {total_seqs_q, h, d},
//                    qkv_type, nullptr, nullptr, nullptr);
//    te_dKV = makeTransformerEngineTensor(dKV.data_ptr(), {total_seqs_kv, 2, h, d},
//                    qkv_type, nullptr, nullptr, nullptr);
//  } else {
//    NVTE_ERROR("Fused attention only supports FP8 and BF16/FP16 data types. \n");
//  }
//
//  TensorWrapper te_Bias;
//  if (Bias.has_value()) {
//    auto bias_shape = Bias.value().sizes().vec();
//    std::vector<size_t> shape{bias_shape.begin(), bias_shape.end()};
//    te_Bias = makeTransformerEngineTensor(
//                    Bias.value().data_ptr(), shape, DType::kFloat32,
//                    nullptr, nullptr, nullptr);
//  }
////  if (bias_type.has_value()) {
////    std::string bias_type = bias_type.value().data();
////  }
//// converts?
//
//  // create output tensor pack
//  // tensor elements are [dQ, dKV]
//  NVTETensorPack nvte_aux_ctx_pack;
//  nvte_tensor_pack_create(&nvte_aux_ctx_pack);
//  nvte_aux_ctx_pack.size = Aux_CTX_Tensors.size();
//  for (size_t i = 0; i < nvte_aux_ctx_pack.size; ++i) {
//    auto tensor = reinterpret_cast<Tensor*>(nvte_output_pack.tensors[i]);
//    tensor->data.dptr = Aux_CTX_Tensors[i].data_ptr();
//    tensor->data.shape = Aux_CTX_Tensors[i].sizes().data_ptr();
//    tensor->data.dtype = GetTransformerEngineDType(Aux_CTX_Tensors[i].scalar_type());
//  }
//  //// set properties for O since O is always an output
//  //// and some of its properties are known at this stage
//  //nvte_output_pack.size = 2;
//  //Tensor *output_dQ = reinterpret_cast<Tensor*>(nvte_output_pack.tensors[0]);
//  //Tensor *output_dKV = reinterpret_cast<Tensor*>(nvte_output_pack.tensors[1]);
//  //output_dQ->data.dptr = nullptr;
//  //output_dQ->data.shape = {total_seqs_q, h, d};
//  //output_dQ->data.dtype = qkv_type;
//  //output_dQ->amax.dptr = nullptr;
//  //output_dQ->scale.dptr = nullptr;
//  //output_dQ->scale_inv.dptr = nullptr;
//  //output_dKV->data.dptr = nullptr;
//  //output_dKV->data.shape = {total_seqs_kv, 2, h, d};
//  //output_dKV->data.dtype = qkv_type;
//  //output_dKV->amax.dptr = nullptr;
//  //output_dKV->scale.dptr = nullptr;
//  //output_dKV->scale_inv.dptr = nullptr;
//  //if (qkv_type == DType::kFloat8E4M3 || qkv_type == DType::kFloat8E5M2) {
//  //  output_dQ->amax.dptr = amax_dQKV.value().data_ptr();
//  //  output_dQ->scale.dptr = scale_dQKV.value().data_ptr();
//  //  output_dKV->amax.dptr = amax_dQKV.value().data_ptr();
//  //  output_dKV->scale.dptr = scale_dQKV.value().data_ptr();
//  //}
//
//  TensorWrapper te_cu_seqlens_q, te_cu_seqlens_kv;
//
//  te_cu_seqlens_q = makeTransformerEngineTensor(cu_seqlens_q, {b+1},
//		    DType::kInt32, nullptr, nullptr, nullptr); 
//  te_cu_seqlens_kv = makeTransformerEngineTensor(cu_seqlens_kv, {b+1},
//		    DType::kInt32, nullptr, nullptr, nullptr); 
//
//  // create workspace
//  TensorWrapper workspace;
//
//  // populate tensors with appropriate shapes and dtypes
//  nvte_fused_attn_bwd_kvpacked(
//                  max_seqlen_q, max_seqlen_kv,
//		  //total_seqs_q, total_seqs_kv, 
//                  attn_scale, p_dropout,
//		  qkv_layout, bias_type, attn_mask_type,
//		  te_cu_seqlens_q.data(),
//		  te_cu_seqlens_kv.data(),
//                  te_Q.data(),
//                  te_KV.data(),
//                  te_Bias.data(),
//                  te_O.data(),
//                  te_dO.data(),
//                  te_S.data(),
//                  te_dS.data(),
//		  &nvte_aux_ctx_pack,
//		  te_dQ.data(),
//		  te_dKV.data(),
//                  //reinterpret_cast<int32_t*>(cu_seqlens.data_ptr()),
//                  //reinterpret_cast<uint64_t*>(rng_state.data_ptr()),
//                  workspace.data(),
//                  at::cuda::getCurrentCUDAStream());
//
//  // allocate memory for workspace and output_pack tensors
//  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
//  workspace = makeTransformerEngineTensor(
//                  workspace_data.data_ptr(),
//                  workspace.shape(), workspace.dtype());
//
//  //std::vector<at::Tensor> output_tensors(nvte_output_pack.size);
//  //for (size_t i = 0; i < nvte_output_pack.size; ++i) {
//  //  auto tensor = reinterpret_cast<Tensor*>(nvte_output_pack.tensors[i]);
//  //  output_tensors[i] = allocateSpace(tensor->data.shape, tensor->data.dtype, false);
//  //  tensor->data.dptr = output_tensors[i].data_ptr();
//  //  if ((i == 0) && (set_zero)) {
//  //    mha_fill(output_tensors[i],
//  //                    cu_seqlens.index({torch::indexing::Slice(-1, torch::indexing::None)}));
//  //  }
//  //}
//
//  // actual call to kernel
//  nvte_fused_attn_bwd(
//                  max_seqlen_q, max_seqlen_kv,
//		  //total_seqs_q, total_seqs_kv, 
//                  attn_scale, p_dropout,
//		  qkv_layout, bias_type, attn_mask_type,
//		  te_cu_seqlens_q.data(),
//		  te_cu_seqlens_kv.data(),
//                  te_Q.data(),
//                  te_KV.data(),
//                  te_Bias.data(),
//                  te_O.data(),
//                  te_dO.data(),
//                  te_S.data(),
//                  te_dS.data(),
//		  &nvte_aux_ctx_pack,
//		  te_dQ.data(),
//		  te_dKV.data(),
//                  //reinterpret_cast<int32_t*>(cu_seqlens.data_ptr()),
//                  //reinterpret_cast<uint64_t*>(rng_state.data_ptr()),
//                  workspace.data(),
//                  at::cuda::getCurrentCUDAStream());
//
//  nvte_tensor_pack_destroy(&nvte_aux_ctx_pack);
//
//  return {dQ, dKV}
//}

void te_gemm(at::Tensor A,
             at::Tensor A_scale_inverse,
             transformer_engine::DType A_type,
             bool transa,
             at::Tensor B,
             at::Tensor B_scale_inverse,
             transformer_engine::DType B_type,
             bool transb,
             at::Tensor D,
             at::Tensor D_scale,
             transformer_engine::DType D_type,
             at::Tensor D_amax,
             at::Tensor bias,
             transformer_engine::DType bias_type,
             at::Tensor pre_gelu_out,
             bool grad,
             at::Tensor workspace,
             size_t workspaceSize,
             bool accumulate,
             bool use_split_accumulator
) {
  using namespace transformer_engine;
  auto te_A = makeTransformerEngineTensor(A.data_ptr(),
                                          {static_cast<size_t>(A.size(0)),
                                           static_cast<size_t>(A.size(1))},
                                          A_type, nullptr, nullptr,
                                          A_scale_inverse.data_ptr());
  auto te_B = makeTransformerEngineTensor(B.data_ptr(),
                                          {static_cast<size_t>(B.size(0)),
                                           static_cast<size_t>(B.size(1))},
                                          B_type, nullptr, nullptr,
                                          B_scale_inverse.data_ptr());
  auto te_D = makeTransformerEngineTensor(D.data_ptr(),
                                          {static_cast<size_t>(D.size(0)),
                                           static_cast<size_t>(D.size(1))},
                                          D_type, D_amax.data_ptr(),
                                          D_scale.data_ptr(), nullptr);
  auto te_bias = makeTransformerEngineTensor(bias.data_ptr(), {static_cast<size_t>(bias.size(0))},
                                             bias_type);

  const auto gelu_shape = pre_gelu_out.data_ptr() == nullptr
                          ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                          : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                static_cast<size_t>(pre_gelu_out.size(1))};
  auto te_pre_gelu_out = makeTransformerEngineTensor(pre_gelu_out.data_ptr(),
                                                     gelu_shape,
                                                     GetTransformerEngineDType(
                                                         pre_gelu_out.scalar_type()));
  auto te_workspace = makeTransformerEngineTensor(workspace.data_ptr(),
                                                  {workspaceSize},
                                                  DType::kByte);

  nvte_cublas_gemm(te_A.data(),
                   te_B.data(),
                   te_D.data(),
                   te_bias.data(),
                   te_pre_gelu_out.data(),
                   transa,
                   transb,
                   grad,
                   te_workspace.data(),
                   accumulate,
                   use_split_accumulator,
                   at::cuda::getCurrentCUDAStream());
}


void fused_cast_transpose(at::Tensor input,
                          at::Tensor scale,
                          at::Tensor amax,
                          at::Tensor scale_inv,
                          at::Tensor input_cast,
                          at::Tensor input_transpose,
                          transformer_engine::DType otype
) {
  using namespace transformer_engine;

  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  auto input_cu            = makeTransformerEngineTensor(input);
  auto output_cast_cu      = makeTransformerEngineTensor(input_cast.data_ptr(), {M, N}, otype,
                                                         amax.data_ptr(), scale.data_ptr(),
                                                         scale_inv.data_ptr());
  auto output_transpose_cu = makeTransformerEngineTensor(input_transpose.data_ptr(), {N, M}, otype,
                                                         amax.data_ptr(), scale.data_ptr(),
                                                         scale_inv.data_ptr());

  nvte_cast_transpose(input_cu.data(), output_cast_cu.data(), output_transpose_cu.data(),
                      at::cuda::getCurrentCUDAStream());
}


std::vector<at::Tensor> fused_cast_transpose_bgrad(at::Tensor grad_output,
                                                   at::Tensor scale,
                                                   at::Tensor amax,
                                                   at::Tensor scale_inv,
                                                   transformer_engine::DType otype
) {
  using namespace transformer_engine;

  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto grad_output_cast =
            allocateTorchTensor(grad_output.size(0),
                                grad_output.size(1),
                                DType::kByte);
  auto grad_output_transpose =
            allocateTorchTensor(grad_output.size(1),
                                grad_output.size(0),
                                DType::kByte);

  auto input_cu             = makeTransformerEngineTensor(grad_output);
  auto cast_output_cu       = makeTransformerEngineTensor(grad_output_cast.data_ptr(), {M, N},
                                                          otype, amax.data_ptr(), scale.data_ptr(),
                                                          scale_inv.data_ptr());
  auto transposed_output_cu = makeTransformerEngineTensor(grad_output_transpose.data_ptr(),
                                                          {N, M}, otype, amax.data_ptr(),
                                                          scale.data_ptr(), scale_inv.data_ptr());
  auto dbias_cu             = makeTransformerEngineTensor(grad_bias);
  transformer_engine::TensorWrapper workspace;

  nvte_cast_transpose_dbias(input_cu.data(), cast_output_cu.data(),
                            transposed_output_cu.data(), dbias_cu.data(),
                            workspace.data(), at::cuda::getCurrentCUDAStream());

  // Fill workspace
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace = makeTransformerEngineTensor(workspace_data.data_ptr(),
                                          workspace.shape(),
                                          workspace.dtype());

  nvte_cast_transpose_dbias(input_cu.data(), cast_output_cu.data(),
                            transposed_output_cu.data(), dbias_cu.data(),
                            workspace.data(), at::cuda::getCurrentCUDAStream());

  return {grad_bias, grad_output_cast, grad_output_transpose};
}


std::vector<at::Tensor> fused_fp8_transpose_bgrad(at::Tensor grad_output,
                                                   at::Tensor scale,
                                                   at::Tensor amax,
                                                   at::Tensor scale_inv,
                                                   transformer_engine::DType otype,
                                                   transformer_engine::DType grad_bias_type
) {
  using namespace transformer_engine;

  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_bias_type);
  auto grad_output_transpose =
            allocateTorchTensor(grad_output.size(1),
                                grad_output.size(0),
                                DType::kByte);
  auto input_cu             = makeTransformerEngineTensor(grad_output.data_ptr(), {M, N},
                                                         otype, amax.data_ptr(), scale.data_ptr(),
                                                         scale_inv.data_ptr());
  auto transposed_output_cu = makeTransformerEngineTensor(grad_output_transpose.data_ptr(),
                                                          {N, M}, otype, amax.data_ptr(),
                                                          scale.data_ptr(), scale_inv.data_ptr());
  auto dbias_cu             = makeTransformerEngineTensor(grad_bias);
  transformer_engine::TensorWrapper workspace;

  nvte_fp8_transpose_dbias(input_cu.data(), transposed_output_cu.data(), dbias_cu.data(),
                            workspace.data(), at::cuda::getCurrentCUDAStream());

  // Fill workspace
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace = makeTransformerEngineTensor(workspace_data.data_ptr(),
                                          workspace.shape(),
                                          workspace.dtype());

  nvte_fp8_transpose_dbias(input_cu.data(), transposed_output_cu.data(), dbias_cu.data(),
                            workspace.data(), at::cuda::getCurrentCUDAStream());

  return {grad_bias, grad_output_transpose};
}



std::vector<at::Tensor> fused_cast_transpose_bgrad_dgelu(at::Tensor grad_output,
                                                         at::Tensor gelu_input,
                                                         at::Tensor scale,
                                                         at::Tensor amax,
                                                         at::Tensor scale_inv,
                                                         transformer_engine::DType otype
) {
  using namespace transformer_engine;

  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto dgelu =
            allocateTorchTensor(grad_output.size(0),
                                grad_output.size(1),
                                DType::kByte);
  auto dgelu_transpose =
            allocateTorchTensor(grad_output.size(1),
                                grad_output.size(0),
                                DType::kByte);

  transformer_engine::TensorWrapper workspace;
  auto gelu_input_cu        = makeTransformerEngineTensor(gelu_input);
  auto input_cu             = makeTransformerEngineTensor(grad_output);
  auto cast_output_cu       = makeTransformerEngineTensor(dgelu.data_ptr(), {M, N},
                                                          otype, amax.data_ptr(), scale.data_ptr(),
                                                          scale_inv.data_ptr());
  auto transposed_output_cu = makeTransformerEngineTensor(dgelu_transpose.data_ptr(), {N, M},
                                                          otype, amax.data_ptr(), scale.data_ptr(),
                                                          scale_inv.data_ptr());
  auto dbias_cu             = makeTransformerEngineTensor(grad_bias);

  nvte_cast_transpose_dbias_dgelu(input_cu.data(), gelu_input_cu.data(),
                                  cast_output_cu.data(), transposed_output_cu.data(),
                                  dbias_cu.data(), workspace.data(),
                                  at::cuda::getCurrentCUDAStream());

  // Fill workspace
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace = makeTransformerEngineTensor(workspace_data.data_ptr(),
                                          workspace.shape(),
                                          workspace.dtype());

  nvte_cast_transpose_dbias_dgelu(input_cu.data(), gelu_input_cu.data(),
                                  cast_output_cu.data(), transposed_output_cu.data(),
                                  dbias_cu.data(), workspace.data(),
                                  at::cuda::getCurrentCUDAStream());

  return {grad_bias, dgelu, dgelu_transpose};
}


void fused_multi_cast_transpose(std::vector<at::Tensor> input_list,
                                std::vector<at::Tensor> scale_list,
                                std::vector<at::Tensor> cast_output_list,
                                std::vector<at::Tensor> transposed_output_list,
                                std::vector<at::Tensor> amax_list,
                                std::vector<at::Tensor> scale_inv_list,
                                transformer_engine::DType otype
) {
  using namespace transformer_engine;

  // Extract properties from PyTorch tensors
  std::vector<void*> input_dptr_list, scale_dptr_list,
    cast_output_dptr_list, transposed_output_dptr_list,
    amax_dptr_list, scale_inv_dptr_list;
  std::vector<std::vector<size_t>> input_shape_list, scale_shape_list,
    cast_output_shape_list, transposed_output_shape_list,
    amax_shape_list, scale_inv_shape_list;
  std::vector<transformer_engine::DType> input_type_list, scale_type_list,
    cast_output_type_list, transposed_output_type_list,
    amax_type_list, scale_inv_type_list;
  auto extract_tensor_props_skip_dtype = [](at::Tensor& tensor,
                                            std::vector<void*>& dptr_list,
                                            std::vector<std::vector<size_t>>& shape_list) {
    dptr_list.push_back(tensor.data_ptr());
    shape_list.push_back({});
    for (int d = 0; d < tensor.dim(); ++d) {
      shape_list.back().push_back(tensor.size(d));
    }
  };
  auto extract_tensor_props = [](at::Tensor& tensor,
                                 std::vector<void*>& dptr_list,
                                 std::vector<std::vector<size_t>>& shape_list,
                                 std::vector<transformer_engine::DType>& type_list) {
    dptr_list.push_back(tensor.data_ptr());
    shape_list.push_back({});
    for (int d = 0; d < tensor.dim(); ++d) {
      shape_list.back().push_back(tensor.size(d));
    }
    type_list.push_back(GetTransformerEngineDType(tensor.scalar_type()));
  };
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    extract_tensor_props(input_list[tensor_id],
                         input_dptr_list,
                         input_shape_list,
                         input_type_list);
    extract_tensor_props(scale_list[tensor_id],
                         scale_dptr_list,
                         scale_shape_list,
                         scale_type_list);
    extract_tensor_props_skip_dtype(cast_output_list[tensor_id],
                                    cast_output_dptr_list,
                                    cast_output_shape_list);
    cast_output_type_list.push_back(otype);
    extract_tensor_props_skip_dtype(transposed_output_list[tensor_id],
                                    transposed_output_dptr_list,
                                    transposed_output_shape_list);
    transposed_output_type_list.push_back(otype);
    extract_tensor_props(amax_list[tensor_id],
                         amax_dptr_list,
                         amax_shape_list,
                         amax_type_list);
    extract_tensor_props(scale_inv_list[tensor_id],
                         scale_inv_dptr_list,
                         scale_inv_shape_list,
                         scale_inv_type_list);
  }

  transformer_engine::TensorWrapper workspace;

  // Construct TE tensors
  std::vector<NVTETensor> nvte_input_list,
    nvte_cast_output_list, nvte_transposed_output_list;
  std::vector<transformer_engine::TensorWrapper> tensor_wrappers;
  auto make_tensor = [&tensor_wrappers](void* dptr,
                                        const std::vector<size_t>& shape,
                                        transformer_engine::DType dtype,
                                        void* amax_dptr,
                                        void* scale_dptr,
                                        void* scale_inv_dptr)
    -> NVTETensor {
    tensor_wrappers.emplace_back(makeTransformerEngineTensor(dptr, shape, dtype, amax_dptr,
                                                             scale_dptr, scale_inv_dptr));
    return tensor_wrappers.back().data();
  };
  for (size_t i = 0; i < input_dptr_list.size(); ++i) {
    nvte_input_list.emplace_back(make_tensor(input_dptr_list[i],
                                             input_shape_list[i],
                                             input_type_list[i],
                                             nullptr,
                                             nullptr,
                                             nullptr));
    nvte_cast_output_list.emplace_back(make_tensor(cast_output_dptr_list[i],
                                                   cast_output_shape_list[i],
                                                   cast_output_type_list[i],
                                                   amax_dptr_list[i],
                                                   scale_dptr_list[i],
                                                   scale_inv_dptr_list[i]));
    nvte_transposed_output_list.emplace_back(make_tensor(transposed_output_dptr_list[i],
                                                         transposed_output_shape_list[i],
                                                         transposed_output_type_list[i],
                                                         amax_dptr_list[i],
                                                         scale_dptr_list[i],
                                                         scale_inv_dptr_list[i]));
  }

  // Check tensor lists
  NVTE_CHECK(nvte_cast_output_list.size() == nvte_input_list.size(),
             "Number of input and C output tensors must match");
  NVTE_CHECK(nvte_transposed_output_list.size() == nvte_input_list.size(),
             "Number of input and T output tensors must match");

  // Launch TE kernel
  nvte_multi_cast_transpose(nvte_input_list.size(),
                            nvte_input_list.data(),
                            nvte_cast_output_list.data(),
                            nvte_transposed_output_list.data(),
                            at::cuda::getCurrentCUDAStream());
}


at::Tensor fp8_transpose(at::Tensor input,
                         transformer_engine::DType otype
) {
  using namespace transformer_engine;

  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  auto output =
            allocateTorchTensor(input.size(1),
                                input.size(0),
                                DType::kByte);

  auto input_cu  = makeTransformerEngineTensor(input.data_ptr(), {M, N}, otype);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), {N, M}, otype);

  nvte_transpose(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}


at::Tensor fp8_gelu(at::Tensor input,
                    at::Tensor scale,
                    at::Tensor amax,
                    at::Tensor scale_inv,
                    transformer_engine::DType otype
) {
  using namespace transformer_engine;

  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  auto output =
            allocateTorchTensor(input.size(0),
                                input.size(1),
                                DType::kByte);

  auto input_cu =  makeTransformerEngineTensor(input);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), {M, N}, otype,
                                               amax.data_ptr(), scale.data_ptr(),
                                               scale_inv.data_ptr());

  nvte_gelu(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}


std::vector<at::Tensor> layernorm_bwd(const at::Tensor &dz,
                                      const at::Tensor &x,
                                      const at::Tensor &mu,
                                      const at::Tensor &rsigma,
                                      const at::Tensor &gamma,
                                      const int sm_margin,
                                      const bool zero_centered_gamma
) {
    auto dx = at::empty_like(x);
    auto dgamma = at::empty_like(gamma);
    auto dbeta = at::empty_like(gamma);
    transformer_engine::TensorWrapper workspace, barrier, dgamma_part, dbeta_part;

    auto dz_cu      = makeTransformerEngineTensor(dz);
    auto x_cu       = makeTransformerEngineTensor(x);
    auto mu_cu      = makeTransformerEngineTensor(mu);
    auto rsigma_cu  = makeTransformerEngineTensor(rsigma);
    auto gamma_cu   = makeTransformerEngineTensor(gamma);
    auto dx_cu      = makeTransformerEngineTensor(dx);
    auto dgamma_cu  = makeTransformerEngineTensor(dgamma);
    auto dbeta_cu   = makeTransformerEngineTensor(dbeta);

    // This call populates tensors with the required config.
    const auto bwd_fun = zero_centered_gamma ? nvte_layernorm1p_bwd : nvte_layernorm_bwd;
    bwd_fun(dz_cu.data(), x_cu.data(), mu_cu.data(), rsigma_cu.data(), gamma_cu.data(),
            dx_cu.data(), dgamma_cu.data(), dbeta_cu.data(), dgamma_part.data(),
            dbeta_part.data(), at::cuda::getCurrentCUDAStream(),
            at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
            workspace.data(), barrier.data());

    // Alloc space for Tensors.
    auto workspace_data     = allocateSpace(workspace.shape(), workspace.dtype());
    auto barrier_data       = allocateSpace(barrier.shape(), barrier.dtype(), true);
    auto dgamma_part_data   = allocateSpace(dgamma_part.shape(), dgamma_part.dtype());
    auto dbeta_part_data    = allocateSpace(dbeta_part.shape(), dbeta_part.dtype());
    workspace   = makeTransformerEngineTensor(workspace_data.data_ptr(),
                                              workspace.shape(),
                                              workspace.dtype());
    barrier     = makeTransformerEngineTensor(barrier_data.data_ptr(),
                                              barrier.shape(),
                                              barrier.dtype());
    dgamma_part = makeTransformerEngineTensor(dgamma_part_data.data_ptr(),
                                              dgamma_part.shape(),
                                              dgamma_part.dtype());
    dbeta_part  = makeTransformerEngineTensor(dbeta_part_data.data_ptr(),
                                              dbeta_part.shape(),
                                              dbeta_part.dtype());

    // Actual call to bwd kernel.
    bwd_fun(dz_cu.data(), x_cu.data(), mu_cu.data(), rsigma_cu.data(), gamma_cu.data(),
            dx_cu.data(), dgamma_cu.data(), dbeta_cu.data(), dgamma_part.data(),
            dbeta_part.data(), at::cuda::getCurrentCUDAStream(),
            at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
            workspace.data(), barrier.data());

    return { dx, dgamma, dbeta };
}


std::vector<at::Tensor> layernorm_fwd_fp8(const at::Tensor &input,
                                          const at::Tensor &weight,
                                          const at::Tensor &bias,
                                          float eps,
                                          at::Tensor scale,
                                          at::Tensor amax,
                                          at::Tensor scale_inv,
                                          transformer_engine::DType otype,
                                          const int sm_margin,
                                          const bool zero_centered_gamma
) {
    using namespace transformer_engine;

    size_t N = static_cast<size_t>(input.size(0));
    size_t H = static_cast<size_t>(input.size(1));

    DType itype = GetTransformerEngineDType(input.scalar_type());

    auto ln_out = at::empty_like(input, at::CUDA(GetATenDType(otype)));
    auto mu = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));
    auto rsigma = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));
    auto input_cu     = makeTransformerEngineTensor(input);
    auto gamma_cu     = makeTransformerEngineTensor(weight);
    auto beta_cu      = makeTransformerEngineTensor(bias);
    auto z_cu         = makeTransformerEngineTensor(ln_out.data_ptr(), {N, H}, otype,
                                                    amax.data_ptr(), scale.data_ptr(),
                                                    scale_inv.data_ptr());
    auto mu_cu        = makeTransformerEngineTensor(mu);
    auto rsigma_cu    = makeTransformerEngineTensor(rsigma);
    transformer_engine::TensorWrapper workspace, barrier;

    // This call populates workspace and barrier tensors with the required config
    const auto func = zero_centered_gamma ? nvte_layernorm1p_fwd : nvte_layernorm_fwd;
    func(input_cu.data(), gamma_cu.data(), beta_cu.data(), eps, z_cu.data(),
         mu_cu.data(), rsigma_cu.data(), at::cuda::getCurrentCUDAStream(),
         at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
         workspace.data(), barrier.data());

    // Fill workspace and barrier
    auto workspace_data = allocateSpace(workspace.shape(),
                                        workspace.dtype());
    auto barrier_data = allocateSpace(barrier.shape(),
                                      barrier.dtype(),
                                      true);
    workspace = makeTransformerEngineTensor(workspace_data.data_ptr(),
                                            workspace.shape(),
                                            workspace.dtype());
    barrier   = makeTransformerEngineTensor(barrier_data.data_ptr(),
                                            barrier.shape(),
                                            barrier.dtype());

    // Actual call to fwd kernel
    func(input_cu.data(), gamma_cu.data(), beta_cu.data(), eps, z_cu.data(),
         mu_cu.data(), rsigma_cu.data(), at::cuda::getCurrentCUDAStream(),
         at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
         workspace.data(), barrier.data());

    return {ln_out, mu, rsigma};
}


at::Tensor layernorm_fwd_fp8_inf(const at::Tensor &input,
                                 const at::Tensor &weight,
                                 const at::Tensor &bias,
                                 float eps,
                                 at::Tensor scale,
                                 at::Tensor amax,
                                 at::Tensor scale_inv,
                                 transformer_engine::DType otype,
                                 const bool zero_centered_gamma
) {
    // This is a specialized version of layernorm_fwd_fp8, optimized for inference,
    // which only returns the normalized output.
    std::vector<at::Tensor> out = layernorm_fwd_fp8(
      input, weight, bias, eps, scale, amax, scale_inv, otype, 0, zero_centered_gamma);
    return out[0];
}


std::vector<at::Tensor> layernorm_fwd(const at::Tensor &input,
                                      const at::Tensor &weight,
                                      const at::Tensor &bias,
                                      float eps,
                                      const int sm_margin,
                                      const bool zero_centered_gamma
) {
    using namespace transformer_engine;

    size_t N = static_cast<size_t>(input.size(0));
    size_t H = static_cast<size_t>(input.size(1));

    DType itype = GetTransformerEngineDType(input.scalar_type());

    auto ln_out = at::empty_like(input, at::CUDA(GetATenDType(itype)));
    auto mu = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));
    auto rsigma = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));
    auto input_cu     = makeTransformerEngineTensor(input);
    auto gamma_cu     = makeTransformerEngineTensor(weight);
    auto beta_cu      = makeTransformerEngineTensor(bias);
    auto z_cu         = makeTransformerEngineTensor(ln_out);
    auto mu_cu        = makeTransformerEngineTensor(mu);
    auto rsigma_cu    = makeTransformerEngineTensor(rsigma);
    transformer_engine::TensorWrapper workspace, barrier;

    // This call populates workspace and barrier tensors with the required config
    const auto func = zero_centered_gamma ? nvte_layernorm1p_fwd : nvte_layernorm_fwd;
    func(input_cu.data(), gamma_cu.data(), beta_cu.data(), eps, z_cu.data(),
         mu_cu.data(), rsigma_cu.data(), at::cuda::getCurrentCUDAStream(),
         at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
         workspace.data(), barrier.data());

    // Fill workspace and barrier
    auto workspace_data = allocateSpace(workspace.shape(),
                                        workspace.dtype());
    auto barrier_data = allocateSpace(barrier.shape(),
                                      barrier.dtype(),
                                      true);
    workspace = makeTransformerEngineTensor(workspace_data.data_ptr(),
                                            workspace.shape(),
                                            workspace.dtype());
    barrier   = makeTransformerEngineTensor(barrier_data.data_ptr(),
                                            barrier.shape(),
                                            barrier.dtype());

    // Actual call to fwd kernel
    func(input_cu.data(), gamma_cu.data(), beta_cu.data(), eps, z_cu.data(),
         mu_cu.data(), rsigma_cu.data(), at::cuda::getCurrentCUDAStream(),
         at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
         workspace.data(), barrier.data());

    return {ln_out, mu, rsigma};
}


at::Tensor layernorm_fwd_inf(const at::Tensor &input,
                             const at::Tensor &weight,
                             const at::Tensor &bias,
                             float eps,
                             const bool zero_centered_gamma
) {
    // This is a specialized version of layernorm_fwd, optimized for inference,
    // which only returns the normalized output.
    std::vector<at::Tensor> out = layernorm_fwd(input, weight, bias, eps, 0, zero_centered_gamma);
    return out[0];
}


at::Tensor cast_to_fp8(const at::Tensor &input,
                       const at::Tensor &scale,
                       at::Tensor amax,
                       at::Tensor scale_inv,
                       transformer_engine::DType otype
) {
    using namespace transformer_engine;
    auto input_shape = input.sizes().vec();
    std::vector<size_t> shape{input_shape.begin(), input_shape.end()};

    auto output = at::empty_like(input, at::CUDA(GetATenDType(otype)));

    auto input_cu     = makeTransformerEngineTensor(input);
    auto output_cu    = makeTransformerEngineTensor(output.data_ptr(), shape, otype,
                                                    amax.data_ptr(), scale.data_ptr(),
                                                    scale_inv.data_ptr());

    nvte_fp8_quantize(input_cu.data(), output_cu.data(),
                      at::cuda::getCurrentCUDAStream());

    return output;
}


at::Tensor cast_from_fp8(const at::Tensor &input,
                         const at::Tensor &scale_inv,
                         transformer_engine::DType itype,
                         transformer_engine::DType otype
) {
    using namespace transformer_engine;
    auto input_shape = input.sizes().vec();
    std::vector<size_t> shape{input_shape.begin(), input_shape.end()};

    auto output = at::empty_like(input, at::CUDA(GetATenDType(otype)));

    auto input_cu     = makeTransformerEngineTensor(input.data_ptr(), shape, itype,
                                                    nullptr, nullptr, scale_inv.data_ptr());
    auto output_cu    = makeTransformerEngineTensor(output);

    nvte_fp8_dequantize(input_cu.data(), output_cu.data(),
                        at::cuda::getCurrentCUDAStream());

    return output;
}


at::Tensor scaled_softmax_forward(at::Tensor input,
                                  float scale_factor
) {
    using namespace transformer_engine;
    AT_ASSERTM(input.dim() == 4, "expected 4D tensor");
    AT_ASSERTM((input.scalar_type() == at::ScalarType::Half) ||
               (input.scalar_type() == at::ScalarType::BFloat16),
               "Only fp16 and bf16 are supported");

    const int batches = input.size(0);
    const int attn_heads = input.size(1);
    const int query_seq_len = input.size(2);
    const int key_seq_len = input.size(3);

    TORCH_CHECK(key_seq_len <= 4096);
    TORCH_CHECK(query_seq_len > 1);

    // Output
  auto act_options = input.options().requires_grad(false);
  auto softmax_results =
      torch::empty({batches, attn_heads, query_seq_len, key_seq_len}, act_options);

  auto input_cu = makeTransformerEngineTensor(input);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_softmax_forward(input_cu.data(), softmax_results_cu.data(), scale_factor,
                              at::cuda::getCurrentCUDAStream());

  return softmax_results;
}


at::Tensor scaled_softmax_backward(at::Tensor output_grad_,
                                   at::Tensor softmax_results_,
                                   float scale_factor
) {
    using namespace transformer_engine;

    auto output_grads = output_grad_.contiguous();
    auto softmax_results = softmax_results_.contiguous();

    AT_ASSERTM(output_grads.dim() == 4, "expected 4D tensor");
    AT_ASSERTM(softmax_results.dim() == 4, "expected 4D tensor");

    AT_ASSERTM((output_grads.scalar_type() == at::ScalarType::Half) ||
        (output_grads.scalar_type() == at::ScalarType::BFloat16),
        "Only fp16 and bf16 are supported");
    AT_ASSERTM((softmax_results.scalar_type() == at::ScalarType::Half) ||
        (softmax_results.scalar_type() == at::ScalarType::BFloat16),
        "Only fp16 and bf16 are supported");

    auto output_grads_cu = makeTransformerEngineTensor(output_grads);
    auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

    // Produce gradients in place.
    nvte_scaled_softmax_backward(
          output_grads_cu.data(), softmax_results_cu.data(), output_grads_cu.data(),
          scale_factor, at::cuda::getCurrentCUDAStream());

    return output_grads;
}


at::Tensor scaled_masked_softmax_forward(at::Tensor input,
                                         at::Tensor mask,
                                         float scale_factor
) {
    using namespace transformer_engine;

    AT_ASSERTM(input.dim() == 4, "expected 4D tensor");
    AT_ASSERTM((input.scalar_type() == at::ScalarType::Half) ||
               (input.scalar_type() == at::ScalarType::BFloat16),
               "Only fp16 and bf16 are supported");
    AT_ASSERTM(mask.dim() == 4, "expected 4D tensor");
    if (!input.is_contiguous())
        input = input.contiguous();
    if (!mask.is_contiguous())
        mask = mask.contiguous();

    const int batches = input.size(0);
    const int pad_batches = mask.size(0);
    const int attn_heads = input.size(1);
    const int query_seq_len = input.size(2);
    const int key_seq_len = input.size(3);
    TORCH_CHECK(key_seq_len <= 4096);
    TORCH_CHECK(query_seq_len > 1);
    TORCH_CHECK(pad_batches == 1 || pad_batches == batches);
    TORCH_CHECK(mask.size(1) == 1);
    TORCH_CHECK(mask.size(2) == query_seq_len);
    TORCH_CHECK(mask.size(3) == key_seq_len);

    auto act_options = input.options().requires_grad(false);
    auto softmax_results =
        torch::empty({batches, attn_heads, query_seq_len, key_seq_len}, act_options);


    auto input_cu = makeTransformerEngineTensor(input);
    auto mask_cu = makeTransformerEngineTensor(mask);
    auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

    nvte_scaled_masked_softmax_forward(
          input_cu.data(), mask_cu.data(), softmax_results_cu.data(),
          scale_factor, at::cuda::getCurrentCUDAStream());

    return softmax_results;
}


at::Tensor scaled_masked_softmax_backward(at::Tensor output_grad_,
                                          at::Tensor softmax_results_,
                                          float scale_factor
) {
    using namespace transformer_engine;

    auto output_grads = output_grad_.contiguous();
    auto softmax_results = softmax_results_.contiguous();

    AT_ASSERTM(output_grads.dim() == 4, "expected 3D tensor");
    AT_ASSERTM(softmax_results.dim() == 4, "expected 3D tensor");

    AT_ASSERTM((output_grads.scalar_type() == at::ScalarType::Half) ||
        (output_grads.scalar_type() == at::ScalarType::BFloat16),
        "Only fp16 and bf16 are supported");
    AT_ASSERTM((softmax_results.scalar_type() == at::ScalarType::Half) ||
        (softmax_results.scalar_type() == at::ScalarType::BFloat16),
        "Only fp16 and bf16 are supported");

    auto output_grads_cu = makeTransformerEngineTensor(output_grads);
    auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

    // Produce gradients in place.
    nvte_scaled_softmax_backward(
          output_grads_cu.data(), softmax_results_cu.data(), output_grads_cu.data(),
          scale_factor, at::cuda::getCurrentCUDAStream());

    return output_grads;
}


at::Tensor scaled_upper_triang_masked_softmax_forward(at::Tensor input,
                                                      float scale_factor
) {
    using namespace transformer_engine;

    AT_ASSERTM(input.dim() == 3, "expected 3D tensor");
    AT_ASSERTM((input.scalar_type() == at::ScalarType::Half) ||
               (input.scalar_type() == at::ScalarType::BFloat16),
               "Only fp16 and bf16 are supported");

    const int attn_batches = input.size(0);
    const int seq_len = input.size(1);
    TORCH_CHECK(seq_len <= 2048);

    // Output
    auto act_options = input.options().requires_grad(false);
    auto softmax_results =
        torch::empty({attn_batches, seq_len, seq_len}, act_options);

    auto input_cu = makeTransformerEngineTensor(input);
    auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

    nvte_scaled_upper_triang_masked_softmax_forward(input_cu.data(),
                                                    softmax_results_cu.data(),
                                                    scale_factor,
                                                    at::cuda::getCurrentCUDAStream());

    return softmax_results;
}


at::Tensor scaled_upper_triang_masked_softmax_backward(at::Tensor output_grads_,
                                                       at::Tensor softmax_results_,
                                                       float scale_factor
) {
    using namespace transformer_engine;

    auto output_grads = output_grads_.contiguous();
    auto softmax_results = softmax_results_.contiguous();

    AT_ASSERTM(output_grads.dim() == 3, "expected 3D tensor");
    AT_ASSERTM(softmax_results.dim() == 3, "expected 3D tensor");

    AT_ASSERTM((output_grads.scalar_type() == at::ScalarType::Half) ||
        (output_grads.scalar_type() == at::ScalarType::BFloat16),
        "Only fp16 and bf16 are supported");
    AT_ASSERTM((softmax_results.scalar_type() == at::ScalarType::Half) ||
        (softmax_results.scalar_type() == at::ScalarType::BFloat16),
        "Only fp16 and bf16 are supported");

    TORCH_CHECK(output_grads.size(1) == output_grads.size(2));

    auto output_grads_cu = makeTransformerEngineTensor(output_grads);
    auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

    // Produce gradients in place.
    nvte_scaled_upper_triang_masked_softmax_backward(output_grads_cu.data(),
                                                     softmax_results_cu.data(),
                                                     output_grads_cu.data(),
                                                     scale_factor,
                                                     at::cuda::getCurrentCUDAStream());

  return output_grads;
}


size_t get_cublasLt_version() {
    return cublasLtGetVersion();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Softmax functions
  m.def("scaled_softmax_forward", &scaled_softmax_forward, "Scaled Softmax FWD");
  m.def("scaled_softmax_backward", &scaled_softmax_backward, "Scaled Softmax BWD");
  m.def("scaled_masked_softmax_forward", &scaled_masked_softmax_forward,
                                                    "Scaled Masked Softmax FWD");
  m.def("scaled_masked_softmax_backward", &scaled_masked_softmax_backward,
                                                    "Scaled Masked Softmax BWD");
  m.def("scaled_upper_triang_masked_softmax_forward",
            &scaled_upper_triang_masked_softmax_forward,
            "Scaled Upper-Triangular Masked Softmax FWD");
  m.def("scaled_upper_triang_masked_softmax_backward",
            &scaled_upper_triang_masked_softmax_backward,
            "Scaled Upper-Triangular Masked Softmax BWD");

  // Other granular functions
  m.def("layernorm_fwd_fp8", &layernorm_fwd_fp8, "LN FWD FP8");
  m.def("layernorm_bwd", &layernorm_bwd, "LN BWD");
  m.def("layernorm_fwd", &layernorm_fwd, "LN FWD");
  m.def("fused_cast_transpose", &fused_cast_transpose, "Fused Cast + Transpose");
  m.def("fused_cast_transpose_bgrad", &fused_cast_transpose_bgrad,
                                              "Fused Cast + Transpose + BGRAD");
  m.def("fused_fp8_transpose_bgrad", &fused_fp8_transpose_bgrad,
                                              "Fused FP8 Transpose + BGRAD");
  m.def("fused_cast_transpose_bgrad_dgelu", &fused_cast_transpose_bgrad_dgelu,
                                              "Fused Cast + Transpose + BGRAD + DGELU");
  m.def("fused_multi_cast_transpose", &fused_multi_cast_transpose,
                                              "Fused Multi-tensor Cast + Transpose");
  m.def("cast_to_fp8", &cast_to_fp8, "Cast to FP8");
  m.def("cast_from_fp8", &cast_from_fp8, "Cast from FP8");
  m.def("te_gemm", &te_gemm, "CublasLt GEMM");
  m.def("fused_attn_fwd_qkvpacked", &fused_attn_fwd_qkvpacked, "Fused Attention FP8/BF16/FP16 FWD");
  m.def("fused_attn_bwd_qkvpacked", &fused_attn_bwd_qkvpacked, "Fused Attention FP8/BF16/FP16 BWD");
//  m.def("fused_attn_fwd_kvpacked", &fused_attn_fwd_kvpacked, "Fused Attention FP8/BF16/FP16 FWD");
//  m.def("fused_attn_bwd_kvpacked", &fused_attn_bwd_kvpacked, "Fused Attention FP8/BF16/FP16 BWD");
  m.def("fp8_transpose", &fp8_transpose, "Transpose with FP8 I/O");
  m.def("fp8_gelu", &fp8_gelu, "GeLU with FP8 output");

  // Misc
  m.def("get_cublasLt_version", &get_cublasLt_version, "Get cublasLt version");

  // Data structures
  py::class_<transformer_engine::FP8TensorMeta>(m, "FP8TensorMeta")
    .def(py::init<>())
    .def_readwrite("scale", &transformer_engine::FP8TensorMeta::scale)
    .def_readwrite("scale_inv", &transformer_engine::FP8TensorMeta::scale_inv)
    .def_readwrite("amax_history", &transformer_engine::FP8TensorMeta::amax_history);

  py::enum_<transformer_engine::DType>(m, "DType", py::module_local())
    .value("kByte", transformer_engine::DType::kByte)
    .value("kInt32", transformer_engine::DType::kInt32)
    .value("kFloat32", transformer_engine::DType::kFloat32)
    .value("kFloat16", transformer_engine::DType::kFloat16)
    .value("kBFloat16", transformer_engine::DType::kBFloat16)
    .value("kFloat8E4M3", transformer_engine::DType::kFloat8E4M3)
    .value("kFloat8E5M2", transformer_engine::DType::kFloat8E5M2);

  py::enum_<transformer_engine::FP8FwdTensors>(m, "FP8FwdTensors")
    .value("GEMM1_INPUT", transformer_engine::FP8FwdTensors::GEMM1_INPUT)
    .value("GEMM1_WEIGHT", transformer_engine::FP8FwdTensors::GEMM1_WEIGHT)
    .value("GEMM1_OUTPUT", transformer_engine::FP8FwdTensors::GEMM1_OUTPUT)
    .value("GEMM2_INPUT", transformer_engine::FP8FwdTensors::GEMM2_INPUT)
    .value("GEMM2_WEIGHT", transformer_engine::FP8FwdTensors::GEMM2_WEIGHT)
    .value("GEMM2_OUTPUT", transformer_engine::FP8FwdTensors::GEMM2_OUTPUT);

  py::enum_<transformer_engine::FP8BwdTensors>(m, "FP8BwdTensors")
    .value("GRAD_OUTPUT1", transformer_engine::FP8BwdTensors::GRAD_OUTPUT1)
    .value("GRAD_INPUT1", transformer_engine::FP8BwdTensors::GRAD_INPUT1)
    .value("GRAD_OUTPUT2", transformer_engine::FP8BwdTensors::GRAD_OUTPUT2)
    .value("GRAD_INPUT2", transformer_engine::FP8BwdTensors::GRAD_INPUT2);
}
