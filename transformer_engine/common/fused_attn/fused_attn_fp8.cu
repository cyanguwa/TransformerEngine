/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/fused_attn_fp8.h"
#include <cudnn_frontend.h>
#include "../common.h"

bool check_device_arch_newer_than(std::string const arch) {

    int arch_major = 6;
    int arch_minor = 0;
    if (arch == "hopper") {arch_major = 9;}
    if (arch == "ampere") {arch_major = 8;}
    if (arch == "turing")  {arch_major = 7; arch_minor = 5;}
    if (arch == "volta")  {arch_major = 7;}
    if (arch == "pascal") {arch_major = 6;}

    struct cudaDeviceProp prop;
    NVTE_CHECK_CUDA(cudaGetDeviceProperties( &prop, 0 ));

    auto device_version = prop.major * 10 + prop.minor;
    auto queried_version = arch_major * 10 + arch_minor;
     if (device_version >= queried_version) {
        return true;
     }
     return false;
}

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
      NVTE_ERROR("Invalid type");
  }
}

MHA_Layout get_mha_layout(const int layout) {
  switch (layout) {
    case 0:
      return MHA_Layout::NOT_INTERLEAVED;
    case 1:
      return MHA_Layout::QKV_INTERLEAVED;
    case 2:
      return MHA_Layout::KV_INTERLEAVED;
    default:
      NVTE_ERROR("Invalid layout");
  }
}

namespace transformer_engine {
namespace fused_attn {

using namespace transformer_engine;

void generateMHAStrides(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d, int64_t* strideA, MHA_Layout layout, MHA_Matrix matrix) {
    CUDNN_FRONTEND_UNUSED(b);
    constexpr int batch_dim_idx   = 0;
    constexpr int head_dim_idx    = 1;
    constexpr int seqlen_dim_idx  = 2;
    constexpr int hidden_dim_idx  = 3;

    constexpr int seqlen_transpose_dim_idx = 3;
    constexpr int hidden_transpose_dim_idx = 2;

    constexpr int seqlen_q_dim_idx = 2;
    constexpr int seqlen_kv_dim_idx = 3;

    switch (matrix)
    {
        case MHA_Matrix::Q_Matrix:
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
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
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
                strideA[seqlen_dim_idx] = 3 * h * d;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 3 * h * d;
            } else if (layout == MHA_Layout::KV_INTERLEAVED) {
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
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
                strideA[seqlen_transpose_dim_idx] = 3 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 3 * h * d;
            } else if (layout == MHA_Layout::KV_INTERLEAVED) {
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
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
                strideA[hidden_dim_idx] = 1;
                strideA[seqlen_dim_idx] = 3 * h * d;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 3 * h * d;
            } else if (layout == MHA_Layout::KV_INTERLEAVED) {
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
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
                    strideA[hidden_transpose_dim_idx] = 1;
                    strideA[seqlen_transpose_dim_idx] = 3 * h * d;
                    strideA[head_dim_idx] = d;
                    strideA[batch_dim_idx] = s_kv * 3 * h * d;
                } else if (layout == MHA_Layout::KV_INTERLEAVED) {
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

struct FADescriptor {
  std::int64_t b;
  std::int64_t h;
  std::int64_t s_q;
  std::int64_t s_kv;
  std::int64_t d;
  float attnScale;
  float dropoutProbability;
  MHA_Layout layout;
  void* seed;
  void* offset;
  cudnnDataType_t tensor_type;

  bool operator<(const FADescriptor &rhs) const {
    return std::tie(b, h, s_q, s_kv, d,
		    attnScale, dropoutProbability,
		    layout,
		    seed, offset,
                    tensor_type) < std::tie(rhs.b, rhs.h, rhs.s_q, rhs.s_kv,
			    rhs.d, rhs.attnScale, rhs.dropoutProbability,
			    rhs.layout, rhs.seed, rhs.offset,
                            rhs.tensor_type);
  }
};

#if (CUDNN_VERSION >= 8900)
std::unordered_map<std::string, int> tensor_name_to_uid = {
    {"Q",                            1},
    {"K",                            2},
    {"V",                            3},
    {"O",                            4},
    {"S",                            5},
    {"B",                            6},
    {"DROPOUT_SCALE",                7},
    {"S_CONST",                      8},
    {"MNK_OVERRIDE",                 9},
    {"dQ",                          11},
    {"dK",                          12},
    {"dV",                          13},
    {"dO",                          14},
    {"MASK_VAL",                    15},
    {"dS",                          16},
    {"O_SEQLEN",                    17},
    {"M",                           18},
    {"Z",                           19},
    {"descaleQ",                    20},
    {"descaleK",                    21},
    {"descaleV",                    22},
    {"descaleS",                    23},
    {"scaleS",                      24},
    {"amaxS",                       25},
    {"amaxO",                       26},
    {"QKV_RAGGED",                  27},
    {"O_RAGGED",                    28},
    {"K_TRANSPOSE",                 29},
    {"AttnScale",                   30},
    {"scaleO",                      31},
    {"Z_INV",                       32},
    {"descaleO",                    33},
    {"descaledO",                   34},
    {"descaledS",                   35},
    {"descaledQ",                   36},
    {"descaledK",                   37},
    {"descaledV",                   38},
    {"scaledS",                     39},
    {"scaledQ",                     40},
    {"scaledK",                     41},
    {"scaledV",                     42},
    {"amaxdS",                      43},
    {"amaxdQ",                      44},
    {"amaxdK",                      45},
    {"amaxdV",                      46},
    {"V_TRANSPOSE",                 47},
    {"AttnScale_dS_K",              48},
    {"AttnScale_dSTranspose_Q",     49},
    {"DROPOUT_SCALE_dOVt_OdO",      50},
    {"DROPOUT_OFFSET",              51},
    {"DROPOUT_SEED",                52},
    {"VIRTUAL",                     80}
};

bool allowAllConfig(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

static cudnn_frontend::Tensor tensor_create(cudnnDataType_t type, int64_t id, int64_t const * dim,
                                int64_t const * stride, bool is_virtual, bool is_value) {
    int nbDims = 4;
    auto tensor_created = cudnn_frontend::TensorBuilder()
            .setDim(nbDims, dim)
            .setStride(nbDims, stride)
            .setId(id)
            .setAlignment(16) // 16B alignment is needed to run a tensor core engine
            .setDataType(type)
            .setVirtual(is_virtual)
            .setByValue(is_value)
            .build();
    return tensor_created;
};

static cudnn_frontend::Tensor tensor_create_with_offset(cudnnDataType_t type, int64_t id, int64_t const * dim,
                                int64_t const * stride, bool is_virtual, bool is_value, std::shared_ptr<cudnn_frontend::Tensor>& raggedOffset) {
    int nbDims = 4;
    auto tensor_created = cudnn_frontend::TensorBuilder()
            .setDim(nbDims, dim)
            .setStride(nbDims, stride)
            .setId(id)
            .setAlignment(16) // 16B alignment is needed to run a tensor core engine
            .setDataType(type)
            .setVirtual(is_virtual)
            .setByValue(is_value)
            .setRaggedOffset(raggedOffset)
            .build();
    return tensor_created;
};

static cudnn_frontend::PointWiseDesc pw_desc_create(cudnnDataType_t type, cudnnPointwiseMode_t mode) {
    auto pw_desc_created = cudnn_frontend::PointWiseDescBuilder()
            .setMode(mode)
            .setComputeType(type)
            .build();

    return pw_desc_created;
}

static cudnn_frontend::Operation unary_pw_op_create(cudnn_frontend::Tensor const &xDesc, cudnn_frontend::Tensor const &yDesc,
                                                    cudnn_frontend::PointWiseDesc const &pwDesc) {
    auto pw_op_created = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(xDesc)
                        .setyDesc(yDesc)
                        .setpwDesc(pwDesc)
                        .build();
    return pw_op_created;
}

static cudnn_frontend::Operation binary_pw_op_create(cudnn_frontend::Tensor const &xDesc, cudnn_frontend::Tensor const &bDesc,
                                                    cudnn_frontend::Tensor const &yDesc, cudnn_frontend::PointWiseDesc const &pwDesc) {
    auto pw_op_created = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(xDesc)
                        .setbDesc(bDesc)
                        .setyDesc(yDesc)
                        .setpwDesc(pwDesc)
                        .build();
    return pw_op_created;
}

static cudnn_frontend::Operation ternary_pw_op_create(cudnn_frontend::Tensor const &xDesc, cudnn_frontend::Tensor const &bDesc, cudnn_frontend::Tensor const &tDesc,
                                            cudnn_frontend::Tensor const &yDesc, cudnn_frontend::PointWiseDesc const &pwDesc) {
    auto pw_op_created = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(xDesc)
                        .setbDesc(bDesc)
                        .settDesc(tDesc)
                        .setyDesc(yDesc)
                        .setpwDesc(pwDesc)
                        .build();
    return pw_op_created;
}

static cudnn_frontend::Tensor
createAmax(const std::string& amax_tensor_name,
            cudnn_frontend::Tensor& prevBlockOutputTensor,
            std::vector<cudnn_frontend::Operation>& ops) {

        // Amax is just a scalar
        int64_t amax_dim [4] = {1, 1, 1, 1};
        int64_t amax_stride [4] = {1, 1, 1, 1};

        auto amaxTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid[amax_tensor_name], amax_dim, amax_stride, false, false);

        // Define the amax descriptor
        auto redunctionDesc = cudnn_frontend::ReductionDescBuilder()
                                  .setMathPrecision(CUDNN_DATA_FLOAT)
                                  .setReductionOp(CUDNN_REDUCE_TENSOR_AMAX)
                                  .build();

        // Create a reduction amax Node.
        auto reduction_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(prevBlockOutputTensor)
                                .setyDesc(amaxTensor)
                                .setreductionDesc(redunctionDesc)
                                .build();
        ops.push_back(std::move(reduction_op));
        return amaxTensor;
}

static cudnn_frontend::Tensor
createScale(cudnn_frontend::Tensor& prevBlockOutputTensor,
            const std::string& scale_tensor_name,
            cudnnDataType_t tensorType,
            bool isOutputVirtual,
            bool isScaleByValue,
            std::vector<cudnn_frontend::Operation>& ops,
            const std::string& output_tensor_name ="") {

    // scale
    int64_t scale_dim [4] = {1, 1, 1, 1};
    int64_t scale_stride [4] = {1, 1, 1, 1};

    int64_t output_dim [4];
    int64_t output_stride [4];

    // output dim and stride should be the same as prev block dim and stride
    for (int i = 0; i < 4; i++) {
        output_dim[i] = prevBlockOutputTensor.getDim()[i];
        output_stride[i] = prevBlockOutputTensor.getStride()[i];
    }

    auto scaleTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid[scale_tensor_name], scale_dim, scale_stride, false, isScaleByValue); // is by value

    // Hack to get the virtual id to not be same for all the virtual tensors
    int64_t outputUID = isOutputVirtual ? tensor_name_to_uid["VIRTUAL"] + tensor_name_to_uid[scale_tensor_name] + 5000 : tensor_name_to_uid[output_tensor_name];
    auto afterScaleKTensor = tensor_create(tensorType, outputUID, output_dim, output_stride, isOutputVirtual, false); // is virtual

    // Define the scale descriptor
    auto scaleDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a Scale Node.
    auto scale_op = binary_pw_op_create(prevBlockOutputTensor, scaleTensor, afterScaleKTensor, scaleDesc);

    ops.push_back(std::move(scale_op));
    return afterScaleKTensor;
}

static cudnn_frontend::Tensor
createScale(cudnn_frontend::Tensor& prevBlockOutputTensor,
            const cudnn_frontend::Tensor& scaleTensor,
            cudnnDataType_t tensorType,
            bool isOutputVirtual,
            bool isScaleByValue,
            std::vector<cudnn_frontend::Operation>& ops,
            int UID_offset,
            const std::string& output_tensor_name ="") {

    CUDNN_FRONTEND_UNUSED(isScaleByValue);
    int64_t output_dim [4];
    int64_t output_stride [4];
    // output dim and stride should be the same as prev block dim and stride
    for (int i = 0; i < 4; i++) {
        output_dim[i] = prevBlockOutputTensor.getDim()[i];
        output_stride[i] = prevBlockOutputTensor.getStride()[i];
    }

    // Hack to get the virtual id to not be same for all the virtual tensors
    int64_t outputUID = isOutputVirtual ? tensor_name_to_uid["VIRTUAL"] + UID_offset : tensor_name_to_uid[output_tensor_name];
    auto afterScaleTensor = tensor_create(tensorType, outputUID, output_dim, output_stride, isOutputVirtual, false); // is virtual

    // Define the scale descriptor
    auto scaleDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a Scale Node.
    auto scale_op = binary_pw_op_create(prevBlockOutputTensor, scaleTensor, afterScaleTensor, scaleDesc);

    ops.push_back(std::move(scale_op));
    return afterScaleTensor;
}

static cudnn_frontend::Tensor
createScaleWithOffset(cudnn_frontend::Tensor& prevBlockOutputTensor,
            const std::string& scale_tensor_name,
            cudnnDataType_t tensorType,
            bool isOutputVirtual,
            bool isScaleByValue,
            std::vector<cudnn_frontend::Operation>& ops,
            std::shared_ptr<cudnn_frontend::Tensor>& offsetTensor,
            const std::string& output_tensor_name ="") {

    // scale
    int64_t scale_dim [4] = {1, 1, 1, 1};
    int64_t scale_stride [4] = {1, 1, 1, 1};

    int64_t output_dim [4];
    int64_t output_stride [4];
    // If output tensor is dQ, dK, or dV, we need to generate QKV interleaved strides
    if (output_tensor_name == "dQ" || output_tensor_name == "dK" || output_tensor_name == "dV") {
        // Dims remain the same from previous block
        for (int i = 0; i < 4; i++) {
            output_dim[i] = prevBlockOutputTensor.getDim()[i];
        }
        // We know that dQ, dK, and dV are dims [batch, head, s_q, embedding_dim]
        // All dQ, dK, and dV strides will follow Q_Matrix stride layout
        generateMHAStrides(output_dim[0], output_dim[1], output_dim[2], 0 /*s_kv = 0 for placeholder*/, output_dim[3], output_stride, MHA_Layout::QKV_INTERLEAVED, MHA_Matrix::Q_Matrix);
    } else {
        // otherwise output dim and stride should be the same as prev block dim and stride
        for (int i = 0; i < 4; i++) {
            output_dim[i] = prevBlockOutputTensor.getDim()[i];
            output_stride[i] = prevBlockOutputTensor.getStride()[i];
        }
    }

    auto scaleTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid[scale_tensor_name], scale_dim, scale_stride, false, isScaleByValue); // is by value

    cudnnDataType_t outputDataType = isOutputVirtual ? CUDNN_DATA_FLOAT : tensorType;
    // Hack to get the virtual id to not be same for all the virtual tensors
    int64_t outputUID = isOutputVirtual ? tensor_name_to_uid["VIRTUAL"] + tensor_name_to_uid[scale_tensor_name] + 7000 : tensor_name_to_uid[output_tensor_name];
    auto afterScaleTensor = tensor_create_with_offset(outputDataType, outputUID, output_dim, output_stride, isOutputVirtual, false, offsetTensor); // is virtual

    // Define the scale descriptor
    auto scaleDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a Scale Node.
    auto scale_op = binary_pw_op_create(prevBlockOutputTensor, scaleTensor, afterScaleTensor, scaleDesc);

    ops.push_back(std::move(scale_op));
    return afterScaleTensor;
}

static cudnn_frontend::Tensor
createSoftmaxForward(int64_t b,
                     int64_t h,
                     int64_t s_q,
                     int64_t s_kv,
                     std::vector<cudnn_frontend::Operation>& ops,
                     cudnn_frontend::Tensor& prevBlockOutputTensor,
                     bool isTraining) {

    int64_t afterBMM1_dim [4] = {b, h, s_q, s_kv};
    int64_t afterBMM1_stride [4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    int64_t afterReduction_dim [4] = {b, h, s_q, 1};
    int64_t afterReduction_stride [4] = {h * s_q, s_q, 1, 1};

    // max (x) (M tensor)
    auto afterMaxReductionTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["M"], afterReduction_dim, afterReduction_stride, !isTraining, false); // not virtual if training is true, virtual if training is false
    // x - max(x)
    auto afterSubtractionTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 151, afterBMM1_dim, afterBMM1_stride, true, false); // is virtual
    // e^(x - max(x))
    auto afterExponentTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 152, afterBMM1_dim, afterBMM1_stride, true, false); // is virtual;
    // sum (e^(x - max(x))) (Z tensor)
    auto zTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["Z"], afterReduction_dim, afterReduction_stride, true, false); // is virtual
    // 1 / sum (e^(x - max(x))) (Z_INV tensor)
    auto zInvTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["Z_INV"], afterReduction_dim, afterReduction_stride, !isTraining, false); // not virtual if training is true, virtual if training is false
    // Final softmax output (After exponent * Z_INV)
    auto beforeDropoutTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 153, afterBMM1_dim, afterBMM1_stride, true, false); // is virtual

    // Define the reduction descriptor
    auto reductionMaxDesc = cudnn_frontend::ReductionDescBuilder()
                                .setComputeType(CUDNN_DATA_FLOAT)
                                .setReductionOp(CUDNN_REDUCE_TENSOR_MAX)
                                .build();

    // Create a reduction max Node.
    auto reductionMax_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(prevBlockOutputTensor)
                                .setyDesc(afterMaxReductionTensor)
                                .setreductionDesc(reductionMaxDesc)
                                .build();

    // Define the subtract descriptor
    auto subtractDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_SUB);

    // Create a subtract Node.
    auto subtract_op = binary_pw_op_create(prevBlockOutputTensor, afterMaxReductionTensor, afterSubtractionTensor, subtractDesc);

    // Define the exponent descriptor
    auto exponentDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_EXP);

    // Create a exponent Node.
    auto exponent_op = unary_pw_op_create(afterSubtractionTensor, afterExponentTensor, exponentDesc);

    // Define the reduction descriptor
    auto reductionAddDesc = cudnn_frontend::ReductionDescBuilder()
                                .setComputeType(CUDNN_DATA_FLOAT)
                                .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                                .build();

    // Create a reduction add Node.
    auto reductionAdd_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(afterExponentTensor)
                                .setyDesc(zTensor)
                                .setreductionDesc(reductionAddDesc)
                                .build();


    // Define the reciprocal descriptor
    auto reciprocalDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_RECIPROCAL);

    // Create a reciprocal Node.
    auto reciprocal_op = unary_pw_op_create(zTensor, zInvTensor, reciprocalDesc);

    // Define the pw multiply descriptor
    auto multiplyDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a multiply Node.
    auto mutliply_op = binary_pw_op_create(afterExponentTensor, zInvTensor, beforeDropoutTensor, multiplyDesc);

    ops.push_back(std::move(reductionMax_op));
    ops.push_back(std::move(subtract_op));
    ops.push_back(std::move(exponent_op));
    ops.push_back(std::move(reductionAdd_op));
    ops.push_back(std::move(reciprocal_op));
    ops.push_back(std::move(mutliply_op));

    return beforeDropoutTensor;
}

static cudnn_frontend::Tensor
createDropoutForward(int64_t b,
              int64_t h,
              int64_t s_q,
              int64_t s_kv,
              double probability,
              std::vector<cudnn_frontend::Operation>& ops,
              cudnn_frontend::Tensor& beforeDropoutTensor) {

    cudnn_frontend::throw_if(ops.size() == 0, "Dropout DAG constructed incorrectly as the first one", CUDNN_STATUS_BAD_PARAM);

    int64_t afterBMM1_dim [4] = {b, h, s_q, s_kv};
    int64_t afterBMM1_stride [4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    int64_t scale_dim [4] = {1, 1, 1, 1};
    int64_t scale_stride [4] = {1, 1, 1, 1};

    // mask for the dropout
    auto dropoutMaskTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 250, afterBMM1_dim, afterBMM1_stride, true, false); // is virtual
    auto dropoutSeedTensor = tensor_create(CUDNN_DATA_INT64, tensor_name_to_uid["DROPOUT_SEED"], scale_dim, scale_stride, false, false); // is by value
    auto dropoutOffsetTensor = tensor_create(CUDNN_DATA_INT64, tensor_name_to_uid["DROPOUT_OFFSET"], scale_dim, scale_stride, false, false); // is by value

    // after dropout tensor befor scale
    auto beforeDropoutScaleTensor = cudnn_frontend::TensorBuilder()
            .setDim(4, afterBMM1_dim)
            .setStride(4, afterBMM1_stride)
            .setId(tensor_name_to_uid["VIRTUAL"] + 201)
            .setAlignment(16) // 16B alignment is needed to run a tensor core engine
            .setDataType(CUDNN_DATA_FLOAT)
            .setVirtual(true)
            .setByValue(false)
            .setReorderType(cudnn_frontend::cudnnBackendTensorReordering_t::CUDNN_TENSOR_REORDERING_F16x16)
            .build();
    // scale after dropout
    auto scaleDropoutTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["DROPOUT_SCALE"], scale_dim, scale_stride, false, true); // is by value
    // after Scale
    auto afterDropout_before_quan_S = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 202, afterBMM1_dim, afterBMM1_stride, true, false); // is virtual

    // Define the reduction descriptor
    auto rngDesc = cudnn_frontend::RngDescBuilder()
                                .setRngDistribution(CUDNN_RNG_DISTRIBUTION_BERNOULLI)
                                .setBernoulliDistProbability(1.0 - probability)
                                .build();

    // Create a rng Node.
    auto rng_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR)
                                .setyDesc(dropoutMaskTensor)
                                .setSeedDesc(dropoutSeedTensor)
                                .setOffsetDesc(dropoutOffsetTensor)
                                .setRngDesc(rngDesc)
                                .build();


    // Define the multiply mask descriptor
    auto maskMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a multiply mask Node.
    auto maskMul_op = binary_pw_op_create(beforeDropoutTensor, dropoutMaskTensor, beforeDropoutScaleTensor, maskMulDesc);

    // Define the multiply scale descriptor
    auto scaleMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a multiply mask Node.
    auto scaleMul_op = binary_pw_op_create(beforeDropoutScaleTensor, scaleDropoutTensor, afterDropout_before_quan_S, scaleMulDesc);

    ops.push_back(std::move(rng_op));
    ops.push_back(std::move(maskMul_op));
    ops.push_back(std::move(scaleMul_op));

    return afterDropout_before_quan_S;
}

static cudnn_frontend::Tensor
createDropoutBackward(int64_t b,
              int64_t h,
              int64_t s_q,
              int64_t s_kv,
              double probability,
              std::vector<cudnn_frontend::Operation>& ops,
              cudnn_frontend::Tensor& beforeDropoutTensor,
              cudnn_frontend::Tensor& dropoutMaskTensor) {

    cudnn_frontend::throw_if(ops.size() == 0, "Dropout DAG constructed incorrectly as the first one", CUDNN_STATUS_BAD_PARAM);

    int64_t afterBMM1_dim [4] = {b, h, s_q, s_kv};
    int64_t afterBMM1_stride [4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    int64_t scale_dim [4] = {1, 1, 1, 1};
    int64_t scale_stride [4] = {1, 1, 1, 1};

    auto dropoutSeedTensor = tensor_create(CUDNN_DATA_INT64, tensor_name_to_uid["DROPOUT_SEED"], scale_dim, scale_stride, false, false); // is by value
    auto dropoutOffsetTensor = tensor_create(CUDNN_DATA_INT64, tensor_name_to_uid["DROPOUT_OFFSET"], scale_dim, scale_stride, false, false); // is by value

    // after dropout tensor befor scale
    auto beforeDropoutScaleTensor = cudnn_frontend::TensorBuilder()
            .setDim(4, afterBMM1_dim)
            .setStride(4, afterBMM1_stride)
            .setId(tensor_name_to_uid["VIRTUAL"] + 201)
            .setAlignment(16) // 16B alignment is needed to run a tensor core engine
            .setDataType(CUDNN_DATA_FLOAT)
            .setVirtual(true)
            .setByValue(false)
            .setReorderType(cudnn_frontend::cudnnBackendTensorReordering_t::CUDNN_TENSOR_REORDERING_F16x16)
            .build();
    // scale after dropout (1 / (1 - p))
    auto scaleDropoutTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["DROPOUT_SCALE"], scale_dim, scale_stride, false, true); // is by value
    // after Scale
    auto afterDropout_before_quan_S = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 202, afterBMM1_dim, afterBMM1_stride, true, false); // is virtual

    // Define the reduction descriptor
    auto rngDesc = cudnn_frontend::RngDescBuilder()
                                .setRngDistribution(CUDNN_RNG_DISTRIBUTION_BERNOULLI)
                                .setBernoulliDistProbability(1.0 - probability)
                                .build();

    // Create a rng Node.
    auto rng_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR)
                                .setyDesc(dropoutMaskTensor)
                                .setSeedDesc(dropoutSeedTensor)
                                .setOffsetDesc(dropoutOffsetTensor)
                                .setRngDesc(rngDesc)
                                .build();


    // Define the multiply mask descriptor
    auto maskMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a multiply mask Node.
    auto maskMul_op = binary_pw_op_create(beforeDropoutTensor, dropoutMaskTensor, beforeDropoutScaleTensor, maskMulDesc);

    // Define the multiply scale descriptor
    auto scaleMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a multiply mask Node.
    auto scaleMul_op = binary_pw_op_create(beforeDropoutScaleTensor, scaleDropoutTensor, afterDropout_before_quan_S, scaleMulDesc);

    ops.push_back(std::move(rng_op));
    ops.push_back(std::move(maskMul_op));
    ops.push_back(std::move(scaleMul_op));

    return afterDropout_before_quan_S;
}

static cudnn_frontend::Tensor
createSoftmaxBackward(int64_t b,
                     int64_t h,
                     int64_t s_q,
                     int64_t s_kv,
                     std::vector<cudnn_frontend::Operation>& ops,
                     cudnn_frontend::Tensor& dyTensor) {
    cudnn_frontend::throw_if(ops.size() == 0, "Softmax backward constructed incorrectly as the first one", CUDNN_STATUS_BAD_PARAM);

    int64_t dx_dim [4] = {b, h, s_q, s_kv};
    int64_t dx_stride [4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    int64_t M_Z_dim [4] = {b, h, s_q, 1};
    int64_t M_Z_stride [4] = {h * s_q, s_q, 1, 1};

    // creating all tensors
    auto MTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["M"], M_Z_dim, M_Z_stride, false, false); // not virtual
    auto ZInvTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["Z_INV"], M_Z_dim, M_Z_stride, false, false); // not virtual
    auto dxAfterSubtractionTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 252, dx_dim, dx_stride, true, false); // is virtual
    auto dxAfterExponentiation = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 253, dx_dim, dx_stride, true, false); // is virtual
    auto dxBeforeDropout_QKt_Tensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 254, dx_dim, dx_stride, true, false); // is virtual

    // creating all ops
    // sub (dy - M)
    auto subtractionDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_SUB);
    auto subtractionOp = binary_pw_op_create(dyTensor, MTensor, dxAfterSubtractionTensor, subtractionDesc);

    // Define the exponent descriptor
    auto exponentDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_EXP);

    // Create a exponent Node. (exp(dy - M))
    auto exponentOp = unary_pw_op_create(dxAfterSubtractionTensor, dxAfterExponentiation, exponentDesc);

    // Define the pw multiply descriptor
    auto multiplyDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a multiply Node.
    auto mutliplyOp = binary_pw_op_create(dxAfterExponentiation, ZInvTensor, dxBeforeDropout_QKt_Tensor, multiplyDesc);

    ops.push_back(std::move(subtractionOp));
    ops.push_back(std::move(exponentOp));
    ops.push_back(std::move(mutliplyOp));

    return dxBeforeDropout_QKt_Tensor;
}

static cudnn_frontend::Tensor
createQKBMM(int64_t b,
           int64_t h,
           int64_t s_q,
           int64_t s_kv,
           int64_t d,
           MHA_Layout layout,
           cudnnDataType_t tensorType,
           std::vector<cudnn_frontend::Operation>& ops,
           const cudnn_frontend::Tensor &qTensor,
           const cudnn_frontend::Tensor &kTensor,
           const cudnn_frontend::Tensor &mnkOverride,
           std::shared_ptr<cudnn_frontend::Tensor>& QKVRaggedOffsetTensor) {
    // Creates the necessary tensor descriptors
    int64_t k_transpose_dim [4] = {b, h, d, s_kv};
    int64_t k_transpose_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, k_transpose_stride, layout, MHA_Matrix::K_Matrix_Transpose);

    int64_t s_dim [4] = {b, h, s_q, s_kv};
    int64_t s_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, s_stride, layout, MHA_Matrix::S_Matrix);

    auto kTransposeTensor = tensor_create_with_offset(tensorType, tensor_name_to_uid["K_TRANSPOSE"], k_transpose_dim, k_transpose_stride, false, false, QKVRaggedOffsetTensor); // is virtual

    // first GEMM output
    auto afterQKTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 1, s_dim, s_stride, true, false); // is virtual

    // Define the matmul desc
    auto matmulDesc = cudnn_frontend::MatMulDescBuilder()
                                    .setComputeType(CUDNN_DATA_FLOAT)
                                    .setPaddingValue(-2000000)
                                    .build();

    // Create reshape node for K -> K.T
    auto reshape_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                            .setxDesc(kTensor)
                            .setyDesc(kTransposeTensor)
                            .build();


    // Create a matmul Node
    auto matmulOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(qTensor)
                            .setbMatDesc(kTransposeTensor)
                            .setcMatDesc(afterQKTensor)
                            .setmOverrideDesc(mnkOverride)
                            .setnOverrideDesc(mnkOverride)
                            .setmatmulDesc(matmulDesc)
                            .build();


    ops.push_back(std::move(reshape_op));
    ops.push_back(std::move(matmulOp));

    return afterQKTensor;
}

static cudnn_frontend::Tensor
createSVBMM(int64_t b,
           int64_t h,
           int64_t s_q,
           int64_t s_kv,
           int64_t d,
           MHA_Layout layout,
           cudnnDataType_t tensorType,
           std::vector<cudnn_frontend::Operation>& ops,
           const cudnn_frontend::Tensor &softmaxTensor,
           const cudnn_frontend::Tensor &mnkOverride,
           std::shared_ptr<cudnn_frontend::Tensor>& QKVRaggedOffsetTensor) {

    cudnn_frontend::throw_if(ops.size() == 0, "BMM2 op constructed incorrectly as the first one", CUDNN_STATUS_BAD_PARAM);

    int64_t v_dim [4] =  {b, h, s_kv, d};
    int64_t v_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, v_stride, layout, MHA_Matrix::V_Matrix);

    int64_t o_dim [4] =  {b, h, s_q, d};
    int64_t o_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, o_stride, layout, MHA_Matrix::O_Matrix);

    auto vTensor = tensor_create_with_offset(tensorType, tensor_name_to_uid["V"], v_dim, v_stride, false, false, QKVRaggedOffsetTensor);
    // second fprop GEMM output
    auto oTensor = tensor_create(tensorType, tensor_name_to_uid["VIRTUAL"] + 300, o_dim, o_stride, true, false); // is virtual

    // Define the matmul desc
    auto matmulDesc = cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();

    // Create a matmul Node
    auto matmulOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(softmaxTensor)
                            .setbMatDesc(vTensor)
                            .setcMatDesc(oTensor)
                            .setmOverrideDesc(mnkOverride)
                            .setkOverrideDesc(mnkOverride)
                            .setmatmulDesc(matmulDesc)
                            .build();

    ops.push_back(std::move(matmulOp));

    return oTensor;
}

static cudnn_frontend::Tensor
createSdOBMM(int64_t b,
           int64_t h,
           int64_t s_q,
           int64_t s_kv,
           int64_t d,
           cudnnDataType_t tensorType,
           std::vector<cudnn_frontend::Operation>& ops,
           const cudnn_frontend::Tensor &softmaxTensor,
           const cudnn_frontend::Tensor &dOTensor,
           const cudnn_frontend::Tensor &mnkOverride) {

    cudnn_frontend::throw_if(ops.size() == 0, "BMM2 op constructed incorrectly as the first one", CUDNN_STATUS_BAD_PARAM);

    int64_t s_dim_transpose [4] =  {b, h, s_kv, s_q};
    int64_t s_stride_transpose [4] = {h * s_kv * s_q, s_kv * s_q, 1, s_kv};

    int64_t v_dim [4] =  {b, h, s_kv, d};
    int64_t v_stride [4] = {h * s_kv * d, d, h * d, 1};

    auto sTransposeTensor = tensor_create(tensorType, tensor_name_to_uid["VIRTUAL"] + 499, s_dim_transpose, s_stride_transpose, true, false); // is virtual
    // S.T * dO
    auto dVTensor_before_dequan_S = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 500, v_dim, v_stride, true, false); // is virtual

    // Create reshape node for softmax -> softmax.T
    auto reshape_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                            .setxDesc(softmaxTensor)
                            .setyDesc(sTransposeTensor)
                            .build();

    // Define the matmul desc
    auto matmulDesc = cudnn_frontend::MatMulDescBuilder()
                                    .setComputeType(CUDNN_DATA_FLOAT)
                                    .setPaddingValue(0)
                                    .build();

    // Create a matmul Node
    auto matmulOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(sTransposeTensor)
                            .setbMatDesc(dOTensor)
                            .setcMatDesc(dVTensor_before_dequan_S)
                            .setmOverrideDesc(mnkOverride)
                            .setkOverrideDesc(mnkOverride)
                            .setmatmulDesc(matmulDesc)
                            .build();


    ops.push_back(std::move(reshape_op));
    ops.push_back(std::move(matmulOp));

    return dVTensor_before_dequan_S;
}

static cudnn_frontend::Tensor
createdOVBMM(int64_t b,
           int64_t h,
           int64_t s_q,
           int64_t s_kv,
           int64_t d,
           MHA_Layout layout,
           cudnnDataType_t tensorType,
           std::vector<cudnn_frontend::Operation>& ops,
           const cudnn_frontend::Tensor &dOTensor,
           const cudnn_frontend::Tensor &mnkOverride,
           std::shared_ptr<cudnn_frontend::Tensor>& QKVRaggedOffsetTensor) {
    // Creates the necessary tensor descriptors
    int64_t v_dim [4] =  {b, h, s_kv, d};
    int64_t v_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, v_stride, layout, MHA_Matrix::V_Matrix);

    int64_t v_transpose_dim [4] = {b, h, d, s_kv};
    int64_t v_transpose_stride [4];
    v_transpose_stride[0] = v_stride[0];
    v_transpose_stride[1] = v_stride[1];
    v_transpose_stride[2] = v_stride[3];
    v_transpose_stride[3] = v_stride[2];

    int64_t s_dim [4] = {b, h, s_q, s_kv};
    int64_t s_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, s_stride, layout, MHA_Matrix::S_Matrix);

    auto vTensor = tensor_create_with_offset(tensorType, tensor_name_to_uid["V"], v_dim, v_stride, false, false, QKVRaggedOffsetTensor);
    auto vTransposeTensor = tensor_create_with_offset(tensorType, tensor_name_to_uid["V_TRANSPOSE"], v_transpose_dim, v_transpose_stride, false, false, QKVRaggedOffsetTensor); // is virtual

    // dO * V.T
    auto afterdOVTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 600, s_dim, s_stride, true, false); // is virtual

    // Define the matmul desc
    auto matmulDesc = cudnn_frontend::MatMulDescBuilder()
                                            .setComputeType(CUDNN_DATA_FLOAT)
                                            .setPaddingValue(0)
                                            .build();

    // Create reshape node for V -> V.T
    auto reshape_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                            .setxDesc(vTensor)
                            .setyDesc(vTransposeTensor)
                            .build();


    // Create a matmul Node
    auto matmulOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(dOTensor)
                            .setbMatDesc(vTransposeTensor)
                            .setcMatDesc(afterdOVTensor)
                            .setmOverrideDesc(mnkOverride)
                            .setnOverrideDesc(mnkOverride)
                            .setmatmulDesc(matmulDesc)
                            .build();

    ops.push_back(std::move(reshape_op));
    ops.push_back(std::move(matmulOp));

    return afterdOVTensor;
}

static cudnn_frontend::Tensor
createdOAndORowReductionChain(int64_t b,
           int64_t h,
           int64_t s_q,
           int64_t s_kv,
           int64_t d,
           MHA_Layout layout,
           std::vector<cudnn_frontend::Operation>& ops,
           const cudnn_frontend::Tensor &O_after_dequan,
           const cudnn_frontend::Tensor &dO_after_dequan,
           const cudnn_frontend::Tensor &dropoutScale_dOVt_OdO_Tensor) {

    int64_t o_dim [4] = {b, h, s_q, d};
    int64_t o_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, o_stride, layout, MHA_Matrix::O_Matrix);
    int64_t o_dim_row_sum [4] = {b, h, s_q, 1};
    int64_t o_dim_row_sum_stride [4] = {s_q * h, s_q, 1, 1};

    auto O_dO_after_pointwise_multiply = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 700, o_dim, o_stride, true, false); // is virtual
    auto O_dO_after_dropout_scale = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 701, o_dim, o_stride, true, false); // is virtual
    auto O_dO_after_rowsum = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 702, o_dim_row_sum, o_dim_row_sum_stride, true, false); // is virtual

    // Define the pw multiply descriptor
    auto multiplyDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a multiply Node.
    auto mutliply_op = binary_pw_op_create(O_after_dequan, dO_after_dequan, O_dO_after_pointwise_multiply, multiplyDesc);

    // Create multiply node with dropout scale
    auto dropout_scale_multiply_op = binary_pw_op_create(O_dO_after_pointwise_multiply, dropoutScale_dOVt_OdO_Tensor, O_dO_after_dropout_scale, multiplyDesc);

    // Define the reduction descriptor
    auto reductionAddDesc = cudnn_frontend::ReductionDescBuilder()
                                .setComputeType(CUDNN_DATA_FLOAT)
                                .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                                .build();

    // Create a reduction add Node.
    auto reductionAdd_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(O_dO_after_dropout_scale)
                                .setyDesc(O_dO_after_rowsum)
                                .setreductionDesc(reductionAddDesc)
                                .build();

    ops.push_back(std::move(mutliply_op));
    ops.push_back(std::move(dropout_scale_multiply_op));
    ops.push_back(std::move(reductionAdd_op));

    return O_dO_after_rowsum;
}

static cudnn_frontend::Tensor
createBiasSubtractionSoftmaxMulChain(int64_t b,
           int64_t h,
           int64_t s_q,
           int64_t s_kv,
           int64_t d,
           MHA_Layout layout,
           std::vector<cudnn_frontend::Operation>& ops,
           const cudnn_frontend::Tensor &dS_after_dropout,
           const cudnn_frontend::Tensor &AfterDropout_before_quan_S,
           const cudnn_frontend::Tensor &O_dO_after_rowsum) {
    // TODO: Add dropout
    int64_t o_dim [4] = {b, h, s_q, s_kv};
    int64_t o_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, o_stride, layout, MHA_Matrix::S_Matrix);
    auto dS_minus_O_dO = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 800, o_dim, o_stride, true, false); // is virtual
    auto S_mul_dS_minus_O_dO = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 801, o_dim, o_stride, true, false); // is virtual

    // Define the pw subtraction descriptor
    auto subDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_SUB);

    // Create a subtraction Node.
    auto sub_op = binary_pw_op_create(dS_after_dropout, O_dO_after_rowsum, dS_minus_O_dO, subDesc);

    // Define the pw multiplication descriptor
    auto multiplyDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a multiply Node.
    auto mutliply_op = binary_pw_op_create(AfterDropout_before_quan_S, dS_minus_O_dO, S_mul_dS_minus_O_dO, multiplyDesc);

    ops.push_back(std::move(sub_op));
    ops.push_back(std::move(mutliply_op));

    return S_mul_dS_minus_O_dO;
}

static cudnn_frontend::Tensor
createdSKBMM(int64_t b,
           int64_t h,
           int64_t s_q,
           int64_t s_kv,
           int64_t d,
           std::vector<cudnn_frontend::Operation>& ops,
           const cudnn_frontend::Tensor &dSTensor,
           const cudnn_frontend::Tensor &kTensor,
           const cudnn_frontend::Tensor &mnkOverride) {
    CUDNN_FRONTEND_UNUSED(s_q);
    // Creates the necessary tensor descriptors
    int64_t after_dSK_dim [4] = {b, h, s_kv, d};
    int64_t after_dSK_stride [4] = {h * s_kv * d, d, h * d, 1};
    // dS * K
    auto After_dS_K = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 875, after_dSK_dim, after_dSK_stride, true, false); // is virtual

    // Define the matmul desc
    auto matmulDesc = cudnn_frontend::MatMulDescBuilder()
                                    .setComputeType(CUDNN_DATA_FLOAT)
                                    .setPaddingValue(0)
                                    .build();

    // Create a matmul Node
    auto matmulOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(dSTensor)
                            .setbMatDesc(kTensor)
                            .setcMatDesc(After_dS_K)
                            .setmOverrideDesc(mnkOverride)
                            .setkOverrideDesc(mnkOverride)
                            .setmatmulDesc(matmulDesc)
                            .build();

    ops.push_back(std::move(matmulOp));

    return After_dS_K;
}

static cudnn_frontend::Tensor
createdSQBMM(int64_t b,
           int64_t h,
           int64_t s_q,
           int64_t s_kv,
           int64_t d,
           MHA_Layout layout,
           std::vector<cudnn_frontend::Operation>& ops,
           const cudnn_frontend::Tensor &dSTensor,
           const cudnn_frontend::Tensor &qTensor,
           const cudnn_frontend::Tensor &mnkOverride) {
    // Creates the necessary tensor descriptors
    int64_t dS_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, dS_stride, layout, MHA_Matrix::S_Matrix);

    int64_t dS_transpose_dim [4] = {b, h, s_kv, s_q};
    int64_t dS_transpose_stride [4];
    dS_transpose_stride[0] = dS_stride[0];
    dS_transpose_stride[1] = dS_stride[1];
    dS_transpose_stride[2] = dS_stride[3];
    dS_transpose_stride[3] = dS_stride[2];

    int64_t after_dSTranspose_Q_dim [4] = {b, h, s_kv, d};
    int64_t after_dSTranspose_Q_stride [4] = {h * s_kv * d, d, h * d, 1};

    auto dSTransposeTensor = tensor_create(CUDNN_DATA_FP8_E5M2, tensor_name_to_uid["VIRTUAL"] + 650, dS_transpose_dim, dS_transpose_stride, true, false); // is virtual

    // dS.T * Q
    auto After_dSTranspose_Q = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 651, after_dSTranspose_Q_dim, after_dSTranspose_Q_stride, true, false); // is virtual

    // Create reshape node for V -> V.T
    auto reshape_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                            .setxDesc(dSTensor)
                            .setyDesc(dSTransposeTensor)
                            .build();

    // Define the matmul desc
    auto matmulDesc = cudnn_frontend::MatMulDescBuilder()
                                            .setComputeType(CUDNN_DATA_FLOAT)
                                            .setPaddingValue(0)
                                            .build();

    // Create a matmul Node
    auto matmulOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(dSTransposeTensor)
                            .setbMatDesc(qTensor)
                            .setcMatDesc(After_dSTranspose_Q)
                            .setmOverrideDesc(mnkOverride)
                            .setkOverrideDesc(mnkOverride)
                            .setmatmulDesc(matmulDesc)
                            .build();


    ops.push_back(std::move(reshape_op));
    ops.push_back(std::move(matmulOp));

    return After_dSTranspose_Q;
}

void
fa_cudnn_fp8_fprop(int64_t b,
                int64_t h,
                int64_t s_q,
                int64_t s_kv,
                int64_t d,
                float attnScale,
                bool isTraining,
                float dropoutProbability,
                MHA_Layout layout,
                void* devPtrQKV,
                void* devPtrM,
                void* devPtrZInv,
                void* devPtrO,
                void* devPtrDropoutSeed,
                void* devPtrDropoutOffset,
                void* devPtrDescaleQ,
                void* devPtrDescaleK,
                void* devPtrDescaleV,
                void* devPtrDescaleS,
                void* devPtrScaleS,
                void* devPtrScaleO,
                void* devPtrAmaxO,
                void* devPtrAmaxS,
                void* devPtrQKVRaggedOffset,
                void* devPtrORaggeDOffset,
                void* devPtrMNKOverride,
                cudnnDataType_t tensorType,
		void* workspace_ptr,
		uint64_t* workspace_size)
{
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        NVTE_CHECK_CUDNN(cudnnCreate(&handle_));

	// FP8 BERT Flash Attention only runs on cudnn v8.9 and above and only on Hopper
        if (check_device_arch_newer_than("hopper") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                    nullptr,
                    CUDNN_STATUS_ARCH_MISMATCH,
                    "Run FP8 BERT Flash Attention: Sample requires Hopper or above GPU");
        }

	FADescriptor descriptor{b,
                              h,
                              s_q,
                              s_kv,
                              d,
			      attnScale,
                              dropoutProbability,
                              layout,
                              devPtrDropoutSeed,
                              devPtrDropoutOffset,
                              tensorType};

        using CacheType = std::map<FADescriptor, cudnn_frontend::ExecutionPlan>;
        static CacheType fa_fprop_cache;

        // Get plan from cache if cache is available, otherwise create one
        auto get_plan = [&](CacheType &cache, const FADescriptor &descriptor) {

            // if hit, return
            auto it = cache.find(descriptor);
            if (it != cache.end()) {
              auto plan = it->second;
              return plan;
            }

	    // otherwise, build the op_graph and the plan. Then update cache
            std::vector<cudnn_frontend::Operation const*> all_ops;
            std::vector<cudnn_frontend::Operation> ops;

            cudnn_frontend::throw_if(dropoutProbability != 0.0f && !isTraining, "Dropout probability should be 0.0f for inference mode", CUDNN_STATUS_BAD_PARAM);
            cudnn_frontend::throw_if(dropoutProbability == 1.0f, "Dropout probability cannot be 1.0", CUDNN_STATUS_BAD_PARAM);

            // Ragged tensors have b + 1 elements
            int64_t raggedDim [4] =  {b + 1, 1, 1, 1};
            int64_t raggedStride [4] = {1, 1, 1, 1};
            // Create offset tensors
            auto QKVOffsetTensor = tensor_create(CUDNN_DATA_INT32, tensor_name_to_uid["QKV_RAGGED"], raggedDim, raggedStride, false, false);
            auto ORaggedOffsetTensor = tensor_create(CUDNN_DATA_INT32, tensor_name_to_uid["O_RAGGED"], raggedDim, raggedStride, false, false);

            int64_t seqlen_dim [4] =  {b, 1, 1, 1};
            int64_t seqlen_stride [4] = {1, 1, 1, 1};
            // Create override tensors
            auto seqlenMNKTensor = tensor_create(CUDNN_DATA_INT32, tensor_name_to_uid["MNK_OVERRIDE"], seqlen_dim, seqlen_stride, false, false);

            // Create shared ptrs to ragged offset tensors for multiple tensors to use ragged offset
            std::shared_ptr<cudnn_frontend::Tensor> QKVRaggedOffsetTensorPtr = std::make_shared<cudnn_frontend::Tensor>(std::move(QKVOffsetTensor));
            std::shared_ptr<cudnn_frontend::Tensor> ORaggedOffsetTensorPtr = std::make_shared<cudnn_frontend::Tensor>(std::move(ORaggedOffsetTensor));

            // Create Q and K tensors that are used in different places
            int64_t q_dim [4] = {b, h, s_q, d};
            int64_t q_stride [4];
            generateMHAStrides(b, h, s_q, s_kv, d, q_stride, layout, MHA_Matrix::Q_Matrix);

            int64_t k_dim [4] =  {b, h, s_kv, d};
            int64_t k_stride [4];
            generateMHAStrides(b, h, s_q, s_kv, d, k_stride, layout, MHA_Matrix::K_Matrix);

            auto qTensor = tensor_create_with_offset(tensorType, tensor_name_to_uid["Q"], q_dim, q_stride, false, false, QKVRaggedOffsetTensorPtr);
            auto kTensor = tensor_create_with_offset(tensorType, tensor_name_to_uid["K"], k_dim, k_stride, false, false, QKVRaggedOffsetTensorPtr);

            // Q * K.T
            auto afterQKTensor = createQKBMM(b, h, s_q, s_kv, d, layout, tensorType, ops, qTensor, kTensor, seqlenMNKTensor, QKVRaggedOffsetTensorPtr);

            // QK.T * attn scale
            auto AfterAttnScale_before_dequan_Q_tensor = createScale(afterQKTensor, // input tensor
                                                                    "AttnScale", // scale tensor
                                                                    CUDNN_DATA_FLOAT,  // output tensor type
                                                                    true, // output is virtual
                                                                    true, // scale is by value
                                                                    ops);

            // QK.T * attn scale * dequant_Q
            auto AfterAttnScale_before_dequan_K_tensor = createScale(AfterAttnScale_before_dequan_Q_tensor, // input tensor
                                                                    "descaleQ", // scale tensor
                                                                    CUDNN_DATA_FLOAT, // output tensor type
                                                                    true, // output is virtual
                                                                    false, // scale is by value
                                                                    ops);

            // QK.T * attn scale * dequant_Q * dequant_K
            auto AfterAttnScale_tensor = createScale(AfterAttnScale_before_dequan_K_tensor, // input tensor
                                                    "descaleK", // scale tensor
                                                    CUDNN_DATA_FLOAT, // output tensor type
                                                    true, // output is virtual
                                                    false, // scale is by value
                                                    ops);


            auto BeforeDropoutTensor = createSoftmaxForward(b, h, s_q, s_kv, ops, AfterAttnScale_tensor, isTraining);

            auto AfterDropout_before_quan_S = createDropoutForward(b, h, s_q, s_kv, dropoutProbability, ops, BeforeDropoutTensor);

            // Amax for S
            createAmax("amaxS", BeforeDropoutTensor, ops);

            // After softmax * dropout * scale S -> fp8 input to next bmm with V
            auto AfterMultiplyDropout = createScale(AfterDropout_before_quan_S, // input tensor
                                                    "scaleS", // scale tensor
                                                    tensorType, // output tensor type
                                                    true, // output is virtual
                                                    false, // scale is by value
                                                    ops);

            // After softmax * Dropout * V
            auto OTensor_before_dequan_S_tensor = createSVBMM(b, h, s_q, s_kv, d, layout, tensorType, ops, AfterMultiplyDropout, seqlenMNKTensor, QKVRaggedOffsetTensorPtr);

            // O * dequant_S
            auto OTensor_before_dequan_V_tensor = createScale(OTensor_before_dequan_S_tensor, // input tensor
                                                            "descaleS", // scale tensor
                                                            CUDNN_DATA_FLOAT, // output tensor type
                                                            true, // output is virtual
                                                            false, // scale is by value
                                                            ops);

            // O * dequant_S * dequant_V
            auto OTensor_before_quan_O_tensor = createScale(OTensor_before_dequan_V_tensor, // input tensor
                                                            "descaleV", // scale tensor
                                                            CUDNN_DATA_FLOAT, // output tensor type
                                                            true, // output is virtual
                                                            false, // scale is by value
                                                            ops);

            // O * dequant_S * dequant_V * scale O
            auto OTensor = createScaleWithOffset(OTensor_before_quan_O_tensor, // input tensor
                                "scaleO", // scale tensor
                                tensorType, // output tensor type
                                false, // output not virtual
                                false, // scale is by value
                                ops,
                                ORaggedOffsetTensorPtr, // ragged offset
                                "O");

            // Amax for O
            createAmax("amaxO", OTensor_before_quan_O_tensor, ops);

            for (unsigned int i = 0; i < ops.size(); i++) {
                all_ops.push_back(&ops[i]);
            }

            // Create an Operation Graph
            auto opGraph = cudnn_frontend::OperationGraphBuilder()
                               .setHandle(handle_)
                               .setOperationGraph(all_ops.size(), all_ops.data())
                               .build();


            cudnn_frontend::EngineConfigList filtered_configs;
            auto statuses = cudnn_frontend::get_heuristics_list<1>({"heuristics_instant"}, opGraph, allowAllConfig, filtered_configs, true);

            if (filtered_configs.size() == 0) {
                cudnn_frontend::set_error_and_throw_exception(
                        nullptr,
                        CUDNN_STATUS_NOT_SUPPORTED,
                        "run_mha_fprop: No config returned by the heuristics");
            }

            auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(filtered_configs[0], opGraph.getTag()).build();
	    cache.insert({descriptor, plan});
            return plan;

	}; // end of get_plan

	auto plan = get_plan(fa_fprop_cache, descriptor);
	*workspace_size = static_cast<uint64_t>(plan.getWorkspaceSize());
	if (workspace_ptr == nullptr)
	{
	    if (*workspace_size > 0)
	    {
	        return;
            }
            else if (*workspace_size == 0)
                return;
        }

	// execute if workspace is not nullptr or the plan require 0 workspace 
        void* devPtrQ = (void *) devPtrQKV; // q points to the top of qkv
        void* devPtrK = (void *)(static_cast<int8_t*>(devPtrQKV) + h * d); // k is at an offset of h * d
        void* devPtrV = (void *)(static_cast<int8_t*>(devPtrQKV) + 2 * h * d); // v is at an offset of 2 * h * d

        float dropoutScale = 1.0f/(1.0f - dropoutProbability); // 1 / (1 - p)

        std::set<std::pair<uint64_t, void*>> data_ptrs;
        // add all the data pointers to be used in the variant pack
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["Q"], devPtrQ));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["K"], devPtrK));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["K_TRANSPOSE"], devPtrK));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["V"], devPtrV));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["AttnScale"], &attnScale));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["DROPOUT_SCALE"], &dropoutScale));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["DROPOUT_SEED"], devPtrDropoutSeed));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["DROPOUT_OFFSET"], devPtrDropoutOffset));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["O"], devPtrO));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["descaleQ"], devPtrDescaleQ));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["descaleK"], devPtrDescaleK));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["descaleV"], devPtrDescaleV));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["descaleS"], devPtrDescaleS));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["scaleS"], devPtrScaleS));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["scaleO"], devPtrScaleO));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["amaxO"], devPtrAmaxO));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["amaxS"], devPtrAmaxS));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["QKV_RAGGED"], devPtrQKVRaggedOffset));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["O_RAGGED"], devPtrORaggeDOffset));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["MNK_OVERRIDE"], devPtrMNKOverride));

        // If training, then we need to write out M and Z_INV
        if (isTraining) {
            data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["M"], devPtrM));
            data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["Z_INV"], devPtrZInv));
        }

        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(data_ptrs)
                               .build();
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        NVTE_CHECK_CUDA(cudaDeviceSynchronize());


        NVTE_CHECK_CUDNN(cudnnDestroy(handle_));

        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

    } catch (cudnn_frontend::cudnnException& e) {
        struct cudaDeviceProp prop;
        NVTE_CHECK_CUDA(cudaGetDeviceProperties( &prop, 0 ));

	// this example is only for GH100 cards (cudnn Version >= 8900)
        if (!((prop.major == 9 && prop.minor == 0 && CUDNN_VERSION >= 8900)) && (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH || e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
            std::cout << "Example is only supported for GH100 (cuDNN >= 8900) GPUs" << std::endl;
        }  else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
        }
    }
}

void
fa_cudnn_fp8_bprop(int64_t b,
                int64_t h,
                int64_t s_q,
                int64_t s_kv,
                int64_t d,
                float attnScale,
                float dropoutProbability,
                MHA_Layout layout,
                void* devPtrQKV,
                void* devPtrM,
                void* devPtrZInv,
                void* devPtrO,
                void* devPtrdO,
                void* devPtrdQKV,
                void* devPtrDropoutSeed,
                void* devPtrDropoutOffset,
                void* devPtrDescaleQ,
                void* devPtrDescaleK,
                void* devPtrDescaleV,
                void* devPtrDescaleO,
                void* devPtrDescaledO,
                void* devPtrDescaleS,
                void* devPtrDescaledS,
                void* devPtrScaleS,
                void* devPtrScaledS,
                void* devPtrScaledQ,
                void* devPtrScaledK,
                void* devPtrScaledV,
                void* devPtrAmaxdS,
                void* devPtrAmaxdQ,
                void* devPtrAmaxdK,
                void* devPtrAmaxdV,
                void* devPtrQKVRaggedOffset,
                void* devPtrORaggeDOffset,
                void* devPtrMNKOverride,
                cudnnDataType_t tensorType,
		void* workspace_ptr,
		uint64_t* workspace_size)
{
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        NVTE_CHECK_CUDNN(cudnnCreate(&handle_));

	// FP8 BERT Flash Attention only runs on cudnn v8.9 and above and only on Hopper
        if (check_device_arch_newer_than("hopper") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                    nullptr,
                    CUDNN_STATUS_ARCH_MISMATCH,
                    "Run FP8 BERT Flash Attention: Sample requires Hopper or above GPU");
        }

	FADescriptor descriptor{b,
                              h,
                              s_q,
                              s_kv,
                              d,
			      attnScale,
                              dropoutProbability,
                              layout,
                              devPtrDropoutSeed,
                              devPtrDropoutOffset,
                              tensorType};

	using CacheType = std::map<FADescriptor, cudnn_frontend::ExecutionPlan>;
        static CacheType fa_bprop_cache;

        // Get plan from cache if cache is available, otherwise create one
        auto get_plan = [&](CacheType &cache, const FADescriptor &descriptor) {

            // if hit, return
            auto it = cache.find(descriptor);
            if (it != cache.end()) {
              auto plan = it->second;
              return plan;
            }

	    // otherwise, build the op_graph and the plan. Then update cache
            std::vector<cudnn_frontend::Operation const*> all_ops;
            std::vector<cudnn_frontend::Operation> ops;

            cudnn_frontend::throw_if(dropoutProbability == 1.0f, "Dropout probability cannot be 1.0", CUDNN_STATUS_BAD_PARAM);

            // Ragged tensors have b + 1 elements
            int64_t raggedDim [4] =  {b + 1, 1, 1, 1};
            int64_t raggedStride [4] = {1, 1, 1, 1};
            // Create offset tensors
            auto QKVOffsetTensor = tensor_create(CUDNN_DATA_INT32, tensor_name_to_uid["QKV_RAGGED"], raggedDim, raggedStride, false, false);
            auto ORaggedOffsetTensor = tensor_create(CUDNN_DATA_INT32, tensor_name_to_uid["O_RAGGED"], raggedDim, raggedStride, false, false);

            // Create shared ptrs to ragged offset tensors for multiple tensors to use ragged offset
            std::shared_ptr<cudnn_frontend::Tensor> QKVRaggedOffsetTensorPtr = std::make_shared<cudnn_frontend::Tensor>(std::move(QKVOffsetTensor));
            std::shared_ptr<cudnn_frontend::Tensor> ORaggedOffsetTensorPtr = std::make_shared<cudnn_frontend::Tensor>(std::move(ORaggedOffsetTensor));

            // Create Q and K tensors that are used in different places
            int64_t q_dim [4] = {b, h, s_q, d};
            int64_t q_stride [4];
            generateMHAStrides(b, h, s_q, s_kv, d, q_stride, layout, MHA_Matrix::Q_Matrix);

            int64_t k_dim [4] =  {b, h, s_kv, d};
            int64_t k_stride [4];
            generateMHAStrides(b, h, s_q, s_kv, d, k_stride, layout, MHA_Matrix::K_Matrix);

            auto qTensor = tensor_create_with_offset(tensorType, tensor_name_to_uid["Q"], q_dim, q_stride, false, false, QKVRaggedOffsetTensorPtr);
            auto kTensor = tensor_create_with_offset(tensorType, tensor_name_to_uid["K"], k_dim, k_stride, false, false, QKVRaggedOffsetTensorPtr);

            int64_t scale_dim [4] = {1, 1, 1, 1};
            int64_t scale_stride [4] = {1, 1, 1, 1};

            // Create attnScale tensor for multiple ops to use
            auto attnScaleTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["AttnScale"], scale_dim, scale_stride, false, true); // is by value

            // Create descale Q K dO dS global tensors since they are used in multiple places
            auto descaleQTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["descaleQ"], scale_dim, scale_stride, false, false);
            auto descaleKTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["descaleK"], scale_dim, scale_stride, false, false);
            auto descaledOTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["descaledO"], scale_dim, scale_stride, false, false);
            auto descaledSTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["descaledS"], scale_dim, scale_stride, false, false);

            int64_t seqlen_dim [4] =  {b, 1, 1, 1};
            int64_t seqlen_stride [4] = {1, 1, 1, 1};
            // Create MNK override tensor
            auto seqlenMNKTensor = tensor_create(CUDNN_DATA_INT32, tensor_name_to_uid["MNK_OVERRIDE"], seqlen_dim, seqlen_stride, false, false);

            int64_t O_dim [4] =  {b, h, s_q, d};
            int64_t O_stride [4];
            generateMHAStrides(b, h, s_q, s_kv, d, O_stride, layout, MHA_Matrix::O_Matrix);
            // Create O and loss tensor
            auto OTensor = tensor_create_with_offset(tensorType, tensor_name_to_uid["O"], O_dim, O_stride, false, false, ORaggedOffsetTensorPtr);
            // dO is used in multiple places and E5M2
            auto dOTensor = tensor_create_with_offset(CUDNN_DATA_FP8_E5M2, tensor_name_to_uid["dO"], O_dim, O_stride, false, false, ORaggedOffsetTensorPtr);

            // Q * K.T
            auto afterQKTensor = createQKBMM(b, h, s_q, s_kv, d, layout, tensorType, ops, qTensor, kTensor, seqlenMNKTensor, QKVRaggedOffsetTensorPtr);

            // QK.T * attn scale
            auto AfterAttnScale_before_dequan_Q_tensor = createScale(afterQKTensor, // input tensor
                                                                    attnScaleTensor, // scale tensor
                                                                    CUDNN_DATA_FLOAT,  // output tensor type
                                                                    true, // output is virtual
                                                                    true, // scale is by value
                                                                    ops,
                                                                    1999 /*UID offset*/);

            // QK.T * attn scale * dequant_Q
            auto AfterAttnScale_before_dequan_K_tensor = createScale(AfterAttnScale_before_dequan_Q_tensor, // input tensor
                                                                    descaleQTensor, // scale tensor
                                                                    CUDNN_DATA_FLOAT, // output tensor type
                                                                    true, // output is virtual
                                                                    false, // scale is by value
                                                                    ops,
                                                                    2000 /*UID offset*/);

            // QK.T * attn scale * dequant_Q * dequant_K
            auto AfterAttnScale_tensor = createScale(AfterAttnScale_before_dequan_K_tensor, // input tensor
                                                    descaleKTensor, // scale tensor
                                                    CUDNN_DATA_FLOAT, // output tensor type
                                                    true, // output is virtual
                                                    false, // scale is by value
                                                    ops,
                                                    2001 /*UID offset*/);

            auto beforeDropout_QKt_Tensor = createSoftmaxBackward(b, h, s_q, s_kv, ops, AfterAttnScale_tensor);

            int64_t afterBMM1_dim [4] = {b, h, s_q, s_kv};
            int64_t afterBMM1_stride [4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

            // mask for the dropout. Used in different places
            auto dropoutMaskTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 200, afterBMM1_dim, afterBMM1_stride, true, false); // is virtual

            auto AfterDropout_before_quan_S = createDropoutBackward(b, h, s_q, s_kv, dropoutProbability, ops, beforeDropout_QKt_Tensor, dropoutMaskTensor);

            // After softmax * scale S -> fp8 input to next bmm with V
            auto AfterMultiply = createScale(AfterDropout_before_quan_S, // input tensor
                                                    "scaleS", // scale tensor
                                                    tensorType, // output tensor type
                                                    true, // output is virtual
                                                    false, // scale is by value
                                                    ops);

            // After softmax * dO
            auto dVTensor_before_dequan_S = createSdOBMM(b, h, s_q, s_kv, d, tensorType, ops, AfterMultiply, dOTensor, seqlenMNKTensor);

            // O * dequant_S
            auto dVTensor_before_dequan_dO = createScale(dVTensor_before_dequan_S, // input tensor
                                                            "descaleS", // scale tensor
                                                            CUDNN_DATA_FLOAT, // output tensor type
                                                            true, // output is virtual
                                                            false, // scale is by value
                                                            ops);

            // O * dequant_S * dequant_dO
            auto dVTensor_before_quan_dV = createScale(dVTensor_before_dequan_dO, // input tensor
                                                            descaledOTensor, // scale tensor
                                                            CUDNN_DATA_FLOAT, // output tensor type
                                                            true, // output is virtual
                                                            false, // scale is by value
                                                            ops,
                                                            2002 /*UID offset*/);

            // O * dequant_S * dequant_dO * scale dV
            auto dVTensor = createScaleWithOffset(dVTensor_before_quan_dV, // input tensor
                                "scaledV", // scale tensor
                                CUDNN_DATA_FP8_E5M2, // output tensor type
                                false, // output not virtual
                                false, // scale is by value
                                ops,
                                QKVRaggedOffsetTensorPtr, // ragged offset
                                "dV" /*Output tensor name*/);

            // Amax for dV
            createAmax("amaxdV", dVTensor_before_quan_dV, ops);

            auto dS_before_dequan_dO_Tensor = createdOVBMM(b, h, s_q, s_kv, d, layout, tensorType, ops, dOTensor, seqlenMNKTensor, QKVRaggedOffsetTensorPtr);


            // dS * dequant_dO
            auto dS_before_dequan_V = createScale(dS_before_dequan_dO_Tensor, // input tensor
                                                            descaledOTensor, // scale tensor
                                                            CUDNN_DATA_FLOAT, // output tensor type
                                                            true, // output is virtual
                                                            false, // scale is by value
                                                            ops,
                                                            2003 /*UID offset*/);

            // O * dequant_S * dequant_dV
            auto dS_after_dequan = createScale(dS_before_dequan_V, // input tensor
                                                            "descaleV", // scale tensor
                                                            CUDNN_DATA_FLOAT, // output tensor type
                                                            true, // output is virtual
                                                            false, // scale is by value
                                                            ops);

            // RNG Multiply
            auto beforeDropoutScale_dOVt_Tensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 350, afterBMM1_dim, afterBMM1_stride, true, false); // is virtual
            // After dropout mask and scale
            auto dS_after_dropout = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 351, afterBMM1_dim, afterBMM1_stride, true, false); // is virtual

            // Define the multiply mask descriptor
            auto mulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

            // Create a multiply mask Node.
            auto maskMul_op = binary_pw_op_create(dS_after_dequan, dropoutMaskTensor, beforeDropoutScale_dOVt_Tensor, mulDesc);

            ops.push_back(std::move(maskMul_op));

            // scale after dropout for dO and O chain
            auto dropoutScale_dOVt_OdO_Tensor = tensor_create(tensorType, tensor_name_to_uid["DROPOUT_SCALE_dOVt_OdO"], scale_dim, scale_stride, false, true); // is by value

            // Create a multiply dropout scale Node.
            auto mul_dropout_scale_op = binary_pw_op_create(beforeDropoutScale_dOVt_Tensor, dropoutScale_dOVt_OdO_Tensor, dS_after_dropout, mulDesc);

            ops.push_back(std::move(mul_dropout_scale_op));

            // O * dequant_O
            auto O_after_dequan_Tensor = createScale(OTensor, // input tensor
                                            "descaleO", // scale tensor
                                            CUDNN_DATA_FLOAT, // output tensor type
                                            true, // output is virtual
                                            false, // scale is by value
                                            ops);

            // dO * dequant_dO
            auto dO_after_dequan_Tensor = createScale(dOTensor, // input tensor
                                            descaledOTensor, // scale tensor
                                            CUDNN_DATA_FLOAT, // output tensor type
                                            true, // output is virtual
                                            false, // scale is by value
                                            ops,
                                            2004 /*UID offset*/);


            // row reduction sum[(dO * dequant_dO) * (O * dequant_O) * (1 - p)]
            auto O_dO_after_rowsum = createdOAndORowReductionChain(b, h, s_q, s_kv, d, layout, ops, O_after_dequan_Tensor, dO_after_dequan_Tensor, dropoutScale_dOVt_OdO_Tensor);

            // (dS_after_dropout - O_dO_after_rowsum) * AfterDropout_before_quan_S
            auto S_mul_dS_minus_O_dO = createBiasSubtractionSoftmaxMulChain(b, h, s_q, s_kv, d, layout, ops, dS_after_dropout, AfterDropout_before_quan_S, O_dO_after_rowsum);



            // S_mul_dS_minus_O_dO * scaledS
            auto S_mul_dS_minus_O_dO_after_quan_dS = createScale(S_mul_dS_minus_O_dO, // input tensor
                                                                "scaledS", // scale tensor
                                                                CUDNN_DATA_FP8_E5M2, // output tensor type
                                                                true, // output is virtual
                                                                false, // scale is by value
                                                                ops);

            // Amax for dS
            createAmax("amaxdS", S_mul_dS_minus_O_dO, ops);

            // dS @ K
            auto After_dS_K = createdSKBMM(b, h, s_q, s_kv, d, ops, S_mul_dS_minus_O_dO_after_quan_dS, kTensor, seqlenMNKTensor);


            // (dS * K) * attn scale ds_K
            auto AfterAttnScale_dS_K_before_dequan_dS = createScale(After_dS_K, // input tensor
                                            attnScaleTensor, // scale tensor
                                            CUDNN_DATA_FLOAT, // output tensor type
                                            true, // output is virtual
                                            true, // scale is by value
                                            ops,
                                            2005 /*UID offset*/);

            // (dS * K) * attn scale ds_K * descale dS
            auto AfterAttnScale_dS_K_before_dequan_K = createScale(AfterAttnScale_dS_K_before_dequan_dS, // input tensor
                                            descaledSTensor, // scale tensor
                                            CUDNN_DATA_FLOAT, // output tensor type
                                            true, // output is virtual
                                            false, // scale is by value
                                            ops,
                                            2006 /*UID offset*/);

            // (dS * K) * attn scale ds_K * descale dS * descale K
            auto AfterAttnScale_dS_K_before_quan_dQ = createScale(AfterAttnScale_dS_K_before_dequan_K, // input tensor
                                            descaleKTensor, // scale tensor
                                            CUDNN_DATA_FLOAT, // output tensor type
                                            true, // output is virtual
                                            false, // scale is by value
                                            ops,
                                            2007 /*UID offset*/);

            // (dS * K) * attn scale ds_K * descale dS * descale K * scale dQ
            auto dQ = createScaleWithOffset(AfterAttnScale_dS_K_before_quan_dQ, // input tensor
                                "scaledQ", // scale tensor
                                CUDNN_DATA_FP8_E5M2, // output tensor type
                                false, // output not virtual
                                false, // scale is by value
                                ops,
                                QKVRaggedOffsetTensorPtr, // ragged offset
                                "dQ");

            // Amax for dQ
            createAmax("amaxdQ", AfterAttnScale_dS_K_before_quan_dQ, ops);

            // dS.T @ Q
            auto After_dSTranspose_Q = createdSQBMM(b, h, s_q, s_kv, d, layout, ops, S_mul_dS_minus_O_dO_after_quan_dS, qTensor, seqlenMNKTensor);

            // (dS.T @ Q) * attn scale ds.T_Q
            auto AfterAttnScale_dSTranspose_Q_before_dequan_dS = createScale(After_dSTranspose_Q, // input tensor
                                            attnScaleTensor, // scale tensor
                                            CUDNN_DATA_FLOAT, // output tensor type
                                            true, // output is virtual
                                            true, // scale is by value
                                            ops,
                                            2008 /*UID offset*/);

            // (dS.T * Q) * attn scale dS.T_Q * descale dS
            auto AfterAttnScale_dSTranspose_Q_before_dequan_Q = createScale(AfterAttnScale_dSTranspose_Q_before_dequan_dS, // input tensor
                                            descaledSTensor, // scale tensor
                                            CUDNN_DATA_FLOAT, // output tensor type
                                            true, // output is virtual
                                            false, // scale is by value
                                            ops,
                                            2009 /*UID offset*/);

            // (dS.T * Q) * attn scale dS.T_Q * descale dS * descale Q
            auto AfterAttnScale_dSTranspose_Q_before_quan_dK = createScale(AfterAttnScale_dSTranspose_Q_before_dequan_Q, // input tensor
                                            descaleQTensor, // scale tensor
                                            CUDNN_DATA_FLOAT, // output tensor type
                                            true, // output is virtual
                                            false, // scale is by value
                                            ops,
                                            2010 /*UID offset*/);

            // (dS.T * Q) * attn scale dS.T_Q * descale dS * descale Q * scale dK
            auto dK = createScaleWithOffset(AfterAttnScale_dSTranspose_Q_before_quan_dK, // input tensor
                                "scaledK", // scale tensor
                                CUDNN_DATA_FP8_E5M2, // output tensor type
                                false, // output not virtual
                                false, // scale is by value
                                ops,
                                QKVRaggedOffsetTensorPtr, // ragged offset
                                "dK");

            // Amax for dK
            createAmax("amaxdK", AfterAttnScale_dSTranspose_Q_before_quan_dK, ops);

            for (unsigned int i = 0; i < ops.size(); i++) {
                all_ops.push_back(&ops[i]);
            }

            // Create an Operation Graph
            auto opGraph = cudnn_frontend::OperationGraphBuilder()
                               .setHandle(handle_)
                               .setOperationGraph(all_ops.size(), all_ops.data())
                               .build();


            cudnn_frontend::EngineConfigList filtered_configs;
            auto statuses = cudnn_frontend::get_heuristics_list<1>({"heuristics_instant"}, opGraph, allowAllConfig, filtered_configs, true);

            if (filtered_configs.size() == 0) {
                cudnn_frontend::set_error_and_throw_exception(
                        nullptr,
                        CUDNN_STATUS_NOT_SUPPORTED,
                        "run_mha_bprop: No config returned by the heuristics");
            }

            auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(filtered_configs[0], opGraph.getTag()).build();
	    cache.insert({descriptor, plan});
            return plan;
	};

	auto plan = get_plan(fa_bprop_cache, descriptor);
	*workspace_size = static_cast<uint64_t>(plan.getWorkspaceSize());

	if (workspace_ptr == nullptr)
	{
	    if (*workspace_size > 0)
	        return;
            else if (*workspace_size == 0)
                return;
        }

	// execute if workspace is not nullptr or the plan require 0 workspace 
        void* devPtrQ = (void *) devPtrQKV; // q points to the top of qkv
        void* devPtrK = (void *)(static_cast<int8_t*>(devPtrQKV) + h * d); // k is at an offset of h * d
        void* devPtrV = (void *)(static_cast<int8_t*>(devPtrQKV) + 2 * h * d); // v is at an offset of 2 * h * d
    
        void* devPtrdQ = (void *) devPtrdQKV; // dQ points to the top of dQKV
        void* devPtrdK = (void *)(static_cast<int8_t*>(devPtrdQKV) + h * d); // dK is at an offset of h * d
        void* devPtrdV = (void *)(static_cast<int8_t*>(devPtrdQKV) + 2 * h * d); // dV is at an offset of 2 * h * d
    
        std::set<std::pair<uint64_t, void*>> data_ptrs;
        float dropoutScale = 1.0f/(1.0f - dropoutProbability); // 1 / (1 - p)
        float dropoutScale_dOVt_OdO = 1.0f - dropoutProbability; // (1 - p)
        // add all the data pointers to be used in the variant pack
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["Q"], devPtrQ));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["K"], devPtrK));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["K_TRANSPOSE"], devPtrK));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["V"], devPtrV));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["V_TRANSPOSE"], devPtrV));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["dQ"], devPtrdQ));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["dK"], devPtrdK));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["dV"], devPtrdV));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["dO"], devPtrdO));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["AttnScale"], &attnScale));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["DROPOUT_SCALE"], &dropoutScale));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["DROPOUT_SCALE_dOVt_OdO"], &dropoutScale_dOVt_OdO));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["DROPOUT_SEED"], devPtrDropoutSeed));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["DROPOUT_OFFSET"], devPtrDropoutOffset));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["M"], devPtrM));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["Z_INV"], devPtrZInv));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["O"], devPtrO));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["descaleQ"], devPtrDescaleQ));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["descaleK"], devPtrDescaleK));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["descaleV"], devPtrDescaleV));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["descaleS"], devPtrDescaleS));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["descaledS"], devPtrDescaledS));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["descaleO"], devPtrDescaleO));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["descaledO"], devPtrDescaledO));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["scaleS"], devPtrScaleS));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["scaledS"], devPtrScaledS));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["scaledQ"], devPtrScaledQ));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["scaledK"], devPtrScaledK));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["scaledV"], devPtrScaledV));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["amaxdS"], devPtrAmaxdS));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["amaxdQ"], devPtrAmaxdQ));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["amaxdK"], devPtrAmaxdK));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["amaxdV"], devPtrAmaxdV));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["QKV_RAGGED"], devPtrQKVRaggedOffset));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["O_RAGGED"], devPtrORaggeDOffset));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["MNK_OVERRIDE"], devPtrMNKOverride));
    
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(data_ptrs)
                               .build();
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        NVTE_CHECK_CUDA(cudaDeviceSynchronize());

        NVTE_CHECK_CUDNN(cudnnDestroy(handle_));
    
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

    } catch (cudnn_frontend::cudnnException& e) {
        struct cudaDeviceProp prop;
        NVTE_CHECK_CUDA(cudaGetDeviceProperties( &prop, 0 ));

	// this example is only for GH100 cards (cudnn Version >= 8900)
        if (!((prop.major == 9 && prop.minor == 0 && CUDNN_VERSION >= 8900)) && (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH || e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
            std::cout << "Example is only supported for GH100 (cuDNN >= 8900) GPUs" << std::endl;
        }  else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
        }
    }
}

#endif

} // namespace fused_attn

void fused_attn_fwd(int64_t b, int64_t max_seq_len,
                int64_t total_seqs, int64_t h, int64_t d,
                float attn_scale, float p_dropout,
		int qkv_layout, bool is_training,
		const Tensor *input_QKV,
		Tensor *input_M,
                Tensor *input_ZInv,
                Tensor *input_S,
                Tensor *input_O,
		int32_t *QKVRaggedOffset,
                int32_t *ORaggedOffset,
		int32_t *Seqlens,
		uint64_t *RngState,
                Tensor *workspace,
                cudaStream_t stream)
{
  void* devPtrQKV = input_QKV->data.dptr;
  void* devPtrDescaleQ = input_QKV->scale_inv.dptr;
  void* devPtrDescaleK = input_QKV->scale_inv.dptr;
  void* devPtrDescaleV = input_QKV->scale_inv.dptr;

  void* devPtrM = input_M->data.dptr;
  void* devPtrZInv = input_ZInv->data.dptr;

  void* devPtrAmaxS = input_S->amax.dptr;
  void* devPtrScaleS = input_S->scale.dptr;
  void* devPtrDescaleS = input_S->scale_inv.dptr;

  void* devPtrO = input_O->data.dptr;
  void* devPtrAmaxO = input_O->amax.dptr;
  void* devPtrScaleO = input_O->scale.dptr;
//  void* devPtrDescaleO = input_O->scale_inv.dptr;

  void* devPtrQKVRaggedOffset = (void *)QKVRaggedOffset;
  void* devPtrORaggedOffset = (void *)ORaggedOffset;
  void* devPtrDropoutSeed = (void *)RngState;
  void* devPtrDropoutOffset = (void *)(RngState + 1);
  void* devPtrMNKOverride = (void *)Seqlens;

  const DType QKV_type = input_QKV->data.dtype;
  MHA_Layout layout = get_mha_layout(qkv_layout);
  uint64_t workspace_size = 0;

  if (((QKV_type == DType::kFloat8E4M3) || (QKV_type == DType::kFloat8E5M2)) && (max_seq_len <= 512))
  {
#if (CUDNN_VERSION >= 8900)
    fused_attn::fa_cudnn_fp8_fprop(b, h, max_seq_len, max_seq_len, d,
		    attn_scale, is_training, p_dropout, layout,
		    devPtrQKV, devPtrM, devPtrZInv, devPtrO,
		    devPtrDropoutSeed, devPtrDropoutOffset,
		    devPtrDescaleQ, devPtrDescaleK, devPtrDescaleV,
		    devPtrDescaleS, 
		    devPtrScaleS, 
		    devPtrScaleO,
		    devPtrAmaxO, devPtrAmaxS,
		    devPtrQKVRaggedOffset, devPtrORaggedOffset,
		    devPtrMNKOverride,
		    get_cudnn_dtype(QKV_type),
		    workspace->data.dptr,
		    &workspace_size);

    if (workspace_size > 0)
    {
      if (workspace->data.dptr == nullptr)
      {
        workspace->data.shape = { workspace_size };
        workspace->data.dtype = DType::kByte;
        return;
      }
    }
    else if (workspace_size == 0)
    {
      workspace->data.shape = { 1 };
      workspace->data.dtype = DType::kByte;
      return;
    }
#else
    printf("Error: CUDNN_VERSION must be >= 8900! \n");
#endif

  }
  else if (((QKV_type == DType::kFloat16) || (QKV_type == DType::kBFloat16)) && (max_seq_len <= 512))
  {
    printf("TO DO: call bf16/fp16 version of cudnn flash attn fwd \n");
  }
  else if (max_seq_len > 512)
  {
    printf("TO DO: call Julien version of cudnn flash attn fwd \n");
  }
  else
  {
    NVTE_ERROR("Invalid combination of data type and sequence length! \n");
  }

}

void fused_attn_bwd(int64_t b, int64_t max_seq_len,
                int64_t total_seqs, int64_t h, int64_t d,
                float attn_scale, float p_dropout, int qkv_layout,
		const Tensor *input_QKV,
		Tensor *output_dQKV,
		const Tensor *input_M,
                const Tensor *input_ZInv,
                const Tensor *input_S,
                Tensor *output_dS,
                const Tensor *input_O,
                const Tensor *input_dO,
		int32_t *QKVRaggedOffset,
                int32_t *ORaggedOffset,
                int32_t *Seqlens,
		uint64_t *RngState,
                Tensor *workspace,
                cudaStream_t stream
) {
  void* devPtrQKV = input_QKV->data.dptr;
  void* devPtrDescaleQ = input_QKV->scale_inv.dptr;
  void* devPtrDescaleK = input_QKV->scale_inv.dptr;
  void* devPtrDescaleV = input_QKV->scale_inv.dptr;

  void* devPtrdQKV = output_dQKV->data.dptr;
  void* devPtrAmaxdQ = output_dQKV->amax.dptr;
  void* devPtrAmaxdK = output_dQKV->amax.dptr;
  void* devPtrAmaxdV = output_dQKV->amax.dptr;
  void* devPtrScaledQ = output_dQKV->scale.dptr;
  void* devPtrScaledK = output_dQKV->scale.dptr;
  void* devPtrScaledV = output_dQKV->scale.dptr;
//  void* devPtrDescaledQ = output_dQKV->scale_inv.dptr;
//  void* devPtrDescaledK = output_dQKV->scale_inv.dptr;
//  void* devPtrDescaledV = output_dQKV->scale_inv.dptr;

  void* devPtrM = input_M->data.dptr;
  void* devPtrZInv = input_ZInv->data.dptr;

  void* devPtrScaleS = input_S->scale.dptr;
  void* devPtrDescaleS = input_S->scale_inv.dptr;
  void* devPtrAmaxdS = output_dS->amax.dptr;
  void* devPtrScaledS = output_dS->scale.dptr;
  void* devPtrDescaledS = output_dS->scale_inv.dptr;

  void* devPtrO = input_O->data.dptr;
  void* devPtrDescaleO = input_O->scale_inv.dptr;
  void* devPtrdO = input_dO->data.dptr;
  void* devPtrDescaledO = input_dO->scale_inv.dptr;

  void* devPtrQKVRaggedOffset = (void *)QKVRaggedOffset;
  void* devPtrORaggedOffset = (void *)ORaggedOffset;
  void* devPtrDropoutSeed = (void *)RngState;
  void* devPtrDropoutOffset = (void *)(RngState + 1);
  void* devPtrMNKOverride = (void *)Seqlens; //nullptr;

  const DType QKV_type = input_QKV->data.dtype;
  MHA_Layout layout = get_mha_layout(qkv_layout);
  uint64_t workspace_size = 0;

  if (((QKV_type == DType::kFloat8E4M3) || (QKV_type == DType::kFloat8E5M2)) && (max_seq_len <= 512))
  {
#if (CUDNN_VERSION >= 8900)
    fused_attn::fa_cudnn_fp8_bprop(b, h, max_seq_len, max_seq_len, d,
		    attn_scale, p_dropout, layout,
		    devPtrQKV, devPtrM, devPtrZInv,
		    devPtrO, devPtrdO, devPtrdQKV,
		    devPtrDropoutSeed, devPtrDropoutOffset,
		    devPtrDescaleQ, devPtrDescaleK, devPtrDescaleV,
		    devPtrDescaleO, devPtrDescaledO,
		    devPtrDescaleS, devPtrDescaledS,
		    devPtrScaleS, devPtrScaledS,
		    devPtrScaledQ, devPtrScaledK, devPtrScaledV,
		    devPtrAmaxdS,
		    devPtrAmaxdQ, devPtrAmaxdK, devPtrAmaxdV,
		    devPtrQKVRaggedOffset, devPtrORaggedOffset,
		    devPtrMNKOverride,
		    get_cudnn_dtype(QKV_type),
		    workspace->data.dptr,
		    &workspace_size);

    if (workspace_size > 0)
    {
      if (workspace->data.dptr == nullptr)
      {
        workspace->data.shape = { workspace_size };
        workspace->data.dtype = DType::kByte;
        return;
      }
    }
    else if (workspace_size == 0)
    {
      workspace->data.shape = { 1 };
      workspace->data.dtype = DType::kByte;
      return;
    }
#else
    printf("Error: CUDNN_VERSION must be >= 8900! \n");
#endif

  }
  else if (((QKV_type == DType::kFloat16) || (QKV_type == DType::kBFloat16)) && (max_seq_len <= 512))
  {
    printf("TO DO: call bf16/fp16 version of cudnn flash attn bwd \n");
  }
  else if (max_seq_len > 512)
  {
    printf("TO DO: call Julien version of cudnn flash attn bwd \n");
  }
  else
  {
    NVTE_ERROR("Invalid combination of data type and sequence length! \n");
  }

}

} // namespace transformer_engine

void nvte_fused_attn_fwd(
		int64_t b, int64_t max_seq_len,
                int64_t total_seqs, int64_t h, int64_t d,
                float attn_scale, float p_dropout,
		int qkv_layout, bool is_training,
		const NVTETensor QKV,
		NVTETensor M,
                NVTETensor ZInv,
                NVTETensor S,
                NVTETensor O,
		int32_t *QKVRaggedOffset,
                int32_t *ORaggedOffset,
                int32_t *Seqlens,
                uint64_t *RngState,
		NVTETensor workspace,
		cudaStream_t stream)
{
  NVTE_API_CALL(nvte_cudnn_flash_attn_fwd);
  using namespace transformer_engine;
  const Tensor *input_QKV = reinterpret_cast<const Tensor*>(QKV);
  Tensor *output_M = reinterpret_cast<Tensor*>(M);
  Tensor *output_ZInv = reinterpret_cast<Tensor*>(ZInv);
  Tensor *output_S = reinterpret_cast<Tensor*>(S);
  Tensor *output_O = reinterpret_cast<Tensor*>(O);
  Tensor *wkspace = reinterpret_cast<Tensor*>(workspace);

  fused_attn_fwd(b, max_seq_len, total_seqs, h, d,
		  attn_scale, p_dropout, qkv_layout, is_training,
		  input_QKV, output_M, output_ZInv, output_S, output_O,
		  QKVRaggedOffset, ORaggedOffset, Seqlens,
		  RngState, wkspace, stream);
}

void nvte_fused_attn_bwd(
		int64_t b, int64_t max_seq_len,
                int64_t total_seqs, int64_t h, int64_t d,
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
		cudaStream_t stream)
{
  NVTE_API_CALL(nvte_cudnn_flash_attn_bwd);
  using namespace transformer_engine;
  const Tensor *input_QKV = reinterpret_cast<const Tensor*>(QKV);
  Tensor *output_dQKV = reinterpret_cast<Tensor*>(dQKV);
  const Tensor *input_M = reinterpret_cast<const Tensor*>(M);
  const Tensor *input_ZInv = reinterpret_cast<const Tensor*>(ZInv);
  const Tensor *input_S = reinterpret_cast<const Tensor*>(S);
  Tensor *output_dS = reinterpret_cast<Tensor*>(dS);
  const Tensor *input_O = reinterpret_cast<const Tensor*>(O);
  const Tensor *input_dO = reinterpret_cast<const Tensor*>(dO);
  Tensor *wkspace = reinterpret_cast<Tensor*>(workspace);

  fused_attn_bwd(b, max_seq_len, total_seqs, h, d, attn_scale, p_dropout, qkv_layout,
		  input_QKV, output_dQKV, input_M, input_ZInv, input_S, output_dS, input_O, input_dO,
		  QKVRaggedOffset, ORaggedOffset, Seqlens,
		  RngState, wkspace, stream);
}