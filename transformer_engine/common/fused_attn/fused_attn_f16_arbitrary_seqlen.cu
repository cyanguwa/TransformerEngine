/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "fused_attn_f16_arbitrary_seqlen.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend_utils.h>
#include <map>
#include <vector>

#include "../common.h"
#include "utils.h"
#include "../util/cuda_runtime.h"
#include "../util/system.h"

#if (CUDNN_VERSION >= 8900)
#define Q_ID 1
#define K_ID 2
#define V_ID 3
#define O_ID 4
#define S_ID 5
#define B_ID 6
#define D_CONST_ID 7
#define S_CONST_ID 8
#define Q_SEQLEN_ID 9
#define K_SEQLEN_ID 10
#define dQ_ID 11
#define dK_ID 12
#define dV_ID 13
#define dO_ID 14
#define MASK_VAL_ID 15
#define dS_ID 16
#define D_SEED_ID 17
#define D_OFFSET_ID 18
#define S_STATS_ID 19
#define S_SUM_ID 20
#define SCALE_PROB 21
#define K_TRANSPOSE_ID 22
#define dQ_ACCUM_ID 23

#define VIRTUAL_ID 30

namespace transformer_engine {
namespace fused_attn {

static cudnn_frontend::Tensor
createScale(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
            NVTE_QKV_Layout layout, cudnnDataType_t tensorType,
            const cudnn_frontend::Tensor& sTensor,
            std::vector<cudnn_frontend::Operation>* ops) {
    // scale
    int64_t scale_dim[4] = {1, 1, 1, 1};
    int64_t scale_stride[4] = {1, 1, 1, 1};

    int64_t s_dim[4] =  {b, h, s_q, s_kv};
    int64_t s_stride[4];
    generateMatrixStrides(b, h, s_q, s_kv, d, s_stride, layout, NVTE_QKV_Matrix::NVTE_S_Matrix);

    auto scaleTensor = tensor_create(
                       tensorType, S_CONST_ID, scale_dim,
                       scale_stride, false, true);  // is by value
    auto sScaleTensor = tensor_create(
                        tensorType, VIRTUAL_ID + 2000, s_dim,
                        s_stride, true, false);  // is virtual

    // Define the scale descriptor
    auto scaleDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a scale node
    auto scale_op = binary_pw_op_create(sTensor, scaleTensor, sScaleTensor, scaleDesc);

    ops->push_back(std::move(scale_op));
    return sScaleTensor;
}

static cudnn_frontend::Tensor
createQKBMM(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
           NVTE_QKV_Layout layout, cudnnDataType_t tensorType,
           std::vector<cudnn_frontend::Operation>* ops) {
    // Creates the necessary tensor descriptors
    int64_t q_dim[4] = {b, h, s_q, d};
    int64_t q_stride[4];
    generateMatrixStrides(b, h, s_q, s_kv, d, q_stride, layout, NVTE_QKV_Matrix::NVTE_Q_Matrix);

    int64_t k_dim[4] =  {b, h, d, s_kv};
    int64_t k_stride[4];
    generateMatrixStrides(
            b, h, s_q, s_kv, d, k_stride, layout, NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose);

    int64_t s_dim[4] = {b, h, s_q, s_kv};
    int64_t s_stride[4];
    generateMatrixStrides(b, h, s_q, s_kv, d, s_stride, layout, NVTE_QKV_Matrix::NVTE_S_Matrix);

    auto qTensor = tensor_create(tensorType, Q_ID, q_dim, q_stride, false, false);
    auto kTransposeTensor = tensor_create(
                            tensorType, K_ID, k_dim, k_stride, false, false);  // is virtual
    // first GEMM output
    auto sTensor = tensor_create(
                   CUDNN_DATA_FLOAT, VIRTUAL_ID + 1, s_dim, s_stride, true, false);  // is virtual

    // Define the matmul 1 desc
    auto matmul_1_Desc = cudnn_frontend::MatMulDescBuilder()
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();

    // Create a matmul 1 node
    auto matmul_op1 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(qTensor)
                            .setbMatDesc(kTransposeTensor)
                            .setcMatDesc(sTensor)
                            .setmatmulDesc(matmul_1_Desc)
                            .build();

    ops->push_back(std::move(matmul_op1));

    return sTensor;
}

static cudnn_frontend::Tensor
createCausalMask(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
           NVTE_QKV_Layout layout, cudnnDataType_t tensorType,
           std::vector<cudnn_frontend::Operation>* ops,
           const cudnn_frontend::Tensor& prevBlockOutputTensor) {
    CUDNN_FRONTEND_UNUSED(d);
    CUDNN_FRONTEND_UNUSED(layout);
    CUDNN_FRONTEND_UNUSED(tensorType);

    NVTE_CHECK(ops->size() != 0, "Padding Mask constructed incorrectly as the first one");

    // subtraction output
    int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
    int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    int64_t maskVal_dim[4] =  {1, 1, 1, 1};
    int64_t maskVal_stride[4] = {1, 1, 1, 1};

    // mask value to put in the masked pixels
    auto maskValTensor = tensor_create(
                            CUDNN_DATA_FLOAT, MASK_VAL_ID, maskVal_dim,
                            maskVal_stride, false, true);  // is by value
    // gen index row output
    auto rowIndexTensor = tensor_create(
                            CUDNN_DATA_FLOAT, VIRTUAL_ID + 100, afterBMM1_dim,
                            afterBMM1_stride, true, false);  // is virtual
    // gen index column output
    auto columnIndexTensor = tensor_create(
                            CUDNN_DATA_FLOAT, VIRTUAL_ID + 101, afterBMM1_dim,
                            afterBMM1_stride, true, false);  // is virtual
    // create causal mask (row >= col)
    auto causalMaskTensor = tensor_create(
                            CUDNN_DATA_BOOLEAN, VIRTUAL_ID + 106, afterBMM1_dim,
                            afterBMM1_stride, true, false);  // is virtual

    // output after masking
    auto maskOutputTensor = tensor_create(
                            CUDNN_DATA_FLOAT, VIRTUAL_ID + 107, afterBMM1_dim,
                            afterBMM1_stride, true, false);  // is virtual

    // Define the gen index for row descriptor
    auto genIndexRowDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_GEN_INDEX)
                            .setAxis(2)
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();

    // Create a gen index node
    auto genIndexRow_op = unary_pw_op_create(
                            prevBlockOutputTensor, rowIndexTensor, genIndexRowDesc);

    // Define the gen index for row descriptor
    auto genIndexColumnDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_GEN_INDEX)
                            .setAxis(3)
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();

    // Create a gen index node
    auto genIndexColumn_op = unary_pw_op_create(
                            prevBlockOutputTensor, columnIndexTensor, genIndexColumnDesc);

    // Define the greater than equal to comparison descriptor
    auto rowGreaterColDesc = pw_desc_create(CUDNN_DATA_BOOLEAN, CUDNN_POINTWISE_CMP_GE);

    // Create a greater than equal to node
    auto rowGreaterCol_op = binary_pw_op_create(
                            rowIndexTensor, columnIndexTensor, causalMaskTensor, rowGreaterColDesc);

    // Define the binary select to perform masking descriptor
    auto maskDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_BINARY_SELECT);

    // Create a binary select node
    auto mask_op = ternary_pw_op_create(
                            prevBlockOutputTensor, maskValTensor,
                            causalMaskTensor, maskOutputTensor, maskDesc);

    ops->push_back(std::move(genIndexRow_op));
    ops->push_back(std::move(genIndexColumn_op));
    ops->push_back(std::move(rowGreaterCol_op));
    ops->push_back(std::move(mask_op));

    return maskOutputTensor;
}

static cudnn_frontend::Tensor
createSoftmaxForward(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, bool isTraining,
                     std::vector<cudnn_frontend::Operation>* ops,
                     const cudnn_frontend::Tensor& sAfterMaskTensor) {
    int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
    int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    int64_t afterReduction_dim[4] = {b, h, s_q, 1};
    int64_t afterReduction_stride[4] = {h * s_q, s_q, 1, 1};

    // max (x)
    auto afterMaxReductionTensor = tensor_create(
                            CUDNN_DATA_FLOAT, VIRTUAL_ID + 150, afterReduction_dim,
                            afterReduction_stride, true, false);  // is virtual

    // x - max(x)
    auto afterSubtractionTensor = tensor_create(
                            CUDNN_DATA_FLOAT, VIRTUAL_ID + 151, afterBMM1_dim,
                            afterBMM1_stride, true, false);  // is virtual

    // e^(x - max(x))
    auto afterExponentTensor = tensor_create(
                            CUDNN_DATA_FLOAT, VIRTUAL_ID + 152, afterBMM1_dim,
                            afterBMM1_stride, true, false);  // is virtual;

    // sum (e^(x - max(x)))
    auto afterAddReductionTensor = tensor_create(
                            CUDNN_DATA_FLOAT, VIRTUAL_ID + 153, afterReduction_dim,
                            afterReduction_stride, true, false);  // is virtual

    // log (sum (e^(x - max(x))))
    auto afterLogLTensor = tensor_create(
                            CUDNN_DATA_FLOAT, VIRTUAL_ID + 154, afterReduction_dim,
                            afterReduction_stride, true, false);

    // M + log (sum (e^(x - max(x))))
    auto softmaxStatsTensor = tensor_create(
                            CUDNN_DATA_FLOAT, S_STATS_ID, afterReduction_dim,
                            afterReduction_stride, !isTraining, false);
                            // not virtual if training is true, virtual if training is false

    // divide (e/ sum(e))
    auto afterSoftmaxTensor = cudnn_frontend::TensorBuilder()
            .setDim(4, afterBMM1_dim)
            .setStride(4, afterBMM1_stride)
            .setId(VIRTUAL_ID + 156)
            .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
            .setDataType(CUDNN_DATA_FLOAT)
            .setVirtual(true)
            .setByValue(false)
            .setReorderType(
                cudnn_frontend::TensorReordering_t::F16x16)
                //cudnn_frontend::cudnnBackendTensorReordering_t::CUDNN_TENSOR_REORDERING_F16x16)
            .build();

    // Define the reduction descriptor
    auto reductionMaxDesc = cudnn_frontend::ReductionDescBuilder()
                                .setComputeType(CUDNN_DATA_FLOAT)
                                .setReductionOp(CUDNN_REDUCE_TENSOR_MAX)
                                .build();

    // Create a reduction max node
    auto reductionMax_op = cudnn_frontend::OperationBuilder(
                                CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(sAfterMaskTensor)
                                .setyDesc(afterMaxReductionTensor)
                                .setreductionDesc(reductionMaxDesc)
                                .build();

    // Define the subtract descriptor
    auto subtractDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_SUB);

    // Create a subtract node
    auto subtract_op = binary_pw_op_create(
                                sAfterMaskTensor, afterMaxReductionTensor,
                                afterSubtractionTensor, subtractDesc);

    // Define the exponent descriptor
    auto exponentDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_EXP);

    // Create a exponent node
    auto exponent_op = unary_pw_op_create(
                                afterSubtractionTensor, afterExponentTensor, exponentDesc);

    // Define the reduction descriptor
    auto reductionAddDesc = cudnn_frontend::ReductionDescBuilder()
                                .setComputeType(CUDNN_DATA_FLOAT)
                                .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                                .build();

    // Create a reduction add node
    auto reductionAdd_op = cudnn_frontend::OperationBuilder(
                                CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(afterExponentTensor)
                                .setyDesc(afterAddReductionTensor)
                                .setreductionDesc(reductionAddDesc)
                                .build();

    // Create log descriptor
    auto logDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_LOG);

    // Create log node
    auto log_op = unary_pw_op_create(afterAddReductionTensor, afterLogLTensor, logDesc);

    // Create add descriptor
    auto addDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_ADD);

    // Create add node
    auto add_op = binary_pw_op_create(
                                afterMaxReductionTensor, afterLogLTensor,
                                softmaxStatsTensor, addDesc);

    // Define the division descriptor
    auto divisionDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_DIV);

    // Create a subtract node
    auto division_op = binary_pw_op_create(
                                afterExponentTensor, afterAddReductionTensor,
                                afterSoftmaxTensor, divisionDesc);

    ops->push_back(std::move(reductionMax_op));
    ops->push_back(std::move(subtract_op));
    ops->push_back(std::move(exponent_op));
    ops->push_back(std::move(reductionAdd_op));
    ops->push_back(std::move(log_op));
    ops->push_back(std::move(add_op));
    ops->push_back(std::move(division_op));

    return afterSoftmaxTensor;
}

static cudnn_frontend::Tensor
createDropoutForward(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
              double probability, cudnnDataType_t tensorType,
              std::vector<cudnn_frontend::Operation>* ops,
              const cudnn_frontend::Tensor& afterSoftmaxTensor) {
    CUDNN_FRONTEND_UNUSED(d);

    NVTE_CHECK(ops->size() != 0, "Dropout DAG constructed incorrectly as the first one");

    int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
    int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    int64_t scale_dim[4] = {1, 1, 1, 1};
    int64_t scale_stride[4] = {1, 1, 1, 1};

    auto dropoutSeed = tensor_create(
                            CUDNN_DATA_INT64, D_SEED_ID, scale_dim,
                            scale_stride, false, false);  // not virtual
    auto dropoutOffset = tensor_create(
                            CUDNN_DATA_INT64, D_OFFSET_ID, scale_dim,
                            scale_stride, false, false);  // not virtual

    // mask for the dropout
    auto dropoutMaskTensor = tensor_create(
                            CUDNN_DATA_FLOAT, VIRTUAL_ID + 200, afterBMM1_dim,
                            afterBMM1_stride, true, false);  // is virtual
    // after dropout tensor
    auto afterDropoutTensor = cudnn_frontend::TensorBuilder()
            .setDim(4, afterBMM1_dim)
            .setStride(4, afterBMM1_stride)
            .setId(VIRTUAL_ID + 201)
            .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
            .setDataType(tensorType)
            .setVirtual(true)
            .setByValue(false)
            .setReorderType(
                cudnn_frontend::TensorReordering_t::F16x16)
                //cudnn_frontend::cudnnBackendTensorReordering_t::CUDNN_TENSOR_REORDERING_F16x16)
            .build();
    // scale after dropout
    auto scaleDropoutTensor = tensor_create(
                            CUDNN_DATA_FLOAT, D_CONST_ID, scale_dim,
                            scale_stride, false, true);  // is by value
    // after Scale
    auto afterScaleTensor = tensor_create(
                            tensorType, VIRTUAL_ID + 202, afterBMM1_dim,
                            afterBMM1_stride, true, false);  // is virtual

    // Define the reduction descriptor
    auto rngDesc = cudnn_frontend::RngDescBuilder()
                            .setRngDistribution(CUDNN_RNG_DISTRIBUTION_BERNOULLI)
                            .setBernoulliDistProbability(1.0 - probability)
                            .build();

    // Create a rng node
    auto rng_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR)
                            .setyDesc(dropoutMaskTensor)
                            .setSeedDesc(dropoutSeed)
                            .setOffsetDesc(dropoutOffset)
                            .setRngDesc(rngDesc)
                            .build();

    // Define the multiply mask descriptor
    auto maskMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a multiply mask node
    auto maskMul_op = binary_pw_op_create(
                            afterSoftmaxTensor, dropoutMaskTensor,
                            afterDropoutTensor, maskMulDesc);

    // Define the multiply scale descriptor
    auto scaleMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a multiply scale node
    auto scaleMul_op = binary_pw_op_create(
                            afterDropoutTensor, scaleDropoutTensor,
                            afterScaleTensor, scaleMulDesc);

    ops->push_back(std::move(rng_op));
    ops->push_back(std::move(maskMul_op));
    ops->push_back(std::move(scaleMul_op));

    return afterScaleTensor;
}

static cudnn_frontend::Tensor
createDropoutBackward(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
              double probability, cudnnDataType_t tensorType,
              std::vector<cudnn_frontend::Operation>* ops,
              const cudnn_frontend::Tensor& afterSoftmaxTensor,
              const cudnn_frontend::Tensor& dropoutMaskTensor) {
    CUDNN_FRONTEND_UNUSED(d);

    NVTE_CHECK(ops->size() != 0, "Dropout DAG constructed incorrectly as the first one");

    int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
    int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    int64_t scale_dim[4] = {1, 1, 1, 1};
    int64_t scale_stride[4] = {1, 1, 1, 1};

    auto dropoutSeed = tensor_create(
                            CUDNN_DATA_INT64, D_SEED_ID, scale_dim,
                            scale_stride, false, false);  // not virtual
    auto dropoutOffset = tensor_create(
                            CUDNN_DATA_INT64, D_OFFSET_ID, scale_dim,
                            scale_stride, false, false);  // not virtual

    // after dropout tensor
    auto afterDropoutTensor = cudnn_frontend::TensorBuilder()
            .setDim(4, afterBMM1_dim)
            .setStride(4, afterBMM1_stride)
            .setId(VIRTUAL_ID + 201)
            .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
            .setDataType(tensorType)
            .setVirtual(true)
            .setByValue(false)
            .setReorderType(
                cudnn_frontend::TensorReordering_t::F16x16)
                //cudnn_frontend::cudnnBackendTensorReordering_t::CUDNN_TENSOR_REORDERING_F16x16)
            .build();
    // scale after dropout
    auto scaleDropoutTensor = tensor_create(
                            CUDNN_DATA_FLOAT, D_CONST_ID, scale_dim,
                            scale_stride, false, true);  // is by value
    // after Scale
    auto afterScaleTensor = tensor_create(
                            tensorType, VIRTUAL_ID + 202, afterBMM1_dim,
                            afterBMM1_stride, true, false);  // is virtual

    // Define the reduction descriptor
    auto rngDesc = cudnn_frontend::RngDescBuilder()
                            .setRngDistribution(CUDNN_RNG_DISTRIBUTION_BERNOULLI)
                            .setBernoulliDistProbability(1.0 - probability)
                            .build();

    // Create a rng node
    auto rng_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR)
                            .setyDesc(dropoutMaskTensor)
                            .setSeedDesc(dropoutSeed)
                            .setOffsetDesc(dropoutOffset)
                            .setRngDesc(rngDesc)
                            .build();

    // Define the multiply mask descriptor
    auto maskMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a multiply mask node
    auto maskMul_op = binary_pw_op_create(
                            afterSoftmaxTensor, dropoutMaskTensor,
                            afterDropoutTensor, maskMulDesc);

    // Define the multiply scale descriptor
    auto scaleMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a multiply scale node
    auto scaleMul_op = binary_pw_op_create(
                            afterDropoutTensor, scaleDropoutTensor,
                            afterScaleTensor, scaleMulDesc);

    ops->push_back(std::move(rng_op));
    ops->push_back(std::move(maskMul_op));
    ops->push_back(std::move(scaleMul_op));

    return afterScaleTensor;
}

static void
createSVBMM(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
           NVTE_QKV_Layout layout, cudnnDataType_t tensorType,
           std::vector<cudnn_frontend::Operation>* ops,
           cudnn_frontend::Tensor const &afterScaleDropoutTensor) {
    NVTE_CHECK(ops->size() != 0, "BMM2 op constructed incorrectly as the first one");

    int64_t v_dim[4] =  {b, h, s_kv, d};
    int64_t v_stride[4];
    generateMatrixStrides(b, h, s_q, s_kv, d, v_stride, layout, NVTE_QKV_Matrix::NVTE_V_Matrix);

    int64_t o_dim[4] =  {b, h, s_q, d};
    int64_t o_stride[4];
    generateMatrixStrides(b, h, s_q, s_kv, d, o_stride, layout, NVTE_QKV_Matrix::NVTE_O_Matrix);

    auto vTensor = tensor_create(tensorType, V_ID, v_dim, v_stride, false, false);
    // second GEMM output
    auto oTensor = tensor_create(tensorType, O_ID, o_dim, o_stride, false, false);

    // Define the matmul 2 desc
    auto matmul_2_Desc = cudnn_frontend::MatMulDescBuilder()
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();

    // Create a matmul 2 node
    auto matmul_op2 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(afterScaleDropoutTensor)
                            .setbMatDesc(vTensor)
                            .setcMatDesc(oTensor)
                            .setmatmulDesc(matmul_2_Desc)
                            .build();

    ops->push_back(std::move(matmul_op2));
}


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
                cudaStream_t stream, cudnnHandle_t handle) {
std::cout << "Enter fwd impl " << std::endl;
    if (cudnnGetCudartVersion() < 12000) {
        NVTE_ERROR("cuDNN MHA Graph (FWD) requires CUDA Toolkit 12.0 or above!");
        return;
    }

    if (cudnnGetVersion() < 8901) {
        NVTE_ERROR("cuDNN MHA Graph (FWD) requires cuDNN 8.9.1 or above!");
        return;
    }

    const int device_id = transformer_engine::cuda::current_device();
    const int sm_arch_ = transformer_engine::cuda::sm_arch(device_id);
    std::cout << "sm_arch " << sm_arch_ << std::endl;
    if (sm_arch_ < 80) {
        NVTE_ERROR("cuDNN MHA Graph (FWD) requires Ampere or above!");
        return;
    }

    bool is_bias = (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS);
    bool is_alibi = false;
    bool is_causal = (mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK);
    bool is_padding = (mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK);
    bool is_dropout_mask = false;
    //CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT
    //bool is_inference         = !is_training; //false;

    namespace fe = cudnn_frontend;
    fe::graph::Graph mha_graph;
    mha_graph.set_io_data_type(tensorType)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

    int64_t q_stride[4];
    int64_t k_stride[4];
    int64_t v_stride[4];
    generateMatrixStrides(b, h, s_q, s_kv, d, q_stride, layout, NVTE_QKV_Matrix::NVTE_Q_Matrix);
    generateMatrixStrides(b, h, s_q, s_kv, d, k_stride, layout, NVTE_QKV_Matrix::NVTE_K_Matrix);
    generateMatrixStrides(b, h, s_q, s_kv, d, v_stride, layout, NVTE_QKV_Matrix::NVTE_V_Matrix);
    std::vector<int64_t> q_strides(q_stride, q_stride + 4);
    std::vector<int64_t> k_strides(k_stride, k_stride + 4);
    std::vector<int64_t> v_strides(v_stride, v_stride + 4);
std::cout << "Get qkv strides " << q_strides.size() << std::endl;
    auto Q = mha_graph.tensor(fe::graph::Tensor_attributes()
                    .set_name("Q")
                    .set_dim({b, h, s_q, d})
                    .set_stride(q_strides));
    auto K = mha_graph.tensor(fe::graph::Tensor_attributes()
                    .set_name("K")
                    .set_dim({b, h, s_kv, d})
                    .set_stride(k_strides));
    auto V = mha_graph.tensor(fe::graph::Tensor_attributes()
                    .set_name("V")
                    .set_dim({b, h, s_kv, d})
                    .set_stride(v_strides));

    auto attn_scale = mha_graph.tensor(fe::graph::Tensor_attributes()
                    .set_name("attn_scale")
                    .set_dim({1, 1, 1, 1})
                    .set_stride({1, 1, 1, 1})
                    .set_is_pass_by_value(true)
                    .set_data_type(fe::DataType_t::FLOAT));

    std::shared_ptr<fe::graph::Tensor_attributes> bias, dropout_seed, dropout_offset;
    if (is_bias) {
        bias = mha_graph.tensor(fe::graph::Tensor_attributes()
                        .set_name("bias")
                        .set_dim({b, 1, s_q, s_kv})
                        .set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));
    }

    if (dropout_probability != 0.0f) {
        dropout_seed = mha_graph.tensor(fe::graph::Tensor_attributes()
                        .set_name("Seed")
                        .set_dim({1, 1, 1, 1})
                        .set_stride({1, 1, 1, 1})
                        .set_data_type(fe::DataType_t::INT32));
        dropout_offset = mha_graph.tensor(fe::graph::Tensor_attributes()
                        .set_name("Offset")
                        .set_dim({1, 1, 1, 1})
                        .set_stride({1, 1, 1, 1})
                        .set_data_type(fe::DataType_t::INT32));
    }

    std::shared_ptr<fe::graph::Tensor_attributes> seq_q, seq_kv;
    if (is_padding) {
        seq_q  = mha_graph.tensor(fe::graph::Tensor_attributes()
                        .set_name("seq_q")
                        .set_dim({b, 1, 1, 1})
                        .set_stride({1, 1, 1, 1})
                        .set_data_type(fe::DataType_t::INT32));
        seq_kv = mha_graph.tensor(fe::graph::Tensor_attributes()
                        .set_name("seq_kv")
                        .set_dim({b, 1, 1, 1})
                        .set_stride({1, 1, 1, 1})
                        .set_data_type(fe::DataType_t::INT32));
    }

std::cout << "Set options seq q/kv" << std::endl;
    auto scaled_dot_product_flash_attention_options = 
            fe::graph::Scaled_dot_product_flash_attention_attributes()
            .set_name("flash_attention")
            .set_is_inference(!is_training)
            .set_causal_mask(is_causal)
            .set_attn_scale(attn_scale);

    if (cudnnGetVersion() >= 8904) {
        scaled_dot_product_flash_attention_options.set_alibi_mask(is_alibi);
    }

    if (cudnnGetVersion() >= 8903) {
        if (is_bias) {
            scaled_dot_product_flash_attention_options.set_bias(bias);
        }
        if (is_padding) {
            scaled_dot_product_flash_attention_options.set_padding_mask(is_padding)
                    .set_seq_len_q(seq_q)
                    .set_seq_len_kv(seq_kv);
        }
    }

    if (dropout_probability != 0.0f) {
        scaled_dot_product_flash_attention_options.set_dropout(
                        dropout_probability, dropout_seed, dropout_offset);
    }

std::cout << "Call sdpa " << std::endl;

    auto [O, Stats] = mha_graph.scaled_dot_product_flash_attention(
                    Q, K, V, scaled_dot_product_flash_attention_options);

    int64_t o_stride[4];
    generateMatrixStrides(b, h, s_q, s_kv, d, o_stride, layout, NVTE_QKV_Matrix::NVTE_O_Matrix);
    std::vector<int64_t> o_strides(o_stride, o_stride + 4);
    O->set_output(true).set_dim({b, h, s_q, d}).set_stride(o_strides);

    if (Stats) {
        Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT)
                .set_dim({b, h, s_q, 1})
                .set_stride({h * s_q, s_q, 1, 1});
    }

std::cout << "Call sdpa validate " << std::endl;
    if (!mha_graph.validate().is_good()) {
        NVTE_ERROR("cuDNN MHA Graph (FWD): validation is unsuccessful!");
        return;
    }

    if (!mha_graph.build_operation_graph(handle).is_good()) {
        NVTE_ERROR("cuDNN MHA Graph (FWD): build is unsuccessful!");
        return;
    }

    auto plans = mha_graph.get_execution_plan_list({fe::HeurMode_t::A});
    
    if (!plans.check_support(handle).is_good()) {
        NVTE_ERROR("cuDNN MHA Graph (FWD): support check is unsuccessful!");
        return;
    }

    if (!mha_graph.set_execution_plans(plans).is_good()) {
        NVTE_ERROR("cuDNN MHA Graph (FWD): setting execution plans is unsuccessful!");
        return;
    }

    auto plan_workspace_size = mha_graph.get_workspace_size();
std::cout << "Plan size " << plan_workspace_size << std::endl;

    // Exit to request upper level API to allocate memory if needed
    size_t actual_seqlen_workspace_size = 2 * b * sizeof(int32_t);
    if (workspace == nullptr) {
        *workspace_size = plan_workspace_size + actual_seqlen_workspace_size;
std::cout << "workspace size " << *workspace_size << std::endl;
        return;
    }

    // Build variant pack
std::cout << "Create variant pack" << std::endl;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {Q, devPtrQ},
        {K, devPtrK},
        {V, devPtrV},
        {attn_scale, &scaling_factor},
        {O, devPtrO}};

    //if (is_bias) {
    //    variant_pack[bias] = devPtrBias;
    //}

    if (cudnnGetVersion() >= 8903) {
std::cout << "Create actual seqs" << std::endl;
        if (is_padding) {
            constexpr size_t nthreads_per_block = 128;
            const size_t grid = (b + nthreads_per_block - 1) / nthreads_per_block;
            void *devActualSeqlenQ = static_cast<int8_t *>(workspace) + plan_workspace_size;
            void *devActualSeqlenKV = static_cast<int8_t *>(devActualSeqlenQ) + b * sizeof(int32_t);
            cu_seqlens_to_actual_seqlens<<<grid, nthreads_per_block, 0, stream>>>(
                b, static_cast<const int32_t *>(devPtrCuSeqlensQ),
                static_cast<const int32_t *>(devPtrCuSeqlensKV),
                static_cast<int32_t *>(devActualSeqlenQ), static_cast<int32_t *>(devActualSeqlenKV));
            variant_pack[seq_q]  = devActualSeqlenQ;
            variant_pack[seq_kv] = devActualSeqlenKV;
        }
    }

    if (dropout_probability != 0.0f) {
        // add (mask, scale) option
        variant_pack[dropout_seed] = devPtrDropoutSeed;
        variant_pack[dropout_offset] = devPtrDropoutOffset;
    }

    if(is_training) {
        variant_pack[Stats] = devPtrSoftmaxStats;
    }
    
std::cout << "Execute " << std::endl;
    if (!mha_graph.execute(handle, variant_pack, workspace).is_good()) {
        NVTE_ERROR("cuDNN MHA Graph (FWD): execution is unsuccessful!");
        return;
    }
}

void fused_attn_arbitrary_seqlen_fwd_impl_old(
                                int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
                                bool is_training, float scaling_factor, float dropout_probability,
                                NVTE_QKV_Layout layout,
                                void *devPtrQ, void *devPtrK, void *devPtrV,
                                void *devPtrSoftmaxStats, void *devPtrO,
                                void* devPtrDropoutSeed, void* devPtrDropoutOffset,
                                void* devPtrCuSeqlensQ, void* devPtrCuSeqlensKV,
                                cudnnDataType_t tensorType,
                                void *workspace, size_t *workspace_size,
                                cudaStream_t stream, cudnnHandle_t handle) {
    try {
        NVTE_CHECK_CUDNN(cudnnSetStream(handle, stream));

        if (!is_training) {
          dropout_probability = 0.0f;
        }

        FADescriptor descriptor{b,           h,
                                s_q,         s_kv,
                                d,           scaling_factor,
                                is_training, dropout_probability,
                                layout,      NVTE_Bias_Type::NVTE_NO_BIAS,
                                NVTE_Mask_Type::NVTE_CAUSAL_MASK,   tensorType,
                                false};

        using CacheType = std::map<FADescriptor, cudnn_frontend::ExecutionPlan>;
        static thread_local CacheType fmha_fprop_cache;

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

            // Q * K^T
            auto sTensor = createQKBMM(b, h, s_q, s_kv, d, layout, tensorType, &ops);

            // Q * K^T * bmmScale
            auto sScaleTensor = createScale(
                                b, h, s_q, s_kv, d, layout, CUDNN_DATA_FLOAT, sTensor, &ops);

            // Causual mask
            auto sAfterMaskTensor = createCausalMask(
                                b, h, s_q, s_kv, d, layout, tensorType, &ops, sScaleTensor);

            NVTE_CHECK(dropout_probability != 1.0f,
                                "Dropout probability cannot be 1.0");

            auto softmax_output = createSoftmaxForward(
                                b, h, s_q, s_kv, is_training, &ops, sAfterMaskTensor);

            // Dropout(softmax)
            auto dropout_output = createDropoutForward(
                                b, h, s_q, s_kv, d,
                                dropout_probability, tensorType, &ops, softmax_output);
            createSVBMM(b, h, s_q, s_kv, d, layout, tensorType, &ops, dropout_output);

            for (unsigned int i = 0; i < ops.size(); i++) {
                all_ops.push_back(&ops[i]);
            }

            // Create an Operation Graph
            auto opGraph = cudnn_frontend::OperationGraphBuilder()
                                .setHandle(handle)
                                .setOperationGraph(all_ops.size(), all_ops.data())
                                .build();

            cudnn_frontend::EngineConfigList filtered_configs;
            auto statuses = cudnn_frontend::get_heuristics_list<1>(
                                {"heuristics_instant"}, opGraph, allowAllConfig,
                                filtered_configs, true);

            if (filtered_configs.size() == 0) {
                cudnn_frontend::set_error_and_throw_exception(
                        nullptr,
                        CUDNN_STATUS_NOT_SUPPORTED,
                        "run_mha_fprop: No config returned by the heuristics");
            }

            auto plan = cudnn_frontend::ExecutionPlanBuilder()
                                .setHandle(handle)
                                .setEngineConfig(filtered_configs[0], opGraph.getTag())
                                .build();

            cache.insert({descriptor, plan});
            return plan;
        };

        auto plan = get_plan(fmha_fprop_cache, descriptor);

        auto plan_workspace_size = plan.getWorkspaceSize();

        // Exit to request upper level API to allocate memory if needed
        if (workspace == nullptr) {
            *workspace_size = plan_workspace_size;
            return;
        }

        std::set<std::pair<uint64_t, void*>> data_ptrs;
        // Add all the data pointers to be used in the variant pack
        float negInfinity = -1.0E+10f;
        float scale_dropout = 1.0f/(1.0f - dropout_probability);

        data_ptrs.insert(std::pair<uint64_t, void*>(Q_ID, devPtrQ));
        data_ptrs.insert(std::pair<uint64_t, void*>(K_ID, devPtrK));
        data_ptrs.insert(std::pair<uint64_t, void*>(V_ID, devPtrV));
        data_ptrs.insert(std::pair<uint64_t, void*>(MASK_VAL_ID, &negInfinity));
        data_ptrs.insert(std::pair<uint64_t, void*>(S_CONST_ID, &scaling_factor));
        data_ptrs.insert(std::pair<uint64_t, void*>(O_ID, devPtrO));
        data_ptrs.insert(std::pair<uint64_t, void*>(D_SEED_ID, devPtrDropoutSeed));
        data_ptrs.insert(std::pair<uint64_t, void*>(D_OFFSET_ID, devPtrDropoutOffset));
        data_ptrs.insert(std::pair<uint64_t, void*>(D_CONST_ID, &scale_dropout));

        // If training mode, we write out softmax stats
        if (is_training) {
            data_ptrs.insert(std::pair<uint64_t, void*>(S_STATS_ID, devPtrSoftmaxStats));
        }

        auto variantPack = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace)
                               .setDataPointers(data_ptrs)
                               .build();

        NVTE_CHECK_CUDNN(
            cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc()));
    } catch (cudnn_frontend::cudnnException &e) {
        NVTE_ERROR(e.what());
    }

}

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
                            cudaStream_t stream, cudnnHandle_t handle, bool use_workspace_opt) {

std::cout << "Enter bwd impl" << std::endl;
    if (cudnnGetCudartVersion() < 12000) {
        NVTE_ERROR("cuDNN MHA Graph (BWD) requires CUDA Toolkit 12.0 or above!");
        return;
    }

    // inconsistent with fwd??
    if (cudnnGetVersion() < 8903) {
        NVTE_ERROR("cuDNN MHA Graph (BWD) requires cuDNN 8.9.3 or above!");
        return;
    }

    const int device_id = transformer_engine::cuda::current_device();
    const int sm_arch_ = transformer_engine::cuda::sm_arch(device_id);
    std::cout << "sm_arch " << sm_arch_ << std::endl;
    if (sm_arch_ < 80) {
        NVTE_ERROR("cuDNN MHA Graph (BWD) requires Ampere or above!");
        return;
    }

    bool is_bias = (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS);
    bool is_alibi = false;
    bool is_causal = (mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK);
    bool is_padding = (mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK);
    bool is_dropout_mask = false;
    //CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT
std::cout << "is bias " << is_bias << " is causal " << is_causal << " is_padding " << is_padding << std::endl; 

    namespace fe = cudnn_frontend;
    fe::graph::Graph mha_graph;
    mha_graph.set_io_data_type(tensorType)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

std::cout << "Get qkv strides" << std::endl;
    int64_t q_stride[4];
    int64_t k_stride[4];
    int64_t v_stride[4];
    int64_t o_stride[4];
    generateMatrixStrides(b, h, s_q, s_kv, d, q_stride, layout, NVTE_QKV_Matrix::NVTE_Q_Matrix);
    generateMatrixStrides(b, h, s_q, s_kv, d, k_stride, layout, NVTE_QKV_Matrix::NVTE_K_Matrix);
    generateMatrixStrides(b, h, s_q, s_kv, d, v_stride, layout, NVTE_QKV_Matrix::NVTE_V_Matrix);
    generateMatrixStrides(b, h, s_q, s_kv, d, o_stride, layout, NVTE_QKV_Matrix::NVTE_O_Matrix);
    std::vector<int64_t> q_strides(q_stride, q_stride + 4);
    std::vector<int64_t> k_strides(k_stride, k_stride + 4);
    std::vector<int64_t> v_strides(v_stride, v_stride + 4);
    std::vector<int64_t> o_strides(o_stride, o_stride + 4);
    auto q = mha_graph.tensor(fe::graph::Tensor_attributes()
                    .set_name("Q")
                    .set_dim({b, h, s_q, d})
                    .set_stride(q_strides));
    auto k = mha_graph.tensor(fe::graph::Tensor_attributes()
                    .set_name("K")
                    .set_dim({b, h, s_kv, d})
                    .set_stride(k_strides));
    auto v = mha_graph.tensor(fe::graph::Tensor_attributes()
                    .set_name("V")
                    .set_dim({b, h, s_kv, d})
                    .set_stride(v_strides));
    auto o = mha_graph.tensor(fe::graph::Tensor_attributes()
                    .set_name("O")
                    .set_dim({b, h, s_q, d})
                    .set_stride(o_strides));
    auto dO = mha_graph.tensor(fe::graph::Tensor_attributes()
                    .set_name("dO")
                    .set_dim({b, h, s_q, d})
                    .set_stride(o_strides));
    auto stats = mha_graph.tensor(fe::graph::Tensor_attributes()
                    .set_name("stats")
                    .set_dim({b, h, s_q, 1})
                    .set_stride({h * s_q, s_q, 1, 1})
                    .set_data_type(fe::DataType_t::FLOAT));

    auto attn_scale = mha_graph.tensor(fe::graph::Tensor_attributes()
                    .set_name("attn_scale")
                    .set_dim({1, 1, 1, 1})
                    .set_stride({1, 1, 1, 1})
                    .set_is_pass_by_value(true)
                    .set_data_type(fe::DataType_t::FLOAT));

    std::shared_ptr<fe::graph::Tensor_attributes> bias, dropout_seed, dropout_offset;
    if (is_bias) {
        bias = mha_graph.tensor(fe::graph::Tensor_attributes()
                        .set_name("bias")
                        .set_dim({b, 1, s_q, s_kv})
                        .set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));
    }
    if (dropout_probability != 0.0f) {
        dropout_seed = mha_graph.tensor(fe::graph::Tensor_attributes()
                        .set_name("Seed")
                        .set_dim({1, 1, 1, 1})
                        .set_stride({1, 1, 1, 1})
                        .set_data_type(fe::DataType_t::INT32));
        dropout_offset = mha_graph.tensor(fe::graph::Tensor_attributes()
                        .set_name("Offset")
                        .set_dim({1, 1, 1, 1})
                        .set_stride({1, 1, 1, 1})
                        .set_data_type(fe::DataType_t::INT32));
    }

    std::shared_ptr<fe::graph::Tensor_attributes> seq_q, seq_kv;
    if (is_padding) {
        seq_q  = mha_graph.tensor(fe::graph::Tensor_attributes()
                        .set_name("seq_q")
                        .set_dim({b, 1, 1, 1})
                        .set_stride({1, 1, 1, 1})
                        .set_data_type(fe::DataType_t::INT32));
        seq_kv = mha_graph.tensor(fe::graph::Tensor_attributes()
                        .set_name("seq_kv")
                        .set_dim({b, 1, 1, 1})
                        .set_stride({1, 1, 1, 1})
                        .set_data_type(fe::DataType_t::INT32));
    }

std::cout << "Create options" << std::endl;
    auto scaled_dot_product_flash_attention_backward_options =
            fe::graph::Scaled_dot_product_flash_attention_backward_attributes()
            .set_name("flash_attention_backward")
            .set_causal_mask(is_causal)
            .set_attn_scale(attn_scale);

    if (cudnnGetVersion() >= 8904) {
        scaled_dot_product_flash_attention_backward_options.set_alibi_mask(is_alibi);
    }

    if (is_bias) {
        scaled_dot_product_flash_attention_backward_options.set_bias(bias);
    }

    if (dropout_probability != 0.0f) {
        scaled_dot_product_flash_attention_backward_options.set_dropout(
                        dropout_probability, dropout_seed, dropout_offset);
    }

std::cout << "Call sdpa" << std::endl;
    auto [dQ, dK, dV] = mha_graph.scaled_dot_product_flash_attention_backward(
                    q, k, v, o, dO, stats, scaled_dot_product_flash_attention_backward_options);

    dQ->set_output(true)
            .set_dim({b, h, s_q, d})
            .set_stride(q_strides);
    dK->set_output(true)
            .set_dim({b, h, s_kv, d})
            .set_stride(k_strides);
    dV->set_output(true)
            .set_dim({b, h, s_kv, d})
            .set_stride(v_strides);

std::cout << "Validate etc" << std::endl;
    if (!mha_graph.validate().is_good()) {
        NVTE_ERROR("cuDNN MHA Graph (BWD): validation is unsuccessful!");
        return;
    }

    if (!mha_graph.build_operation_graph(handle).is_good()) {
        NVTE_ERROR("cuDNN MHA Graph (BWD): build is unsuccessful!");
        return;
    }

    auto plans = mha_graph.get_execution_plan_list({fe::HeurMode_t::A});
    
    if (!plans.check_support(handle).is_good()) {
        NVTE_ERROR("cuDNN MHA Graph (BWD): support check is unsuccessful!");
        return;
    }

    if (!mha_graph.set_execution_plans(plans).is_good()) {
        NVTE_ERROR("cuDNN MHA Graph (BWD): setting execution plans is unsuccessful!");
        return;
    }

    auto plan_workspace_size = mha_graph.get_workspace_size();
std::cout << "Plan size " << plan_workspace_size << std::endl;

    // Exit to request upper level API to allocate memory if needed
    if (workspace == nullptr) {
        *workspace_size = plan_workspace_size;
        return;
    }

    // build variant pack
std::cout << "Variant Pack" << plan_workspace_size << std::endl;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {q, devPtrQ},
        {k, devPtrKTranspose},
        {v, devPtrVTranspose},
        {o, devPtrO},
        {dO, devPtrdO},
        {stats, devPtrSoftmaxStats},
        {dQ, devPtrdQ},
        {dK, devPtrdK},
        {dV, devPtrdV},
        {attn_scale, &scaling_factor}
    };

    //if (is_bias) {
    //    variant_pack[bias] = devPtrBias;
    //}

    if (dropout_probability != 0.0f) {
        variant_pack[dropout_seed] = devPtrDropoutSeed;
        variant_pack[dropout_offset] = devPtrDropoutOffset;
    }

std::cout << "Bwd execute " << plan_workspace_size << std::endl;
    if (!mha_graph.execute(handle, variant_pack, workspace).is_good()) {
        NVTE_ERROR("cuDNN MHA graph (BWD): execution is unsuccessful!");
        return;
    }
}

void fused_attn_arbitrary_seqlen_bwd_impl_old(
                            int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
                            float scaling_factor, float dropout_probability, NVTE_QKV_Layout layout,
                            void* devPtrQ, void* devPtrKTranspose, void* devPtrVTranspose,
                            void* devPtrO, void* devPtrSoftmaxStats,
                            void* devPtrdQ, void* devPtrdK, void* devPtrdV, void* devPtrdO,
                            void* devPtrDropoutSeed, void* devPtrDropoutOffset,
                            void* devPtrCuSeqlensQ, void* devPtrCuSeqlensKV,
                            cudnnDataType_t tensorType, void *workspace, size_t *workspace_size,
                            cudaStream_t stream, cudnnHandle_t handle, bool use_workspace_opt) {
    try {
        NVTE_CHECK_CUDNN(cudnnSetStream(handle, stream));

        FADescriptor descriptor{b,           h,
                                s_q,         s_kv,
                                d,           scaling_factor,
                                true,        dropout_probability,
                                layout,      NVTE_Bias_Type::NVTE_NO_BIAS,
                                NVTE_Mask_Type::NVTE_CAUSAL_MASK,   tensorType,
                                use_workspace_opt};

        using CacheType = std::map<FADescriptor, cudnn_frontend::ExecutionPlan>;
        static thread_local CacheType fmha_bprop_cache;

        auto get_plan = [&](CacheType &cache, const FADescriptor &descriptor) {
            auto it = cache.find(descriptor);
            if (it != cache.end()) {
                return it->second;
            }

            std::vector<cudnn_frontend::Operation const*> all_ops;
            std::vector<cudnn_frontend::Operation> ops;

            // Creates the necessary tensor descriptors
            int64_t q_dim[4] = {b, h, s_q, d};
            int64_t q_stride[4];
            generateMatrixStrides(
                            b, h, s_q, s_kv, d, q_stride,
                            layout, NVTE_QKV_Matrix::NVTE_Q_Matrix);

            int64_t k_transpose_dim[4] =  {b, h, d, s_kv};
            int64_t k_transpose_stride[4];
            generateMatrixStrides(
                            b, h, s_q, s_kv, d, k_transpose_stride,
                            layout, NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose);

            int64_t v_transpose_dim[4] =  {b, h, d, s_kv};
            int64_t v_transpose_stride[4];
            generateMatrixStrides(
                            b, h, s_q, s_kv, d, v_transpose_stride,
                            layout, NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose);

            int64_t p_dim[4] = {b, h, s_q, s_kv};
            int64_t p_stride[4];
            generateMatrixStrides(
                            b, h, s_q, s_kv, d, p_stride,
                            layout, NVTE_QKV_Matrix::NVTE_S_Matrix);

            int64_t p_transpose_dim[4] = {b, h, s_kv, s_q};
            int64_t p_transpose_stride[4];
            p_transpose_stride[0] = p_stride[0];
            p_transpose_stride[1] = p_stride[1];
            p_transpose_stride[2] = p_stride[3];
            p_transpose_stride[3] = p_stride[2];

            int64_t o_dim[4] =  {b, h, s_q, d};
            int64_t o_stride[4];
            generateMatrixStrides(
                            b, h, s_q, s_kv, d, o_stride,
                            layout, NVTE_QKV_Matrix::NVTE_O_Matrix);

            int64_t dqAccum_dim[4] =  {b, h, s_q, d};
            int64_t dqAccum_stride[4];
            generateMatrixStrides(b, h, s_q, s_kv, d, dqAccum_stride,
                            layout, NVTE_QKV_Matrix::NVTE_O_Matrix);

            int64_t scale_dim[4] = {1, 1, 1, 1};
            int64_t scale_stride[4] = {1, 1, 1, 1};

            /*******************************************************************************
             *                          Dot product dO * O                                */ 

            // output and gradient of the output
            auto oTensor = tensor_create(tensorType, O_ID, o_dim, o_stride, false, false);
            auto dOTensor = tensor_create(tensorType, dO_ID, o_dim, o_stride, false, false);

            auto dotProductTensor = tensor_create(
                            CUDNN_DATA_FLOAT, VIRTUAL_ID, o_dim,
                            o_stride, true, false);  // is virtual

            // Create pointwise mul
            auto multiplyDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

            // do * O
            auto dotProductOp = binary_pw_op_create(
                            dOTensor, oTensor, dotProductTensor, multiplyDesc);
            ops.push_back(std::move(dotProductOp));

            /*******************************************************************************
             *                         Reduction(dO * O)                                  */

            int64_t reduction_dim[4] = {b, h, s_q, 1};
            int64_t reduction_stride[4] = {h * s_q, s_q, 1, 1};

            // reduction(dO * O)
            auto afterReductionTensor = tensor_create(
                            CUDNN_DATA_FLOAT, VIRTUAL_ID + 1, reduction_dim,
                            reduction_stride, true, false);  // is virtual
            auto reductionAddDesc = cudnn_frontend::ReductionDescBuilder()
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                            .build();

            // Create a reduction add node
            auto reductionAdd_op = cudnn_frontend::OperationBuilder(
                            CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                            .setxDesc(dotProductTensor)
                            .setyDesc(afterReductionTensor)
                            .setreductionDesc(reductionAddDesc)
                            .build();
            ops.push_back(std::move(reductionAdd_op));


            /*******************************************************************************
             *                        reduction(dO * O) * scale prob -> softmaxSum         */

            auto softmaxSumTensor = tensor_create(
                            CUDNN_DATA_FLOAT, S_SUM_ID, reduction_dim,
                            reduction_stride, false, false);  // not virtual
            auto scaleProbTensor = tensor_create(
                            CUDNN_DATA_FLOAT, SCALE_PROB, scale_dim,
                            scale_stride, false, true);  // not virtual
            auto softmaxSumOp = binary_pw_op_create(
                            afterReductionTensor, scaleProbTensor,
                            softmaxSumTensor, multiplyDesc);
            ops.push_back(std::move(softmaxSumOp));

            /*******************************************************************************
             *                        Q @ K.T -> P                                        */

            // Inputs from fprop
            auto qTensor = tensor_create(tensorType, Q_ID, q_dim, q_stride, false, false);
            auto kTransposeTensor = tensor_create(
                            tensorType, K_ID, k_transpose_dim,
                            k_transpose_stride, false, false);
            auto pTensor = tensor_create(
                            CUDNN_DATA_FLOAT, VIRTUAL_ID + 2, p_dim,
                            p_stride, true, false);  // is virtual

            // matmul to calculate dvTensor
            auto matmul_0_Desc = cudnn_frontend::MatMulDescBuilder()
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();

            auto matmul_op0 = cudnn_frontend::OperationBuilder(
                            CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(qTensor)
                            .setbMatDesc(kTransposeTensor)
                            .setcMatDesc(pTensor)
                            .setmatmulDesc(matmul_0_Desc)
                            .build();

            ops.push_back(std::move(matmul_op0));

            /*******************************************************************************
             *                        P * bmmScale -> pAfterScale                         */

            auto bmmScaleTensor = tensor_create(
                            CUDNN_DATA_FLOAT, S_CONST_ID, scale_dim,
                            scale_stride, false, true);  // not virtual and by value
            auto pAfterScaleTensor = tensor_create(
                            CUDNN_DATA_FLOAT, VIRTUAL_ID + 2000, p_dim,
                            p_stride, true, false);  // virtual
            auto scaleOp = binary_pw_op_create(
                            pTensor, bmmScaleTensor, pAfterScaleTensor, multiplyDesc);
            ops.push_back(std::move(scaleOp));

            /*******************************************************************************
             *                          Causal masking -> pAfterMaskTensor                */

            auto pAfterMaskTensor = createCausalMask(
                            b, h, s_q, s_kv, d, layout, tensorType, &ops, pAfterScaleTensor);

            /*******************************************************************************
             *                          pAfterMaskTensor - softmaxStats -> pAfterSubtract */

            auto pAfterSubtractTensor = tensor_create(
                            CUDNN_DATA_FLOAT, VIRTUAL_ID + 3, p_dim,
                            p_stride, true, false);  // is virtual
            auto softmaxStatsTensor = tensor_create(
                            CUDNN_DATA_FLOAT, S_STATS_ID, reduction_dim,
                            reduction_stride, false, false);  // not virtual
            auto subtractDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_SUB);
            auto subtract_op = binary_pw_op_create(
                            pAfterMaskTensor, softmaxStatsTensor,
                            pAfterSubtractTensor, subtractDesc);
            ops.push_back(std::move(subtract_op));

            /*******************************************************************************
             *                          e^(pAfterSubtract) -> pAfterSoftmax               */

            auto pAfterSoftmaxTensor = tensor_create(
                            CUDNN_DATA_FLOAT, VIRTUAL_ID + 4, p_dim,
                            p_stride, true, false);  // is virtual
            auto expDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_EXP);
            auto exp_op = unary_pw_op_create(
                            pAfterSubtractTensor, pAfterSoftmaxTensor, expDesc);
            ops.push_back(std::move(exp_op));

            /*******************************************************************************
             *                          Dropout -> afterScaleDropout                      */

            auto dropoutMaskTensor = tensor_create(
                            CUDNN_DATA_FLOAT, VIRTUAL_ID + 5, p_dim,
                            p_stride, true, false);  // is virtual
            auto afterScaleDropoutTensor = createDropoutBackward(
                            b, h, s_q, s_kv, d, dropout_probability, tensorType,
                            &ops, pAfterSoftmaxTensor, dropoutMaskTensor);

            /*******************************************************************************
             *                          afterScaleDropout -> sTransposeTensor             */

            auto sTransposeTensor = tensor_create(
                            tensorType, VIRTUAL_ID + 6, p_transpose_dim,
                            p_transpose_stride, true, false);  // is virtual
            auto reshape_op = cudnn_frontend::OperationBuilder(
                            CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                            .setxDesc(afterScaleDropoutTensor)
                            .setyDesc(sTransposeTensor)
                            .build();
            ops.push_back(std::move(reshape_op));

            // Outputs of bprop
            int64_t dq_dim[4] = {b, h, s_q, d};
            int64_t dq_stride[4];
            generateMatrixStrides(b, h, s_q, s_kv, d, dq_stride,
                            layout, NVTE_QKV_Matrix::NVTE_Q_Matrix);

            int64_t dk_dim[4] = {b, h, s_kv, d};
            int64_t dk_stride[4];
            generateMatrixStrides(b, h, s_q, s_kv, d, dk_stride,
                            layout, NVTE_QKV_Matrix::NVTE_K_Matrix);

            int64_t dv_dim[4] = {b, h, s_kv, d};
            int64_t dv_stride[4];
            generateMatrixStrides(b, h, s_q, s_kv, d, dv_stride,
                            layout, NVTE_QKV_Matrix::NVTE_V_Matrix);

            // Outputs of backprop
            auto dQTensor = tensor_create(tensorType, dQ_ID, dq_dim, dq_stride, false, false);
            auto dKTensor = tensor_create(tensorType, dK_ID, dk_dim, dk_stride, false, false);
            auto dVTensor = tensor_create(tensorType, dV_ID, dv_dim, dv_stride, false, false);
                            // not virtual

            /*******************************************************************************
             *                          sTransposeTensor @ dO -> dV                       */

            auto matmul_1_Desc = cudnn_frontend::MatMulDescBuilder()
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();

            auto matmul_op1 = cudnn_frontend::OperationBuilder(
                            CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(sTransposeTensor)
                            .setbMatDesc(dOTensor)
                            .setcMatDesc(dVTensor)
                            .setmatmulDesc(matmul_1_Desc)
                            .build();

            ops.push_back(std::move(matmul_op1));

            /*******************************************************************************
             *                          dO @ V.T -> dS                                    */

            auto vTransposeTensor = tensor_create(
                            tensorType, V_ID, v_transpose_dim,
                            v_transpose_stride, false, false);
            auto dSTensor = tensor_create(
                            CUDNN_DATA_FLOAT, VIRTUAL_ID + 7, p_dim,
                            p_stride, true, false);  // is virtual

            auto matmul_2_Desc = cudnn_frontend::MatMulDescBuilder()
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();

            auto matmul_op2 = cudnn_frontend::OperationBuilder(
                            CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(dOTensor)
                            .setbMatDesc(vTransposeTensor)
                            .setcMatDesc(dSTensor)
                            .setmatmulDesc(matmul_2_Desc)
                            .build();

            ops.push_back(std::move(matmul_op2));

            /*******************************************************************************
             *                          dS * dropoutMask -> dSAfterDropout                */

            auto dSAfterDropoutTensor = tensor_create(
                            CUDNN_DATA_FLOAT, VIRTUAL_ID + 8, p_dim,
                            p_stride, true, false);  // is virtual
            auto multiply_op = binary_pw_op_create(
                            dSTensor, dropoutMaskTensor,
                            dSAfterDropoutTensor, multiplyDesc);
            ops.push_back(std::move(multiply_op));

            /*******************************************************************************
             *                          dSAfterDropout - softmaxSum -> dsAfterSubtract    */

            auto dsAfterSubtractTensor = tensor_create(
                            CUDNN_DATA_FLOAT, VIRTUAL_ID + 9, p_dim,
                            p_stride, true, false);  // is virtual
            auto subtract_op2 = binary_pw_op_create(
                            dSAfterDropoutTensor, softmaxSumTensor,
                            dsAfterSubtractTensor, subtractDesc);
            ops.push_back(std::move(subtract_op2));

            /*******************************************************************************
             *                          dsAfterSubtract * afterSoftmax -> dP              */

            auto dPTensor = tensor_create(
                            CUDNN_DATA_FLOAT, VIRTUAL_ID + 10, p_dim,
                            p_stride, true, false);  // is virtual
            auto multiply_op2 = binary_pw_op_create(
                            dsAfterSubtractTensor, pAfterSoftmaxTensor,
                            dPTensor, multiplyDesc);
            ops.push_back(std::move(multiply_op2));

            /*******************************************************************************
             *                          dP * scaleDropout -> dPAfterDropoutScale          */

            auto dPAfterDropoutScaleTensor = tensor_create(
                            CUDNN_DATA_FLOAT, VIRTUAL_ID + 11, p_dim,
                            p_stride, true, false);  // is virtual
            auto scaleDropoutTensor = tensor_create(
                            CUDNN_DATA_FLOAT, D_CONST_ID, scale_dim,
                            scale_stride, false, true);  // is by value
            auto multiply_op3 = binary_pw_op_create(
                            dPTensor, scaleDropoutTensor,
                            dPAfterDropoutScaleTensor, multiplyDesc);
            ops.push_back(std::move(multiply_op3));

            /*******************************************************************************
             *                          dPAfterDropoutScale * bmmScale -> dPScaledTensor  */

            auto dPScaledTensor = tensor_create(
                            CUDNN_DATA_FLOAT, VIRTUAL_ID + 12, p_dim,
                            p_stride, true, false);  // is virtual
            auto multiply_op4 = binary_pw_op_create(
                            dPAfterDropoutScaleTensor, bmmScaleTensor,
                            dPScaledTensor, multiplyDesc);
            ops.push_back(std::move(multiply_op4));

            /*******************************************************************************
             *                          K.T -> K                                          */

            int64_t kDim[4] = {b, h, s_kv, d};
            int64_t kStride[4];
            generateMatrixStrides(
                            b, h, s_q, s_kv, d, kStride,
                            layout, NVTE_QKV_Matrix::NVTE_K_Matrix);
            auto kTensor = tensor_create(
                            tensorType, VIRTUAL_ID + 13, kDim,
                            kStride, true, false);  // is virtual
            auto reshape_op2 = cudnn_frontend::OperationBuilder(
                            CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                            .setxDesc(kTransposeTensor)
                            .setyDesc(kTensor)
                            .build();
            ops.push_back(std::move(reshape_op2));

            /*******************************************************************************
             *                          dP @ K -> dqAccumTensor / dqTensor                */

            auto dqAccumTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, dqAccum_dim)
                .setStride(4, dqAccum_stride)
                .setId(dQ_ACCUM_ID)
                .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                .setDataType(CUDNN_DATA_FLOAT)
                .setVirtual(false)
                .setByValue(false)
                .setReorderType(
                cudnn_frontend::TensorReordering_t::F16x16)
                //cudnn_frontend::cudnnBackendTensorReordering_t::CUDNN_TENSOR_REORDERING_F16x16)
                .build();

            auto matmul_3_Desc = cudnn_frontend::MatMulDescBuilder()
                                .setComputeType(CUDNN_DATA_FLOAT)
                                .build();

            if (!use_workspace_opt) {
                auto matmul_op3 = cudnn_frontend::OperationBuilder(
                                    CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                                    .setaMatDesc(dPScaledTensor)
                                    .setbMatDesc(kTensor)
                                    .setcMatDesc(dqAccumTensor)
                                    .setmatmulDesc(matmul_3_Desc)
                                    .build();

                ops.push_back(std::move(matmul_op3));
            } else {
                auto matmul_op3 = cudnn_frontend::OperationBuilder(
                                    CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                                    .setaMatDesc(dPScaledTensor)
                                    .setbMatDesc(kTensor)
                                    .setcMatDesc(dQTensor)
                                    .setmatmulDesc(matmul_3_Desc)
                                    .build();

                ops.push_back(std::move(matmul_op3));
            }

            /*******************************************************************************
             *                          dP.T @ Q -> dK                                    */

            auto dPTransposeTensor = tensor_create(
                                CUDNN_DATA_FLOAT, VIRTUAL_ID + 14, p_transpose_dim,
                                p_transpose_stride, true, false);  // is virtual
            auto reshape_op3 = cudnn_frontend::OperationBuilder(
                                CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                                .setxDesc(dPScaledTensor)
                                .setyDesc(dPTransposeTensor)
                                .build();
            ops.push_back(std::move(reshape_op3));

            auto matmul_4_Desc = cudnn_frontend::MatMulDescBuilder()
                                .setComputeType(CUDNN_DATA_FLOAT)
                                .build();
            auto matmul_op4 = cudnn_frontend::OperationBuilder(
                                CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                                .setaMatDesc(dPTransposeTensor)
                                .setbMatDesc(qTensor)
                                .setcMatDesc(dKTensor)
                                .setmatmulDesc(matmul_4_Desc)
                                .build();

            ops.push_back(std::move(matmul_op4));

            /*******************************************************************************
             *                          dqAccumTensor @ identity -> dqTensor              */

            if (!use_workspace_opt) {
                auto identityDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_IDENTITY);
                auto identity_op = unary_pw_op_create(dqAccumTensor, dQTensor, identityDesc);
                ops.push_back(std::move(identity_op));
            }

            for (unsigned int i = 0; i < ops.size(); i++) {
                all_ops.push_back(&ops[i]);
            }

            // Create an Operation Graph
            auto opGraph = cudnn_frontend::OperationGraphBuilder()
                               .setHandle(handle)
                               .setOperationGraph(all_ops.size(), all_ops.data())
                               .build();

            cudnn_frontend::EngineConfigList filtered_configs;
            auto statuses = cudnn_frontend::get_heuristics_list<1>(
                {"heuristics_instant"}, opGraph, allowAllConfig, filtered_configs, true);

            if (filtered_configs.size() == 0) {
                cudnn_frontend::set_error_and_throw_exception(
                    nullptr, CUDNN_STATUS_NOT_SUPPORTED,
                    "run_mha_bprop: No config returned by the heuristics");
            }

            auto plan = cudnn_frontend::ExecutionPlanBuilder()
                            .setHandle(handle)
                            .setEngineConfig(filtered_configs[0], opGraph.getTag())
                            .build();

            cache.insert({descriptor, plan});
            return plan;
        };

        auto plan = get_plan(fmha_bprop_cache, descriptor);

        auto plan_workspace_size = plan.getWorkspaceSize();

        // Exit to request upper level API to allocate memory if needed
        size_t softmaxSum_workspace_size = b * h * s_q * sizeof(float);
        size_t dqAccum_workspace_size = b * s_q * h * d * sizeof(float);
        if (workspace == nullptr) {
            if (use_workspace_opt) {
                *workspace_size = plan_workspace_size + softmaxSum_workspace_size;
            } else {
                *workspace_size = plan_workspace_size + softmaxSum_workspace_size
                                  + dqAccum_workspace_size;
            }
            return;
        }

        void *devPtrSoftmaxSum = static_cast<int8_t *>(workspace) + plan_workspace_size;
        void *devPtrdQAccumulator = nullptr;
        if (!use_workspace_opt) {
            devPtrdQAccumulator = static_cast<int8_t *>(devPtrSoftmaxSum)
                                        + softmaxSum_workspace_size;
            NVTE_CHECK_CUDA(cudaMemsetAsync(
                                  devPtrdQAccumulator, 0, dqAccum_workspace_size, stream));
        }

        std::set<std::pair<uint64_t, void *>> data_ptrs;
        // add all the data pointers to be used in the variant pack
        float negInfinity = -1.0E+10f;
        float scale_dropout = 1.0f/(1.0f - dropout_probability);
        data_ptrs.insert(std::pair<uint64_t, void*>(dQ_ID, devPtrdQ));
        if (!use_workspace_opt) {
            data_ptrs.insert(std::pair<uint64_t, void*>(dQ_ACCUM_ID, devPtrdQAccumulator));
        }
        data_ptrs.insert(std::pair<uint64_t, void*>(dK_ID, devPtrdK));
        data_ptrs.insert(std::pair<uint64_t, void*>(dV_ID, devPtrdV));

        data_ptrs.insert(std::pair<uint64_t, void*>(Q_ID, devPtrQ));
        data_ptrs.insert(std::pair<uint64_t, void*>(K_ID, devPtrKTranspose));
        data_ptrs.insert(std::pair<uint64_t, void*>(V_ID, devPtrVTranspose));
        data_ptrs.insert(std::pair<uint64_t, void*>(O_ID, devPtrO));
        data_ptrs.insert(std::pair<uint64_t, void*>(dO_ID, devPtrdO));
        data_ptrs.insert(std::pair<uint64_t, void*>(S_STATS_ID, devPtrSoftmaxStats));
        data_ptrs.insert(std::pair<uint64_t, void*>(S_SUM_ID, devPtrSoftmaxSum));
        data_ptrs.insert(std::pair<uint64_t, void*>(D_SEED_ID, devPtrDropoutSeed));
        data_ptrs.insert(std::pair<uint64_t, void*>(D_OFFSET_ID, devPtrDropoutOffset));
        data_ptrs.insert(std::pair<uint64_t, void*>(MASK_VAL_ID, &negInfinity));

        float scaleProb = 1.0f - dropout_probability;
        data_ptrs.insert(std::pair<uint64_t, void*>(D_CONST_ID, &scale_dropout));
        data_ptrs.insert(std::pair<uint64_t, void*>(S_CONST_ID, &scaling_factor));
        data_ptrs.insert(std::pair<uint64_t, void*>(SCALE_PROB, &scaleProb));

        auto variantPack = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace)
                               .setDataPointers(data_ptrs)
                               .build();

        NVTE_CHECK_CUDNN(
            cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc()));
    } catch (cudnn_frontend::cudnnException &e) {
        NVTE_ERROR(e.what());
    }
}

}  // namespace fused_attn

using namespace transformer_engine::fused_attn;
void fused_attn_arbitrary_seqlen_fwd_qkvpacked(
    size_t batch, size_t max_seqlen, size_t num_head, size_t head_dim, bool is_training,
    float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, const Tensor *input_QKV, const Tensor *input_Bias, Tensor *output_O,
    NVTETensorPack *Aux_CTX_Tensors, const Tensor *cu_seqlens, const Tensor *rng_state,
    Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle) {
    using namespace transformer_engine;

    // QKV shape is [b, s, 3, h, d]
    void *devPtrQKV = input_QKV->data.dptr;
    const auto stride = 2 * num_head * head_dim;

    void *devPtrQ = static_cast<void *>(devPtrQKV);
    void *devPtrK = static_cast<void *>(static_cast<int8_t *>(devPtrQKV) + stride);
    void *devPtrV = static_cast<void *>(static_cast<int8_t *>(devPtrQKV) + 2 * stride);
    Tensor *input_Q, *input_K, *input_V;
    input_Q->data.dptr = devPtrQ;
    input_Q->data.shape = {batch, max_seqlen, num_head, head_dim};
    input_Q->data.dtype = input_QKV->data.dtype;
    input_K->data.dptr = devPtrK;
    input_K->data.shape = {batch, max_seqlen, num_head, head_dim};
    input_K->data.dtype = input_QKV->data.dtype;
    input_V->data.dptr = devPtrV;
    input_V->data.shape = {batch, max_seqlen, num_head, head_dim};
    input_V->data.dtype = input_QKV->data.dtype;

    fused_attn_arbitrary_seqlen_fwd(
        batch, max_seqlen, max_seqlen, num_head, head_dim, is_training,
        attn_scale, p_dropout, qkv_layout, bias_type,
        mask_type, input_Q, input_K, input_V, input_Bias, output_O,
        Aux_CTX_Tensors, cu_seqlens, cu_seqlens, rng_state,
        workspace, stream, handle);

    //void *devPtrO = output_O->data.dptr;

    //void *devPtrS = nullptr;

    //void *devPtrCuSeqlens = cu_seqlens->data.dptr;

    //if (Aux_CTX_Tensors->size == 0) {
    //    Aux_CTX_Tensors->size = 2;
    //    Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
    //    output_S->data.dptr = nullptr;
    //    output_S->data.shape = {batch, num_head, max_seqlen, 1};
    //    output_S->data.dtype = DType::kFloat32;
    //    Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
    //    output_rng_state->data.dptr = nullptr;
    //    output_rng_state->data.shape = {2};
    //    output_rng_state->data.dtype = DType::kInt64;
    //} else if (Aux_CTX_Tensors->size == 2) {
    //    Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
    //    devPtrS = output_S->data.dptr;
    //    Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
    //    output_rng_state->data.dptr = rng_state->data.dptr;
    //} else {
    //    NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
    //}

    //void* devPtrDropoutSeed = rng_state->data.dptr;
    //void* devPtrDropoutOffset = reinterpret_cast<void *>(
    //                reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

    //const DType QKV_type = input_QKV->data.dtype;
    //size_t workspace_size = 0;

    //fused_attn_arbitrary_seqlen_fwd_impl(batch, num_head, max_seqlen, max_seqlen, head_dim,
    //                            is_training, attn_scale, p_dropout, qkv_layout,
    //                            bias_type, mask_type,
    //                            devPtrQ, devPtrK, devPtrV, devPtrS, devPtrO,
    //                            devPtrDropoutSeed, devPtrDropoutOffset,
    //                            devPtrCuSeqlens, devPtrCuSeqlens,  
    //                            get_cudnn_fe_dtype(QKV_type),
    //                            workspace->data.dptr, &workspace_size, stream, handle);

    //if (workspace_size > 0) {
    //    if (workspace->data.dptr == nullptr) {
    //        workspace->data.shape = {workspace_size};
    //        workspace->data.dtype = DType::kByte;
    //        return;
    //    }
    //} else if (workspace_size == 0) {
    //    workspace->data.shape = {1};
    //    workspace->data.dtype = DType::kByte;
    //    return;
    //} else {
    //    NVTE_ERROR("Unexpected workspace_size.");
    //}
}

void fused_attn_arbitrary_seqlen_bwd_qkvpacked(size_t batch, size_t max_seqlen, size_t num_head,
                                  size_t head_dim, float attn_scale, float p_dropout,
                                  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                                  NVTE_Mask_Type mask_type,
                                  const Tensor *input_QKV, const Tensor *input_O,
                                  const Tensor *input_dO, Tensor *output_S,
                                  Tensor *output_dQKV, Tensor *output_dBias,
                                  const Tensor *cu_seqlens, const Tensor *rng_state,
                                  Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle) {
    using namespace transformer_engine;

    // QKV shape is [b, s, 3, h, d]
    void *devPtrQKV = input_QKV->data.dptr;

    auto stride = 2 * num_head * head_dim;
    void *devPtrQ = devPtrQKV;
    void *devPtrK = static_cast<void *>(static_cast<int8_t *>(devPtrQKV) + stride);
    void *devPtrV = static_cast<void *>(static_cast<int8_t *>(devPtrQKV) + 2 * stride);

//    void* devPtrO = input_O->data.dptr;
//    void *devPtrdO = input_dO->data.dptr;

    // dQKV shape is [b, s, 3, h, d]
    void *devPtrdQKV = output_dQKV->data.dptr;
    void *devPtrdQ = devPtrdQKV;
    void *devPtrdK = static_cast<void *>(static_cast<int8_t *>(devPtrdQKV) + stride);
    void *devPtrdV = static_cast<void *>(static_cast<int8_t *>(devPtrdQKV) + 2 * stride);

    Tensor *input_Q, *input_K, *input_V;
    input_Q->data.dptr = devPtrQ;
    input_Q->data.shape = {batch, max_seqlen, num_head, head_dim};
    input_Q->data.dtype = input_QKV->data.dtype;
    input_K->data.dptr = devPtrK;
    input_K->data.shape = {batch, max_seqlen, num_head, head_dim};
    input_K->data.dtype = input_QKV->data.dtype;
    input_V->data.dptr = devPtrV;
    input_V->data.shape = {batch, max_seqlen, num_head, head_dim};
    input_V->data.dtype = input_QKV->data.dtype;

    Tensor *output_dQ, *output_dK, *output_dV;
    output_dQ->data.dptr = devPtrdQ;
    output_dQ->data.shape = {batch, max_seqlen, num_head, head_dim};
    output_dQ->data.dtype = output_dQKV->data.dtype;
    output_dK->data.dptr = devPtrdK;
    output_dK->data.shape = {batch, max_seqlen, num_head, head_dim};
    output_dK->data.dtype = output_dQKV->data.dtype;
    output_dV->data.dptr = devPtrdV;
    output_dV->data.shape = {batch, max_seqlen, num_head, head_dim};
    output_dV->data.dtype = output_dQKV->data.dtype;

    fused_attn_arbitrary_seqlen_bwd(
        batch, max_seqlen, max_seqlen, num_head, head_dim,
        attn_scale, p_dropout, qkv_layout, bias_type, mask_type,
        input_Q, input_K, input_V, input_O, input_dO,
        output_S, output_dQ, output_dK, output_dV, output_dBias,
        cu_seqlens, cu_seqlens, rng_state,
        workspace, stream, handle);

//    void *devPtrSoftmaxStats = nullptr;
//    devPtrSoftmaxStats = output_S->data.dptr;
//
//    void *devPtrCuSeqlens = cu_seqlens->data.dptr;
//
//    void* devPtrDropoutSeed = rng_state->data.dptr;
//    void* devPtrDropoutOffset = reinterpret_cast<void *>(
//                    reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);
//
//    const auto qkv_type = input_QKV->data.dtype;
//    size_t workspace_size = 0;
//
//    bool use_workspace_opt = false;
//#if (CUDNN_VERSION >= 8905)
//    const int device_id = cuda::current_device();
//    const int sm_arch_ = cuda::sm_arch(device_id);
//    if (sm_arch_ >= 90) {
//        // quick estimate of dp workspace size
//        size_t max_seqlen_div_up_q = ((max_seqlen + 64 - 1) / 64) * 64;
//        size_t max_seqlen_div_up_kv = ((max_seqlen + 64 - 1) / 64) * 64;
//        size_t required_dp_workspace =
//        (batch * num_head * max_seqlen_div_up_q * max_seqlen_div_up_kv * 2 + 1048576 - 1) / 1048576;
//        // default upper limit for dp workspace 256MB
//        size_t max_allowed_dp_workspace = 256;
//        if (required_dp_workspace <= max_allowed_dp_workspace) {
//            use_workspace_opt = true;
//        }
//        use_workspace_opt = transformer_engine::getenv<bool>(
//            "NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT", use_workspace_opt);
//#if (CUDNN_VERSION < 8906)
//        NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
//        if ((layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD)
//            || (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_H2D)) {
//                use_workspace_opt = false;
//        }
//#endif
//    }
//#endif
//
//    fused_attn_arbitrary_seqlen_bwd_impl(batch, num_head, max_seqlen, max_seqlen, head_dim,
//                                attn_scale, p_dropout, qkv_layout,
//                                bias_type, mask_type,
//                                devPtrQ, devPtrK, devPtrV, devPtrO, devPtrSoftmaxStats,
//                                devPtrdQ, devPtrdK, devPtrdV, devPtrdO,
//                                devPtrDropoutSeed, devPtrDropoutOffset,
//                                devPtrCuSeqlens, devPtrCuSeqlens,  
//                                get_cudnn_fe_dtype(qkv_type), workspace->data.dptr,
//                                &workspace_size, stream, handle, use_workspace_opt);
//
//    if (workspace_size > 0) {
//        if (workspace->data.dptr == nullptr) {
//            workspace->data.shape = {workspace_size};
//            workspace->data.dtype = DType::kByte;
//            return;
//        }
//    } else if (workspace_size == 0) {
//        workspace->data.shape = {1};
//        workspace->data.dtype = DType::kByte;
//        return;
//    } else {
//        NVTE_ERROR("Unexpected workspace_size.");
//    }
}

void fused_attn_arbitrary_seqlen_fwd(
    size_t batch, size_t max_seqlen_q, size_t max_seqlen_kv,
    size_t num_head, size_t head_dim, bool is_training,
    float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, const Tensor *input_Q, const Tensor *input_K,
    const Tensor *input_V, const Tensor *input_Bias, Tensor *output_O,
    NVTETensorPack *Aux_CTX_Tensors, const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
    const Tensor *rng_state,
    Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle) {
    using namespace transformer_engine;

    const DType QKV_type = input_Q->data.dtype;
    void *devPtrQ = input_Q->data.dptr;
    void *devPtrK = input_K->data.dptr;
    void *devPtrV = input_V->data.dptr;
    void *devPtrBias = input_Bias->data.dptr;
    void *devPtrO = output_O->data.dptr;
    void *devPtrS = nullptr;

    void *devPtrCuSeqlensQ = cu_seqlens_q->data.dptr;
    void *devPtrCuSeqlensKV = cu_seqlens_kv->data.dptr;

    if (Aux_CTX_Tensors->size == 0) {
        Aux_CTX_Tensors->size = 2;
        Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
        output_S->data.dptr = nullptr;
        output_S->data.shape = {batch, num_head, max_seqlen_q, 1};
        output_S->data.dtype = DType::kFloat32;
        Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
        output_rng_state->data.dptr = nullptr;
        output_rng_state->data.shape = {2};
        output_rng_state->data.dtype = DType::kInt64;
    } else if (Aux_CTX_Tensors->size == 2) {
        Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
        devPtrS = output_S->data.dptr;
        Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
        output_rng_state->data.dptr = rng_state->data.dptr;
    } else {
        NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
    }

    void* devPtrDropoutSeed = rng_state->data.dptr;
    void* devPtrDropoutOffset = reinterpret_cast<void *>(
                    reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

    size_t workspace_size = 0;

    bool use_old_impl = false;
    use_old_impl = transformer_engine::getenv<bool>("NVTE_FUSED_ATTN_OLD", use_old_impl);
    if (use_old_impl) {
    fused_attn_arbitrary_seqlen_fwd_impl_old(batch, num_head, max_seqlen_q, max_seqlen_kv, head_dim,
                                is_training, attn_scale, p_dropout, qkv_layout,
                                devPtrQ, devPtrK, devPtrV, devPtrS, devPtrO,
                                devPtrDropoutSeed, devPtrDropoutOffset,
                                devPtrCuSeqlensQ, devPtrCuSeqlensKV,
                                get_cudnn_dtype(QKV_type),
                                workspace->data.dptr, &workspace_size, stream, handle);
    } else {
    fused_attn_arbitrary_seqlen_fwd_impl(batch, num_head, max_seqlen_q, max_seqlen_kv, head_dim,
                                is_training, attn_scale, p_dropout, qkv_layout,
                                bias_type, mask_type,
                                devPtrQ, devPtrK, devPtrV, devPtrS, devPtrO,
                                devPtrDropoutSeed, devPtrDropoutOffset,
                                devPtrCuSeqlensQ, devPtrCuSeqlensKV,
                                get_cudnn_fe_dtype(QKV_type),
                                workspace->data.dptr, &workspace_size, stream, handle);
    }

    if (workspace_size > 0) {
        if (workspace->data.dptr == nullptr) {
            workspace->data.shape = {workspace_size};
            workspace->data.dtype = DType::kByte;
            return;
        }
    } else if (workspace_size == 0) {
        workspace->data.shape = {1};
        workspace->data.dtype = DType::kByte;
        return;
    } else {
        NVTE_ERROR("Unexpected workspace_size.");
    }
}

void fused_attn_arbitrary_seqlen_bwd(size_t batch, size_t max_seqlen_q, size_t max_seqlen_kv,
                                  size_t num_head, size_t head_dim, float attn_scale,
                                  float p_dropout, NVTE_QKV_Layout qkv_layout,
                                  NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                                  const Tensor *input_Q, const Tensor *input_K,
                                  const Tensor *input_V, const Tensor *input_O,
                                  const Tensor *input_dO, Tensor *output_S,
                                  Tensor *output_dQ, Tensor *output_dK, Tensor *output_dV,
                                  Tensor *output_dBias, const Tensor *cu_seqlens_q,
                                  const Tensor *cu_seqlens_kv,
                                  const Tensor *rng_state, Tensor *workspace,
                                  cudaStream_t stream, cudnnHandle_t handle) {
    using namespace transformer_engine;

    const auto QKV_type = input_Q->data.dtype;
    void *devPtrQ = input_Q->data.dptr;
    void *devPtrK = input_K->data.dptr;
    void *devPtrV = input_V->data.dptr;
    void* devPtrO = input_O->data.dptr;
    void *devPtrdO = input_dO->data.dptr;

    void *devPtrdQ = output_dQ->data.dptr;
    void *devPtrdK = output_dK->data.dptr;
    void *devPtrdV = output_dV->data.dptr;
    void *devPtrSoftmaxStats = nullptr;
    devPtrSoftmaxStats = output_S->data.dptr;
//    void *devPtrdBias = output_dBias->data.dptr;

    void *devPtrCuSeqlensQ = cu_seqlens_q->data.dptr;
    void *devPtrCuSeqlensKV = cu_seqlens_kv->data.dptr;

    void* devPtrDropoutSeed = rng_state->data.dptr;
    void* devPtrDropoutOffset = reinterpret_cast<void *>(
                    reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

    size_t workspace_size = 0;

    bool use_workspace_opt = false;
#if (CUDNN_VERSION >= 8905)
    const int device_id = cuda::current_device();
    const int sm_arch_ = cuda::sm_arch(device_id);
    if (sm_arch_ >= 90) {
        // quick estimate of dp workspace size
        size_t max_seqlen_div_up_q = ((max_seqlen_q + 64 - 1) / 64) * 64;
        size_t max_seqlen_div_up_kv = ((max_seqlen_kv + 64 - 1) / 64) * 64;
        size_t required_dp_workspace =
        (batch * num_head * max_seqlen_div_up_q * max_seqlen_div_up_kv * 2 + 1048576 - 1) / 1048576;
        // default upper limit for dp workspace 256MB
        size_t max_allowed_dp_workspace = 256;
        if (required_dp_workspace <= max_allowed_dp_workspace) {
            use_workspace_opt = true;
        }
        use_workspace_opt = transformer_engine::getenv<bool>(
            "NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT", use_workspace_opt);
#if (CUDNN_VERSION < 8906)
        NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
        if ((layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD)
            || (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_H2D)) {
                use_workspace_opt = false;
        }
#endif
    }
#endif

    bool use_old_impl = false;
    use_old_impl = transformer_engine::getenv<bool>("NVTE_FUSED_ATTN_OLD", use_old_impl);
    if (use_old_impl) {
    fused_attn_arbitrary_seqlen_bwd_impl_old(batch, num_head, max_seqlen_q, max_seqlen_kv, head_dim,
                                attn_scale, p_dropout, qkv_layout,
                                devPtrQ, devPtrK, devPtrV, devPtrO, devPtrSoftmaxStats,
                                devPtrdQ, devPtrdK, devPtrdV, devPtrdO,
                                devPtrDropoutSeed, devPtrDropoutOffset,
                                devPtrCuSeqlensQ, devPtrCuSeqlensKV,
                                get_cudnn_dtype(QKV_type), workspace->data.dptr,
                                &workspace_size, stream, handle, use_workspace_opt);
    } else {
    fused_attn_arbitrary_seqlen_bwd_impl(batch, num_head, max_seqlen_q, max_seqlen_kv, head_dim,
                                attn_scale, p_dropout, qkv_layout,
                                bias_type, mask_type,
                                devPtrQ, devPtrK, devPtrV, devPtrO, devPtrSoftmaxStats,
                                devPtrdQ, devPtrdK, devPtrdV, devPtrdO,
                                devPtrDropoutSeed, devPtrDropoutOffset,
                                devPtrCuSeqlensQ, devPtrCuSeqlensKV,
                                get_cudnn_fe_dtype(QKV_type), workspace->data.dptr,
                                &workspace_size, stream, handle, use_workspace_opt);
    }

    if (workspace_size > 0) {
        if (workspace->data.dptr == nullptr) {
            workspace->data.shape = {workspace_size};
            workspace->data.dtype = DType::kByte;
            return;
        }
    } else if (workspace_size == 0) {
        workspace->data.shape = {1};
        workspace->data.dtype = DType::kByte;
        return;
    } else {
        NVTE_ERROR("Unexpected workspace_size.");
    }
}
}  // namespace transformer_engine
#endif  // CUDNN_VERSION >= 8900
