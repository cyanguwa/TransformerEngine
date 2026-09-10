/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file fused_attn_uids.h
 *  \brief Stable UIDs for every logical fused-attention tensor.
 *
 *  cuDNN-frontend graphs identify their inputs/outputs either by the
 *  ``shared_ptr<Tensor_attributes>`` returned at build time, or by an integer
 *  UID assigned via ``Tensor_attributes::set_uid``. The C++ builders currently
 *  key their ``variant_pack`` maps by the ``shared_ptr`` objects, which forces
 *  the executor to own the frontend tensor handles.
 *
 *  To let a Python graph builder hand ``execute()`` a plain ``{UID: ptr}`` map
 *  (i.e. ``cudnn.pygraph`` + ``variant_pack`` keyed by int) without owning any
 *  frontend tensor objects, every logical tensor is given a fixed UID here.
 *  These enumerators are the single source of truth shared by the C++ builders
 *  and the future Python builders, so the two agree on which pointer feeds
 *  which graph slot.
 *
 *  Rules:
 *   - Values are explicit and MUST remain stable; never renumber an existing
 *     tensor. Append new tensors with the next unused value.
 *   - UIDs only need to be unique *within a single graph*. Forward and backward
 *     graphs are distinct, so a UID may legitimately appear in only one pass.
 *   - F16 and FP8 use separate enums because their tensor sets diverge
 *     substantially (FP8 carries descale/scale/amax and transpose tensors).
 *     MXFP8 reuses the FP8 graph path and therefore the FP8 UIDs.
 */

#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_UIDS_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_UIDS_H_

#include <cstdint>

namespace transformer_engine {
namespace fused_attn {

/*! \enum FusedAttnUIDF16
 *  \brief UIDs for the F16/BF16 arbitrary-seqlen graphs (forward and backward).
 */
enum class FusedAttnUIDF16 : int64_t {
  // Core data tensors
  kQ = 1,
  kK = 2,
  kV = 3,
  kO = 4,
  kdO = 5,
  kdQ = 6,
  kdK = 7,
  kdV = 8,
  // Bias
  kBias = 9,
  kdBias = 10,
  // Softmax statistics
  kStats = 11,     // softmax LSE (forward S1 / backward stats)
  kMaxLogit = 12,  // forward S2, only when return_max_logit
  // Host scalar (attention scale), passed through the variant pack
  kAttnScale = 13,
  // Sequence lengths (cu_seqlens or materialized actual_seqlens)
  kSeqQ = 14,
  kSeqKV = 15,
  // Paged-KV page tables
  kPageTableK = 16,
  kPageTableV = 17,
  // Ragged offsets
  kOffsetQ = 18,
  kOffsetK = 19,
  kOffsetV = 20,
  kOffsetO = 21,
  kOffsetStats = 22,
  // Dropout RNG state
  kDropoutSeed = 23,
  kDropoutOffset = 24,
  // Learnable softmax offset (sink attention)
  kSoftmaxOffset = 25,
  kdSoftmaxOffset = 26,
};

/*! \enum FusedAttnUIDFP8
 *  \brief UIDs for the FP8 graphs (forward and backward). Also used by MXFP8.
 */
enum class FusedAttnUIDFP8 : int64_t {
  // Core data tensors
  kQ = 1,
  kK = 2,
  kV = 3,
  kO = 4,
  kdO = 5,
  kdQ = 6,
  kdK = 7,
  kdV = 8,
  kStats = 9,
  // Bias
  kBias = 10,
  kdBias = 11,
  // Descales for inputs
  kDescaleQ = 12,
  kDescaleK = 13,
  kDescaleV = 14,
  kDescaleO = 15,
  kDescaledO = 16,
  kDescaleS = 17,
  kDescaledP = 18,
  // Scales
  kScaleO = 19,
  kScaleS = 20,
  kScaledP = 21,
  kScaledQ = 22,
  kScaledK = 23,
  kScaledV = 24,
  // Amaxes
  kAmaxO = 25,
  kAmaxS = 26,
  kAmaxdP = 27,
  kAmaxdQ = 28,
  kAmaxdK = 29,
  kAmaxdV = 30,
  // Transposed / f16 helper tensors (backward)
  kQt = 31,
  kKt = 32,
  kdOf16 = 33,
  kdOt = 34,
  kDescaleQt = 35,
  kDescaleKt = 36,
  kDescaledOt = 37,
  // Sequence lengths
  kSeqQ = 38,
  kSeqKV = 39,
  // Ragged offsets
  kOffsetQ = 40,
  kOffsetK = 41,
  kOffsetV = 42,
  kOffsetO = 43,
  kOffsetStats = 44,
  // Dropout RNG state
  kDropoutSeed = 45,
  kDropoutOffset = 46,
  // Learnable softmax offset (sink attention)
  kSoftmaxOffset = 47,
  kdSoftmaxOffset = 48,
};

}  // namespace fused_attn
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_UIDS_H_
