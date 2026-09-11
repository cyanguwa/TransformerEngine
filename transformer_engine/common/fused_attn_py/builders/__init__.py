# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuDNN Python graph builders for fused attention.

Each builder turns a derived ``FusedAttnConfig`` plus per-tensor BHSD dim/stride
descriptors into a built ``cudnn.pygraph``, mirroring the corresponding C++
``create_graph_*`` function. The framework glue
(PyTorch / JAX) supplies the descriptors and the variant-pack pointers; the
builders themselves are framework-neutral and depend only on the ``cudnn``
Python package (injected, never imported at module load).
"""
