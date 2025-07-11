# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Methods needed for distributed training (DP/TP)."""
from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager, AbstractContextManager, ContextDecorator
from functools import lru_cache
from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import torch
from torch.cuda import _lazy_call, _lazy_init
from torch.utils.checkpoint import detach_variable, noop_context_fn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import _get_module_fsdp_state
from torch.distributed.fsdp._traversal_utils import _get_fsdp_states_with_modules

try:
    import torch.distributed._symmetric_memory as symm_mem

    HAS_TORCH_SYMMETRIC = True
except ImportError:
    HAS_TORCH_SYMMETRIC = False

import transformer_engine_torch as tex

from . import torch_version
from .utils import (
    is_non_tn_fp8_gemm_supported,
    safely_set_viewless_tensor_data,
    needs_quantized_gemm,
)
from .constants import dist_group_type
from .fp8 import FP8GlobalStateManager, fp8_autocast
from .tensor.float8_tensor import Float8Quantizer, Float8Tensor, Float8CurrentScalingQuantizer
from .tensor.mxfp8_tensor import MXFP8Quantizer
from .tensor.float8_blockwise_tensor import Float8BlockQuantizer
from .tensor.quantized_tensor import QuantizedTensor, Quantizer
from .tensor._internal.float8_tensor_base import Float8TensorBase
from .tensor._internal.mxfp8_tensor_base import MXFP8TensorBase
from .tensor._internal.float8_blockwise_tensor_base import Float8BlockwiseQTensorBase
from ..debug.pytorch.debug_quantization import DebugQuantizedTensor, DebugQuantizer


__all__ = ["checkpoint", "CudaRNGStatesTracker"]


_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    "tensor_model_parallel": False,
    "partition_dim": -1,
    "partition_stride": 1,
}

_USE_REENTRANT_ACTIVATION_RECOMPUTE = True

_FP8_ACTIVATION_RECOMPUTE_ENABLED = False
_FP8_ACTIVATION_RECOMPUTE_PHASE = False


_ALL_ACTIVE_RNG_STATES = {}


def get_all_rng_states() -> bool:
    """Returns all generator states used by `CudaRNGStatesTracker`."""
    return _ALL_ACTIVE_RNG_STATES


def set_all_rng_states(states: List) -> None:
    """Updates all generator states used by `CudaRNGStatesTracker`."""
    global _ALL_ACTIVE_RNG_STATES
    _ALL_ACTIVE_RNG_STATES = states


def graph_safe_rng_available() -> bool:
    """Returns whether cuda graph safe RNG state manipulation is supported."""
    return (
        hasattr(torch.cuda.CUDAGraph, "register_generator_state")
        and hasattr(torch.Generator, "graphsafe_set_state")
        and hasattr(torch.Generator, "graphsafe_get_state")
        and hasattr(torch.Generator, "clone_state")
    )


def _get_cuda_rng_state(
    device: Union[int, str, torch.device] = "cuda",
    clone: bool = False,
    graph_safe: bool = True,
) -> torch.Tensor:
    """Return the random number generator state of the specified GPU."""

    _lazy_init()
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("cuda", device)
    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()
    default_generator = torch.cuda.default_generators[idx]
    if graph_safe_rng_available() and graph_safe:
        if clone:
            # Reference to the cloned generator state
            return default_generator.clone_state()
        # Reference to the current generator state
        return default_generator.graphsafe_get_state()
    return default_generator.get_state()


def _set_cuda_rng_state(
    new_state: torch.Tensor,
    device: Union[int, str] = -1,
    graph_safe=True,
) -> None:
    """Sets the random number generator state of the current GPU."""

    if device == -1:
        device = torch.device("cuda")
    elif isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("cuda", device)

    def cb() -> None:
        idx = device.index
        if idx is None:
            idx = torch.cuda.current_device()
        default_generator = torch.cuda.default_generators[idx]
        if graph_safe_rng_available() and graph_safe:
            default_generator.graphsafe_set_state(new_state)
            return
        default_generator.set_state(new_state)

    _lazy_call(cb)


def set_tensor_model_parallel_attributes(
    tensor: torch.Tensor, is_parallel: bool, dim: int, stride: int
) -> None:
    """set attributes needed for TP"""
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, "tensor_model_parallel", is_parallel)
    setattr(tensor, "partition_dim", dim)
    setattr(tensor, "partition_stride", stride)


@lru_cache
def get_distributed_world_size(group: Optional[dist_group_type] = None) -> int:
    """Return world size for the distributed group."""
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group=group)


@lru_cache
def get_distributed_rank(group: Optional[dist_group_type] = None) -> int:
    """Return my rank for the distributed group."""
    assert torch.distributed.is_initialized(), "torch.distributed is not initialized."
    return torch.distributed.get_rank(group=group)


def initialize_affine_weight_gpu(
    weight: torch.Tensor,
    init_method: Callable,
    get_rng_state_tracker: Callable,
    partition_dim: int = 0,
    stride: int = 1,
    set_tp_attributes: bool = True,
) -> None:
    """Initialize affine weight for model parallel on GPU."""

    if set_tp_attributes:
        set_tensor_model_parallel_attributes(
            tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
        )

    if get_rng_state_tracker is None:
        init_method(weight)
        return

    with get_rng_state_tracker().fork():
        init_method(weight)


def split_tensor_into_1d_equal_chunks(
    tensor: torch.Tensor, tp_group: dist_group_type, new_buffer: bool = False
) -> torch.Tensor:
    """Break a tensor into equal 1D chunks."""
    partition_size = torch.numel(tensor) // get_distributed_world_size(tp_group)
    start_index = partition_size * get_distributed_rank(tp_group)
    end_index = start_index + partition_size
    if new_buffer:
        data = torch.empty(
            partition_size,
            dtype=tensor.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
        data.copy_(tensor.view(-1)[start_index:end_index])
    else:
        data = tensor.view(-1)[start_index:end_index]
    return data


def gather_split_1d_tensor(tensor: torch.Tensor, tp_group: dist_group_type) -> torch.Tensor:
    """Opposite of above function, gather values from model parallel ranks."""
    numel_gathered = torch.numel(tensor) * get_distributed_world_size(tp_group)
    gathered = torch.empty(
        numel_gathered,
        dtype=tensor.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    torch.distributed.all_gather_into_tensor(gathered, tensor, group=tp_group)
    return gathered


class activation_recompute_forward(AbstractContextManager, ContextDecorator):
    """Context manager used to control the forward runtime behavior when executed
    under the `CheckpointFunction` function. For running FP8, the forward pass will
    run without storing intermediate activations. Instead, the forward pass saves
    the inputs tuple and the calling function. In the backwards pass, these are
    retrieved, and the forward pass is computed again while tracking the intermediate
    activations, followed by calculation of gradients using these values.
    """

    _is_first_fp8_module: List = []

    def __init__(self, activation_recompute: bool = False, recompute_phase: bool = False):
        super().__init__()
        self.activation_recompute = activation_recompute
        self.recompute_phase = recompute_phase

    def __enter__(self):
        global _FP8_ACTIVATION_RECOMPUTE_ENABLED, _FP8_ACTIVATION_RECOMPUTE_PHASE
        _FP8_ACTIVATION_RECOMPUTE_ENABLED = (
            self.activation_recompute and FP8GlobalStateManager.is_fp8_enabled()
        )
        _FP8_ACTIVATION_RECOMPUTE_PHASE = self.recompute_phase

        if self.activation_recompute and not self.recompute_phase:
            activation_recompute_forward._is_first_fp8_module.append(
                FP8GlobalStateManager.IS_FIRST_FP8_MODULE
            )
        if self.activation_recompute and self.recompute_phase:
            FP8GlobalStateManager.IS_FIRST_FP8_MODULE = (
                activation_recompute_forward._is_first_fp8_module.pop(0)
            )

    def __exit__(self, *exc_details):
        global _FP8_ACTIVATION_RECOMPUTE_ENABLED, _FP8_ACTIVATION_RECOMPUTE_PHASE
        _FP8_ACTIVATION_RECOMPUTE_ENABLED = False
        _FP8_ACTIVATION_RECOMPUTE_PHASE = False


def is_fp8_activation_recompute_enabled() -> bool:
    """Return global boolean"""
    return _FP8_ACTIVATION_RECOMPUTE_ENABLED


def in_fp8_activation_recompute_phase() -> bool:
    """Return global boolean"""
    return _FP8_ACTIVATION_RECOMPUTE_PHASE


def _get_active_autocast_contexts():
    """
    Returns new CPU and GPU torch.amp.autocast(..) contexts that match the active autocast state
    at the time of this function's execution.
    """
    autocast_cached = torch.is_autocast_cache_enabled()

    if torch_version() >= (2, 4, 0):
        gpu_autocast_enabled = torch.is_autocast_enabled("cuda")
        gpu_autocast_dtype = torch.get_autocast_dtype("cuda")
        gpu_autocast_ctx = torch.amp.autocast(
            "cuda",
            enabled=gpu_autocast_enabled,
            dtype=gpu_autocast_dtype,
            cache_enabled=autocast_cached,
        )

        cpu_autocast_enabled = torch.is_autocast_enabled("cpu")
        cpu_autocast_dtype = torch.get_autocast_dtype("cpu")
        cpu_autocast_ctx = torch.amp.autocast(
            "cpu",
            enabled=cpu_autocast_enabled,
            dtype=cpu_autocast_dtype,
            cache_enabled=autocast_cached,
        )
    else:
        gpu_autocast_enabled = torch.is_autocast_enabled()
        gpu_autocast_dtype = torch.get_autocast_gpu_dtype()
        gpu_autocast_ctx = torch.cuda.amp.autocast(
            gpu_autocast_enabled, gpu_autocast_dtype, autocast_cached
        )

        cpu_autocast_enabled = torch.is_autocast_cpu_enabled()
        cpu_autocast_dtype = torch.get_autocast_cpu_dtype()
        cpu_autocast_ctx = torch.cpu.amp.autocast(
            cpu_autocast_enabled, cpu_autocast_dtype, autocast_cached
        )

    return gpu_autocast_ctx, cpu_autocast_ctx


class _CheckpointFunction(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
    two main changes:
        1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
        2) the states in the model parallel tracker are also properly
           tracked/set/reset.
    """

    @staticmethod
    def forward(
        ctx,
        run_function: Callable,
        distribute_saved_activations: bool,
        get_rng_state_tracker: Union[Callable, None],
        tp_group: Union[dist_group_type, None],
        context_fn: Union[Callable, None],
        kwargs: Dict[str, Any],
        *args: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """Call forward function while saving state to be able to
        redo the computation later."""
        ctx.run_function = run_function
        ctx.distribute_saved_activations = distribute_saved_activations

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = _get_cuda_rng_state(graph_safe=False)
        if get_rng_state_tracker is not None:
            ctx.fwd_cuda_rng_state_tracker = get_rng_state_tracker().get_states()

        if context_fn is not None:
            forward_ctx, recompute_ctx = context_fn()
        else:
            forward_ctx, recompute_ctx = noop_context_fn()

        # Preserve torch autocast context for the backward pass
        torch_gpu_amp_ctx, torch_cpu_amp_ctx = _get_active_autocast_contexts()

        with torch.no_grad(), forward_ctx:
            with activation_recompute_forward(activation_recompute=True, recompute_phase=False):
                outputs = run_function(*args, **kwargs)

        # Divide hidden states across model parallel group and only keep
        # the chunk corresponding to the current rank.
        if distribute_saved_activations:
            ctx.input_0_shape = args[0].data.shape
            safely_set_viewless_tensor_data(
                args[0],
                split_tensor_into_1d_equal_chunks(args[0].data, tp_group, new_buffer=True),
            )

        # Store everything.
        ctx.inputs = [arg if not torch.is_tensor(arg) else None for arg in args]
        tensor_inputs = [arg if torch.is_tensor(arg) else None for arg in args]
        ctx.save_for_backward(*tensor_inputs)

        fp8 = FP8GlobalStateManager.is_fp8_enabled()
        ctx.get_rng_state_tracker = get_rng_state_tracker
        ctx.tp_group = tp_group
        ctx.recompute_ctx = recompute_ctx
        ctx.torch_gpu_amp_ctx = torch_gpu_amp_ctx
        ctx.torch_cpu_amp_ctx = torch_cpu_amp_ctx
        ctx.fp8 = fp8
        ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
        ctx.kwargs = kwargs

        return outputs

    @staticmethod
    def backward(
        ctx, *args: Tuple[Union[torch.Tensor, None], ...]
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """Call backward function with activation recomputation."""
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), please use .backward() if possible"
            )

        inputs = tuple(
            t if t is not None else arg for (t, arg) in zip(ctx.saved_tensors, ctx.inputs)
        )

        get_rng_state_tracker = ctx.get_rng_state_tracker

        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(
                inputs[0],
                gather_split_1d_tensor(inputs[0].data, ctx.tp_group).view(ctx.input_0_shape),
            )

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = _get_cuda_rng_state(graph_safe=False)
        if get_rng_state_tracker is not None:
            bwd_cuda_rng_state_tracker = get_rng_state_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state, graph_safe=False)
        if get_rng_state_tracker is not None:
            get_rng_state_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # Compute the forward pass.
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad(), ctx.recompute_ctx, ctx.torch_gpu_amp_ctx, ctx.torch_cpu_amp_ctx, activation_recompute_forward(
            activation_recompute=True, recompute_phase=True
        ), fp8_autocast(
            enabled=ctx.fp8, fp8_recipe=ctx.fp8_recipe
        ):
            outputs = ctx.run_function(*detached_inputs, **ctx.kwargs)

        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state, graph_safe=False)
        if get_rng_state_tracker is not None:
            get_rng_state_tracker().set_states(bwd_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        outputs_with_grad = []
        args_with_grad = []
        for i, output in enumerate(outputs):
            if torch.is_tensor(output) and output.requires_grad:
                outputs_with_grad.append(output)
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "none of output has requires_grad=True, this checkpoint() is not necessary"
            )

        # backward does not require entering autocast context because
        # backward implementations already retrieve fp8 recipe and
        # enablement from stored ctx.
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None for inp in detached_inputs
        )
        return (None, None, None, None, None, None) + grads


class _CheckpointFrame:
    """
    Storage frame for forward RNG states and detached activations from the forward recompute.
    """

    def __init__(self, recompute_fn: Callable, get_rng_state_tracker: Callable):
        self.recompute_fn = recompute_fn
        self.recomputed = []
        self.count = 0
        self.get_rng_state_tracker = get_rng_state_tracker
        self.fwd_rng_states = None
        self.bwd_rng_states = None

    def cache_rng_states(self, forward=True):
        """Cache fwd/bwd RNG states in the frame to restore later."""
        rng_states = (
            torch.get_rng_state(),
            _get_cuda_rng_state(graph_safe=False),
        )
        if self.get_rng_state_tracker is not None:
            rng_states += (self.get_rng_state_tracker().get_states(),)

        if forward:
            self.fwd_rng_states = rng_states
        else:
            self.bwd_rng_states = rng_states

    def restore_rng_states(self, forward=True):
        """Restore fwd/bwd RNG states that were previously cached into the frame."""
        if forward:
            rng_states = self.fwd_rng_states
        else:
            rng_states = self.bwd_rng_states

        torch.set_rng_state(rng_states[0])
        _set_cuda_rng_state(rng_states[1], graph_safe=False)
        if self.get_rng_state_tracker is not None:
            self.get_rng_state_tracker().set_states(rng_states[2])


class _recomputation_hook(
    torch.autograd.graph.saved_tensors_hooks
):  # pylint: disable=too-few-public-methods
    """torch.autograd hook for packing/unpacking tensors during the activation recompute phase."""

    def __init__(self, frame):

        def pack_hook(x):
            """
            Packing hook for each recomputed activation passed into the `ctx.save_for_backward()`
            call in the forward recomputation.
            """
            frame.recomputed.append(x.detach())
            return x.detach()

        def unpack_hook(x):
            """
            No-op unpack hook that will never be called because the backward pass for the
            forward recomputation is never triggered.
            """
            return x

        super().__init__(pack_hook, unpack_hook)


class _checkpoint_hook(
    torch.autograd.graph.saved_tensors_hooks
):  # pylint: disable=too-few-public-methods
    """torch.autograd hook for packing/unpacking tensors during the checkpointed forward pass."""

    def __init__(self, frame, args, kwargs):

        def pack_hook(x):
            """
            Packing hook for each tensor passed into `ctx.save_for_backward()` call in the
            forward pass. Since this is the first forward pass, we discard the tensor and instead
            pack a placeholder tensor index into the autograd engine context.
            """
            del x
            idx = frame.count
            frame.count += 1
            return idx

        def unpack_hook(idx):
            """
            Unpacking hook for each tensor that comes out of the `ctx.saved_tensors` call in the
            backward pass. The first time this is called, the _recomputation_hook will save all the
            activation tensors from `ctx.save_for_backward()` in the forward recomputation into the
            _CheckpointFrame. Subsequent calls will simply return the already recomputed activation
            tensor at the given index of the _CheckpointFrame storage.
            """

            if not frame.recomputed:
                # Store current RNG states in the backward pass
                frame.cache_rng_states(forward=False)

                # Set RNG states to what we saved before the forward pass
                frame.restore_rng_states(forward=True)

                # Recompute the forward pass
                with _recomputation_hook(frame):
                    frame.recompute_fn(*args, **kwargs)

                # Restore RNG states back to the backward pass
                frame.restore_rng_states(forward=False)

            # Return the already recomputed activation tensor at the given index
            activation = frame.recomputed[idx]
            frame.recomputed[idx] = None
            return activation

        super().__init__(pack_hook, unpack_hook)


def use_reentrant_activation_recompute():
    """Returns `True` if activation recompute is using the 'reentrant' method."""
    return _USE_REENTRANT_ACTIVATION_RECOMPUTE


def get_activation_recompute_contexts():
    """Returns context objects for the checkpointed forward pass and the forward recompute phase."""
    forward_ctx = activation_recompute_forward(
        activation_recompute=True,
        recompute_phase=False,
    )
    recompute_ctx = activation_recompute_forward(
        activation_recompute=True,
        recompute_phase=True,
    )
    return forward_ctx, recompute_ctx


def has_te_modules(network):
    """
    Check if there are any Transformer Engine modules in the network.
    """
    from .module import LayerNorm, RMSNorm
    from .module.base import TransformerEngineBaseModule
    from .attention.dot_product_attention.backends import UnfusedDotProductAttention
    from .attention.dot_product_attention.dot_product_attention import DotProductAttention
    from .attention.multi_head_attention import MultiheadAttention
    from .transformer import TransformerLayer

    te_classes_list = [
        LayerNorm,
        RMSNorm,
        TransformerEngineBaseModule,
        UnfusedDotProductAttention,
        DotProductAttention,
        MultiheadAttention,
        TransformerLayer,
    ]

    if isinstance(network, torch.nn.Module):
        for module in network.modules():
            if any(isinstance(module, te_class) for te_class in te_classes_list):
                return True
        return False

    # Cannot check for TE modules inside a custom class/callable that's not a torch.nn.Module,
    # so just assume that it has TE modules just to be safe.
    return True


@torch._disable_dynamo
def checkpoint(
    function: Callable,
    *args: Tuple[torch.Tensor, ...],
    **kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, ...]:
    """
    Checkpoint a part of the model by trading compute for memory. This function is based on
    `torch.utils.checkpoint.checkpoint <https://pytorch.org/docs/stable/checkpoint.html>`_.

    .. warning::

        It is the user's responsibility to ensure identical behavior when calling
        :attr:`function` from the forward and backward pass. If different output is
        produced (e.g. due to global state), then the checkpointed version won't
        be numerically equivalent.

    .. warning::
        `use_reentrant=False` does not support early stopping, and will execute the entire forward
        pass for the checkpointed module when recomputing activations in the backward pass.

    Parameters
    ----------
    function: Callable
            pytorch module used to run the forward and backward passes using
            the specified :attr:`args` and :attr:`kwargs`.
    distribute_saved_activations: bool, default = False
            if set to `True` and `use_reentrant=True`, first tensor argument is distributed
            across the specified tensor parallel group (`tp_group`) before saving it for the
            backward pass. This has no effect when `use_reentrant=False`.
    get_rng_state_tracker: `Callable`, default = None
            python callable which returns an instance of :func:`CudaRNGStatesTracker`.
    tp_group : ProcessGroup, default = None
            tensor parallel process group. Used only when `distribute_saved_activations=True`
            and `use_reentrant=True`. If `None`, it falls back to the default group.
    use_reentrant : bool, default = True
            perform checkpointing in reentrant mode.
    args : tuple
            tuple of torch tensors for inputs to :attr:`function`.
    kwargs : dict
            dictionary of string keys for keyword arguments to :attr:`function`.
    """
    # Pop out te.distributed.checkpoint() arguments
    global _USE_REENTRANT_ACTIVATION_RECOMPUTE
    _USE_REENTRANT_ACTIVATION_RECOMPUTE = kwargs.pop("use_reentrant", True)
    distribute_saved_activations = kwargs.pop("distribute_saved_activations", False)
    tp_group = kwargs.pop("tp_group", None)
    get_rng_state_tracker = kwargs.pop("get_rng_state_tracker", None)

    # Ensure backward compatibility.
    if (
        len(args) > 3
        and isinstance(args[0], bool)
        and callable(args[1])
        and isinstance(args[2], None | dist_group_type)
    ):
        warnings.warn(
            "Passing non-tensor non-keyword arguments is deprecated and support will be removed in "
            "future releases of TransformerEngine. `distribute_saved_activations`, `tp_group`, and "
            "`get_rng_state_tracker` must be passed as keyword arguments to `checkpoint`.",
            DeprecationWarning,
            stacklevel=2,
        )
        distribute_saved_activations = args[0]
        get_rng_state_tracker = args[1]
        tp_group = args[2]
        args = args[3:]

    # Trigger the native PyTorch checkpoint if the function is not or does not contain a
    # Transformer Engine module.
    context_fn = kwargs.pop("context_fn", noop_context_fn)
    determinism_check = kwargs.pop("determinism_check", "default")
    debug = kwargs.pop("debug", False)
    if not has_te_modules(function):
        return torch.utils.checkpoint.checkpoint(
            function,
            *args,
            use_reentrant=_USE_REENTRANT_ACTIVATION_RECOMPUTE,
            context_fn=context_fn,
            determinism_check=determinism_check,
            debug=debug,
            **kwargs,
        )

    from .module.base import TransformerEngineBaseModule

    if isinstance(function, TransformerEngineBaseModule):
        # If this TE module is FSDP-wrapped, clear its FSDP group information because there's no need
        # to scatter/gather activations that we will recompute anyway.
        setattr(function, "fsdp_wrapped", False)
        setattr(function, "fsdp_group", None)

    # Otherwise discard unused te.utils.checkpoint.checkpoint() arguments
    # and execute TE's own checkpointing
    # NOTE: This logic uses the TE checkpoint on all custom callable `function` handles because we
    #       cannot be sure there are no TE modules inside the function. It also means we might run
    #       the TE checkpoint for non-TE modules, so the TE checkpoint has to support a potential
    #       user context function.
    del determinism_check, debug
    if _USE_REENTRANT_ACTIVATION_RECOMPUTE:
        # If saved activations need to be distributed but there is no process group,
        # default to the world group.
        if distribute_saved_activations:
            assert torch.distributed.is_initialized(), "torch.distributed is not initialized."
            tp_group = torch.distributed.GroupMember.WORLD if tp_group is None else tp_group

        return _CheckpointFunction.apply(
            function,
            distribute_saved_activations,
            get_rng_state_tracker,
            tp_group,
            context_fn,
            kwargs,
            *args,
        )

    if distribute_saved_activations:
        warnings.warn(
            "`distribute_saved_activations=True` has no effect when `use_reentrant=False`. "
            "The non-reentrant checkpoint implementation does not manually store forward "
            "inputs for the activation recompute in the backward pass, and instead leverages "
            "the autograd engine's pack/unpack hooks."
        )

    user_forward_ctx, user_recompute_ctx = context_fn()
    te_forward_ctx, te_recompute_ctx = get_activation_recompute_contexts()

    # Preserve the torch autocast contexts from the forward pass during recompute phase.
    torch_gpu_amp_forward_ctx, torch_cpu_amp_forward_ctx = _get_active_autocast_contexts()

    fp8 = FP8GlobalStateManager.is_fp8_enabled()
    fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None

    def recompute_fn(*args, **kwargs):
        with torch.autograd.enable_grad(), (
            te_recompute_ctx
        ), user_recompute_ctx, torch_gpu_amp_forward_ctx, torch_cpu_amp_forward_ctx, fp8_autocast(
            enabled=fp8, fp8_recipe=fp8_recipe
        ):
            function(*args, **kwargs)

    # Initialize a new checkpoint frame for each new forward pass.
    new_frame = _CheckpointFrame(
        recompute_fn,
        get_rng_state_tracker,
    )
    new_frame.cache_rng_states(forward=True)

    with _checkpoint_hook(new_frame, args, kwargs), te_forward_ctx, user_forward_ctx:
        out = function(*args, **kwargs)

    return out


class CudaRNGStatesTracker:
    """
    For model parallelism, multiple RNG states need to simultaneously exist in order
    to execute operations in or out of the model parallel region. This class keeps
    track of the various RNG states and provides utility methods to maintain them and
    execute parts of the model under a given RNG setting. Using the `add` method, a
    cuda rng state is initialized based on the input `seed` and is assigned to `name`.
    Later, by forking the rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self):
        # Map from a string name to the cuda rng state.
        self.states_ = {}
        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def reset(self):
        """
        Set to the initial state (no tracker).
        """
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self) -> Dict[str, torch.Tensor]:
        """
        Get rng states. Copy the dictionary so we have direct pointers
        to the states, not just a pointer to the dictionary.
        """
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states: Dict[str, torch.Tensor]) -> None:
        """
        Set the rng states. For efficiency purposes, we do not
        check the size of seed for compatibility.

        states: Dict[str, torch.Tensor]
               A mapping from string names to RNG states.
        """
        self.states_ = states

    def add(self, name: str, seed: int) -> None:
        """
        Adds a new RNG state.

        name: str
             string identifier for the RNG state.
        seed: int
             PyTorch seed for the RNG state.
        """
        # Check seed is not already used.
        if seed in self.seeds_:
            raise RuntimeError(f"seed {seed} already exists")
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise RuntimeError(f"cuda rng state {name} already exists")

        if graph_safe_rng_available():
            new_state = _get_cuda_rng_state(clone=True)
            new_state.manual_seed(seed)
            self.states_[name] = new_state
            # Update global states.
            set_all_rng_states(self.states_)
        else:
            # Get the current rng state.
            orig_rng_state = _get_cuda_rng_state()
            # Set the new state and store it.
            torch.cuda.manual_seed(seed)
            self.states_[name] = _get_cuda_rng_state(clone=True)
            # Reset rng state to what it was.
            _set_cuda_rng_state(orig_rng_state)
            # Update global states.
            set_all_rng_states(self.states_)

    @contextmanager
    def fork(self, name: str = "model-parallel-rng"):
        """
        Fork the cuda rng state, perform operations, and exit with
        the original state.

        name: str
             string identifier for the RNG state.
        """
        # Check if we have added the state
        if name not in self.states_:
            raise KeyError(f"cuda rng state {name} is not added")
        # Get the reference to current rng state.
        orig_cuda_rng_state = _get_cuda_rng_state()
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name])
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # this is redundant with graph-safe API
            if not graph_safe_rng_available():
                self.states_[name] = _get_cuda_rng_state()
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state)


def reduce_scatter_along_first_dim(
    inp: torch.Tensor, tp_group: dist_group_type, async_op: bool = False
) -> Tuple[torch.Tensor, Optional[torch.distributed.Work]]:
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = get_distributed_world_size(tp_group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return inp, None

    dim_size = list(inp.size())
    assert (
        dim_size[0] % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"

    dim_size[0] = dim_size[0] // world_size

    output = torch.empty(dim_size, dtype=inp.dtype, device=torch.cuda.current_device())
    handle = torch.distributed.reduce_scatter_tensor(
        output, inp.contiguous(), group=tp_group, async_op=async_op
    )
    return output, handle


def _all_gather_fp8(
    inp: torch.Tensor,
    process_group: dist_group_type,
    *,
    async_op: bool = False,
    quantizer: Optional[Quantizer] = None,
    out_shape: Optional[list[int]] = None,
) -> tuple[Float8TensorBase, Optional[torch.distributed.Work]]:
    """All-gather FP8 tensor along first dimension."""
    world_size = get_distributed_world_size(process_group)

    # Check that quantizer is valid
    if quantizer is not None and not isinstance(
        quantizer, (Float8Quantizer, Float8CurrentScalingQuantizer)
    ):
        raise ValueError(f"Got non-FP8 quantizer ({quantizer.__class__.__name__})")

    # Output tensor dims
    if out_shape is None:
        out_shape = list(inp.size())
        out_shape[0] *= world_size

    # Cast input tensor to FP8 if needed
    # Note: We cannot directly all-gather the transposed FP8 tensor,
    # so temporarily modify quantizer to avoid creating FP8 transpose.
    if not isinstance(inp, Float8TensorBase):
        assert isinstance(quantizer, (Float8Quantizer, Float8CurrentScalingQuantizer))
        # we cannot directly gather the transposed fp8 tensor
        # so we need to disable columnwise usage for the quantizer
        # and then set it back to the original value after quantizing
        init_rowwise_usage = quantizer.rowwise_usage
        init_columnwise_usage = quantizer.columnwise_usage
        quantizer.set_usage(rowwise=True, columnwise=False)
        inp = quantizer(inp)
        quantizer.set_usage(
            rowwise=init_rowwise_usage,
            columnwise=init_columnwise_usage,
        )

    # Construct output tensor
    out: Float8TensorBase
    if quantizer is not None:
        dtype = torch.float32
        device = "cuda"
        if isinstance(inp, Float8Tensor):
            dtype = inp.dtype
            device = inp.device
        out = quantizer.make_empty(out_shape, dtype=dtype, device=device)
    elif isinstance(inp, Float8Tensor):
        out = inp.make_like(inp, shape=out_shape)
        out._data = torch.empty(
            out_shape,
            dtype=torch.uint8,
            device=inp.device,
        )
        out._transpose = None
        out._transpose_invalid = True
    else:
        raise RuntimeError("FP8TensorBase is not supported yet without Quantizer")

    # Assume scaling factors are identical across ranks
    out._scale_inv = inp._scale_inv

    # Perform communication
    handle = torch.distributed.all_gather_into_tensor(
        out._data,
        inp._data.contiguous(),
        group=process_group,
        async_op=async_op,
    )

    # Make sure FP8 transpose is populated if needed
    needs_transpose = (
        quantizer is not None and quantizer.columnwise_usage and not is_non_tn_fp8_gemm_supported()
    )
    if needs_transpose:
        if handle is not None:
            handle.wait()
            handle = None
        out._create_transpose()

    return out, handle


def _set_quantizer_format(quantizer: Quantizer, compact: bool = False) -> None:
    """Make quantizer compact"""
    _quantizer = quantizer
    if isinstance(quantizer, DebugQuantizer):
        _quantizer = quantizer.parent_quantizer
    if isinstance(_quantizer, Float8BlockQuantizer):
        _quantizer.all_gather_usage = compact


def _post_process_fp8_blockwise_gather(
    out: Float8BlockwiseQTensorBase,
    quantizer: Float8BlockQuantizer,
    handle: Optional[torch.distributed.Work] = None,
) -> Float8BlockwiseQTensorBase:
    """Post-process FP8 blockwise gather."""
    if handle is not None:
        handle.wait()
        handle = None

    if out._is_gemm_ready_format():
        return out

    needs_columnwise_data_transpose = (
        quantizer is not None and quantizer.columnwise_usage and not is_non_tn_fp8_gemm_supported()
    )
    need_rowwise_scale_transpose = (
        quantizer is not None and quantizer.rowwise_usage and not is_non_tn_fp8_gemm_supported()
    )

    # CuBLAS requires transpose of the scale inv tensor, suppose orig input is 256x1024
    # columnwise compact format means doing 128x1 quantization of it
    # so quantized tensor is 256x1024, scale inv is 2x1024
    # If we were doing GEMM_READY format, then it's equivalent to do 1x128 quantization
    # on a transposed 1024x256 tensor, so scale inv is 1024x2, cublas requries 2x1024
    # Thereforce, it turns out we don't need to transpose the scale inv, only columnwise data
    if needs_columnwise_data_transpose:
        out._transpose_columnwise_data()
    if need_rowwise_scale_transpose:
        out._rowwise_scale_inv = out._rowwise_scale_inv.transpose(-2, -1).contiguous()
    out._data_format = tex.Float8BlockScaleTensorFormat.GEMM_READY
    return out


@dataclass
class _FP8BlockwiseAllGatherAsyncHandle:
    """Handle for asynchronous FP8 blockwise all-gather."""

    tensor: Float8BlockwiseQTensorBase
    quantizer: Float8BlockQuantizer
    async_handle: torch.distributed.Work
    _synchronized: bool = False

    def wait(self) -> None:
        """Wait for the async operation to complete and post-process the tensor."""
        if self._synchronized:
            return
        self.async_handle.wait()
        _post_process_fp8_blockwise_gather(self.tensor, self.quantizer)
        self._synchronized = True


def _all_gather_fp8_blockwise(
    inp: torch.Tensor,
    process_group: dist_group_type,
    *,
    async_op: bool = False,  # pylint: disable=unused-argument
    quantizer: Optional[Quantizer] = None,
    out_shape: Optional[list[int]] = None,
) -> tuple[torch.Tensor, Optional[torch.distributed.Work]]:
    """
    All-gather FP8 tensor along first dimension for blockwise quantization.

    Returns: quantizer(gather(inp))

    NOTE: The implementation is only going to honor async_op=True for FP8 gather case.
    In the case where tensor shape is not divisible by 128, the implementation will fall back
    to synchronous gather and invoke the quantizer.
    """

    # Input tensor attributes
    device: torch.device
    dtype: torch.dtype
    if isinstance(inp, torch.Tensor):
        device = inp.device
        dtype = inp.dtype
    elif isinstance(inp, Float8BlockwiseQTensorBase):
        if inp._rowwise_data is not None:
            device = inp._rowwise_data.device
        elif inp._columnwise_data is not None:
            device = inp._columnwise_data.device
        else:
            raise ValueError("Got Float8BlockwiseQTensorBase input tensor without any data")
        dtype = torch.bfloat16  # Only has fp8 dtype. Guess BF16 for dequant.
    else:
        raise ValueError(
            "Invalid type for input tensor (expected torch.Tensor or Float8BlockwiseQTensorBase, "
            f"found {inp.__class__.__name__})"
        )
    world_size = get_distributed_world_size(process_group)

    # Check that quantizer is valid
    if quantizer is not None and not isinstance(quantizer, Float8BlockQuantizer):
        raise ValueError(f"Got non-FP8 blockwise quantizer ({quantizer.__class__.__name__})")
    if not (quantizer.block_scaling_dim == 1 and quantizer.block_len == 128):
        raise NotImplementedError("Only 1D blockwise quantization is supported for allgather")

    # Output tensor dims
    if out_shape is None:
        out_shape = list(inp.size())
        out_shape[0] *= world_size

    # Doing BF16 gather for now as baseline because it's simpler
    if (
        not isinstance(inp, Float8BlockwiseQTensorBase)
        and quantizer is not None
        and not quantizer.is_quantizable(inp)
    ):
        out = torch.empty(
            out_shape,
            dtype=dtype,
            device=device,
            memory_format=torch.contiguous_format,
        )
        torch.distributed.all_gather_into_tensor(out, inp, group=process_group, async_op=False)
        orig_all_gather_usage = quantizer.all_gather_usage
        quantizer.all_gather_usage = False
        out = quantizer(out)
        quantizer.all_gather_usage = orig_all_gather_usage
        return out, None

    # Implementation of fp8 gather needs to account for:
    # * Getting columnwise data as a transpose of how it is stored for GEMMS.
    # * Gathering non GEMM swizzled scales.

    # Cast input tensor to Float8BlockwiseQTensor with required data
    # Set to compact usage in case the quantizer is not correctly configured
    orig_all_gather_usage = quantizer.all_gather_usage
    quantizer.all_gather_usage = True
    if not isinstance(inp, Float8BlockwiseQTensorBase):
        inp = quantizer(inp)
    elif (quantizer.rowwise_usage and inp._rowwise_data is None) or (
        quantizer.columnwise_usage and inp._columnwise_data is None
    ):
        warnings.warn(
            "Input and quantizer do not have matching usages. "
            "Dequantizing and requantizing to Float8BlockwiseQTensor."
        )
        inp = quantizer(inp.dequantize())
    quantizer.all_gather_usage = orig_all_gather_usage

    # Begin to do network communication, need to make sure compact format
    if inp._data_format != tex.Float8BlockScaleTensorFormat.COMPACT:
        raise RuntimeError(
            "All-gather with FP8 block-wise quantized tensor requires compact data format, "
            f"but found data_format={inp._data_format}"
        )

    # Construct Float8BlockwiseQTensor output tensor
    out = quantizer.make_empty(out_shape, dtype=dtype, device=device)

    # Coalesce NCCL collectives
    with torch.distributed._coalescing_manager(
        group=process_group,
        device=device,
        async_ops=async_op,
    ) as coalescing_manager:

        # Gather Float8BlockwiseQTensor data for row-wise usage
        if quantizer.rowwise_usage:
            # Launch all-gathers
            torch.distributed.all_gather_into_tensor(
                out._rowwise_scale_inv,
                inp._rowwise_scale_inv,
                group=process_group,
            )
            torch.distributed.all_gather_into_tensor(
                out._rowwise_data,
                inp._rowwise_data,
                group=process_group,
            )

        # Gather Float8BlockwiseQTensor data for column-wise usage
        if quantizer.columnwise_usage:
            # Launch all-gathers
            torch.distributed.all_gather_into_tensor(
                out._columnwise_scale_inv,
                inp._columnwise_scale_inv,
                group=process_group,
            )
            torch.distributed.all_gather_into_tensor(
                out._columnwise_data,
                inp._columnwise_data,
                group=process_group,
            )

    handle = coalescing_manager if async_op else None

    # Unlike MXFP8, this fp8 blockwise tensor primarily works with Hopper
    # This means that we need to transpose the gathered columnwise data
    # Example usage is grad_output tensor, ie. dY in linear backward
    # We want to gather two FP8 tensors (rowwise and columnwise) along dim0
    # and then transpose the columnwise data to match the rowwise data
    # Make sure FP8 transpose is populated if needed

    if async_op:
        handle = _FP8BlockwiseAllGatherAsyncHandle(out, quantizer, handle)
    else:
        # if it's a sync op, we need to do the transpose here as post processing step
        _post_process_fp8_blockwise_gather(out, quantizer, handle)

    return out, handle


def _all_gather_mxfp8(
    inp: torch.Tensor,
    process_group: dist_group_type,
    *,
    async_op: bool = False,
    quantizer: MXFP8Quantizer,
    out_shape: Optional[list[int]] = None,
) -> tuple[MXFP8TensorBase, Optional[torch.distributed.Work]]:
    """All-gather MXFP8 tensor along first dimension."""

    # Input tensor attributes
    in_shape: Iterable[int]
    device: torch.device
    dtype: torch.dtype
    if isinstance(inp, torch.Tensor):
        in_shape = inp.size()
        device = inp.device
        dtype = inp.dtype
    elif isinstance(inp, MXFP8TensorBase):
        if inp._rowwise_data is not None:
            in_shape = inp._rowwise_data.size()
            device = inp._rowwise_data.device
            dtype = inp._rowwise_data.dtype
        elif inp._columnwise_data is not None:
            in_shape = inp._columnwise_data.size()
            device = inp._columnwise_data.device
            dtype = inp._columnwise_data.dtype
        else:
            raise ValueError("Got MXFP8 input tensor without any data")
        dtype = torch.bfloat16
    else:
        raise ValueError(
            "Invalid type for input tensor (expected torch.Tensor or MXFP8TensorBase, "
            f"found {inp.__class__.__name__})"
        )

    # Output tensor shape
    world_size = get_distributed_world_size(process_group)
    if out_shape is None:
        out_shape = [in_shape[0] * world_size] + in_shape[1:]

    # For cases where inp has dimensions that cannot be quantized,
    # we gather in high precision followed by a cast to FP8.
    if (
        not isinstance(inp, MXFP8TensorBase)
        and quantizer is not None
        and not quantizer.is_quantizable(inp)
    ):
        out = torch.empty(
            out_shape,
            dtype=dtype,
            device=device,
            memory_format=torch.contiguous_format,
        )
        torch.distributed.all_gather_into_tensor(out, inp, group=process_group)
        out = quantizer(out)
        return out, None

    # Cast input tensor to MXFP8 with required data
    if not isinstance(inp, MXFP8TensorBase):
        inp = quantizer(inp)
    elif (quantizer.rowwise_usage and inp._rowwise_data is None) or (
        quantizer.columnwise_usage and inp._columnwise_data is None
    ):
        warnings.warn(
            "Input and quantizer do not have matching usages. "
            "Dequantizing and requantizing to MXFP8."
        )
        inp = quantizer(inp.dequantize())

    # Construct MXFP8 output tensor
    out = quantizer.make_empty(out_shape, dtype=dtype, device=device)

    # Coalesce NCCL collectives
    with torch.distributed._coalescing_manager(
        group=process_group,
        device=device,
        async_ops=async_op,
    ) as coalescing_manager:

        # Gather MXFP8 data for row-wise usage
        if quantizer.rowwise_usage:

            # Remove padding from MXFP8 scale-inverses
            in_scale_inv = inp._rowwise_scale_inv
            out_scale_inv = out._rowwise_scale_inv
            flattened_in_shape0 = math.prod(in_shape[:-1])
            if in_scale_inv.size(0) != flattened_in_shape0:
                in_scale_inv = in_scale_inv[:flattened_in_shape0]
                out_scale_inv[flattened_in_shape0 * world_size :].zero_()
                out_scale_inv = out_scale_inv[: flattened_in_shape0 * world_size]

            # Launch all-gathers
            torch.distributed.all_gather_into_tensor(
                out_scale_inv,
                in_scale_inv,
                group=process_group,
            )
            torch.distributed.all_gather_into_tensor(
                out._rowwise_data,
                inp._rowwise_data,
                group=process_group,
            )

        # Gather MXFP8 data for column-wise usage
        if quantizer.columnwise_usage:

            # Remove padding from MXFP8 scale-inverses
            in_scale_inv = inp._columnwise_scale_inv
            out_scale_inv = out._columnwise_scale_inv
            flattened_in_shape0 = math.prod(in_shape[:-1]) // 32
            if in_scale_inv.size(0) != flattened_in_shape0:
                in_scale_inv = in_scale_inv[:flattened_in_shape0]
                out_scale_inv[flattened_in_shape0 * world_size :].zero_()
                out_scale_inv = out_scale_inv[: flattened_in_shape0 * world_size]

            # Launch all-gathers
            torch.distributed.all_gather_into_tensor(
                out_scale_inv,
                in_scale_inv,
                group=process_group,
            )
            torch.distributed.all_gather_into_tensor(
                out._columnwise_data,
                inp._columnwise_data,
                group=process_group,
            )

    handle = coalescing_manager if async_op else None
    return out, handle


def gather_along_first_dim(
    inp: torch.Tensor,
    process_group: dist_group_type,
    async_op: bool = False,
    quantizer: Optional[Quantizer] = None,
) -> tuple[torch.Tensor, Optional[torch.distributed.Work]]:
    """
    All-gather tensors and concatenate along first dimension.
    """

    # Return immediately if no communication is required
    world_size = get_distributed_world_size(process_group)
    if world_size == 1:
        if quantizer is not None and not isinstance(inp, QuantizedTensor):
            inp = quantizer(inp)
        return inp, None

    # Output tensor dims
    out_shape = list(inp.size())
    out_shape[0] *= world_size

    # FP8 case: delayed scaling or current scaling
    if isinstance(inp, Float8TensorBase) or isinstance(
        quantizer, (Float8Quantizer, Float8CurrentScalingQuantizer)
    ):
        return _all_gather_fp8(
            inp,
            process_group,
            async_op=async_op,
            quantizer=quantizer,
            out_shape=out_shape,
        )

    # FP8 block scaling case, block length = 128
    if isinstance(inp, Float8BlockwiseQTensorBase) or isinstance(quantizer, Float8BlockQuantizer):
        return _all_gather_fp8_blockwise(
            inp,
            process_group,
            async_op=async_op,
            quantizer=quantizer,
            out_shape=out_shape,
        )

    # MXFP8 case
    if isinstance(inp, MXFP8TensorBase) or isinstance(quantizer, MXFP8Quantizer):
        assert isinstance(quantizer, MXFP8Quantizer)
        return _all_gather_mxfp8(
            inp,
            process_group,
            async_op=async_op,
            quantizer=quantizer,
            out_shape=out_shape,
        )

    # Debug case - call gather_along_first_dim on each tensor
    if isinstance(inp, DebugQuantizedTensor):
        out_obj = inp
        rowwise = inp.get_tensor(False)
        columnwise = inp.get_tensor(True)
        final_quantizer = (
            None if not needs_quantized_gemm(inp, rowwise=True) else quantizer.parent_quantizer
        )
        # Temporary fix for TP communication of Float8BlockwiseQTensorBase
        if isinstance(rowwise, Float8BlockwiseQTensorBase):
            rowwise = inp._original_tensor
        rowwise_total = gather_along_first_dim(rowwise, process_group, False, final_quantizer)[0]
        out_obj.rowwise_gemm_tensor = rowwise_total
        if rowwise is not columnwise:
            final_quantizer_columnwise = (
                None if not needs_quantized_gemm(inp, rowwise=False) else quantizer.parent_quantizer
            )
            # Temporary fix for TP communication of Float8BlockwiseQTensorBase
            if isinstance(columnwise, Float8BlockwiseQTensorBase):
                columnwise = inp._original_tensor
            columnwise_total, _ = gather_along_first_dim(
                columnwise, process_group, False, final_quantizer_columnwise
            )
            out_obj.columnwise_gemm_tensor = columnwise_total
        else:
            out_obj.rowwise_gemm_tensor = out_obj.rowwise_gemm_tensor
        return out_obj, None

    # High-precision communication for quantized tensors
    if quantizer is not None:
        warnings.warn(
            "Attempting to all-gather an unsupported quantized tensor. "
            "Falling back to high-precision all-gather."
        )
        if isinstance(inp, QuantizedTensor):
            inp = inp.dequantize()
        # Falling back to high-precision all-gather for Float8BlockQuantizer
        # means that it should directly output GEMM_READY format
        _set_quantizer_format(quantizer, compact=False)
        out = torch.empty(
            out_shape,
            dtype=inp.dtype,
            device=inp.device,
            memory_format=torch.contiguous_format,
        )
        torch.distributed.all_gather_into_tensor(out, inp, group=process_group)
        out = quantizer(out)
        return out, None

    # Dequantize quantized tensor if not supported
    if isinstance(inp, QuantizedTensor):
        warnings.warn(
            "Attempting to all-gather an unsupported quantized tensor. "
            "Falling back to high-precision all-gather."
        )
        inp = inp.dequantize()

    # Communication for plain PyTorch tensors
    out = torch.empty(
        out_shape,
        dtype=inp.dtype,
        device=inp.device,
        memory_format=torch.contiguous_format,
    )
    handle = torch.distributed.all_gather_into_tensor(
        out,
        inp.contiguous(),
        group=process_group,
        async_op=async_op,
    )
    return out, handle


# Global cache to store symmetric memory tensors
symmetric_mem_cache = {}


def get_symmetric_memory_tensor(tensor_numel, tensor_dtype, tensor_device, tp_group, tag=None):
    """
    Gets or creates a symmetric memory tensor with specified properties.

    Reuses cached tensors when available to avoid redundant creation and rendezvous operations.

    Note: This function always returns a 1D tensor.

    Parameters
    ----------
    tensor_numel : int
        Number of elements in the tensor.
    tensor_dtype : torch.dtype
        Data type of the tensor.
    tensor_device : torch.device
        Device on which to allocate the tensor.
    tp_group : dist_group_type
        Process group for rendezvous operation.
    tag : Any, optional
        Optional identifier to further distinguish tensors.

    Returns
    -------
    torch.Tensor
        A symmetric memory tensor with the specified properties.
    """
    # Create a cache key based on tensor properties and group
    cache_key = (tensor_numel, tensor_dtype, tensor_device, tp_group.group_name, tag)

    # Check if we already have a symmetric memory tensor for this configuration
    if cache_key not in symmetric_mem_cache:
        # Create a new symmetric memory tensor if not in cache
        msg = symm_mem.empty(
            tensor_numel,
            dtype=tensor_dtype,
            device=tensor_device,
        )
        # Perform the rendezvous once for this tensor
        symm_mem.rendezvous(msg, group=tp_group)
        # Store in cache
        symmetric_mem_cache[cache_key] = msg
    else:
        # Reuse the existing symmetric memory tensor
        msg = symmetric_mem_cache[cache_key]

    return msg


def symmetric_all_reduce(
    inp: torch.Tensor,
    tp_group: Optional[dist_group_type] = None,
    async_op: bool = False,
    all_reduce_type: str = "multimem_all_reduce",
):
    """
    Performs an all-reduce operation across multiple processes using symmetric memory.
    If the input tensor is already in the symmetric memory cache we can avoid copy
    overheads by just directly using the input tensor for all reduce.  Externally
    created symmetric memory tensors not in the cache currently will not be able to
    avoid the extra copies.

    Parameters
    ----------
    inp : torch.Tensor
        The input tensor to be reduced. The operation is performed in-place.

    tp_group : Optional[dist_group_type], default=None
        The process group over which to perform the all-reduce operation.
        If None, the default process group is used.

    async_op : bool, default=False
        Whether to perform the operation asynchronously.
        Note: Currently only synchronous operations are supported for symmetric memory variants.

    all_reduce_type : str, default="multimem_all_reduce"
        The type of all-reduce implementation to use. Options include:
        - "nccl": Standard PyTorch distributed all-reduce
        - "multimem_all_reduce": multimem symmetric all-reduce
        - "two_shot": Two-shot symmetric all-reduce
        - "one_shot": One-shot symmetric all-reduce

    Returns
    -------
    Tuple[torch.Tensor, Optional[torch.distributed.Work]]
        - The first element is the input tensor with the all-reduce result.
        - The second element is the async work handle if async_op=True,
          otherwise None.
    """
    assert async_op is False, "Async symmetric ops no supported yet"
    assert HAS_TORCH_SYMMETRIC, "Could not import symetric memory from torch"

    if get_distributed_world_size(tp_group) == 1:
        return inp, None

    if all_reduce_type == "nccl":
        # Standard all-reduce implementation
        handle = torch.distributed.all_reduce(inp, group=tp_group, async_op=async_op)
        return inp, handle

    all_reduce_impl = None
    if all_reduce_type == "multimem_all_reduce":
        all_reduce_impl = torch.ops.symm_mem.multimem_all_reduce_
    elif all_reduce_type == "two_shot":
        all_reduce_impl = torch.ops.symm_mem.two_shot_all_reduce_
    elif all_reduce_type == "one_shot":
        all_reduce_impl = torch.ops.symm_mem.one_shot_all_reduce
    else:
        raise TypeError(f"All reduce type {all_reduce_type} is not supported.")

    group_name = tp_group.group_name
    tensor_shape = inp.shape
    tensor_numel = inp.numel()
    tensor_dtype = inp.dtype
    tensor_device = inp.device

    input_id = id(inp)
    is_cached = any(id(cached_tensor) == input_id for cached_tensor in symmetric_mem_cache.values())
    # Check if the input tensor is already in the symmetric memory cache. If it is we can avoid copy overheads.
    if is_cached:
        all_reduce_impl(
            inp,
            "sum",
            group_name,
        )
    else:
        # Get symmetric memory tensor. Build or retrieve from cache.
        msg = get_symmetric_memory_tensor(tensor_numel, tensor_dtype, tensor_device, tp_group)

        msg.copy_(inp.reshape(-1))

        all_reduce_impl(
            msg,
            "sum",
            group_name,
        )

        # Copy the result back to the input tensor
        inp.copy_(msg.reshape(tensor_shape))

    return inp, None


def allreduce(
    inp: torch.Tensor,
    tp_group: Optional[dist_group_type] = None,
    async_op: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.distributed.Work]]:
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_distributed_world_size(tp_group) == 1:
        return inp, None

    # All-reduce.
    handle = torch.distributed.all_reduce(inp, group=tp_group, async_op=async_op)

    return inp, handle


def _fsdp_scatter_tensors(
    fsdp_group: dist_group_type,
    *tensors: torch.Tensor,
):
    shapes = []
    if fsdp_group is not None:
        for t in tensors:
            if isinstance(t, torch.Tensor):
                targets = t.get_data_tensors() if isinstance(t, QuantizedTensor) else [t]
                for target in targets:
                    shapes.append(target.data.shape)
                    safely_set_viewless_tensor_data(
                        target,
                        split_tensor_into_1d_equal_chunks(target.data, fsdp_group, new_buffer=True),
                    )
            else:
                shapes.append(None)
    return shapes


def _fsdp_gather_tensors(
    fsdp_group: dist_group_type,
    shapes: List[Tuple[int, ...]],
    *tensors: torch.Tensor,
):
    if fsdp_group is not None:
        assert len(shapes) == len(tensors), "Number of tensors and tensor shapes must be equal."
        for s, t in zip(shapes, tensors):
            if isinstance(t, torch.Tensor):
                assert s is not None, "Internal TE error."
                targets = t.get_data_tensors() if isinstance(t, QuantizedTensor) else [t]
                for target in targets:
                    safely_set_viewless_tensor_data(
                        target, gather_split_1d_tensor(target.data, fsdp_group).view(s)
                    )


def _is_te_module(module):
    """
    Check if given module is a Transformer Engine module that requires the TE checkpoint
    implementation for activation recompute.
    """
    from .module import LayerNorm, RMSNorm
    from .module.base import TransformerEngineBaseModule
    from .attention.dot_product_attention.dot_product_attention import DotProductAttention
    from .attention.dot_product_attention.backends import UnfusedDotProductAttention
    from .attention.multi_head_attention import MultiheadAttention
    from .transformer import TransformerLayer

    te_classes_list = [
        LayerNorm,
        RMSNorm,
        TransformerEngineBaseModule,
        UnfusedDotProductAttention,
        DotProductAttention,
        MultiheadAttention,
        TransformerLayer,
    ]
    is_te_module = False
    for te_class in te_classes_list:
        if isinstance(module, te_class):
            is_te_module = True
            break
    return is_te_module


def prepare_te_modules_for_fsdp(fsdp_root: torch.nn.Module) -> None:
    """
    Inject FSDP process gorup references into FSDP-wrapped TE modules in an FSDP-wrapped root
    module in order to scatter/gather the Fp8 weight copies at the same time FSDP scatters/gathers
    its `FlatParameters`.

    Parameters
    ----------
    fsdp_root: torch.nn.Module
               FSDP-wrapped root module that may contain FSDP-wrapped TE modules.
    """
    assert isinstance(fsdp_root, FSDP), "Root module must be FSDP-wrapped."

    # If the root module is a TE module, inject FSDP information into it
    if _is_te_module(fsdp_root.module):
        if hasattr(fsdp_root, "primary_weights_in_fp8"):
            assert not fsdp_root.primary_weights_in_fp8, (
                "TE modules with primary weights in FP8 cannot be FSDP-wrapped. "
                "Please initialize your model without the te.fp8_model_init(...) context."
            )
        root_state = _get_module_fsdp_state(fsdp_root)
        assert root_state is not None, "Root module does not have a valid _FSDPState."
        setattr(fsdp_root.module, "fsdp_group", root_state.process_group)

    # Iterate through all FSDP-wrapped submodules and inject FSDP information into TE modules
    fsdp_states, fsdp_modules = _get_fsdp_states_with_modules(fsdp_root)
    for state, fsdp_module in zip(fsdp_states, fsdp_modules):
        if _is_te_module(fsdp_module.module):
            if hasattr(fsdp_module.module, "primary_weights_in_fp8"):
                assert not fsdp_module.module.primary_weights_in_fp8, (
                    "TE modules with primary weights in FP8 cannot be FSDP-wrapped. "
                    "Please initialize your model without the te.fp8_model_init(...) context."
                )
            setattr(fsdp_module.module, "fsdp_group", state.process_group)


class FullyShardedDataParallel(FSDP):
    """
    Transformer Engine wrapper around `torch.distributed.fsdp.FullyShardedDataParallel` that
    extracts necessary information out of the FSDP wrap for TE modules to scatter their
    activation tensors after each forward pass and gather them before the backward pass.
    """

    def __init__(self, module, *args, **kwargs):
        super().__init__(module, *args, **kwargs)
        prepare_te_modules_for_fsdp(self)
