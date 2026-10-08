"""One numerical policy for forward, backward and checkpoint recomputation."""
from contextlib import contextmanager
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


def configure_fp32():
    if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
        raise RuntimeError("Autocast must be disabled for bridge optimization")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


@contextmanager
def fp32_context():
    configure_fp32()
    with sdpa_kernel(SDPBackend.MATH):
        yield


def checkpoint_contexts():
    return fp32_context(), fp32_context()


def check_tensor_tree(value, *, finite=True):
    if isinstance(value, torch.Tensor):
        if value.is_complex():
            raise TypeError("Complex state is unsupported")
        if value.is_floating_point():
            if value.dtype != torch.float32:
                raise TypeError(f"Expected FP32, got {value.dtype}")
            if finite and value.device.type != "meta" and not torch.isfinite(value).all():
                raise FloatingPointError("Nonfinite tensor")
    elif isinstance(value, Mapping):
        for item in value.values():
            check_tensor_tree(item, finite=finite)
    elif isinstance(value, (tuple, list)):
        for item in value:
            check_tensor_tree(item, finite=finite)


def check_module(module: nn.Module, *, finite=True):
    check_tensor_tree(tuple(module.parameters()), finite=finite)
    check_tensor_tree(tuple(module.buffers()), finite=finite)


def freeze(module: nn.Module):
    """Freeze parameters, not the module's derivatives with respect to inputs."""
    check_module(module)
    module.eval().requires_grad_(False)
    for parameter in module.parameters():
        parameter.grad = None
    for child in module.modules():
        for name, value in child.named_buffers(recurse=False):
            if value.requires_grad:
                child._buffers[name] = value.detach()
    return module
