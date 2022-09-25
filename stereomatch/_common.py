"""
Common utils used internally in the project.
"""
from typing import Optional

import torch


def _is_tensor_incompatible(reuse_tensor, sizes, dtype, device):
    return (reuse_tensor is None or reuse_tensor.size() != sizes
            or dtype != reuse_tensor.dtype
            or device != reuse_tensor.device)


def empty_tensor(*sizes, dtype: torch.dtype = torch.float32,
                 reuse_tensor: Optional[torch.Tensor] = None,
                 device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Returns a `torch.empty` tensor trying to reuse the `reuse_tensor` if possible.

    Args:
        sizes: The tensor shape.
        dtype: The tensor type.
        reuse_tensor: If possible, it will fill with zeros this tensor and return it.
        device: The tensor device.

    Returns:
        An empty tensor. If both tensors are compatible, then it will return the `reuse_tensor`.
    """
    if _is_tensor_incompatible(reuse_tensor, sizes, dtype, device):
        del reuse_tensor
        return torch.empty(sizes, dtype=dtype, device=device)
    return reuse_tensor


def zeros_tensor_like(base_tensor, reuse_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Returns a `torch.zeros_like` tensor trying to reuse the `reuse_tensor` if possible.

    Args:
        base_tensor: Creates (or reuse) a tensor with the same sizes, type and in the same device.
        reuse_tensor: If possible, it will fill with zeros this tensor and return it.

    Returns:
        A zeros tensor. If both tensors are compatible, then it will return the `reuse_tensor`.
    """
    if _is_tensor_incompatible(reuse_tensor, base_tensor.size(),
                               base_tensor.dtype, base_tensor.device):
        del reuse_tensor
        return torch.zeros_like(base_tensor)

    reuse_tensor.zero_()
    return reuse_tensor
