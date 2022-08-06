from typing import Optional

import torch


def empty_tensor(*sizes, dtype: torch.dtype = torch.float32,
                 reuse_tensor: Optional[torch.Tensor] = None,
                 device: torch.device = torch.device("cpu")) -> torch.Tensor:
    if reuse_tensor is None or (reuse_tensor.size() != sizes
                                or dtype != reuse_tensor.dtype
                                or dtype != reuse_tensor.device):
        return torch.empty(sizes, dtype=dtype, device=device)
    return reuse_tensor
