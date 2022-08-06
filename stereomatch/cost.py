"""Functions to generate matching costs
"""

from typing import Optional

import torch

from ._cstereomatch import CostOps as _CostOps
from ._common import empty_tensor

def ssd(left_image: torch.Tensor, right_image: torch.Tensor, max_disparity: int,
        kernel_size: int = 7, cost_volume: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the sum of squared distances cost for a given pair of rectified images.

    Args:
        left_image: A [HxW] tensor representating the left image.
        right_image: A [HxW] tensor representating the right image.
        max_disparity: The maximum disparity that it should compute the costs for.
        kernel_size: The SSD kernel size.
        cost_volume: If passed, it will use this tensor as output tensor instead
         of allocating a new one.

    Return:
        A cost volume with shape [HxWx max_disparity].
    """
    cost_volume = empty_tensor(left_image.size(0), left_image.size(1), max_disparity,
                               dtype=torch.float32,
                               device=left_image.device,
                               reuse_tensor=cost_volume)
    _CostOps.compute_ssd(left_image.float(),
                         right_image.float(), cost_volume, kernel_size)

    return cost_volume


def ssd_texture(left_image, right_image, max_disparity, kernel_size=7, cost_volume: Optional[torch.Tensor] = None):
    cost_volume = empty_tensor(left_image.height, left_image.width,
                               max_disparity,
                               dtype=torch.float32,
                               device=torch.device("cuda:0"),
                               reuse_tensor=cost_volume)
    _CostOps.compute_ssd(left_image, right_image, cost_volume, kernel_size)

    return cost_volume
