"""Functions to generate matching costs
"""

from typing import Optional

import torch

from .cuda_texture import CUDATexture
from ._cstereomatch import CostOps as _CostOps
from ._common import empty_tensor


class SSD:
    """
    Attributes:

        max_disparity: The maximum disparity that it should compute the costs for.
        kernel_size: The SSD kernel size.
    """

    def __init__(self, max_disparity: int, kernel_size: int = 7,
                 cost_volume_dtype: torch.dtype = torch.float):
        self.max_disparity = max_disparity
        self.kernel_size = kernel_size
        self.cost_volume_dtype = cost_volume_dtype

    def __call__(self, left_image: torch.Tensor, right_image: torch.Tensor,
                 cost_volume: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the sum of squared distances cost for a given pair of rectified images.

        Args:
            left_image: A [HxW] tensor representating the left image.
            right_image: A [HxW] tensor representating the right image.
            cost_volume: If passed, it will use this tensor as output tensor instead
             of allocating a new one.

        Return:
            A cost volume with shape [HxWx max_disparity].
        """
        cost_volume = empty_tensor(left_image.size(0), left_image.size(1), self.max_disparity,
                                   dtype=self.cost_volume_dtype,
                                   device=left_image.device,
                                   reuse_tensor=cost_volume)
        _CostOps.compute_ssd(left_image, right_image, cost_volume,
                             self.kernel_size)

        return cost_volume


class SSDTexture:
    """
    Attributes:

        max_disparity: The maximum disparity that it should compute the costs for.
        kernel_size: The SSD kernel size.
    """

    def __init__(self, max_disparity: int, kernel_size: int = 7):
        self.max_disparity = max_disparity
        self.kernel_size = kernel_size

    def __call__(self, left_image: CUDATexture, right_image: CUDATexture, cost_volume: Optional[torch.Tensor] = None) -> torch.Tensor:

        if left_image.use_normalized_coords or right_image.use_normalized_coords:
            raise RuntimeError(
                "Texture coordinates can't be normalized for this implementation")
        cost_volume = empty_tensor(left_image.height, left_image.width,
                                   self.max_disparity,
                                   dtype=torch.float32,
                                   device=torch.device("cuda:0"),
                                   reuse_tensor=cost_volume)
        _CostOps.compute_ssd(left_image, right_image,
                             cost_volume, self.kernel_size)

        return cost_volume


class Birchfield:
    """
    Attributes:

        max_disparity: The maximum disparity that it should compute the costs for.
        kernel_size: The SSD kernel size.
    """

    def __init__(self, max_disparity: int, kernel_size: int = 7):
        self.max_disparity = max_disparity
        self.kerne_size = kernel_size

    def __call__(self, left_image: torch.Tensor, right_image: torch.Tensor,
                 cost_volume: Optional[torch.Tensor] = None) -> torch.Tensor:
        cost_volume = empty_tensor(left_image.size(0), left_image.size(1), self.max_disparity,
                                   dtype=torch.float32,
                                   device=left_image.device,
                                   reuse_tensor=cost_volume)
        _CostOps.compute_birchfield(
            left_image.float(), right_image.float(), cost_volume, 4)

        return cost_volume
