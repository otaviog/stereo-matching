"""
Implements methods for aggregating the cost volumes.
"""
from typing import Optional

import torch

from stereomatch._cstereomatch import (AggregationOps as _AggregationOps)
from ._common import zeros_tensor_like


class Semiglobal:
    """
    The semiglobal aggregation purposed in
    Hirschmuller, Heiko. "Accurate and efficient stereo processing by semi-global matching and
    mutual information." In 2005 IEEE Computer Society Conference on Computer Vision and Pattern
    Recognition (CVPR'05), vol. 2, pp. 807-814. IEEE, 2005.

    This implementation uses:
        - 6 path directions: Horizontal (left-right and right-left),
          Vertical (top-bottom and bottom-top) and diagonals (forward and backward).
        - Adaptive second penalty based on the image gradient.

    """

    def __init__(self, penalty1: float = 0.1, penalty2: float = 0.2):
        """
        Args:
            penalty1: The cost penalty for changing the disparity by one level.
            penalty2: The cost penalty for changing the disparity to other levels.
        """
        self.penalty1 = penalty1
        self.penalty2 = penalty2

    def __call__(self, cost_volume: torch.Tensor, left_image: torch.Tensor,
                 sga_volume: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Executes the algorithm. Node that the dispairty dimension (D) must be power of two
        for GPU tensors.

        Args:
            cost_volume: The input cost volume, shape is [HxWxD]
            left_image: The left image. Shape must be [HxW].
            sga_volume: The aggregation output. Shape must be[HxWxD]. If a value is passed, the
             function will try to reuse its space.

        Returns:
            The SGA aggregation volume. If the `sga_volume` could be reused,
             its same pointer is returned, else a new allocated one is return.
        """
        sga_volume = zeros_tensor_like(cost_volume, reuse_tensor=sga_volume)
        _AggregationOps.run_semiglobal(
            cost_volume, left_image,
            self.penalty1, self.penalty2,
            sga_volume)

        return sga_volume
