from typing import Optional

import torch

from stereomatch._cstereomatch import (
    DisparityReduceOps as _DisparityReduceOps)
from ._common import empty_tensor


def _is_power_of_two(n):
    """
    https://stackoverflow.com/a/57027610
    """
    return (n != 0) and (n & (n-1) == 0)


class WinnerTakesAll:
    """
    Implements the winner takes all disparity computation which takes the
     disparity with lowest cost.
    """

    def __call__(self, cost_volume: torch.Tensor, disparity_img: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the disparity.

        Args:
            cost_volume: A [HxWxD] tensor with the matching cost per pixel at different disparities.
            disparity_img: A optional [HxW] tensor to which the disparities are written. If not supplied
             or the expected tensor properities (shape, device, type) does not match, then the function will allocate
             a new tensor to output.

        Returns:
            A disparity map with shape [HxW] and type int32.
        """
        disparity_img = empty_tensor(
            cost_volume.size(0), cost_volume.size(1),
            dtype=torch.int32, device=cost_volume.device, reuse_tensor=disparity_img)

        if cost_volume.is_cuda and not _is_power_of_two(cost_volume.size(2)):
            raise RuntimeError(
                ("Winner takes all requires max disparity "
                 "(`cost_volume.size(2)`) to be a power of 2."))

        _DisparityReduceOps.run_winners_take_all(cost_volume, disparity_img)
        return disparity_img


class DynamicProgramming:
    """
    Uses the classical technique of dynamic programming to compute the disparity map.
    """

    def __call__(self, cost_volume: torch.Tensor, disparity_img: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the disparity.

        Args:
            cost_volume: A [HxWxD] tensor with the matching cost per pixel at different disparities.
            disparity_img: A optional [HxW] tensor to which the disparities are written. If not supplied
             or the expected tensor properities (shape, device, type) does not match, then the function will allocate
             a new tensor to output.

        Returns:
            A disparity map with shape [HxW] and type int32.
        """

        height, width, max_disparity = cost_volume.shape

        disparity_img = empty_tensor(
            height, width,
            dtype=torch.int32, device=cost_volume.device,
            reuse_tensor=disparity_img)

        # Auxiliar data
        path_volume = torch.empty(
            *cost_volume.size(),
            dtype=torch.int8, device=cost_volume.device)
        row_final_costs = torch.empty(
            height, max_disparity,
            dtype=cost_volume.dtype,
            device=cost_volume.device
        )

        _DisparityReduceOps.run_dynamic_programming(
            cost_volume, path_volume, row_final_costs,
            disparity_img)

        return disparity_img
