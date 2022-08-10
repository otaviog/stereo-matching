import torch

from stereomatch._cstereomatch import (AggregationOps as _AggregationOps)


def _is_power_of_two(n):
    """
    https://stackoverflow.com/a/57027610
    """
    return (n != 0) and (n & (n-1) == 0)


class WinnerTakesAll:
    def estimate(self, cost_volume: torch.Tensor) -> torch.Tensor:
        disparity_img = torch.empty(
            cost_volume.size(0), cost_volume.size(1),
            dtype=torch.int32, device=cost_volume.device)

        if cost_volume.is_cuda and not _is_power_of_two(cost_volume.size(2)):
            raise RuntimeError(
                ("Winner takes all requires max disparity "
                 "(`cost_volume.size(2)`) to be a power of 2."))

        _AggregationOps.run_winners_take_all(cost_volume, disparity_img)
        return disparity_img


class DynamicProgramming:
    def estimate(self, cost_volume):
        # TODO: post dynamic programming to use DxHxW order.
        height, width, max_disparity = cost_volume.shape

        disparity_img = torch.empty(
            height, width,
            dtype=torch.int32, device=cost_volume.device)

        # Auxiliar data
        path_volume = torch.empty(
            *cost_volume.size(),
            dtype=torch.int8, device=cost_volume.device)
        row_final_costs = torch.empty(
            height, max_disparity,
            dtype=cost_volume.dtype,
            device=cost_volume.device
        )

        _AggregationOps.run_dynamic_programming(
            cost_volume, path_volume, row_final_costs,
            disparity_img)

        return disparity_img


class SemiglobalAggregation:
    def estimate(self, cost_volume, left_image):
        sga_volume = torch.zeros_like(cost_volume)

        _AggregationOps.run_semiglobal(
            cost_volume, left_image,
            0.1, 0.2,
            sga_volume)

        return sga_volume
