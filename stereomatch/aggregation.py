import torch

from stereomatch._cstereomatch import (AggregationOps as _AggregationOps)


class WinnerTakesAll:
    def estimate(self, cost_volume):
        disparity_img = torch.empty(
            cost_volume.size(0), cost_volume.size(1),
            dtype=torch.int32, device=cost_volume.device)

        _AggregationOps.run_winners_take_all(cost_volume, disparity_img)
        return disparity_img


class DynamicProgramming:
    def estimate(self, cost_volume):
        # TODO: post dynamic programming to use DxHxW order.
        cost_volume = cost_volume.permute(2, 0, 1)
        disparity_img = torch.empty(
            cost_volume.size(1), cost_volume.size(2),
            dtype=torch.int32, device=cost_volume.device)

        # Auxiliar data
        path_volume = torch.empty(
            *cost_volume.size(),
            dtype=torch.int8, device=cost_volume.device)
        disp_costsum_per_row = torch.empty(
            cost_volume.size(0), cost_volume.size(1),
            dtype=cost_volume.dtype,
            device=cost_volume.device
        )

        _AggregationOps.run_dynamic_programming(
            cost_volume, path_volume, disp_costsum_per_row,
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
