"""Function to generate matching costs
"""
import torch

from ._cstereomatch import CostOps as _CostOps


def ssd(left_image, right_image, max_disparity, kernel_size=7):
    cost_volume = torch.full(
        (max_disparity, left_image.size(0), left_image.size(1)), float("inf"),
        dtype=torch.float, device=left_image.device)
    _CostOps.compute_ssd(left_image.float(),
                         right_image.float(), cost_volume, kernel_size)

    return cost_volume

def ssd_texture(left_image, right_image, max_disparity, kernel_size=7):
    cost_volume = torch.full(
        (max_disparity, left_image.height, left_image.width), float("inf"),
        dtype=torch.float, device="cuda:0")
    _CostOps.compute_ssd(left_image, right_image, cost_volume, kernel_size)

    return cost_volume
