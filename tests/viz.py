"""
Visualization utilities for the tests.
"""

from pathlib import Path

import torch
import matplotlib.pyplot as plt


def save_depthmap(depthmap: torch.Tensor, exp_prefix: str):
    """
    Saves a depthmap figure.

    Args:
        depthmap: The resulting depthmap, it will use its device to determine
         the image's file suffix.
        exp_prefix: The image file prefix. Its suffix is determined by the
         depthmap device.
    """
    plt.figure()
    plt.imshow(depthmap.cpu().numpy())
    plt.axis("off")

    device = "gpu" if depthmap.is_cuda else "cpu"
    Path(exp_prefix).parent.mkdir(exist_ok=True, parents=True)
    exp_prefix = str(exp_prefix)
    plt.savefig(f'{exp_prefix}-{device}.png')
    plt.close()
