"""
Pipeline to compose stereo matching.
"""

from typing import Callable, Optional, Union

import torch

import stereomatch
from .cost import SSDTexture
from .cuda_texture import CUDATexture


TensorOrTexture = Union[torch.Tensor, CUDATexture]
CostFunction = Callable[[TensorOrTexture, TensorOrTexture, torch.Tensor],
                        torch.Tensor]
AggregationFunction = Callable[[torch.Tensor, torch.Tensor, torch.Tensor],
                               torch.Tensor]
DisparityReduceFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class _TexCostFunctionWrapper:
    def __init__(self, cost_function):
        self.cost_function = cost_function

    def __call__(self, left_image: torch.Tensor,
                 right_image: torch.Tensor, cost_volume: torch.Tensor):
        return self.cost_function(
            stereomatch.cuda_texture.CUDATexture.from_tensor(
                left_image.cuda()),
            stereomatch.cuda_texture.CUDATexture.from_tensor(
                right_image.cuda()),
            cost_volume=cost_volume)


class Pipeline:
    """
    Class for composing stereo matching pipelines.
    """

    def __init__(self, cost: CostFunction,
                 disparity_reduce: DisparityReduceFunction,
                 aggregation: Optional[AggregationFunction] = None):
        """
        Setups the pipeline.

        Args:
            cost: The cost function, it should accept the left and right images,
             and a output cost volume. It should return a [HxWxD] cost volume tensor.
            disparity_reduce: The disparity reduce method. It should accept a cost volume,
             a output disparity image, and should return a [HxW] disparity image.
            aggregation: The aggregation function. It accepts the cost volume, the left image,
             and the output aggregation volume.
        """
        if isinstance(cost, SSDTexture):
            self.cost = _TexCostFunctionWrapper(cost)
        else:
            self.cost = cost

        self.disparity_reduce = disparity_reduce
        self.aggregation = aggregation

        self.aggregation = aggregation

        self._cost_volume = None
        self._aggregation_volume = None
        self._disparity_image = None

    def estimate(self, left_image: torch.Tensor, right_image: torch.Tensor) -> torch.Tensor:
        """
        Executes the pipeline.

        Args:
            left_image: The left image. It shape and type must be compatible with the cost method.
            right_image: The right image. It shape and type must be compatible with the cost method.

        Returns:
            The disparity map.
        """
        self._cost_volume = self.cost(left_image, right_image,
                                      cost_volume=self._cost_volume)

        if self.aggregation is not None:
            # TODO: fix the Double upload of left_image to GPU
            self._aggregation_volume = self.aggregation(
                self._cost_volume, left_image.to(self._cost_volume.device),
                self._aggregation_volume)
        else:
            self._aggregation_volume = self._cost_volume

        self._disparity_image = self.disparity_reduce(
            self._aggregation_volume, self._disparity_image)

        return self._disparity_image
