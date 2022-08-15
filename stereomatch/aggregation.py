from typing import Optional

import torch

from stereomatch._cstereomatch import (AggregationOps as _AggregationOps)
from ._common import zeros_tensor_like


class Semiglobal:
    def __init__(self, penalty1: float = 0.1, penalty2: float = 0.2):
        self.penalty1 = penalty1
        self.penalty2 = penalty2

    def __call__(self, cost_volume: torch.Tensor, left_image: torch.Tensor,
                 sga_volume: Optional[torch.Tensor] = None) -> torch.Tensor:
        sga_volume = zeros_tensor_like(cost_volume, reuse_tensor=sga_volume)

        _AggregationOps.run_semiglobal(
            cost_volume, left_image,
            self.penalty1, self.penalty2,
            sga_volume)

        return sga_volume
