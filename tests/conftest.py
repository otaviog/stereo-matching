from pathlib import Path
import dataclasses

import numpy as np
from PIL import Image
import pytest
import torch

import stereomatch


@dataclasses.dataclass
class CostFixture:
    volume: torch.Tensor
    left_image: torch.Tensor

    def to(self, device):
        return CostFixture(self.volume.to(device), self.left_image.to(device))


@pytest.fixture
def ssd_cost():
    cache_file = Path(__file__).parent / \
        "test_cache/cost_volume_teddy.torch"

    image_base_dir = Path(__file__).parent.parent / \
        "test-data/middleburry/teddy/"

    left_image = torch.from_numpy(
        np.array(Image.open(image_base_dir / "im2.png").convert('L')))

    if cache_file.exists():
        return CostFixture(volume=torch.load(str(cache_file)),
                           left_image=left_image)

    right_image = torch.from_numpy(
        np.array(Image.open(image_base_dir / "im6.png").convert('L')))

    cost_volume = stereomatch.cost.ssd(
        left_image, right_image, 128)

    cache_file.parent.mkdir(exist_ok=True, parents=True)
    torch.save(cost_volume, str(cache_file))
    return CostFixture(volume=cost_volume, left_image=left_image)


def pytest_configure():
    pytest.STM_MAX_DISPARITY = 32
