"""
Common testing.
"""
from pathlib import Path
import dataclasses

import numpy as np
from PIL import Image
import pytest
import torch

import stereomatch


@pytest.fixture
def sample_stereo_pair():
    """
    Fixture with sample stereo pair to use during testing.
    """
    image_base_dir = (Path(__file__).parent.parent /
                      "tests/data/middleburry/teddy/")

    target_size = (512, 256)
    left_image = torch.from_numpy(
        np.array(Image.open(image_base_dir /
                            "im2.png").convert('L').resize(target_size))).float() / 255.0
    right_image = torch.from_numpy(
        np.array(Image.open(image_base_dir /
                 "im6.png").convert('L').resize(target_size))).float() / 255.0

    return left_image, right_image


@dataclasses.dataclass
class CostFixture:
    """
    Input for aggregation and disparity reduce methods.

    Attributes:
        volume: A cost volume with shape [HxWxD], where D is the maximum disparity.
        left_image: The left image that some methods like SGM uses to adapt its calculations.
    """
    volume: torch.Tensor
    left_image: torch.Tensor

    def to(self, device):
        """
        Returns a new instance with its tensor attributes to the given device.
        """
        return CostFixture(self.volume.to(device), self.left_image.to(device))


@pytest.fixture
def ssd_cost():
    """
    Fixture for aggregation and disparity reduce methods to have a common SSD cost volume to test.
    """
    cache_file = (Path(__file__).parent /
                  "test_cache/cost_volume_teddy.torch")

    image_base_dir = (Path(__file__).parent.parent /
                      "tests/data/middleburry/teddy/")

    left_image = torch.from_numpy(
        np.array(Image.open(image_base_dir / "im2.png").convert('L')))

    if cache_file.exists():
        return CostFixture(volume=torch.load(str(cache_file)),
                           left_image=left_image)

    right_image = torch.from_numpy(
        np.array(Image.open(image_base_dir / "im6.png").convert('L')))

    cost_volume = stereomatch.cost.SSD(128)(left_image, right_image)

    cache_file.parent.mkdir(exist_ok=True, parents=True)
    torch.save(cost_volume, str(cache_file))
    return CostFixture(volume=cost_volume, left_image=left_image)


def pytest_configure():
    """
    Configures global variables used during testing.
    """
    pytest.STM_TEST_OUTPUT_PATH = Path(__file__).parent / "test-result"
    pytest.STM_MAX_DISPARITY = 32
